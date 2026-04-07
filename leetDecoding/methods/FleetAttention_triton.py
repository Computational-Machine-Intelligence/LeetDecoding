import triton.language as tl
import math
import torch
import triton



@triton.jit
def FleetAttention_kernel_v2(B_ptr, C_ptr, V_ptr, ans_ptr,
               seqlen: tl.constexpr,
               dim: tl.constexpr,
               rank: tl.constexpr,
               stride_vbh: tl.constexpr,
               stride_bbh: tl.constexpr,
               dim_BLOCK: tl.constexpr):
    # grid: (bsz*heads, dim//dim_BLOCK)
    # rank and seq loops are both inside — no race condition, supports all dtypes.
    # ans_ptr points to a float32 accumulation buffer to avoid bf16 truncation per rank iter.
    bz = tl.program_id(axis=0)
    dim_block_idx = tl.program_id(axis=1)
    off_dim = tl.arange(0, dim_BLOCK)

    for rank_idx in range(rank):
        cv = tl.zeros([1, dim_BLOCK], dtype=tl.float32)
        for seq_idx in range(seqlen):
            offs_bc = bz * stride_bbh + seq_idx * rank + rank_idx
            v_ptrs = V_ptr + bz * stride_vbh + seq_idx * dim + dim_block_idx * dim_BLOCK + off_dim[None, :]
            ans_ptrs = ans_ptr + bz * stride_vbh + seq_idx * dim + dim_block_idx * dim_BLOCK + off_dim[None, :]

            b = tl.load(B_ptr + offs_bc).to(tl.float32)
            c = tl.load(C_ptr + offs_bc).to(tl.float32)
            v = tl.load(v_ptrs, mask=(off_dim[None, :] < dim), other=0).to(tl.float32)

            cv = c * v + cv
            o = b * cv

            # ans_ptr is float32 buffer: no precision loss across rank iterations
            ans = tl.load(ans_ptrs, mask=(off_dim[None, :] < dim), other=0.0)
            tl.store(ans_ptrs, ans + o, mask=(off_dim[None, :] < dim))


@triton.jit
def FleetAttention_with_decay_kernel_v2(B_ptr, C_ptr, V_ptr, gamma_ptr, ans_ptr,
               heads: tl.constexpr,
               seqlen: tl.constexpr,
               dim: tl.constexpr,
               rank: tl.constexpr,
               stride_vbh: tl.constexpr,
               stride_bbh: tl.constexpr,
               dim_BLOCK: tl.constexpr):
    # grid: (bsz*heads, dim//dim_BLOCK)
    # ans_ptr is a float32 accumulation buffer.
    bz = tl.program_id(axis=0)
    dim_block_idx = tl.program_id(axis=1)
    off_dim = tl.arange(0, dim_BLOCK)

    head_idx = bz % heads
    gamma = tl.load(gamma_ptr + head_idx).to(tl.float32)

    for rank_idx in range(rank):
        cv = tl.zeros([1, dim_BLOCK], dtype=tl.float32)
        for seq_idx in range(seqlen):
            offs_bc = bz * stride_bbh + seq_idx * rank + rank_idx
            v_ptrs = V_ptr + bz * stride_vbh + seq_idx * dim + dim_block_idx * dim_BLOCK + off_dim[None, :]
            ans_ptrs = ans_ptr + bz * stride_vbh + seq_idx * dim + dim_block_idx * dim_BLOCK + off_dim[None, :]

            b = tl.load(B_ptr + offs_bc).to(tl.float32)
            c = tl.load(C_ptr + offs_bc).to(tl.float32)
            v = tl.load(v_ptrs, mask=(off_dim[None, :] < dim), other=0).to(tl.float32)

            cv = c * v + cv * gamma
            o = b * cv

            ans = tl.load(ans_ptrs, mask=(off_dim[None, :] < dim), other=0.0)
            tl.store(ans_ptrs, ans + o, mask=(off_dim[None, :] < dim))


def FleetAttention_triton(B, C, V, gamma=None):
    bsz, heads, seqlen, rank = B.shape
    dim = V.shape[-1]
    dim_BLOCK = 32
    B = B.contiguous()
    C = C.contiguous()
    V = V.contiguous()
    orig_dtype = B.dtype
    # Use float32 accumulation buffer to avoid bf16 truncation across rank iterations
    ans = torch.zeros((bsz, heads, seqlen, dim), device=B.device, dtype=torch.float32)
    grid = (bsz * heads, dim // dim_BLOCK)
    if gamma is None:
        FleetAttention_kernel_v2[grid](B, C, V, ans,
                         seqlen, dim, rank, seqlen * dim, seqlen * rank, dim_BLOCK)
    else:
        FleetAttention_with_decay_kernel_v2[grid](B, C, V, gamma, ans,
                         heads, seqlen, dim, rank, seqlen * dim, seqlen * rank, dim_BLOCK)
    return ans.to(orig_dtype)
