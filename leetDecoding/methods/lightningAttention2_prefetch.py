# Copyright (c) 2024 Doraemonzzz

"""Lightning Attention 2 with only explicit K/V prefetching added.

This variant intentionally preserves the launch configuration, data types,
tiling, and fallback behavior of lightningAttention2. The only kernel-level
change is loop rotation: the current K/V tile is loaded before the loop and the
next K/V tile is loaded before computing the current iteration.
"""

import torch
import triton
import triton.language as tl

GPU_MAP = {
    # Blackwell (sm_120), including the RTX 5090. Use the conservative
    # block size used by the RTX A6000 until a larger tuned value is known.
    "NVIDIA GeForce RTX 5090": 32,
    "NVIDIA RTX A6000": 32,
    "NVIDIA A100-PCIE-40GB": 128, 
    "NVIDIA A100 80GB PCIe": 128, 
    "NVIDIA A800-SXM4-80GB": 64,
    "NVIDIA H20": 64,
}

# Compute the causal linear attention of the ordinary mask
@triton.jit
def _fwd_kernel_without_s_prefetch(
    Q,
    K,
    V,
    Out,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    BLOCK_MODEL: tl.constexpr,
):
    ##### get offset
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_e = tl.program_id(1)
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    # channel offset
    e_offset = off_e * BLOCK_MODEL

    ##### get block ptr
    Q_block_ptr = Q + qk_offset + tl.arange(0, d)[None, :]
    K_trans_block_ptr = K + qk_offset + tl.arange(0, d)[:, None]
    V_block_ptr = V + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    O_block_ptr = Out + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]

    ##### init diag decay(Lambda); q, k decay; kv
    # q, k decay
    off_block = tl.arange(
        0, BLOCK
    )  # Not bug, this is a bit different from algorithm 1, but is mathematically equivalent
    # diag decay
    index = off_block[:, None] - off_block[None, :]
    diag_decay = tl.where(index >= 0, 1, 0)
    kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32)

    # Explicit prefetch: prime the loop with the first K/V tile.
    k_trans_curr = tl.load(
        K_trans_block_ptr + off_block[None, :] * d,
        mask=off_block[None, :] < n,
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        V_block_ptr + off_block[:, None] * e,
        mask=off_block[:, None] < n,
        other=0.0,
    ).to(tl.float32)

    ##### compute
    for i in range(NUM_BLOCK):
        # Start loading the next K/V tile before consuming the current one.
        # On the final iteration next_off_block is entirely masked out.
        next_off_block = off_block + BLOCK
        k_trans_next = tl.load(
            K_trans_block_ptr + next_off_block[None, :] * d,
            mask=next_off_block[None, :] < n,
            other=0.0,
        ).to(tl.float32)
        v_next = tl.load(
            V_block_ptr + next_off_block[:, None] * e,
            mask=next_off_block[:, None] < n,
            other=0.0,
        ).to(tl.float32)

        # Q remains loaded in its original iteration.
        q = tl.load(
            Q_block_ptr + off_block[:, None] * d, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)

        # compute
        qk = tl.dot(q, k_trans_curr) * diag_decay
        o_intra = tl.dot(qk, v)
        o_inter = tl.dot(q, kv)
        o = o_intra + o_inter

        # save and update
        tl.store(
            O_block_ptr + off_block[:, None] * e,
            o.to(O_block_ptr.dtype.element_ty),
            mask=off_block[:, None] < n,
        )
        kv = kv + tl.dot(k_trans_curr, v)
        k_trans_curr = k_trans_next
        v = v_next
        off_block += BLOCK


# Compute causal linear attention of mask with weight decay
@triton.jit
def _fwd_kernel_prefetch(
    Q,
    K,
    V,
    Out,
    S,  # log lambda
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    BLOCK_MODEL: tl.constexpr,
):
    ##### get offset
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_e = tl.program_id(1)
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    # channel offset
    e_offset = off_e * BLOCK_MODEL

    ##### get block ptr
    Q_block_ptr = Q + qk_offset + tl.arange(0, d)[None, :]
    K_trans_block_ptr = K + qk_offset + tl.arange(0, d)[:, None]
    V_block_ptr = V + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    O_block_ptr = Out + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    S_block_ptr = S + off_h

    ##### init diag decay(Lambda); q, k decay; kv
    s = tl.load(S_block_ptr)
    # q, k decay
    off_block = tl.arange(
        0, BLOCK
    )  # Not bug, this is a bit different from algorithm 1, but is mathematically equivalent
    q_decay = tl.exp(-s.to(tl.float32) * off_block[:, None])
    k_trans_decay = tl.exp(-s.to(tl.float32) * (BLOCK - off_block[None, :]))
    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)
    # diag decay
    index = off_block[:, None] - off_block[None, :]
    s_index = (s * index).to(tl.float32)
    s_index = tl.where(index >= 0, -s_index, float("-inf"))
    diag_decay = tl.exp(s_index)
    kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32)

    # Explicit prefetch: prime the loop with the first K/V tile.
    k_trans_curr = tl.load(
        K_trans_block_ptr + off_block[None, :] * d,
        mask=off_block[None, :] < n,
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        V_block_ptr + off_block[:, None] * e,
        mask=off_block[:, None] < n,
        other=0.0,
    ).to(tl.float32)

    ##### compute
    for i in range(NUM_BLOCK):
        # Start loading the next K/V tile before consuming the current one.
        # On the final iteration next_off_block is entirely masked out.
        next_off_block = off_block + BLOCK
        k_trans_next = tl.load(
            K_trans_block_ptr + next_off_block[None, :] * d,
            mask=next_off_block[None, :] < n,
            other=0.0,
        ).to(tl.float32)
        v_next = tl.load(
            V_block_ptr + next_off_block[:, None] * e,
            mask=next_off_block[:, None] < n,
            other=0.0,
        ).to(tl.float32)

        # Q remains loaded in its original iteration.
        q = tl.load(
            Q_block_ptr + off_block[:, None] * d, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)

        # compute
        qk = tl.dot(q, k_trans_curr) * diag_decay
        o_intra = tl.dot(qk, v)
        o_inter = tl.dot(q, kv) * q_decay
        o = o_intra + o_inter

        # save and update
        tl.store(
            O_block_ptr + off_block[:, None] * e,
            o.to(O_block_ptr.dtype.element_ty),
            mask=off_block[:, None] < n,
        )
        kv = block_decay * kv + tl.dot(k_trans_curr * k_trans_decay, v)
        k_trans_curr = k_trans_next
        v = v_next
        off_block += BLOCK


def _shared_mem_limit():
    props = torch.cuda.get_device_properties(0)
    return getattr(props, 'shared_memory_per_block_optin', None) or props.shared_memory_per_block


def _default_block_size(dtype, head_dim):
    name = torch.cuda.get_device_name(0)
    base = GPU_MAP.get(name, 32)
    if dtype == torch.float32:
        block = max(base // 2, 8)
    else:
        block = base
    # RTX 5090 (and similar) opt-in shared memory is 101376 B. Default Triton
    # num_stages plus large head dim (RetNet d=256) overshoots that limit.
    smem = _shared_mem_limit()
    if smem is not None and smem <= 101376 and head_dim >= 128:
        block = min(block, 16)
    return block


class LightningAttention2Prefetch(torch.autograd.Function):
    lightning_block_size = GPU_MAP.get(
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else '', 32)

    @staticmethod
    def forward(ctx, q, k, v, s=None):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        if s is not None:
            s = s.contiguous()
        b, h, n, d = q.shape
        e = v.shape[-1]
        o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
        BLOCK = _default_block_size(q.dtype, d)
        BLOCK_MODEL = min(triton.next_power_of_2(e), 32)
        kernel = _fwd_kernel_without_s_prefetch if s is None else _fwd_kernel_prefetch
        from triton.runtime.errors import OutOfResources
        while True:
            NUM_BLOCK = triton.cdiv(n, BLOCK)
            grid = (b * h, triton.cdiv(e, BLOCK_MODEL))
            try:
                if s is None:
                    kernel[grid](
                        q, k, v, o, b, h, n, d, e,
                        BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK, BLOCK_MODEL=BLOCK_MODEL,
                        num_warps=4, num_stages=1,
                    )
                else:
                    kernel[grid](
                        q, k, v, o, s, b, h, n, d, e,
                        BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK, BLOCK_MODEL=BLOCK_MODEL,
                        num_warps=4, num_stages=1,
                    )
                break
            except OutOfResources:
                if BLOCK <= 8 and BLOCK_MODEL <= 16:
                    raise
                if BLOCK_MODEL > 16:
                    BLOCK_MODEL = max(16, BLOCK_MODEL // 2)
                else:
                    BLOCK = max(8, BLOCK // 2)
        return o
    
lightning_attn2_prefetch = LightningAttention2Prefetch.apply


if __name__=='__main__':
    dtype = torch.float32
    Q = torch.randn(2,32,8000,128,dtype=dtype,device='cuda:0')
    K = torch.randn(2,32,8000,128,dtype=dtype,device='cuda:0')
    V = torch.randn(2,32,8000,128,dtype=dtype,device='cuda:0')
    gamma = torch.full((32,),0.9,device='cuda:0',dtype=dtype)
    ans =lightning_attn2_prefetch(Q,K,V,gamma)
    correct_ans = torch.matmul(torch.tril(torch.matmul(Q,K.transpose(2,3))) ,V)
    print('ours norm:',torch.norm(ans),'\ncorrect norm:',torch.norm(correct_ans),'\ndifference norm:',torch.norm(correct_ans-ans))