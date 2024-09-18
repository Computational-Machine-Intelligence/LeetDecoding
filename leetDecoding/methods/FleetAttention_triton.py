import triton.language as tl
import math
import torch
import triton


@triton.jit
def FleetAttention_kernel(B_ptr,C_ptr,V_ptr,ans_ptr,
               seqlen:tl.constexpr,
               dim:tl.constexpr,
               rank:tl.constexpr,
               stride_vbh:tl.constexpr,
               stride_bbh:tl.constexpr,
               dim_BLOCK:tl.constexpr):
    rank_idx = tl.program_id(axis=0)
    bz = tl.program_id(axis=1)
    dim_block_idx = tl.program_id(axis=2)
    off_b = tl.arange(0,1) 
    off_dim = tl.arange(0,dim_BLOCK) # 一块dim的索引
    cv = tl.zeros([1,dim_BLOCK], dtype=tl.float32)
    o = tl.zeros([1,dim_BLOCK], dtype=tl.float32)
    for seq_idx in range(seqlen):
        offs_bc = bz * stride_bbh + seq_idx * rank + rank_idx + off_b[None,:]
        offs_v = bz * stride_vbh + seq_idx * dim + dim_block_idx * dim_BLOCK + off_dim[None,:]
        ans_ptrs = ans_ptr + bz * stride_vbh + seq_idx * dim + dim_block_idx * dim_BLOCK + off_dim[None,:]
        v_ptrs = V_ptr + offs_v # v是一个block
        b_ptr = B_ptr + offs_bc # b 只是一个值
        c_ptr = C_ptr + offs_bc # c 只是一个值
        b = tl.load(b_ptr, mask=(off_b[None,:]<1), other=0)
        c = tl.load(c_ptr, mask=(off_b[None,:]<1), other=0)
        v = tl.load(v_ptrs, mask=(off_dim[None,:]<dim), other=0)
        cv = c * v + cv
        o = b * cv    
        # tl.atomic_add(ans_ptrs, o, mask=(off_dim[None,:]<dim))
        ans = tl.load(ans_ptrs, mask=(off_dim[None,:]<dim), other=0)
        tl.store(ans_ptrs, ans+o, mask=(off_dim[None,:]<dim))



@triton.jit
def FleetAttention_with_decay_kernel(B_ptr,C_ptr,V_ptr,gamma_ptr,ans_ptr,
               heads:tl.constexpr,
               seqlen:tl.constexpr,
               dim:tl.constexpr,
               rank:tl.constexpr,
               stride_vbh:tl.constexpr,
               stride_bbh:tl.constexpr,
               dim_BLOCK:tl.constexpr):
    rank_idx = tl.program_id(axis=0)
    bz = tl.program_id(axis=1)
    dim_block_idx = tl.program_id(axis=2)
    off_b = tl.arange(0,1) 
    off_dim = tl.arange(0,dim_BLOCK) # 一块dim的索引
    off_gamma = tl.full((1,),bz%heads, dtype=tl.int32)
    cv = tl.zeros([1,dim_BLOCK], dtype=tl.float32)
    o = tl.zeros([1,dim_BLOCK], dtype=tl.float32)
    gamma = tl.load(gamma_ptr+off_gamma,mask=(off_gamma<heads),other=0)
    
    for seq_idx in range(seqlen):
        offs_bc = bz * stride_bbh + seq_idx * rank + rank_idx + off_b[None,:]
        offs_v = bz * stride_vbh + seq_idx * dim + dim_block_idx * dim_BLOCK + off_dim[None,:]
        ans_ptrs = ans_ptr + bz * stride_vbh + seq_idx * dim + dim_block_idx * dim_BLOCK + off_dim[None,:]
        v_ptrs = V_ptr + offs_v # v是一个block
        b_ptr = B_ptr + offs_bc # b 只是一个值
        c_ptr = C_ptr + offs_bc # c 只是一个值
        b = tl.load(b_ptr, mask=(off_b[None,:]<1), other=0)
        c = tl.load(c_ptr, mask=(off_b[None,:]<1), other=0)
        v = tl.load(v_ptrs, mask=(off_dim[None,:]<dim), other=0)
        cv = c * v + cv * gamma
        o = b * cv    
        # tl.atomic_add(ans_ptrs, o, mask=(off_dim[None,:]<dim))
        
        ans = tl.load(ans_ptrs, mask=(off_dim[None,:]<dim), other=0)
        tl.store(ans_ptrs, ans+o, mask=(off_dim[None,:]<dim))


def FleetAttention_triton(B,C,V,gamma=None):
    bsz, heads, seqlen, rank = B.shape
    dim = V.shape[-1] 
    dim_BLOCK = 32
    B = B.contiguous()
    C = C.contiguous()
    V = V.contiguous()
    ans = torch.zeros((bsz, heads, seqlen, dim), device=B.device, dtype=B.dtype)
    grid = lambda META: (rank, bsz * heads, dim // dim_BLOCK) 
    if gamma is None:
        FleetAttention_kernel[grid](B,C,V,ans,
                         seqlen,dim,rank,seqlen*dim,seqlen*rank,dim_BLOCK)
    else:
        FleetAttention_with_decay_kernel[grid](B,C,V,gamma,ans,
                         heads,seqlen,dim,rank,seqlen*dim,seqlen*rank,dim_BLOCK)
    return ans
