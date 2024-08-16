# Copyright (c) 2024 Doraemonzzz

import torch
import triton
import triton.language as tl
import pycuda.driver as drv
import pycuda.autoinit

GPU_MAP = {
    "NVIDIA RTX A6000": 32, 
    "NVIDIA A100-PCIE-40GB": 128, 
    'NVIDIA A100 80GB PCIe': 128, 
    'NVIDIA A800-SXM4-80GB': 128,
}

# Compute the causal linear attention of the ordinary mask
@triton.jit
def _fwd_kernel_without_s(
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

    ##### compute
    for i in range(NUM_BLOCK):
        # load
        q = tl.load(
            Q_block_ptr + off_block[:, None] * d, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)
        k_trans = tl.load(
            K_trans_block_ptr + off_block[None, :] * d,
            mask=off_block[None, :] < n,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_block_ptr + off_block[:, None] * e, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)

        # compute
        qk = tl.dot(q, k_trans) * diag_decay
        o_intra = tl.dot(qk, v)
        o_inter = tl.dot(q, kv)
        o = o_intra + o_inter

        # save and update
        tl.store(
            O_block_ptr + off_block[:, None] * e,
            o.to(O_block_ptr.dtype.element_ty),
            mask=off_block[:, None] < n,
        )
        kv = kv + tl.dot(k_trans, v)
        off_block += BLOCK


# Compute causal linear attention of mask with weight decay
@triton.jit
def _fwd_kernel(
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
    s_index = s * index
    s_index = tl.where(index >= 0, -s_index, float("-inf"))
    diag_decay = tl.exp(s_index)
    kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32)

    ##### compute
    for i in range(NUM_BLOCK):
        # load
        q = tl.load(
            Q_block_ptr + off_block[:, None] * d, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)
        k_trans = tl.load(
            K_trans_block_ptr + off_block[None, :] * d,
            mask=off_block[None, :] < n,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_block_ptr + off_block[:, None] * e, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)

        # compute
        qk = tl.dot(q, k_trans) * diag_decay
        o_intra = tl.dot(qk, v)
        o_inter = tl.dot(q, kv) * q_decay
        o = o_intra + o_inter

        # save and update
        tl.store(
            O_block_ptr + off_block[:, None] * e,
            o.to(O_block_ptr.dtype.element_ty),
            mask=off_block[:, None] < n,
        )
        kv = block_decay * kv + tl.dot(k_trans * k_trans_decay, v)
        off_block += BLOCK


class LightningAttention2(torch.autograd.Function):
    lightning_block_size = GPU_MAP[drv.Device(0).name()] # By default, the first GPU is selected as the computing device.
    
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
        if q.dtype == torch.float16 or q.dtype ==torch.bfloat16:
            BLOCK = LightningAttention2.lightning_block_size # The larger the BLOCK, the more shared memory is required.
        elif q.dtype == torch.float32:
            BLOCK = LightningAttention2.lightning_block_size // 2
        NUM_BLOCK = triton.cdiv(q.shape[2], BLOCK)
        # parallel over channel
        BLOCK_MODEL = min(triton.next_power_of_2(e), 32)
        grid = (b * h, triton.cdiv(e, BLOCK_MODEL))
        if s is None:
            _fwd_kernel_without_s[grid](
                q,
                k,
                v,
                o,
                b,
                h,
                n,
                d,
                e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                BLOCK_MODEL=BLOCK_MODEL,
            )
        else:
            _fwd_kernel[grid](
                q,
                k,
                v,
                o,
                s,
                b,
                h,
                n,
                d,
                e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                BLOCK_MODEL=BLOCK_MODEL,
            )
        return o
    
lightning_attn2 = LightningAttention2.apply


if __name__=='__main__':
    dtype = torch.float16
    Q = torch.randn(2,32,129,128,dtype=dtype,device='cuda:0')
    K = torch.randn(2,32,129,128,dtype=dtype,device='cuda:0')
    V = torch.randn(2,32,129,128,dtype=dtype,device='cuda:0')
    gamma = torch.full((32,),0.9,device='cuda:0',dtype=dtype)
    ans =lightning_attn2(Q,K,V,gamma)
    correct_ans = torch.matmul(torch.tril(torch.matmul(Q,K.transpose(2,3))) ,V)
    print('ours norm:',torch.norm(ans),'\ncorrect norm:',torch.norm(correct_ans),'\ndifference norm:',torch.norm(correct_ans-ans))

