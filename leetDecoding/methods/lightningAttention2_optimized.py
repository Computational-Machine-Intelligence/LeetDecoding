import torch
import triton
import triton.language as tl




@triton.autotune(
    configs=[
        # small BLOCK_MODEL
        triton.Config({'BLOCK': 64,  'BLOCK_MODEL': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 128, 'BLOCK_MODEL': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 256, 'BLOCK_MODEL': 16}, num_warps=8, num_stages=3),
        # triton.Config({'BLOCK': 512, 'BLOCK_MODEL': 16}, num_warps=8, num_stages=4),

        # middle BLOCK_MODEL
        triton.Config({'BLOCK': 64,  'BLOCK_MODEL': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 128, 'BLOCK_MODEL': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 256, 'BLOCK_MODEL': 32}, num_warps=8, num_stages=3),
        # triton.Config({'BLOCK': 512, 'BLOCK_MODEL': 32}, num_warps=8, num_stages=4),

        # large BLOCK_MODEL
        triton.Config({'BLOCK': 64,  'BLOCK_MODEL': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 128, 'BLOCK_MODEL': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 256, 'BLOCK_MODEL': 64}, num_warps=8, num_stages=3),
        # triton.Config({'BLOCK': 512, 'BLOCK_MODEL': 64}, num_warps=8, num_stages=4),
    ],
    key=['d', 'e']  # choose best config according to shape
)
@triton.jit
def _fwd_kernel_without_s(
    Q, K, V, Out,
    b: tl.constexpr, h: tl.constexpr, n: tl.constexpr,
    d: tl.constexpr, e: tl.constexpr,
    BLOCK: tl.constexpr, BLOCK_MODEL: tl.constexpr,
    DTYPE: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_e = tl.program_id(1)
    NUM_BLOCK = tl.cdiv(n, BLOCK)

    qk_offset = off_bh * n * d
    v_offset  = off_bh * n * e
    o_offset  = off_bh * n * e
    e_offset  = off_e * BLOCK_MODEL

    # 衰减常量（无 s 时都为 1）
    off_block = tl.arange(0, BLOCK)
    q_decay = tl.full((BLOCK, 1), 1.0, dtype=tl.float32)
    k_trans_decay = tl.full((1, BLOCK), 1.0, dtype=tl.float32)
    block_decay = 1.0
    rel_pos = off_block[:, None] - off_block[None, :]
    diag_decay = tl.where(rel_pos >= 0, 1.0, 0.0).to(tl.float32)

    Q_ptr = Q + qk_offset + tl.arange(0, d)[None, :]
    K_ptr = K + qk_offset + tl.arange(0, d)[:, None]
    V_ptr = V + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    O_ptr = Out + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]

    kv = tl.zeros((d, BLOCK_MODEL), dtype=tl.float32)

    # 初次预取
    k_curr = tl.load(K_ptr + off_block[None, :] * d,
                     mask=off_block[None, :] < n,
                     other=0.0)
    v_curr = tl.load(V_ptr + off_block[:, None] * e,
                     mask=off_block[:, None] < n,
                     other=0.0)

    for i in range(NUM_BLOCK):
        q_tile = tl.load(Q_ptr + off_block[:, None] * d,
                         mask=off_block[:, None] < n,
                         other=0.0)

        # QK^T
        qk = tl.dot(q_tile, k_curr, out_dtype=tl.float32) * diag_decay
        o_intra = tl.dot(qk.to(DTYPE), v_curr, out_dtype=tl.float32)
        o_inter = tl.dot(q_tile, kv.to(DTYPE), out_dtype=tl.float32) * q_decay
        o_tile = o_intra + o_inter

        tl.store(O_ptr + off_block[:, None] * e,
                 o_tile.to(DTYPE),
                 mask=off_block[:, None] < n)

        # KV 更新
        k_decayed = k_curr * k_trans_decay.to(DTYPE)
        kv = block_decay * kv + tl.dot(k_decayed, v_curr, out_dtype=tl.float32)

        if i < NUM_BLOCK - 1:
            next_off = off_block + BLOCK
            k_curr = tl.load(K_ptr + next_off[None, :] * d,
                             mask=next_off[None, :] < n,
                             other=0.0)
            v_curr = tl.load(V_ptr + next_off[:, None] * e,
                             mask=next_off[:, None] < n,
                             other=0.0)

        off_block += BLOCK



@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 64,  'BLOCK_MODEL': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 128, 'BLOCK_MODEL': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 256, 'BLOCK_MODEL': 16}, num_warps=8, num_stages=3),
        # triton.Config({'BLOCK': 512, 'BLOCK_MODEL': 16}, num_warps=8, num_stages=4),

        triton.Config({'BLOCK': 64,  'BLOCK_MODEL': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 128, 'BLOCK_MODEL': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 256, 'BLOCK_MODEL': 32}, num_warps=8, num_stages=3),
        # triton.Config({'BLOCK': 512, 'BLOCK_MODEL': 32}, num_warps=8, num_stages=4),

        triton.Config({'BLOCK': 64,  'BLOCK_MODEL': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 128, 'BLOCK_MODEL': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 256, 'BLOCK_MODEL': 64}, num_warps=8, num_stages=3),
        # triton.Config({'BLOCK': 512, 'BLOCK_MODEL': 64}, num_warps=8, num_stages=4),
    ],
    key=['d', 'e'] 
)
@triton.jit
def _fwd_kernel_all(
    Q, K, V, Out, S,
    b: tl.constexpr, h: tl.constexpr, n: tl.constexpr,
    d: tl.constexpr, e: tl.constexpr,
    BLOCK: tl.constexpr, BLOCK_MODEL: tl.constexpr,
    DTYPE: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_e = tl.program_id(1)
    NUM_BLOCK = tl.cdiv(n, BLOCK)

    qk_offset = off_bh * n * d
    v_offset  = off_bh * n * e
    o_offset  = off_bh * n * e
    e_offset  = off_e * BLOCK_MODEL

    s = tl.load(S + (off_bh % h)).to(tl.float32)
    off_block = tl.arange(0, BLOCK)
    q_decay = tl.exp(-s * off_block[:, None])
    k_trans_decay = tl.exp(-s * (BLOCK - off_block[None, :]))
    block_decay = tl.exp(-s * BLOCK)

    rel_pos = off_block[:, None] - off_block[None, :]
    diag_decay = tl.where(rel_pos >= 0, tl.exp(-s * rel_pos), 0.0).to(tl.float32)

    Q_ptr = Q + qk_offset + tl.arange(0, d)[None, :]
    K_ptr = K + qk_offset + tl.arange(0, d)[:, None]
    V_ptr = V + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    O_ptr = Out + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]

    kv = tl.zeros((d, BLOCK_MODEL), dtype=tl.float32)

    k_curr = tl.load(K_ptr + off_block[None, :] * d,
                     mask=off_block[None, :] < n,
                     other=0.0)
    v_curr = tl.load(V_ptr + off_block[:, None] * e,
                     mask=off_block[:, None] < n,
                     other=0.0)

    for i in range(NUM_BLOCK):
        q_tile = tl.load(Q_ptr + off_block[:, None] * d,
                         mask=off_block[:, None] < n,
                         other=0.0)

        qk = tl.dot(q_tile, k_curr, out_dtype=tl.float32) * diag_decay
        o_intra = tl.dot(qk.to(DTYPE), v_curr, out_dtype=tl.float32)
        o_inter = tl.dot(q_tile, kv.to(DTYPE), out_dtype=tl.float32) * q_decay
        o_tile = o_intra + o_inter

        tl.store(O_ptr + off_block[:, None] * e,
                 o_tile.to(DTYPE),
                 mask=off_block[:, None] < n)

        k_decayed = k_curr * k_trans_decay.to(DTYPE)
        kv = block_decay * kv + tl.dot(k_decayed, v_curr, out_dtype=tl.float32)

        if i < NUM_BLOCK - 1:
            next_off = off_block + BLOCK
            k_curr = tl.load(K_ptr + next_off[None, :] * d,
                             mask=next_off[None, :] < n,
                             other=0.0)
            v_curr = tl.load(V_ptr + next_off[:, None] * e,
                             mask=next_off[:, None] < n,
                             other=0.0)

        off_block += BLOCK



class LightningAttention2(torch.autograd.Function):
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

        max_block_model = max(cfg.kwargs['BLOCK_MODEL'] for cfg in _fwd_kernel_without_s.configs)
        grid = (b * h, triton.cdiv(e, max_block_model))
        if s is None:
            if True:
                dtype_triton = tl.float32
                if q.dtype == torch.float16:
                    dtype_triton = tl.float16
                elif q.dtype == torch.bfloat16:
                    dtype_triton = tl.bfloat16
                _fwd_kernel_without_s[grid](q, k, v, o, b, h, n, d, e, DTYPE=dtype_triton)
        else:
            dtype_triton = tl.float32
            if q.dtype == torch.float16:
                dtype_triton = tl.float16
            elif q.dtype == torch.bfloat16:
                dtype_triton = tl.bfloat16
            _fwd_kernel_all[grid](
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
                DTYPE=dtype_triton,
            )
        return o


lightning_attn2_optimized = LightningAttention2.apply