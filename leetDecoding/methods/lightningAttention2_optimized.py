import torch
import triton
import triton.language as tl


def make_fwd_kernel_without_s(configs=None):
    if configs is None:
        configs = [
            # small BLOCK_MODEL
            triton.Config({'BLOCK': 16,  'BLOCK_MODEL': 16}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK': 32,  'BLOCK_MODEL': 16}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK': 64,  'BLOCK_MODEL': 16}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK': 128, 'BLOCK_MODEL': 16}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK': 256, 'BLOCK_MODEL': 16}, num_warps=8, num_stages=3),

            # middle BLOCK_MODEL
            triton.Config({'BLOCK': 16,  'BLOCK_MODEL': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK': 32,  'BLOCK_MODEL': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK': 64,  'BLOCK_MODEL': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK': 128, 'BLOCK_MODEL': 32}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK': 256, 'BLOCK_MODEL': 32}, num_warps=8, num_stages=3),

            # large BLOCK_MODEL
            triton.Config({'BLOCK': 16,  'BLOCK_MODEL': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK': 32,  'BLOCK_MODEL': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK': 64,  'BLOCK_MODEL': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK': 128, 'BLOCK_MODEL': 64}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK': 256, 'BLOCK_MODEL': 64}, num_warps=8, num_stages=3),
        ]

    @triton.autotune(configs=configs, key=['d', 'e'])
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

    return _fwd_kernel_without_s


def make_fwd_kernel_all(configs=None):
    if configs is None:
        configs = [
            triton.Config({'BLOCK': block, "BLOCK_MODEL": bm}, num_warps=num_warps, num_stages=num_stages)
            for block in [32, 64]
            for bm in [32, 64]
            for num_warps in [2, 4, 8]
            for num_stages in [2, 3, 4]
        ]

    @triton.autotune(configs=configs, key=['d', 'e'])
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

    return _fwd_kernel_all


# -----------------------------
# LightningAttention2 with configurable autotune
# -----------------------------

class LightningAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, s=None, autotune_configs=None):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        if s is not None:
            s = s.contiguous()

        b, h, n, d = q.shape
        e = v.shape[-1]
        o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

        # Choose kernel based on s
        if s is None:
            kernel = make_fwd_kernel_without_s(autotune_configs)
        else:
            kernel = make_fwd_kernel_all(autotune_configs)

        max_block_model = max(cfg.kwargs['BLOCK_MODEL'] for cfg in kernel.configs)
        grid = (b * h, triton.cdiv(e, max_block_model))

        dtype_triton = tl.float32
        if q.dtype == torch.float16:
            dtype_triton = tl.float16
        elif q.dtype == torch.bfloat16:
            dtype_triton = tl.bfloat16

        if s is None:
            kernel[grid](q, k, v, o, b, h, n, d, e, DTYPE=dtype_triton)
        else:
            kernel[grid](q, k, v, o, s, b, h, n, d, e, DTYPE=dtype_triton)

        return o


lightning_attn2_optimized = LightningAttention2.apply