import torch
import triton
import triton.language as tl


def _shared_mem_limit():
    props = torch.cuda.get_device_properties(0)
    return getattr(props, 'shared_memory_per_block_optin', None) or props.shared_memory_per_block


def _default_autotune_configs():
    configs = [
        # RTX 5090 (sm_120) safe fallbacks: opt-in shared memory per block is only
        # 101376 B (99 KB). At large d (e.g. rank=512 / RetNet d=256) the kv
        # accumulator plus staged K/V tiles blow the limit for A100-sized configs.
        triton.Config({'BLOCK': 16, 'BLOCK_MODEL': 16}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK': 16, 'BLOCK_MODEL': 16}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK': 16, 'BLOCK_MODEL': 32}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK': 32, 'BLOCK_MODEL': 16}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK': 32,  'BLOCK_MODEL': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 32,  'BLOCK_MODEL': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 32,  'BLOCK_MODEL': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 32,  'BLOCK_MODEL': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 64,  'BLOCK_MODEL': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 64,  'BLOCK_MODEL': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 64,  'BLOCK_MODEL': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK': 64,  'BLOCK_MODEL': 64}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK': 128, 'BLOCK_MODEL': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK': 128, 'BLOCK_MODEL': 32}, num_warps=8, num_stages=1),
    ]
    smem = _shared_mem_limit()
    if smem is not None and smem <= 101376:
        configs = [
            c for c in configs
            if c.kwargs['BLOCK'] <= 32 and c.kwargs['BLOCK_MODEL'] <= 32 and c.num_stages <= 1
        ]
    return configs


def make_fwd_kernel_without_s(configs=None):
    if configs is None:
        configs = _default_autotune_configs()

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
                             other=0.0).to(tl.float32)

            qk = tl.dot(q_tile, k_curr.to(tl.float32), out_dtype=tl.float32) * diag_decay
            o_intra = tl.dot(qk, v_curr.to(tl.float32), out_dtype=tl.float32)
            o_inter = tl.dot(q_tile, kv, out_dtype=tl.float32) * q_decay
            o_tile = o_intra + o_inter

            tl.store(O_ptr + off_block[:, None] * e,
                     o_tile.to(DTYPE),
                     mask=off_block[:, None] < n)

            k_decayed = k_curr.to(tl.float32) * k_trans_decay
            kv = block_decay * kv + tl.dot(k_decayed, v_curr.to(tl.float32))

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
        configs = _default_autotune_configs()

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
                             other=0.0).to(tl.float32)

            qk = tl.dot(q_tile, k_curr.to(tl.float32), out_dtype=tl.float32) * diag_decay
            o_intra = tl.dot(qk, v_curr.to(tl.float32), out_dtype=tl.float32)
            o_inter = tl.dot(q_tile, kv, out_dtype=tl.float32) * q_decay
            o_tile = o_intra + o_inter

            tl.store(O_ptr + off_block[:, None] * e,
                     o_tile.to(DTYPE),
                     mask=off_block[:, None] < n)

            k_decayed = k_curr.to(tl.float32) * k_trans_decay
            kv = block_decay * kv + tl.dot(k_decayed, v_curr.to(tl.float32))

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

# 缓存 kernel，避免每次 forward 重新创建
_CACHED_KERNEL_WITH_S = None
_CACHED_KERNEL_WITHOUT_S = None


class LightningAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, s=None, autotune_configs=None):
        global _CACHED_KERNEL_WITH_S, _CACHED_KERNEL_WITHOUT_S
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        if s is not None:
            s = s.contiguous()

        b, h, n, d = q.shape
        e = v.shape[-1]
        o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

        # 使用缓存的 kernel
        if s is None:
            if _CACHED_KERNEL_WITHOUT_S is None:
                _CACHED_KERNEL_WITHOUT_S = make_fwd_kernel_without_s(autotune_configs)
            kernel = _CACHED_KERNEL_WITHOUT_S
        else:
            if _CACHED_KERNEL_WITH_S is None:
                _CACHED_KERNEL_WITH_S = make_fwd_kernel_all(autotune_configs)
            kernel = _CACHED_KERNEL_WITH_S

        # grid 必须是 callable，让 triton 在 autotune 确定 BLOCK_MODEL 后再计算，
        # 否则用 max(BLOCK_MODEL) 预固定会导致部分 e 维度的 program 根本不会被启动。
        grid = lambda meta: (b * h, triton.cdiv(e, meta['BLOCK_MODEL']))

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