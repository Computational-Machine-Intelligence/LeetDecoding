"""Profile attention breakdown for RetNet / TransNormerLLM models.

Measures per-layer CUDA time for:
  - attn_module  (inclusive: QKV proj + RoPE/LRPE + attn_core + gate + output proj)
  - attn_core    (pure causal linear attention kernel)
  - mlp          (FFN / GLU)
  - total        (end-to-end forward pass)

Then computes:
  - attn_non_core = attn_module - attn_core
  - other         = total - attn_module - mlp

Usage:
    python -m leetDecoding.test.profile_breakdown \
        --model_path pretrained_models/retnet_1_3b \
        --method FleetAttention \
        --batch_size 1 \
        --seq_len 2048 \
        --dtype float32 \
        --repeats 10 \
        --output_dir outputs/profile_breakdown
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_PARENT_ROOT = os.path.dirname(_REPO_ROOT)

if _PARENT_ROOT not in sys.path:
    sys.path.insert(0, _PARENT_ROOT)

from toy_retnet_1_3b.retnet.configuration_retnet import load_config_from_yaml
from toy_retnet_1_3b.retnet.modeling_retnet import RetNetModelWithLMHead
from transnormerLLM.modeling_transnormer import TransnormerForCausalLM
from leetDecoding.test.profiling_timer import CUDABlockTimer

# ---------------------------------------------------------------------------
# Model instrumentation
# ---------------------------------------------------------------------------

def _find_retnet_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Return RetNetBlock modules in order."""
    # model.model.blocks  (RetNetModel.blocks)
    if hasattr(model, "model") and hasattr(model.model, "blocks"):
        return list(model.model.blocks)
    # Fallback: search all RetNetBlock instances
    blocks: List[torch.nn.Module] = []
    for m in model.modules():
        if m.__class__.__name__ == "RetNetBlock":
            blocks.append(m)
    return blocks


def _find_transnormer_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Return TransnormerDecoderLayer modules in order."""
    for m in model.modules():
        if m.__class__.__name__ == "TransnormerDecoderLayer":
            yield m


def instrument_retnet(model: torch.nn.Module, timer: CUDABlockTimer,
                      instrument_core: str = "accelerated") -> None:
    """Hook layer.msr (attn_module), layer.ffn (mlp), and
    monkey-patch the attention core for per-layer attn_core timing.

    Parameters
    ----------
    instrument_core : str
        "accelerated" — patch parallel_retention_update (the kernel dispatch).
        "origin"      — patch parallel_retention (the matmul + decay mask path).
        "none"        — skip core instrumentation (attn_core will be 0).
    """

    layers = _find_retnet_layers(model)
    if not layers:
        raise RuntimeError("Could not find RetNetBlock layers in model.")

    # (a) Module-level hooks (always applied)
    for idx, layer in enumerate(layers):
        timer.add_module(layer.msr, tag="attn_module", name=f"L{idx:02d}.attn")
        timer.add_module(layer.ffn, tag="mlp",         name=f"L{idx:02d}.mlp")

    if instrument_core == "none":
        return

    # (b) Monkey-patch the appropriate attention core method
    if instrument_core == "origin":
        _original_func = layers[0].msr.parallel_retention.__func__

        for idx, layer in enumerate(layers):
            _layer_idx = idx

            def _make_patched_origin(orig_func, lidx):
                def _patched_retention(self, q, k, v, decay_mask, attention_method):
                    name = f"L{lidx:02d}.attn_core"
                    with timer.range("attn_core", name):
                        return orig_func(self, q, k, v, decay_mask, attention_method)
                return _patched_retention

            layer.msr.parallel_retention = _make_patched_origin(
                _original_func, _layer_idx
            ).__get__(layer.msr, type(layer.msr))
        return

    # instrument_core == "accelerated"
    _original_func = layers[0].msr.parallel_retention_update.__func__

    for idx, layer in enumerate(layers):
        _layer_idx = idx  # capture per-iteration

        def _make_patched(orig_func, lidx):
            def _patched_update(self, q, k, v, attention_method):
                name = f"L{lidx:02d}.attn_core"
                with timer.range("attn_core", name):
                    return orig_func(self, q, k, v, attention_method)
            return _patched_update

        layer.msr.parallel_retention_update = _make_patched(
            _original_func, _layer_idx
        ).__get__(layer.msr, type(layer.msr))


def instrument_transnormer(model: torch.nn.Module, timer: CUDABlockTimer) -> None:
    """Hook layer.token_mixer (attn_module), layer.channel_mixer (mlp),
    and wrap the functional attention-method dispatch inside
    NormLinearAttention.inference() to record attn_core.

    This measures attn_core for the custom attention_method branches such as
    BCMV_vanilla, FleetAttention, recursion, etc. For the origin path
    (attention_method=None), the inline blockwise attention computation is
    copied here and wrapped as attn_core because it is not a separate function.
    """
    layers = list(_find_transnormer_layers(model))
    if not layers:
        raise RuntimeError("Could not find TransnormerDecoderLayer layers in model.")

    for idx, layer in enumerate(layers):
        timer.add_module(layer.token_mixer,   tag="attn_module", name=f"L{idx:02d}.attn")
        timer.add_module(layer.channel_mixer, tag="mlp",         name=f"L{idx:02d}.mlp")

    dispatch_names = [
        "blockBased",
        "FleetAttention",
        "FleetAttention_triton",
        "lightning_attn2",
        "lightning_attn2_optimized",
        "causal_dot_product_torch",
        "linear_attn",
        "causal_dot_product",
        "lightningAttention2_torch",
        "recursion",
    ]

    for idx, layer in enumerate(layers):
        token_mixer = layer.token_mixer
        original_inference = token_mixer.inference.__func__

        def _make_patched_inference(orig_func, lidx):
            model_globals = orig_func.__globals__
            rearrange = model_globals["rearrange"]
            block_size = model_globals["BLOCK"]

            def _run_origin_inference(
                self,
                x,
                attn_mask=None,
                attn_padding_mask=None,
                output_attentions=False,
                past_key_value=None,
                use_cache=False,
                slope_rate=None,
                attention_method=None,
            ):
                n = x.shape[-2]
                q, k, v, u = self.qkvu_proj(x).chunk(4, dim=-1)
                q, k, v = map(
                    lambda tensor: rearrange(
                        tensor, "b n (h d) -> b h n d", h=self.num_heads
                    ),
                    [q, k, v],
                )
                q = self.act(q)
                k = self.act(k)

                if self.linear_use_lrpe:
                    q = self.lrpe(q, offset=self.offset)
                    k = self.lrpe(k, offset=self.offset)

                if past_key_value is None:
                    self.offset = q.shape[-2]
                else:
                    self.offset += 1

                ratio = torch.exp(-slope_rate)
                core_name = f"L{lidx:02d}.attn_core"
                with timer.range("attn_core", core_name):
                    if past_key_value is None:
                        if attn_padding_mask is not None:
                            v = v.masked_fill(
                                (1 - attn_padding_mask)
                                .unsqueeze(1)
                                .unsqueeze(-1)
                                .to(torch.bool),
                                0,
                            )
                        num_blocks = (n + block_size - 1) // block_size
                        batch, heads, n, dim = q.shape
                        value_dim = v.shape[-1]
                        array = torch.arange(block_size).to(q)
                        q_decay = torch.exp(-slope_rate * array.reshape(-1, 1))
                        k_decay = torch.exp(
                            -slope_rate * (block_size - array.reshape(-1, 1))
                        )
                        index = array[:, None] - array[None, :]
                        s_index = slope_rate * index[None, None]
                        s_index = torch.where(index >= 0, -s_index, float("-inf"))
                        diag_decay = torch.exp(s_index)

                        kv = torch.zeros(
                            batch, heads, dim, value_dim
                        ).to(torch.float32).to(q.device)
                        output = torch.empty(
                            (batch, heads, n, value_dim), dtype=q.dtype, device=q.device
                        )
                        for block_idx in range(num_blocks):
                            start_idx = block_idx * block_size
                            end_idx = min(start_idx + block_size, n)
                            block_len = end_idx - start_idx

                            qi = q[:, :, start_idx:end_idx].contiguous()
                            ki = k[:, :, start_idx:end_idx].contiguous()
                            vi = v[:, :, start_idx:end_idx].contiguous()
                            qkv_none_diag = torch.matmul(
                                qi.to(torch.float32) * q_decay[:, :block_len], kv
                            ).to(torch.float32)

                            qk = torch.matmul(qi, ki.transpose(-1, -2)).to(
                                torch.float32
                            ) * diag_decay[:, :, :block_len, :block_len]
                            qkv_diag = torch.matmul(qk, vi.to(torch.float32))
                            block_decay = torch.exp(-slope_rate * block_len)
                            output[:, :, start_idx:end_idx] = qkv_none_diag + qkv_diag
                            kv = block_decay * kv + torch.matmul(
                                (ki * k_decay[:, -block_len:])
                                .transpose(-1, -2)
                                .to(vi.dtype),
                                vi,
                            )
                    else:
                        kv = past_key_value
                        output_chunks = []
                        for token_idx in range(n):
                            kv = ratio * kv + torch.einsum(
                                "... n d, ... n e -> ... d e",
                                k[:, :, token_idx:token_idx + 1],
                                v[:, :, token_idx:token_idx + 1],
                            )
                            qkv = torch.einsum(
                                "... n e, ... e d -> ... n d",
                                q[:, :, token_idx:token_idx + 1],
                                kv.to(q.dtype),
                            )
                            output_chunks.append(qkv)
                        output = torch.concat(output_chunks, dim=-2)

                output = rearrange(output, "b h n d -> b n (h d)")
                output = self.norm(output)
                output = u * output
                output = self.out_proj(output)
                return output, None, kv

            def _patched_inference(
                self,
                x,
                attn_mask=None,
                attn_padding_mask=None,
                output_attentions=False,
                past_key_value=None,
                use_cache=False,
                slope_rate=None,
                attention_method=None,
            ):
                if attention_method is None:
                    return _run_origin_inference(
                        self,
                        x,
                        attn_mask=attn_mask,
                        attn_padding_mask=attn_padding_mask,
                        output_attentions=output_attentions,
                        past_key_value=past_key_value,
                        use_cache=use_cache,
                        slope_rate=slope_rate,
                        attention_method=attention_method,
                    )

                saved_globals = {}

                def _wrap_dispatch(fn):
                    def _wrapped(*args, **kwargs):
                        with timer.range("attn_core", f"L{lidx:02d}.attn_core"):
                            return fn(*args, **kwargs)
                    return _wrapped

                for name in dispatch_names:
                    if name in orig_func.__globals__:
                        saved_globals[name] = orig_func.__globals__[name]
                        orig_func.__globals__[name] = _wrap_dispatch(saved_globals[name])

                try:
                    return orig_func(
                        self,
                        x,
                        attn_mask=attn_mask,
                        attn_padding_mask=attn_padding_mask,
                        output_attentions=output_attentions,
                        past_key_value=past_key_value,
                        use_cache=use_cache,
                        slope_rate=slope_rate,
                        attention_method=attention_method,
                    )
                finally:
                    for name, fn in saved_globals.items():
                        orig_func.__globals__[name] = fn

            return _patched_inference

        token_mixer.inference = _make_patched_inference(
            original_inference,
            idx,
        ).__get__(token_mixer, type(token_mixer))


def _load_tokenizer_for_model(model_path: str):
    """Load a tokenizer matching the model at *model_path*.

    Mirrors the logic in main.py:load_model_and_tokenizer.
    """
    if "transnormer" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = "<unk>"
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        tokenizer.pad_token = "<unk>"
    return tokenizer


def _resolve_model_type(model_path: str, model: torch.nn.Module) -> str:
    if "retnet" in model_path.lower():
        return "retnet"
    if "transnormer" in model_path.lower():
        return "transnormer"
    # Try introspection
    try:
        _find_retnet_layers(model)
        return "retnet"
    except Exception:
        pass
    try:
        list(_find_transnormer_layers(model))
        return "transnormer"
    except Exception:
        pass
    raise RuntimeError("Cannot determine model type from path or structure.")


def profile_breakdown(
    model: torch.nn.Module,
    model_type: str,
    seq_len: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_method: str,
    repeats: int = 10,
    warmup: int = 5,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> dict:
    """Run instrumented forward passes and return breakdown dict.

    Parameters
    ----------
    input_ids : Optional[torch.Tensor]
        Pre-tokenized input IDs from real data. If None, random dummy
        inputs are generated (backward-compatible behaviour).
    attention_mask : Optional[torch.Tensor]
        Attention mask matching *input_ids*. Only used when input_ids
        is provided.
    """
    timer = CUDABlockTimer()

    is_origin = (attention_method == "origin")
    model_method = None if is_origin else attention_method
    core_mode = "origin" if is_origin else "accelerated"

    if model_type == "retnet":
        instrument_retnet(model, timer, instrument_core=core_mode)
    elif model_type == "transnormer":
        instrument_transnormer(model, timer)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    timer.enabled = False

    # Create or use provided input
    use_real_inputs = input_ids is not None
    if use_real_inputs:
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device)
    else:
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        attention_mask = None

    # Build kwargs for model forward
    def _forward_kwargs():
        kw = {"input_ids": input_ids, "attention_method": model_method}
        if attention_mask is not None:
            kw["attention_mask"] = attention_mask
        return kw

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(**_forward_kwargs())
    torch.cuda.synchronize()

    # Free memory accumulated during warmup (e.g. KV caches, intermediate tensors)
    timer.cleanup_cuda()

    # Timed runs
    all_records: List[dict] = []
    total_times: List[float] = []

    for run_idx in range(repeats):
        timer.clear()
        timer.enabled = True

        torch.cuda.synchronize()
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record()

        with torch.no_grad():
            _ = model(**_forward_kwargs())

        total_end.record()
        torch.cuda.synchronize()

        timer.enabled = False

        total_ms = total_start.elapsed_time(total_end)
        total_times.append(total_ms)

        by_tag, by_name = timer.summary()

        record = {
            "run": run_idx,
            "total_ms": total_ms,
            "by_tag": dict(by_tag),
            "by_name": {f"{t}||{n}": v for (t, n), v in by_name.items()},
        }

        # Compute derived fields
        attn_module_ms = by_tag.get("attn_module", 0.0)
        attn_core_ms = by_tag.get("attn_core", 0.0)
        mlp_ms = by_tag.get("mlp", 0.0)

        record["derived"] = {
            "attn_module_ms": attn_module_ms,
            "attn_core_ms": attn_core_ms,
            "attn_non_core_ms": attn_module_ms - attn_core_ms,
            "mlp_ms": mlp_ms,
            "other_ms": total_ms - attn_module_ms - mlp_ms,
        }

        all_records.append(record)

        # Aggressively free CUDA memory between repeats to avoid OOM
        # when using large batch sizes or long sequences with real inputs.
        timer.cleanup_cuda()

        # Compute derived fields
        attn_module_ms = by_tag.get("attn_module", 0.0)
        attn_core_ms = by_tag.get("attn_core", 0.0)
        mlp_ms = by_tag.get("mlp", 0.0)

        record["derived"] = {
            "attn_module_ms": attn_module_ms,
            "attn_core_ms": attn_core_ms,
            "attn_non_core_ms": attn_module_ms - attn_core_ms,
            "mlp_ms": mlp_ms,
            "other_ms": total_ms - attn_module_ms - mlp_ms,
        }

        all_records.append(record)

    # Aggregate across runs
    keys = ["attn_module_ms", "attn_core_ms", "attn_non_core_ms", "mlp_ms", "other_ms"]
    agg: Dict[str, dict] = {}
    for key in keys:
        vals = [r["derived"][key] for r in all_records]
        agg[key] = {
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
        }

    # Per-layer aggregation (average over runs)
    per_layer: Dict[int, dict] = defaultdict(lambda: defaultdict(float))
    for record in all_records:
        for full_name, ms in record["by_name"].items():
            tag, name = full_name.split("||", 1)
            # Extract layer index from name like "L05.attn" or "L05.attn_core"
            import re
            m = re.match(r"L(\d+)", name)
            if m:
                layer_idx = int(m.group(1))
                per_layer[layer_idx][tag] += ms / repeats

    result = {
        "config": {
            "model_type": model_type,
            "attention_method": attention_method,
            "batch_size": batch_size,
            "seq_len": input_ids.shape[-1] if use_real_inputs else seq_len,
            "dtype": str(dtype),
            "repeats": repeats,
            "warmup": warmup,
            "use_real_inputs": use_real_inputs,
        },
        "total_ms": {
            "mean": sum(total_times) / len(total_times),
            "min": min(total_times),
            "max": max(total_times),
        },
        "breakdown": agg,
        "per_layer": {
            str(k): dict(v) for k, v in sorted(per_layer.items())
        },
        "raw": all_records,
    }

    timer.remove()
    # Final cleanup to release all profiling memory
    CUDABlockTimer.cleanup_cuda()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _normalize_method_name(raw: str) -> str:
    """Normalize user-friendly method names to what the model expects.

    The single-layer CLI (test_method.py) accepts lowercase names like
    'blockbased' while the RetNet model's dispatch expects 'BlockBased'.
    Pass 'origin' or 'none' to use the original non-accelerated path.
    """
    if raw.lower() in ("origin", "none"):
        return "origin"  # sentinel; callers will pass None to model

    # Map of common variants to the canonical model name
    _CANONICAL: dict = {
        "blockbased": "BlockBased",
        "fleetattention": "FleetAttention",
        "fleetattention_torch": "FleetAttention_torch",
        "fleetattention_triton": "FleetAttention",
        "lightningattention2": "lightningAttention2",
        "lightningattention2_origin": "lightningAttention2_origin",
        "lightningattention2_optimized": "lightningAttention2_optimized",
        "lightningattention2_torch": "lightningAttention2_torch",
        "causal_dot_product": "causal_dot_product",
        "causal_dot_product_torch": "causal_dot_product_torch",
        "bcmv_vanilla": "BCMV_vanilla",
        "recursion": "recursion",
    }
    return _CANONICAL.get(raw.lower(), raw)


def main():
    parser = argparse.ArgumentParser(description="Profile attention breakdown.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--method", type=str, default="FleetAttention")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Sequence length for dummy (random) inputs. "
                             "Ignored when --data_path is provided.")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="./outputs/profile_breakdown")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to real dataset (e.g. data/trec.jsonl). "
                             "When provided together with --task, real tokenized "
                             "inputs are used instead of random dummy tokens.")
    parser.add_argument("--task", type=str, default=None,
                        choices=["trec"],
                        help="Task type for real-data profiling (currently: trec).")
    parser.add_argument("--sample_indices", type=str, default=None,
                        help="Comma-separated list of dataset indices to use "
                             "(e.g. '96,97'). When provided, overrides the "
                             "default behaviour of taking the first --batch_size "
                             "samples. The number of indices must match "
                             "--batch_size.")

    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Normalize method name to what the model's parallel_retention_update expects.
    # The single-layer CLI in test_method.py uses lowercase; the model uses camelCase.
    method = _normalize_method_name(args.method)
    if method != args.method:
        print(f"Normalized method name: {args.method} -> {method}")

    print(f"Loading model from {args.model_path} ...")

    # Auto-detect model type from path (same logic as main.py)
    if "transnormer" in args.model_path.lower():
        model = TransnormerForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device,
        )
    else:
        model = RetNetModelWithLMHead.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device,
        )
    model.eval()

    model_type = _resolve_model_type(args.model_path, model)
    print(f"Detected model type: {model_type}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Real-input pipeline (optional)
    # ------------------------------------------------------------------
    real_input_ids: Optional[torch.Tensor] = None
    real_attention_mask: Optional[torch.Tensor] = None

    if args.data_path is not None and args.task is not None:
        print(f"Loading tokenizer for real-input profiling ...")
        tokenizer = _load_tokenizer_for_model(args.model_path)
        print(f"Loading data from {args.data_path} (task={args.task}) ...")

        if args.task == "trec":
            dataset = []
            with open(args.data_path, "r") as f:
                for line in f:
                    dataset.append(json.loads(line))

            # Determine which sample indices to use
            if args.sample_indices is not None:
                sample_indices = [int(x.strip()) for x in args.sample_indices.split(",")]
                if len(sample_indices) != args.batch_size:
                    raise ValueError(
                        f"Number of --sample_indices ({len(sample_indices)}) "
                        f"does not match --batch_size ({args.batch_size})."
                    )
            else:
                sample_indices = list(range(args.batch_size))

            # Build one batch from the requested samples
            batch_prompts = []
            for j in sample_indices:
                if j >= len(dataset):
                    raise ValueError(
                        f"Sample index {j} out of range (dataset has {len(dataset)} samples)."
                    )
                item = dataset[j]
                all_classes = item.get("all_classes", [])
                context = item.get("context", "")
                input_text = item.get("input", "")

                prompt = "There are some choice you can choose.\n"
                for cls_name in all_classes:
                    prompt += f"{cls_name}\n"
                prompt += "\nThese are some examples:\n" + context
                prompt += "\nPlease answer the following question:\n" + input_text
                batch_prompts.append(prompt)

            encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True)
            real_input_ids = encoded.input_ids
            real_attention_mask = encoded.attention_mask
            print(f"Tokenized real batch: input_ids shape={real_input_ids.shape}, "
                  f"attention_mask shape={real_attention_mask.shape}")
        else:
            raise ValueError(f"Unsupported task for real-data profiling: {args.task}")

    # ------------------------------------------------------------------
    # Run profiling
    # ------------------------------------------------------------------
    result = profile_breakdown(
        model=model,
        model_type=model_type,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        dtype=dtype,
        device=device,
        attention_method=method,
        repeats=args.repeats,
        warmup=args.warmup,
        input_ids=real_input_ids,
        attention_mask=real_attention_mask,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Total time: {result['total_ms']['mean']:.3f} ms")
    print(f"{'='*60}")
    for key, stats in result["breakdown"].items():
        pct = stats["mean"] / result["total_ms"]["mean"] * 100
        print(f"  {key:<22s}: {stats['mean']:8.3f} ms  ({pct:5.1f}%)")
    print(f"{'='*60}")

    # Per-layer summary
    print(f"\nPer-layer breakdown (mean over {args.repeats} runs):")
    print(f"{'Layer':<8} {'attn_module':>12} {'attn_core':>12} {'attn_non_core':>14} {'mlp':>10}")
    print("-" * 60)
    per_layer = result["per_layer"]
    for lidx_str in sorted(per_layer.keys(), key=lambda x: int(x)):
        ld = per_layer[lidx_str]
        attn_mod = ld.get("attn_module", 0)
        attn_core = ld.get("attn_core", 0)
        mlp = ld.get("mlp", 0)
        print(f"L{lidx_str:<6} {attn_mod:12.3f} {attn_core:12.3f} {attn_mod - attn_core:14.3f} {mlp:10.3f}")

    # Save results
    actual_seq_len = real_input_ids.shape[-1] if real_input_ids is not None else args.seq_len
    out_name = f"{model_type}_{method}_b{args.batch_size}_n{actual_seq_len}_{args.dtype}.json"
    out_path = os.path.join(args.output_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
