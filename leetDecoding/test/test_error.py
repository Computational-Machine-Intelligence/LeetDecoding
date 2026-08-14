import argparse
import json
import math
import os

import torch

from leetDecoding.methods.BlockBased import blockBased
from leetDecoding.methods.FleetAttention import FleetAttention
from leetDecoding.methods.FleetAttention_triton import FleetAttention_triton
from leetDecoding.methods.Recursion import recursion
from leetDecoding.methods.causal_dot_product import causal_dot_product
from leetDecoding.methods.causal_dot_product_torch import causal_dot_product_torch
from leetDecoding.methods.lightningAttention2 import lightning_attn2
from leetDecoding.methods.lightningAttention2_optimized import lightning_attn2_optimized
from leetDecoding.methods.lightningAttention2_prefetch import lightning_attn2_prefetch
from leetDecoding.methods.lightningAttention2_torch import lightningAttention2_torch
from leetDecoding.methods.linear_attn import _build_slope_tensor, linear_attn


METHOD_MAP = {
    'FleetAttention_torch': FleetAttention,
    'FleetAttention': FleetAttention_triton,
    'lightningAttention2': lightning_attn2,
    'lightningAttention2_prefetch': lightning_attn2_prefetch,
    'LA_prefetch': lightning_attn2_prefetch,
    'causal_dot_product_torch': causal_dot_product_torch,
    'BCMV_vanilla': linear_attn,
    'recursion': recursion,
    'blockbased': blockBased,
    'causal_dot_product': causal_dot_product,
    'lightningAttention2_torch': lightningAttention2_torch,
    'lightningAttention2_optimized': lightning_attn2_optimized,
}

METHODS_USING_SLOPE = {
    'BCMV_vanilla',
    'lightningAttention2',
    'lightningAttention2_prefetch',
    'LA_prefetch',
    'recursion',
    'blockbased',
    'lightningAttention2_torch',
    'lightningAttention2_optimized',
}

DEFAULT_METHODS = [
    'BCMV_vanilla',
    'FleetAttention',
    'FleetAttention_torch',
    'lightningAttention2',
    'lightningAttention2_prefetch',
    'lightningAttention2_torch',
    'lightningAttention2_optimized',
    'causal_dot_product',
    'causal_dot_product_torch',
    'recursion',
    'blockbased',
]

DTYPE_MAP = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float64': torch.float64,
}


def resolve_method(method_name):
    if method_name == 'lightningAttention2_origin':
        from lightning_attn.ops import lightning_attn_func
        return lightning_attn_func
    if method_name not in METHOD_MAP:
        raise ValueError(f'Unimplemented method name: {method_name}')
    return METHOD_MAP[method_name]


# Row-chunk size for the exact FP64 reference (query dimension).
_REF_CHUNK = 1024


def build_decay_mask_fp64(seqlen, slopes, device, start=0):
    # Row-chunked mask builder: the full mask is (heads, seqlen, seqlen) fp64,
    # which needs ~16 GiB at seqlen=8192. Chunking the query rows keeps memory
    # at O(chunk * seqlen) and yields the same per-row values.
    end = min(start + _REF_CHUNK, seqlen)
    positions = torch.arange(seqlen, device=device, dtype=torch.float64)
    row_positions = torch.arange(start, end, device=device, dtype=torch.float64)
    raw_distance = row_positions[:, None] - positions[None, :]
    distance = raw_distance.clamp_min(0)
    causal = (raw_distance >= 0).to(torch.float64)
    decay = torch.exp(-slopes.to(torch.float64).reshape(-1, 1, 1) * distance.unsqueeze(0))
    return decay * causal.unsqueeze(0)


def bcmv_vanilla_fp64(q, k, v, slopes=None):
    # Exact FP64 reference, computed in query-row chunks. The naive version
    # materializes scores/mask of shape (b, h, seqlen, seqlen) fp64 (~16 GiB at
    # seqlen=8192) and OOMs on 32 GB GPUs. Chunking the query dimension only
    # changes GEMM blocking, not the per-row reduction, so results are unchanged
    # up to fp64 rounding.
    seqlen = q.shape[2]
    k_t = k.transpose(2, 3)
    out = torch.empty_like(v)
    if slopes is None:
        positions = torch.arange(seqlen, device=q.device)
        for start in range(0, seqlen, _REF_CHUNK):
            end = min(start + _REF_CHUNK, seqlen)
            scores_chunk = torch.matmul(q[:, :, start:end], k_t)
            mask_chunk = (positions[start:end, None] >= positions[None, :]).to(torch.float64)
            out[:, :, start:end] = torch.matmul(scores_chunk * mask_chunk, v)
    else:
        for start in range(0, seqlen, _REF_CHUNK):
            end = min(start + _REF_CHUNK, seqlen)
            scores_chunk = torch.matmul(q[:, :, start:end], k_t)
            decay_chunk = build_decay_mask_fp64(seqlen, slopes, q.device, start=start)
            out[:, :, start:end] = torch.matmul(scores_chunk * decay_chunk, v)
    return out


def make_trial_inputs(batch_size, heads, seqlen, rank, dim, device, seed, is_weight_decay):
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)

    scale = math.sqrt(rank)
    q = torch.randn(batch_size, heads, seqlen, rank, device=device, dtype=torch.float64) / scale
    k = torch.randn(batch_size, heads, seqlen, rank, device=device, dtype=torch.float64) / scale
    v = torch.randn(batch_size, heads, seqlen, dim, device=device, dtype=torch.float64) / scale

    slopes = None
    if is_weight_decay:
        slopes = _build_slope_tensor(heads).reshape(heads).to(device=device, dtype=torch.float64)

    return q, k, v, slopes


def run_method(method_name, q, k, v, slopes=None):
    method = resolve_method(method_name)
    if slopes is None:
        out = method(q, k, v)
    elif method_name in METHODS_USING_SLOPE:
        out = method(q, k, v, slopes)
    else:
        out = method(q, k, v, torch.exp(-slopes))
    if isinstance(out, tuple):
        out = out[0]
    return out


def relative_frobenius_error(exact_output, approx_output):
    exact_output = exact_output.to(torch.float64)
    approx_output = approx_output.to(torch.float64)
    numerator = torch.linalg.vector_norm((exact_output - approx_output).reshape(-1))
    denominator = torch.linalg.vector_norm(exact_output.reshape(-1)).clamp_min(torch.finfo(torch.float64).tiny)
    return (numerator / denominator).item()


def summarize_results(results, trials):
    summary = []
    for method_name, dtype_results in results.items():
        for dtype_name, stats in dtype_results.items():
            errors = stats['errors']
            item = {
                'method': method_name,
                'dtype': dtype_name,
                'successful_trials': len(errors),
                'failed_trials': stats['failed_trials'],
                'mean_rel_error': None,
                'max_rel_error': None,
                'last_error': stats['last_error'],
            }
            if errors:
                item['mean_rel_error'] = sum(errors) / len(errors)
                item['max_rel_error'] = max(errors)
            summary.append(item)
    summary.sort(key=lambda item: (item['method'], item['dtype']))
    print(f'Reference: BCMV_vanilla exact FP64 on dtype-quantized inputs over {trials} trial(s)')
    print('| Method | Dtype | Mean Rel. Error | Max Rel. Error | Success |')
    print('| --- | --- | --- | --- | --- |')
    for item in summary:
        if item['mean_rel_error'] is None:
            mean_error = 'FAILED'
            max_error = 'FAILED'
        else:
            mean_error = f"{item['mean_rel_error']:.2e}"
            max_error = f"{item['max_rel_error']:.2e}"
        success = f"{item['successful_trials']}/{trials}"
        print(f"| {item['method']} | {item['dtype']} | {mean_error} | {max_error} | {success} |")
        if item['last_error'] is not None and item['successful_trials'] != trials:
            print(f"  last error: {item['last_error']}")
    return summary


def maybe_save_json(output_path, payload):
    if output_path is None:
        return
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def maybe_save_exact_output(save_path, exact_output):
    if save_path is None:
        return
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if isinstance(exact_output, dict):
        payload = {dtype_name: tensor.detach().cpu() for dtype_name, tensor in exact_output.items()}
    else:
        payload = exact_output.detach().cpu()
    torch.save(payload, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='Batch size.')
    parser.add_argument('--heads', type=int, default=32, help='Number of attention heads.')
    parser.add_argument('--n', type=int, default=1024, help='Sequence length.')
    parser.add_argument('--rank', type=int, default=128, help='Feature dimension of q and k.')
    parser.add_argument('--dim', type=int, default=128, help='Feature dimension of v.')
    parser.add_argument('--trials', type=int, default=100, help='Number of random trials.')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed.')
    parser.add_argument(
        '--methods',
        nargs='+',
        default=DEFAULT_METHODS,
        help='Methods to compare against the BCMV_vanilla FP64 reference.',
    )
    parser.add_argument(
        '--dtypes',
        nargs='+',
        default=['float32', 'bfloat16'],
        help='Approximate dtypes to test. Typical use is float32 and bfloat16.',
    )
    parser.add_argument('--is-weight-decay', action='store_true', help='Enable exponentially decaying mask.')
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device used for the experiment.',
    )
    parser.add_argument('--output-path', type=str, default=None, help='Optional JSON path for summary results.')
    parser.add_argument(
        '--save-exact-output',
        type=str,
        default=None,
        help='Optional .pt path to save the first-trial FP64 reference. Saves a dict keyed by dtype when multiple dtypes are requested.',
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    for dtype_name in args.dtypes:
        if dtype_name not in DTYPE_MAP:
            raise ValueError(f'Unsupported dtype: {dtype_name}')

    results = {
        method_name: {
            dtype_name: {'errors': [], 'failed_trials': 0, 'last_error': None}
            for dtype_name in args.dtypes
        }
        for method_name in args.methods
    }

    saved_exact_output = False
    for trial_idx in range(args.trials):
        q64, k64, v64, slopes64 = make_trial_inputs(
            args.batch,
            args.heads,
            args.n,
            args.rank,
            args.dim,
            device,
            args.seed + trial_idx,
            args.is_weight_decay,
        )
        first_trial_exact_outputs = {} if not saved_exact_output else None

        for dtype_name in args.dtypes:
            dtype = DTYPE_MAP[dtype_name]
            q = q64.to(dtype=dtype)
            k = k64.to(dtype=dtype)
            v = v64.to(dtype=dtype)
            slopes = None if slopes64 is None else slopes64.to(dtype=dtype)
            q_ref = q.to(torch.float64)
            k_ref = k.to(torch.float64)
            v_ref = v.to(torch.float64)
            slopes_ref = None if slopes is None else slopes.to(torch.float64)
            exact_output = bcmv_vanilla_fp64(q_ref, k_ref, v_ref, slopes_ref)
            if first_trial_exact_outputs is not None:
                first_trial_exact_outputs[dtype_name] = exact_output

            for method_name in args.methods:
                stats = results[method_name][dtype_name]
                try:
                    approx_output = run_method(method_name, q, k, v, slopes)
                    error = relative_frobenius_error(exact_output, approx_output)
                    stats['errors'].append(error)
                except Exception as exc:
                    stats['failed_trials'] += 1
                    stats['last_error'] = str(exc)

        if first_trial_exact_outputs is not None:
            exact_output_payload = first_trial_exact_outputs
            if len(first_trial_exact_outputs) == 1:
                exact_output_payload = next(iter(first_trial_exact_outputs.values()))
            maybe_save_exact_output(args.save_exact_output, exact_output_payload)
            saved_exact_output = True

    summary = summarize_results(results, args.trials)
    maybe_save_json(
        args.output_path,
        {
            'reference': 'BCMV_vanilla FP64 on dtype-quantized inputs',
            'config': {
                'batch': args.batch,
                'heads': args.heads,
                'seqlen': args.n,
                'rank': args.rank,
                'dim': args.dim,
                'trials': args.trials,
                'seed': args.seed,
                'device': str(device),
                'is_weight_decay': args.is_weight_decay,
                'methods': args.methods,
                'dtypes': args.dtypes,
            },
            'summary': summary,
        },
    )


# python -m leetDecoding.test.test_error --methods BCMV_vanilla FleetAttention lightningAttention2_optimized --dtypes float32 bfloat16 --n 1024 --trials 100 --output-path ./outputs/error_summary.json