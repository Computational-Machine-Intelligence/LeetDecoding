from leetDecoding.methods.lightningAttention2 import lightning_attn2, GPU_MAP
from leetDecoding.methods.causal_dot_product import causal_dot_product
import torch
import pynvml
import pycuda.driver as drv
from leetDecoding.methods.linear_attn import linear_attn
from leetDecoding.methods.causal_dot_product_torch import causal_dot_product_torch
from leetDecoding.methods.BlockBased import blockBased
from leetDecoding.methods.Recursion import recursion
from leetDecoding.methods.lightningAttention2_torch import lightningAttention2_torch
from leetDecoding.methods.lightningAttention2_optimized import lightning_attn2_optimized
from leetDecoding.methods.FleetAttention import FleetAttention
from leetDecoding.methods.FleetAttention_triton import FleetAttention_triton


DTYPE_MAP = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4
}

ATTENTION_MAP = {
    "vanilla": linear_attn,
    'lightningAttention2': lightning_attn2,
    'lightningAttention2_torch': lightningAttention2_torch,
    'lightningAttention2_optimized': lightning_attn2_optimized,
    'FleetAttention_torch': FleetAttention,
    'FleetAttention': FleetAttention_triton,
    'causal_dot_product': causal_dot_product,
    'causal_dot_product_torch': causal_dot_product_torch,
    'recursion': recursion,
    "block-based": blockBased,
}


def get_gpu_memory(gpu_idx):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return mem_info.total, mem_info.free


def causal_linear_decoder(
    q, k, v,
    is_mask_weight=False,
    gamma=None,
    is_need_exp=True,
    attn_method=None,
    autotune_configs=None,
):
    """_summary_

    Args:
        q (tensor): The shape of q is (batch_size, heads, seqlen, rank)
        k (tensor): The shape of k is (batch_size, heads, seqlen, rank)
        v (tensor): The shape of v is (batch_size, heads, seqlen, dim)
        is_mask_weight (bool): Whether to use the mask weight, False uses the normal causal mask, True uses a mask with weight decay.
        gamma (tensor): The shape of gamma is (heads). The scaling factor for the attention weights.
        is_need_exp: To obtain the weights for the mask, whether we need to apply the operation exp(-gamma) to the gamma parameter.
        autotune_configs: Optional list of triton.Config for autotuning.
    """
    batch_size, heads, seqlen, rank = q.shape
    dim = v.shape[-1]
    if gamma is not None:
        if not is_mask_weight:
            raise Exception("The gamma parameter must be passed when using the mask weight")
        assert gamma.shape[0] == heads, 'The shape of gamma must be consistent with the head of q,k,v'
    type = q.dtype
    device = q.device
    gpu_memory, gpu_memory_free = get_gpu_memory(device.index)
    if 2 * batch_size * heads * seqlen * rank * DTYPE_MAP[type] + batch_size * heads * seqlen * dim * DTYPE_MAP[type] > gpu_memory_free:
        raise Exception("GPU memory is not enough, please use smaller data.")

    call_kwargs = {'autotune_configs': autotune_configs}

    if gamma is not None and not is_need_exp:
        if type == torch.float32:
            if attn_method is not None:
                attention_method = ATTENTION_MAP[attn_method]
                ans = attention_method(q, k, v, gamma)
            else:
                ans = lightning_attn2_optimized(q, k, v, gamma, **call_kwargs)
        else:
            if attn_method is not None:
                attention_method = ATTENTION_MAP[attn_method]
                ans = attention_method(q, k, v, gamma)
            else:
                ans = lightning_attn2_optimized(q, k, v, gamma, **call_kwargs).to(type)
    else:
        if type == torch.float16:
            if seqlen > GPU_MAP[drv.Device(0).name()]:
                if attn_method is not None:
                    attention_method = ATTENTION_MAP[attn_method]
                    ans = attention_method(q, k, v, gamma)
                else:
                    ans = lightning_attn2_optimized(q, k, v, gamma, **call_kwargs)
            else:
                gamma_input = gamma if gamma is None else torch.exp(-gamma)
                if attn_method is not None:
                    attention_method = ATTENTION_MAP[attn_method]
                    ans = attention_method(q, k, v, gamma_input)
                else:
                    ans = lightning_attn2_optimized(q, k, v, gamma_input, **call_kwargs)
        elif type == torch.float32:
            if batch_size > 1 and seqlen >= 1024:
                gamma_input = torch.exp(-gamma) if gamma is not None else None
                if attn_method is not None:
                    attention_method = ATTENTION_MAP[attn_method]
                    ans = attention_method(q, k, v, gamma_input)
                else:
                    ans = lightning_attn2_optimized(q, k, v, gamma_input, **call_kwargs)
            else:
                if attn_method is not None:
                    attention_method = ATTENTION_MAP[attn_method]
                    ans = attention_method(q, k, v, gamma)
                else:
                    ans = lightning_attn2_optimized(q, k, v, gamma, **call_kwargs)
        else:
            raise Exception('Not implement the type', type)
    return ans


# -----------------------------
# Example usage
# -----------------------------

if __name__ == '__main__':
    Q = torch.randn(1, 32, 1024, 128, dtype=torch.float16, device='cuda:0')
    K = torch.randn(1, 32, 1024, 128, dtype=torch.float16, device='cuda:0')
    V = torch.randn(1, 32, 1024, 128, dtype=torch.float16, device='cuda:0')
    gamma = torch.randn((32,), dtype=torch.float16, device='cuda:0')


    ans1 = causal_linear_decoder(Q, K, V, gamma=gamma, is_mask_weight=True)

    custom_configs = [
        triton.Config({'BLOCK': 64, 'BLOCK_MODEL': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 128, 'BLOCK_MODEL': 64}, num_warps=8, num_stages=3),
    ]
    ans2 = causal_linear_decoder(Q, K, V, gamma=gamma, is_mask_weight=True, autotune_configs=custom_configs)

    correct_ans = torch.matmul(torch.tril(torch.matmul(Q, K.transpose(2, 3))), V)
    print('ours norm:', torch.norm(ans1), '\ncorrect norm:', torch.norm(correct_ans), '\ndifference norm:', torch.norm(correct_ans - ans1))