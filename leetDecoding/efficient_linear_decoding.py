from leetDecoding.methods.lightningAttention2 import lightning_attn2, GPU_MAP
from leetDecoding.methods.causal_dot_product import causal_dot_product
import torch
import pynvml
import pycuda.driver as drv


DTYPE_MAP = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4
}


def get_gpu_memory(gpu_idx):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return mem_info.total,mem_info.free


def causal_linear_decoder(q,k,v,is_mask_weight=False, gamma=None,is_need_exp=True):
    """_summary_

    Args:
        q (tensor): The shape of q is (batch_size, heads, seqlen, rank)
        k (tensor): The shape of k is (batch_size, heads, seqlen, rank)
        v (tensor): The shape of v is (batch_size, heads, seqlen, dim)
        is_mask_weight (bool): Whether to use the mask weight, False uses the normal causal mask, True uses a mask with weight decay.
        gamma (tensor): The shape of gamma is (heads). The scaling factor for the attention weights.
        is_need_exp: To obtain the weights for the mask, whether we need to apply the operation exp(-gamma) to the gamma parameter.
    """
    batch_size, heads, seqlen, rank= q.shape
    dim = v.shape[-1]
    if gamma is not None:
        if not is_mask_weight:
            raise Exception("The gamma parameter must be passed when using the mask weight")
        assert gamma.shape[0]==heads, 'The shape of gamma must be consistent with the head of q,k,v' 
    type = q.dtype
    device = q.device
    gpu_memory, gpu_memory_free = get_gpu_memory(device.index)
    if 2 * batch_size * heads * seqlen * rank * DTYPE_MAP[type] + batch_size * heads * seqlen * dim * DTYPE_MAP[type] > gpu_memory_free:
        raise Exception("GPU memory is not enough, please use smaller data.")
    if gamma is not None and not is_need_exp:
        if type==torch.float32:
            ans = causal_dot_product(q,k,v,gamma)
        else:
            ans = causal_dot_product(q.to(torch.float32),k.to(torch.float32),v.to(torch.float32),gamma.to(torch.float32)).to(type)
    else:
        if type == torch.float16:
            if seqlen > GPU_MAP[drv.Device(0).name()]: # sequence length must be larger than the lightningAttention block size
                ans = lightning_attn2(q,k,v,gamma)
            else:
                ans = causal_dot_product(q,k,v,gamma if gamma is None else torch.exp(-gamma))
        elif type == torch.float32:
            if batch_size > 1 and seqlen>=1024: 
                if gamma is None:
                    ans = causal_dot_product(q,k,v)
                else:
                    ans = causal_dot_product(q,k,v,torch.exp(-gamma))
            else:
                ans = lightning_attn2(q,k,v,gamma)
        else:
            raise Exception('Not implement the type',type)
    return ans


if __name__=='__main__':
    Q = torch.randn(1,32,1024,128,dtype=torch.float16,device='cuda:0')
    K = torch.randn(1,32,1024,128,dtype=torch.float16,device='cuda:0')
    V = torch.randn(1,32,1024,128,dtype=torch.float16,device='cuda:0')
    gamma = torch.randn((32,),dtype=torch.float16,device='cuda:0')
    ans = causal_linear_decoder(Q,K,V,gamma=gamma,is_mask_weight=True)
    correct_ans = torch.matmul(torch.tril(torch.matmul(Q,K.transpose(2,3))) ,V)
    print('ours norm:',torch.norm(ans),'\ncorrect norm:',torch.norm(correct_ans),'\ndifference norm:',torch.norm(correct_ans-ans))
