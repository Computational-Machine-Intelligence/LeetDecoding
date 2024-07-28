from methods.FleetAttention import fleetAttention
from methods.lightningAttention2 import lightning_attn2
from methods.causal_dot_product import causal_dot_product
import torch
import pynvml


DTYPE_MAP = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4
}


# 获取一个GPU当前剩余可用的显存大小（以字节为单位）
def get_gpu_memory(gpu_idx):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return mem_info.total,mem_info.free

def causal_linear_decoder(q,k,v,is_mask_weight=False, gamma=None):
    """_summary_

    Args:
        q (tensor): The shape of q is (batch_size, heads, seqlen, rank)
        k (tensor): The shape of k is (batch_size, heads, seqlen, rank)
        v (tensor): The shape of v is (batch_size, heads, seqlen, dim)
        is_mask_weight (bool): Whether to use the mask weight, False uses the normal causal mask, True uses a mask with weight decay.
        gamma (tensor): The shape of gamma is (heads). The scaling factor for the attention weights.
    """
    batch_size, heads, seqlen, rank= q.shape
    dim = v.shape[-1]
    
    assert gamma.shape[0]==heads, 'The shape of gamma must be consistent with the head of q,k,v' 
    type = q.dtype
    device = q.device
    gpu_memory, gpu_memory_free = get_gpu_memory(device.index)
    if 2 * batch_size * heads * seqlen * rank * DTYPE_MAP[type] + batch_size * heads * seqlen * dim * DTYPE_MAP[type] > gpu_memory_free:
        raise Exception("GPU memory is not enough, please use smaller data.")
    if type == torch.float16:
        if is_mask_weight:
            assert gamma is not None, 'The gamma parameter must be passed when using the mask weight'
        ans = lightning_attn2(q,k,v,gamma)
    elif type == torch.float32:
        if is_mask_weight:
            assert gamma is not None, 'The gamma parameter must be passed when using the mask weight'
        if batch_size > 1 and seqlen>=1024: # 如果batch_size>1并且采用mask, 则可以使用带有cuda 优化的FleetAttention
            if gamma is None:
                # 采用causal-dot-product with cuda optimization
                ans = causal_dot_product(q,k,v)
            else:
                ans = fleetAttention(q,k,v,gamma)
        else:
            ans = lightning_attn2(q,k,v,gamma)
    else:
        raise Exception('Not implement the type',type)
    return ans


if __name__=='__main__':
    # 获取并打印第一个GPU的显存大小（以字节为单位）
    gpu_memory,gpu_memory_free = get_gpu_memory(1)
    print(f"Total memory of GPU 0: {gpu_memory / (1024 ** 3):.2f} GB")
    print(f'Free memory of GPU 0: {gpu_memory_free / (1024 ** 3):.2f} GB')

    # 获取设备的数量
    device_count = drv.Device.count()
    x = torch.tensor((1,2),device=torch.device('cuda:1'),dtype=torch.float16)
    for i in range(device_count):
        # 获取设备句柄
        device = drv.Device(i)
        
        # 获取设备属性
        attrs = device.get_attributes()
        
        # 打印设备信息
        print(f"Device {i}: {device.name()}")
        print(f"  Shared memory per block: {attrs[drv.device_attribute.SHARED_MEMORY_PER_BLOCK]} bytes")