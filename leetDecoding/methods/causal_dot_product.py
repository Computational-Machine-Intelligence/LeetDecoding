import torch
from torch.utils.cpp_extension import load
import os
import time


current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)

cuda_module_with_decay_fp32_filePath = os.path.join(current_file_dir,'causal_product_cuda_with_decay_fp32.cu')
cuda_module_with_decay_fp16_filePath = os.path.join(current_file_dir,'causal_product_cuda_with_decay_fp16.cu')
cuda_module_causal_dot_product_filePath = os.path.join(current_file_dir,'causal_product_cuda.cu')


# load cuda module
cuda_module_with_decay_fp32 = load(name="causal_product_cuda_with_decay_fp32_cuda",sources=cuda_module_with_decay_fp32_filePath,verbose=False)
cuda_module_with_decay_fp16 = load(name="causal_product_cuda_with_decay_fp16_cuda",sources=cuda_module_with_decay_fp16_filePath,verbose=False)
cuda_module_causal_dot_product = load(name='causal_dot_product_cuda',sources=cuda_module_causal_dot_product_filePath,verbose=False)


class CausalDotProduct(torch.autograd.Function):
    """Compute the weighted sum of values but attending only to previous
    values."""

    @staticmethod
    def forward(ctx, Q, K, V, gamma=None):
        # Create the output tensor
        dtype = Q.dtype
        device = Q.device
        N, H, L, _ = Q.shape
        _, _, _, M = V.shape
        product = torch.zeros((N, H, L, M), device=device, dtype=torch.float32)
        if gamma is None:
            if dtype == torch.float32:
                cuda_module_causal_dot_product.causal_dot_product(
                    Q.data,
                    K.data,
                    V.data,
                    product.data
                )
            elif dtype == torch.float16:
                gamma = torch.ones((H,),dtype=dtype,device=device)
                cuda_module_with_decay_fp16.causal_dot_product(
                    Q.data,
                    K.data,
                    V.data,
                    gamma.data,
                    product
                )
            else:
                print('dtype not supported')
        else:
            if dtype == torch.float32:
                cuda_module_with_decay_fp32.causal_dot_product(
                    Q.data,
                    K.data,
                    V.data,
                    gamma.data,
                    product
                )
            elif dtype == torch.float16:
                cuda_module_with_decay_fp16.causal_dot_product(
                    Q.data,
                    K.data,
                    V.data,
                    gamma.data,
                    product
                )
            else:
                print('dtype not supported')
        return product


causal_dot_product = CausalDotProduct.apply
