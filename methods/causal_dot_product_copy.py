import torch
from torch.utils.cpp_extension import load
import os
import time
from methods.FleetAttention import fleetAttention
from methods.linear_attn import _build_slope_tensor
import sys
import os


current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)


cuda_module_fp32_filePath = os.path.join(current_file_dir,'causal_product_cuda.cu')


# load cuda module
causal_dot_product_fp32 = load(name="causal_dot_product_fp32_cuda",sources=cuda_module_fp32_filePath,verbose=False)


class CausalDotProduct(torch.autograd.Function):
    """Compute the weighted sum of values but attending only to previous
    values."""

    @staticmethod
    def forward(ctx, Q, K, V):
        # print(CausalDotProduct.dot[Q.device.type])
        # Save the inputs for the gradient computation
        # ctx.save_for_backward(Q, K, V)

        # Create the output tensor
        device = Q.device
        N, H, L, _ = Q.shape
        _, _, _, M = V.shape
        product = torch.zeros((N, H, L, M), device=device)
        
        # print(product.shape)
        # Actually perform the dot product
        causal_dot_product_fp32.causal_dot_product(
            Q.data,
            K.data,
            V.data,
            product
        )
        # print(product.shape)
        return product

# Alias the autograd functions to python style snake case naming
causal_dot_product = CausalDotProduct.apply




if __name__ == '__main__':
    b,h,n,r=1,32,8192,128
    d=128
    type = torch.float32
    device = torch.device("cuda:2")
    # gamma = torch.full((h,),0.5,dtype=type,device=device)
    gamma = _build_slope_tensor(h).to(dtype=type,device=device).reshape(h)
    B = torch.randn(b,h,n,r,dtype=type,device=device)
    C = torch.randn(b,h,n,r,dtype=type,device=device)
    V = torch.randn(b,h,n,d,dtype=type,device=device)
    # start_time = torch.cuda.Event(enable_timing=True)
    # end_time = torch.cuda.Event(enable_timing=True)
    # start_time.record()
    for i in range(40):
        # ans = causal_dot_product(B,C,V)
        ans = fleetAttention(B,C,V,gamma)
    torch.cuda.synchronize()
    start_time = time.time()
    # ans = causal_dot_product(B,C,V)
    ans2 = fleetAttention(B,C,V,gamma)
    # end_time.record()
    # torch.cuda.synchronize()
    
    # start_time2 = time.time()
    # end_time2 = torch.cuda.Event(enable_timing=True)
    # start_time2.record()
    # ans2 = fleetAttention(B,C,V,gamma)
    # ans = causal_dot_product(B,C,V)
    # end_time2.record()
    torch.cuda.synchronize()
    end_time2 = time.time()
    
    # print('method time:',start_time2-start_time)
    print('method2 time:',(end_time2-start_time))
    # print('norm difference:',torch.norm(ans2-ans)/torch.norm(ans))



    # with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
    #                scheduler = (2, 5),
    #                on_trace_ready = profiler.export_chrome_tracing('./profiler_log'),
    #                timer_only = False) as p:
    #     ans2 = fleetAttention(B,C,V,gamma)
    #     p.step()
