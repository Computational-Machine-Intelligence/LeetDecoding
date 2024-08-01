import torch
import time
import math
from methods.causal_dot_product import causal_dot_product
from methods.RowBased import rowBased
from methods.lightningAttention2 import lightning_attn2
from methods.BlockBased import blockBased
from methods.Recursion import recursion
from methods.FleetAttention import FleetAttention
from methods.linear_attn import linear_attn, _build_slope_tensor
import argparse
import torch.utils.benchmark as benchmark


# Analyze the performance of the forward function
def benchmark_forward(
    fn, *inputs, repeats=10, desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function.
        fn: The function to benchmark.
        inputs: The inputs to the function.
        repeats": The number of times to repeat the benchmark.
        desc: A description of the benchmark.
        verbose: Whether to print the benchmark results.
        amp: Whether to use AMP.
        amp_dtype: The AMP data type.
        kwinputs: Additional keyword arguments to pass to the function.
    """
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)
    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


# Detailed analysis of the time of each part (operator)
def pytorch_profiler(
    fn,
    *inputs,
    trace_filename=None,
    backward=False,
    amp=False,
    amp_dtype=torch.float16,
    cpu=False,
    verbose=True,
    **kwinputs,
):
    """Wrap benchmark functions in Pytorch profiler to see CUDA information.
        fn: The function to benchmark.
        inputs: The inputs to the function.
        trace_filename: The filename to save the trace.
        backward: Whether to run the backward pass.
        amp: Whether to use AMP.
        amp_dtype: The AMP data type.
        cpu: Whether to use CPU.
        verbose: Whether to print the profiler results."""
    if backward:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
            g = torch.randn_like(out)
    for _ in range(30):  # Warm up
        if backward:
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    x.grad = None
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
        # Backward should be done outside autocast
        if backward:
            out.backward(g, retain_graph=True)
    activities = ([torch.profiler.ProfilerActivity.CPU] if cpu else []) + [
        torch.profiler.ProfilerActivity.CUDA
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        # profile_memory=True,
        with_stack=True,
    ) as prof:
        if backward:
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    x.grad = None
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
        if backward:
            out.backward(g, retain_graph=True)
    if verbose:
        # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
        print(prof.key_averages().table(row_limit=50))
    if trace_filename is not None:
        prof.export_chrome_trace(trace_filename)


# Analyze the memory used by the program
def benchmark_memory(fn, *inputs, desc="", verbose=True, **kwinputs):
    """
        This function is used to benchmark the video memory usage of the function.
        fn: The function to benchmark.
        inputs: The inputs to the function.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*inputs, **kwinputs)
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / ((2**20) * 1000)
    if verbose:
        print(f"{desc} max memory: {mem}GB")
    torch.cuda.empty_cache()
    return mem
    
    
def test_BCMV_by_random(onlyMethod,b,h,n,r,d,method,type,device,is_weight_decay):
    B = torch.randn(b,h,n,r,dtype=type,device=device)
    C = torch.randn(b,h,n,r,dtype=type,device=device)
    V = torch.randn(b,h,n,d,dtype=type,device=device)
    if is_weight_decay:
        s = _build_slope_tensor(h).to(dtype=type,device=device).reshape(h)
    if onlyMethod:
        if is_weight_decay:
            if method !=lightning_attn2 and method != recursion and method!=blockBased:
                benchmark_forward(method,B,C,V,torch.exp(-s),verbose=True)
                benchmark_memory(method,B,C,V,torch.exp(-s),verbose=True)
            else:
                benchmark_forward(method,B,C,V,s,verbose=True)
                benchmark_memory(method,B,C,V,s,verbose=True)
        else:
            benchmark_forward(method,B,C,V,verbose=True)
            benchmark_memory(method,B,C,V,verbose=True)
    else:
        if is_weight_decay:
            correct_BCMV = linear_attn(B,C,V,s)
            if method !=lightning_attn2 and method!=linear_attn and method!=recursion and method!=blockBased:
                BCMV = method(B,C,V,torch.exp(-s))
            else:
                BCMV = method(B,C,V,s)
            print('method norm:',torch.norm(BCMV),'vanilla norm:',torch.norm(correct_BCMV),'difference norm:',torch.norm(BCMV-correct_BCMV))
        else:
            M = torch.tril(torch.ones((n,n),dtype=type,device=device))
            correct_BCMV = torch.matmul(torch.matmul(B,C.transpose(2,3)) * M,V)
            BCMV = method(B,C,V)
            print('method norm:',torch.norm(BCMV),'vanilla norm:',torch.norm(correct_BCMV),'difference norm:',torch.norm(BCMV-correct_BCMV))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch',help='The batch size of the test, the default is 1',default=1)
    parser.add_argument('--n',help='The sequence length of the test, the default is 8192',default=8192)
    parser.add_argument('--method', help='The method under test will execute both vanilla and the corresponding method. The method can be FleetAttention, lightningAttention, BCMV_vanilla, rowbased, recursion, blockbased, causal_dot_product.')
    parser.add_argument('--type',help='Numeric type, including float16, float32, the default is float16.',default='float16')
    parser.add_argument('--gpu',help='gpu number, default is 0.',default='0')
    parser.add_argument('--is_weight_decay',action='store_true',help='Whether to use weight decay. If it is turned on, it means weight decay is used. If it is not turned on, it means weight decay is not used. The default is not to use weight decay.')
    parser.add_argument('--onlyMethod',action='store_true',help='Whether to test only methods and not vanilla. If enabled, only methods are tested. If disabled, other methods are tested.') 
    args = parser.parse_args()
    
    b,h,n,r,d =int(args.batch), 32, int(args.n), 128, 128
    if args.type=='float16':
        type = torch.float16
    elif args.type=='float32':
        type = torch.float32
    else:
        pass
    gpu = "cuda:"+args.gpu
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)
    
    func = None 
    if args.method=='FleetAttention':
        func = fleetAttention
    elif args.method=='lightningAttention':
        func = lightning_attn2
    elif args.method=='rowbased':
        func = rowBased
    elif args.method=='BCMV_vanilla':
        func = linear_attn
    elif args.method=='recursion':
        func = recursion
    elif args.method=='blockbased':
        func = blockBased
    elif args.method=='causal_dot_product':
        func = causal_dot_product
    else:
        raise Exception("Unimplemented Method Name.")
    print('fixed value.')
    test_BCMV_by_fixed_value(args.onlyMethod,b,h,n,r,d,func,type,device)