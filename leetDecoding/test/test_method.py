import torch
import time
import math
from leetDecoding.methods.causal_dot_product import causal_dot_product
from leetDecoding.methods.causal_dot_product_torch import causal_dot_product_torch
from leetDecoding.methods.lightningAttention2 import lightning_attn2
from leetDecoding.methods.BlockBased import blockBased
from leetDecoding.methods.Recursion import recursion
from leetDecoding.methods.FleetAttention import FleetAttention
from leetDecoding.methods.lightningAttention2_torch import lightningAttention2_torch
from leetDecoding.methods.FleetAttention_triton import FleetAttention_triton
from leetDecoding.methods.linear_attn import linear_attn, _build_slope_tensor
from leetDecoding.methods.lightningAttention2_optimized import lightning_attn2_optimized
import argparse
import torch.utils.benchmark as benchmark
import os
import json
import pandas as pd


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
    # if verbose:
    #     print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)
    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    # if verbose:
    #     print(m.times[0])
    return t, m.times[0]


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
    
    
def test_BCMV_by_random(speedup_check,b,h,n,r,d,method,type,is_weight_decay,output_path,turns):
    device='cuda'
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    B = torch.randn(b,h,n,r,dtype=type,device=device)
    C = torch.randn(b,h,n,r,dtype=type,device=device)
    V = torch.randn(b,h,n,d,dtype=type,device=device)
    # B = torch.rand(b,h,n,r,dtype=type,device=device) * 0.2
    # C = torch.rand(b,h,n,r,dtype=type,device=device) * 0.2
    # V = torch.rand(b,h,n,d,dtype=type,device=device) * 0.2

    # low = 0
    # high = 2
    # B = torch.randint(low, high, (b, h, n, r), dtype=torch.int32, device=device).to(type)
    # C = torch.randint(low, high, (b, h, n, r), dtype=torch.int32, device=device).to(type)
    # V = torch.randint(low, high, (b, h, n, d), dtype=torch.int32, device=device).to(type)



    if is_weight_decay:
        s = _build_slope_tensor(h).to(dtype=type,device=device).reshape(h)
    if speedup_check:
        res = {}
        for i in range(turns):
            if is_weight_decay:
                if method !=lightning_attn2 and method != recursion and method!=blockBased and method !=lightningAttention2_torch and method != lightning_attn2_optimized:
                    _,t = benchmark_forward(method,B,C,V,torch.exp(-s),verbose=True)
                    benchmark_memory(method,B,C,V,torch.exp(-s),verbose=False)
                else:
                    _,t = benchmark_forward(method,B,C,V,s,verbose=True)
                    benchmark_memory(method,B,C,V,s,verbose=False)
            else:
                _,t = benchmark_forward(method,B,C,V,verbose=True)
                benchmark_memory(method,B,C,V,verbose=False)
            res[i] = t
        # compute avg and variance
        times = list(res.values())
        mean = sum(times) / len(times)
        variance = sum((x - mean) ** 2 for x in times) / len(times)
        max_time = max(times)
        min_time = min(times)
        res['avg']=mean
        res['variance']=variance
        res['upper_bound']=max_time-mean
        res['lower_bound']=mean-min_time
        print('avg:',mean,' variance:',variance,'std:',math.sqrt(variance),'upper_bound:',max_time-mean,'lower_bound:',mean-min_time)
        # with open(os.path.join(output_path),'w') as f:
        #     json.dump(res,f,ensure_ascii=False)                
    else:
        if is_weight_decay:
            correct_BCMV = linear_attn(B,C,V,s)
            if method !=lightning_attn2 and method!=linear_attn and method!=recursion and method!=blockBased and method !=lightningAttention2_torch and method != lightning_attn2_optimized:
                BCMV = method(B,C,V,torch.exp(-s))
                benchmark_memory(method,B,C,V,torch.exp(-s),verbose=True)
            else:
                BCMV = method(B,C,V,s)
                benchmark_memory(method,B,C,V,s,verbose=True)
            # import pdb 
            # pdb.set_trace()
            print(f'method norm:{torch.norm(BCMV)}      vanilla norm:{torch.norm(correct_BCMV)}     difference norm:{torch.norm(BCMV-correct_BCMV)} {torch.allclose(BCMV,correct_BCMV,rtol=1e-2)}')
        else:
            M = torch.tril(torch.ones((n,n),dtype=type,device=device))
            correct_BCMV = torch.matmul(torch.matmul(B,C.transpose(2,3)) * M,V)
            BCMV = method(B,C,V)
            benchmark_memory(method,B,C,V,verbose=True)
            print(f'method norm:{torch.norm(BCMV)}      vanilla norm:{torch.norm(correct_BCMV)}     difference norm:{torch.norm(BCMV-correct_BCMV)} {torch.allclose(BCMV,correct_BCMV,rtol=1e-2)}')



def benchmark_method(method,device='cuda',dtype='float32',batch_size=1,is_weight_decay=False):
    if dtype=='float32':
        dtype=torch.float32
    elif dtype=='float16':
        dtype=torch.float16
    else:
        raise Exception('Not implement the type',dtype)
    n_list = [128, 500]
    data = {
        'seqlen': [],
        'time': [],
    }
    for n in n_list:
        t = 'OOM'
        try:
            B = torch.rand((batch_size,32,n,128),dtype=type,device=device)
            C = torch.rand((batch_size,32,n,128),dtype=type,device=device)
            V = torch.rand((batch_size,32,n,128),dtype=type,device=device)
            if is_weight_decay:
                s = torch.full((h,1),0.95,dtype=type,device=device)
                _,time = benchmark_forward(method,B,C,V,torch.exp(-s),verbose=True)
            else:
                _,time = benchmark_forward(method,B,C,V,verbose=True)
            t = f"{time:.6f}"
        except MemoryError as me:
            print('Out of memory.')
        except Exception as e:
            print(f'There is a error: {e}')
        finally:
            data['time'].append(t)
            data['seqlen'].append(n)
    df = pd.DataFrame(data)
    print('Benchmark Performance:\n',df.T)
    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch',help='The batch size of the test, the default is 1',default=1)
    parser.add_argument('--n',help='The sequence length of the test, the default is 2048',default=2048)
    parser.add_argument('--method',default='FleetAttention', help='The method under test will execute both vanilla and the corresponding method. The method can be FleetAttention, lightningAttention2, BCMV_vanilla, causal_dot_product_torch, recursion, blockbased, causal_dot_product,lightningAttention2_torch.')
    parser.add_argument('--type',help='Numeric type, including float16, float32, the default is float16.',default='float16')
    parser.add_argument('--is-weight-decay',action='store_true',help='Whether to use weight decay. If it is turned on, it means weight decay is used. If it is not turned on, it means weight decay is not used. The default is not to use weight decay.')
    parser.add_argument('--speedup-check',action='store_true',help='Whether to check speedup. If enabled, speed are tested. If disabled,it will check the precision between vanilla and the corresponding method.') 
    parser.add_argument('--output-dir',type=str,default='/home/wjp/projects/LA/outputs/single_layer')
    parser.add_argument('--turns',type=int,default=15)
    args = parser.parse_args()

    b,h,n,r,d =int(args.batch), 32, int(args.n), 128, 128
    if args.type=='float16':
        type = torch.float16
    elif args.type=='float32':
        type = torch.float32
    elif args.type=="bfloat16":
        type = torch.bfloat16
    else:
        pass
    # gpu = "cuda:"+str(args.gpu)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # torch.cuda.set_device(gpu)
    # device = torch.device(gpu)
    
    func = None 
    if args.method=='FleetAttention_torch':
        func = FleetAttention
    elif args.method=='FleetAttention':
        func = FleetAttention_triton
    elif args.method=='lightningAttention2':
        func = lightning_attn2
    elif args.method=='causal_dot_product_torch':
        func = causal_dot_product_torch
    elif args.method=='BCMV_vanilla':
        func = linear_attn
    elif args.method=='recursion':
        func = recursion
    elif args.method=='blockbased':
        func = blockBased
    elif args.method=='causal_dot_product':
        func = causal_dot_product
    elif args.method=='lightningAttention2_torch':
        func = lightningAttention2_torch
    elif args.method=='lightningAttention2_origin':
        from lightning_attn.ops import lightning_attn_func
        func = lightning_attn_func
    elif args.method=="lightningAttention2_optimized":
        func = lightning_attn2_optimized
    else:
        raise Exception("Unimplemented Method Name.")
    output_dir = os.path.join(args.output_dir,str(args.batch),str(args.n),args.type)
    if args.is_weight_decay:
        output_dir = os.path.join(output_dir,'weight_decay')
    else:
        output_dir = os.path.join(output_dir,'no_weight_decay')
    os.makedirs(output_dir,exist_ok=True)
    output_path = os.path.join(output_dir,args.method+'.json')
    test_BCMV_by_random(args.speedup_check,b,h,n,r,d,func,type,args.is_weight_decay,output_path,args.turns)


# python test/test_method.py --batch 1 --n 1000 --method lightningAttention2_optimized --type bfloat16 --gpu 7  --is-weight-decay