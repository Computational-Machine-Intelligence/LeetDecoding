//
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
//

//
// For modifications made inside namespace nvidia (authored by jdemouth):
//
// Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <torch/extension.h>
#include <assert.h>
#include <stdio.h>
#include <cuda_fp16.h> 
#include <ATen/ATen.h>


typedef torch::PackedTensorAccessor32<at::Half, 4, torch::RestrictPtrTraits> half_accessor;
typedef torch::PackedTensorAccessor32<at::Half, 1, torch::RestrictPtrTraits> gamma_accessor;
typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> float_accessor;


#define E_BLOCK_SIZE 8

__global__ void causal_dot_product_kernel(
    const half_accessor queries,
    const half_accessor keys,
    const half_accessor values,
    const gamma_accessor gamma,
    float_accessor result,
    const int N,
    const int H,
    const int L,
    const int E,
    const int M
) {
    int n = blockIdx.y;
    int h = blockIdx.z;

    int e_start = blockIdx.x * E_BLOCK_SIZE;
    int m = threadIdx.x % M;

    extern __shared__ at::Half shared_mem[];
    at::Half* shared_kv = shared_mem;

    for (int e_local = 0; e_local < E_BLOCK_SIZE && e_local + e_start < E; e_local++) {
      shared_kv[m + e_local * M] = __float2half(0.0);
    }

    for (int t=0; t<L; t++) {
      at::Half res = __float2half(0.0);
      for (int e_local = 0; e_local < E_BLOCK_SIZE && e_local + e_start < E; e_local++) {
        shared_kv[e_local*M + m] = __hadd(shared_kv[e_local*M + m], __hmul(keys[n][h][t][e_local + e_start], values[n][h][t][m]));
        res = __hadd(res, __hmul(queries[n][h][t][e_local + e_start], shared_kv[e_local*M + m]));
        shared_kv[e_local*M + m] = __hmul(shared_kv[e_local*M + m], gamma[h]);
      }
      atomicAdd(
          &result[n][h][t][m],
          __half2float(res)
      );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void causal_dot_product_(const torch::Tensor queries,
                         const torch::Tensor keys,
                         const torch::Tensor values,
                         const torch::Tensor gamma,
                         torch::Tensor product) {
    // Make sure that we are using the correct GPU device
    torch::DeviceGuard _guard(queries.device());

    int N = queries.size(0);
    int H = queries.size(1);
    int L = queries.size(2);
    int E = queries.size(3);
    int M = values.size(3);

    const int blocks_per_sequence = (E + E_BLOCK_SIZE - 1) / E_BLOCK_SIZE;

    dim3 blockDim(M, 1, 1);
    dim3 gridDim(blocks_per_sequence, N, H);
    const int shared_mem_forward = E_BLOCK_SIZE * M * sizeof(__half);

    causal_dot_product_kernel<<<gridDim, blockDim, shared_mem_forward>>>(
      queries.packed_accessor32<at::Half, 4, torch::RestrictPtrTraits>(),
      keys.packed_accessor32<at::Half, 4, torch::RestrictPtrTraits>(),
      values.packed_accessor32<at::Half, 4, torch::RestrictPtrTraits>(),
      gamma.packed_accessor32<at::Half, 1, torch::RestrictPtrTraits>(),
      product.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      N, H, L, E, M
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void causal_dot_product(const torch::Tensor queries,
                        const torch::Tensor keys,
                        const torch::Tensor values,
                        const torch::Tensor gamma,
                        torch::Tensor product) 
{
  causal_dot_product_(queries, keys, values, gamma, product);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "causal_dot_product",
        &causal_dot_product,
        "Compute the weighted sum of values but attending only to previous "
        "values."
    );
}
