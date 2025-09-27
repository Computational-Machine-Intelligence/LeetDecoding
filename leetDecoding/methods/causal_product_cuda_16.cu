#include <torch/extension.h>
#include <assert.h>
#include <stdio.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h> 
#include <ATen/ATen.h>

// --- 定义各种 Accessor 类型 ---
// FP32 Accessors
typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> float_accessor;
typedef torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> gamma_float_accessor;

// FP16 (Half) Accessors
typedef torch::PackedTensorAccessor32<at::Half, 4, torch::RestrictPtrTraits> half_accessor;
typedef torch::PackedTensorAccessor32<at::Half, 1, torch::RestrictPtrTraits> gamma_half_accessor;

// BF16 (BFloat16) Accessors
typedef torch::PackedTensorAccessor32<at::BFloat16, 4, torch::RestrictPtrTraits> bf16_accessor;
typedef torch::PackedTensorAccessor32<at::BFloat16, 1, torch::RestrictPtrTraits> gamma_bf16_accessor;


#define E_BLOCK_SIZE 8

// =================================================================================
// Kernel for FP16 (at::Half) Inputs -> FP32 Output
// =================================================================================
__global__ void causal_dot_product_kernel_fp16(
    const half_accessor queries, const half_accessor keys, const half_accessor values,
    const gamma_half_accessor gamma, float_accessor result,
    const int N, const int H, const int L, const int E, const int M) 
{
    int n = blockIdx.y;
    int h = blockIdx.z;
    int e_start = blockIdx.x * E_BLOCK_SIZE;
    int m = threadIdx.x % M;

    extern __shared__ float shared_mem[];
    float* shared_kv = shared_mem;

    for (int e_local = 0; e_local < E_BLOCK_SIZE && e_local + e_start < E; e_local++) {
      shared_kv[m + e_local * M] = 0.0f;
    }
    
    const float gamma_h = __half2float(gamma[h]);

    for (int t=0; t<L; t++) {
      float res = 0.0f;
      for (int e_local = 0; e_local < E_BLOCK_SIZE && e_local + e_start < E; e_local++) {
        float k_val = __half2float(keys[n][h][t][e_local + e_start]);
        float v_val = __half2float(values[n][h][t][m]);
        
        shared_kv[e_local*M + m] += k_val * v_val;

        float q_val = __half2float(queries[n][h][t][e_local + e_start]);
        res += q_val * shared_kv[e_local*M + m];

        shared_kv[e_local*M + m] *= gamma_h;
      }
      atomicAdd(&result[n][h][t][m], res);
    }
}

// =================================================================================
// Kernel for FP32 (float) Inputs -> FP32 Output
// =================================================================================
__global__ void causal_dot_product_kernel_fp32(
    const float_accessor queries, const float_accessor keys, const float_accessor values,
    const gamma_float_accessor gamma, float_accessor result,
    const int N, const int H, const int L, const int E, const int M) 
{
    int n = blockIdx.y;
    int h = blockIdx.z;
    int e_start = blockIdx.x * E_BLOCK_SIZE;
    int m = threadIdx.x % M;

    extern __shared__ float shared_mem[];
    float* shared_kv = shared_mem;

    for (int e_local = 0; e_local < E_BLOCK_SIZE && e_local + e_start < E; e_local++) {
      shared_kv[m + e_local * M] = 0.0f;
    }
    
    const float gamma_h = gamma[h]; // No conversion needed

    for (int t=0; t<L; t++) {
      float res = 0.0f;
      for (int e_local = 0; e_local < E_BLOCK_SIZE && e_local + e_start < E; e_local++) {
        // No conversion needed
        shared_kv[e_local*M + m] += keys[n][h][t][e_local + e_start] * values[n][h][t][m];
        res += queries[n][h][t][e_local + e_start] * shared_kv[e_local*M + m];
        shared_kv[e_local*M + m] *= gamma_h;
      }
      atomicAdd(&result[n][h][t][m], res);
    }
}


// --- Kernel Launchers ---

void causal_dot_product_fp16_(const torch::Tensor &queries, const torch::Tensor &keys, const torch::Tensor &values,
                              const torch::Tensor &gamma, torch::Tensor &product) {
    int N = queries.size(0); int H = queries.size(1); int L = queries.size(2);
    int E = queries.size(3); int M = values.size(3);
    const int blocks_per_sequence = (E + E_BLOCK_SIZE - 1) / E_BLOCK_SIZE;

    dim3 blockDim(M, 1, 1);
    dim3 gridDim(blocks_per_sequence, N, H);
    const int shared_mem_forward = E_BLOCK_SIZE * M * sizeof(float);

    causal_dot_product_kernel_fp16<<<gridDim, blockDim, shared_mem_forward>>>(
      queries.packed_accessor32<at::Half, 4, torch::RestrictPtrTraits>(),
      keys.packed_accessor32<at::Half, 4, torch::RestrictPtrTraits>(),
      values.packed_accessor32<at::Half, 4, torch::RestrictPtrTraits>(),
      gamma.packed_accessor32<at::Half, 1, torch::RestrictPtrTraits>(),
      product.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      N, H, L, E, M);
}

void causal_dot_product_fp32_(const torch::Tensor &queries, const torch::Tensor &keys, const torch::Tensor &values,
                              const torch::Tensor &gamma, torch::Tensor &product) {
    int N = queries.size(0); int H = queries.size(1); int L = queries.size(2);
    int E = queries.size(3); int M = values.size(3);
    const int blocks_per_sequence = (E + E_BLOCK_SIZE - 1) / E_BLOCK_SIZE;

    dim3 blockDim(M, 1, 1);
    dim3 gridDim(blocks_per_sequence, N, H);
    const int shared_mem_forward = E_BLOCK_SIZE * M * sizeof(float);

    causal_dot_product_kernel_fp32<<<gridDim, blockDim, shared_mem_forward>>>(
      queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      gamma.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      product.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      N, H, L, E, M);
}

__global__ void causal_dot_product_kernel_bf16(
    const bf16_accessor queries, const bf16_accessor keys, const bf16_accessor values,
    const gamma_bf16_accessor gamma, float_accessor result,
    const int N, const int H, const int L, const int E, const int M) 
{
    int n = blockIdx.y;
    int h = blockIdx.z;
    int e_start = blockIdx.x * E_BLOCK_SIZE;
    int m = threadIdx.x % M;

    // middle computation result stored in fp32
    extern __shared__ float shared_mem[];
    float* shared_kv = shared_mem;

    for (int e_local = 0; e_local < E_BLOCK_SIZE && e_local + e_start < E; e_local++) {
      shared_kv[m + e_local * M] = 0.0f;
    }
    
    // convert at:BFloat16 gamma -> float
    const float gamma_h = float(gamma[h]);

    for (int t=0; t<L; t++) {
      float res = 0.0f;
      for (int e_local = 0; e_local < E_BLOCK_SIZE && e_local + e_start < E; e_local++) {

        float k_val = float(keys[n][h][t][e_local + e_start]);
        float v_val = float(values[n][h][t][m]);
        
        shared_kv[e_local*M + m] += k_val * v_val;

        float q_val = float(queries[n][h][t][e_local + e_start]);
        res += q_val * shared_kv[e_local*M + m];

        shared_kv[e_local*M + m] *= gamma_h;
      }
      atomicAdd(&result[n][h][t][m], res);
    }
}


// =================================================================================
//  BF16 Kernel Launcher
// =================================================================================
void causal_dot_product_bf16_(const torch::Tensor &queries, const torch::Tensor &keys, const torch::Tensor &values,
                              const torch::Tensor &gamma, torch::Tensor &product) {
    int N = queries.size(0); int H = queries.size(1); int L = queries.size(2);
    int E = queries.size(3); int M = values.size(3);
    const int blocks_per_sequence = (E + E_BLOCK_SIZE - 1) / E_BLOCK_SIZE;

    dim3 blockDim(M, 1, 1);
    dim3 gridDim(blocks_per_sequence, N, H);
    // 共享内存用于累加，所以是 float 类型
    const int shared_mem_forward = E_BLOCK_SIZE * M * sizeof(float);

    causal_dot_product_kernel_bf16<<<gridDim, blockDim, shared_mem_forward>>>(
      queries.packed_accessor32<at::BFloat16, 4, torch::RestrictPtrTraits>(),
      keys.packed_accessor32<at::BFloat16, 4, torch::RestrictPtrTraits>(),
      values.packed_accessor32<at::BFloat16, 4, torch::RestrictPtrTraits>(),
      gamma.packed_accessor32<at::BFloat16, 1, torch::RestrictPtrTraits>(),
      product.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      N, H, L, E, M);
}


// =================================================================================
void causal_dot_product(const torch::Tensor queries,
                        const torch::Tensor keys,
                        const torch::Tensor values,
                        const torch::Tensor gamma,
                        torch::Tensor product) 
{
  torch::DeviceGuard _guard(queries.device());

  auto dtype = queries.scalar_type();
  if (dtype == at::kHalf) {
      causal_dot_product_fp16_(queries, keys, values, gamma, product);
  } else if (dtype == at::kFloat) {
      causal_dot_product_fp32_(queries, keys, values, gamma, product);
  } else if (dtype == at::kBFloat16) {
      causal_dot_product_bf16_(queries, keys, values, gamma, product);
  }
  else {
      TORCH_CHECK(false, "Unsupported input data type for causal_dot_product");
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "causal_dot_product",
        &causal_dot_product,
        "Compute the weighted sum of values but attending only to previous "
        "values. Dispatches based on input tensor type."
    );
}