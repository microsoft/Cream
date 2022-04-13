#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <THC/THCAtomics.cuh>
#include <vector>

using index_t = int;

const int HIP_MAX_GRID_NUM = 65535;
const int HIP_MAX_NUM_THREADS = 512;

inline int HIP_GET_NUM_THREADS(const int n) {
  return std::min(HIP_MAX_NUM_THREADS, ((n + 31) / 32) * 32);
}

inline int HIP_GET_BLOCKS(const int n, const int num_threads) {
  return std::min(HIP_MAX_GRID_NUM, n + num_threads - 1) / num_threads;
}

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void rpe_index_forward_gpu_kernel(
    index_t n, scalar_t *p_Y, const scalar_t *__restrict__ p_input,
    const index_t *__restrict__ p_index, index_t num_buckets, index_t H,
    index_t L_query, index_t L_key, index_t L_qk, index_t s0, index_t s1,
    index_t s2, index_t s3) {
  CUDA_KERNEL_LOOP(i, n) {
    index_t gi = i / L_key;
    const index_t qi = gi % L_query;
    gi /= L_query;
    const index_t hi = gi % H;
    gi /= H;
    const index_t bi = gi;
    const index_t ind = bi * s0 + hi * s1 + qi * s2 + p_index[i % L_qk] * s3;
    p_Y[i] = __ldg(&p_input[ind]);
  }
}

template <typename scalar_t>
__global__ void rpe_index_backward_gpu_kernel(
    index_t n, scalar_t *p_grad_input, const index_t *__restrict__ p_index,
    const scalar_t *__restrict__ p_grad_output, index_t num_buckets,
    index_t L_key, index_t L_qk) {
  CUDA_KERNEL_LOOP(i, n) {
    const index_t input_i = i / L_key * num_buckets + p_index[i % L_qk];
    const scalar_t v = p_grad_output[i];
    gpuAtomicAdd(p_grad_input + input_i, v);
  }
}

at::Tensor rpe_index_forward_gpu(torch::Tensor input, torch::Tensor index) {
  /*
  - Inputs
      input: float32 (B, H, L_query, num_buckets)
      index: index_t (L_query, L_key)
  - Outputs
      Y: float32 (B, H, L_query, L_key)
   */
  AT_ASSERTM(input.device().is_cuda(), "input must be a GPU tensor");
  AT_ASSERTM(index.device().is_cuda(), "index must be a GPU tensor");
  AT_ASSERTM(input.ndimension() == 4, "input must be a 4D tensor");
  AT_ASSERTM(index.ndimension() == 2, "index must be a 2D tensor");
  AT_ASSERTM(index.scalar_type() == at::kInt, "index must be Int type");
  AT_ASSERTM(index.is_contiguous(), "index should be contiguous");
  const index_t B = input.size(0);
  const index_t H = input.size(1);
  const index_t num_buckets = input.size(3);
  const index_t L_query = index.size(0);
  const index_t L_key = index.size(1);
  const index_t L_qk = L_query * L_key;
  at::Tensor Y = at::empty({B, H, L_query, L_key}, input.options());
  const index_t numel = Y.numel();
  const at::IntArrayRef strides = input.strides();

  const int threadsPerBlock = HIP_GET_NUM_THREADS(numel);
  const int blocks = HIP_GET_BLOCKS(numel, threadsPerBlock);

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "rpe_index_forward_gpu", [&] {
        const scalar_t *p_input = input.data_ptr<scalar_t>();
        const index_t *p_index = index.data_ptr<index_t>();
        scalar_t *p_Y = Y.data_ptr<scalar_t>();
        rpe_index_forward_gpu_kernel<<<blocks, threadsPerBlock, 0, stream>>>(
            numel, p_Y, p_input, p_index, num_buckets, H, L_query, L_key, L_qk,
            strides[0], strides[1], strides[2], strides[3]);
      });
  return Y;
}

void rpe_index_backward_gpu(torch::Tensor grad_input, torch::Tensor grad_output,
                            torch::Tensor index) {
  /*
  - Inputs
      grad_output: float32 (B, H, L_query, L_key)
      index: index_t (L_query, L_key)
  - Outputs
      grad_input: float32 (B, H, L_query, num_buckets)
   */
  AT_ASSERTM(grad_input.device().is_cuda(), "grad_input must be a GPU tensor");
  AT_ASSERTM(grad_output.device().is_cuda(),
             "grad_output must be a GPU tensor");
  AT_ASSERTM(index.device().is_cuda(), "grad_index must be a GPU tensor");
  AT_ASSERTM(grad_input.ndimension() == 4, "input must be a 4D tensor");
  AT_ASSERTM(grad_output.ndimension() == 4, "input must be a 4D tensor");
  AT_ASSERTM(index.ndimension() == 2, "index must be a 2D tensor");
  AT_ASSERTM(index.scalar_type() == at::kInt, "index must be Int type");

  const index_t num_buckets = grad_input.size(3);
  const index_t L_query = grad_output.size(2);
  const index_t L_key = grad_output.size(3);
  const index_t L_qk = L_query * L_key;

  auto grad_input_ = grad_input.contiguous();
  auto grad_output_ = grad_output.contiguous();
  auto index_ = index.contiguous();

  const index_t numel = grad_output.numel();

  const int threadsPerBlock = HIP_GET_NUM_THREADS(numel);
  const int blocks = HIP_GET_BLOCKS(numel, threadsPerBlock);

  at::cuda::CUDAGuard device_guard(grad_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "rpe_index_backward_gpu", [&] {
        scalar_t *p_grad_input = grad_input_.data_ptr<scalar_t>();
        const index_t *p_index = index_.data_ptr<index_t>();
        const scalar_t *p_grad_output = grad_output_.data_ptr<scalar_t>();
        rpe_index_backward_gpu_kernel<<<blocks, threadsPerBlock, 0, stream>>>(
            numel, p_grad_input, p_index, p_grad_output, num_buckets, L_key,
            L_qk);
      });
}
