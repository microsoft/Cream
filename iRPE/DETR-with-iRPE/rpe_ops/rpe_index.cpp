#include <torch/extension.h>

#include <string>
#include <vector>

using index_t = int;

at::Tensor rpe_index_forward_cpu(torch::Tensor input, torch::Tensor index) {
  /*
  - Inputs
      input: float32 (B, H, L_query, num_buckets)
      index: index_t (L_query, L_key)
  - Outputs
      Y: float32 (B, H, L_query, L_key)
   */
  AT_ASSERTM(input.device().is_cpu(), "input must be a CPU tensor");
  AT_ASSERTM(index.device().is_cpu(), "index must be a CPU tensor");
  AT_ASSERTM(input.ndimension() == 4, "input must be a 4D tensor");
  AT_ASSERTM(index.ndimension() == 2, "index must be a 2D tensor");
  AT_ASSERTM(index.scalar_type() == at::kInt, "index must be Int type");
  const index_t B = input.size(0);
  const index_t H = input.size(1);
  const index_t num_buckets = input.size(3);
  const index_t L_query = index.size(0);
  const index_t L_key = index.size(1);
  const index_t L_qk = L_query * L_key;
  at::Tensor Y = at::empty({B, H, L_query, L_key}, input.options());
  auto input_ = input.contiguous();
  auto index_ = index.contiguous();
  const index_t grain_size = 3000;
  const index_t numel = Y.numel();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "rpe_index_forward_cpu", [&] {
        const scalar_t *p_input = input_.data_ptr<scalar_t>();
        const index_t *p_index = index_.data_ptr<index_t>();
        scalar_t *p_Y = Y.data_ptr<scalar_t>();
        at::parallel_for(0, numel, grain_size, [&](index_t begin, index_t end) {
          /*
          // we optimize the following function to
          // reduce the number of operators, namely divide and multiply.
          for (index_t i = begin; i < end; ++i) {
            p_Y[i] = p_input[i / L_key * num_buckets + p_index[i % L_qk]];
          }
          */

          index_t aligned_begin = (begin + L_qk - 1) / L_qk * L_qk;
          if (aligned_begin > end) aligned_begin = end;
          index_t aligned_end = end / L_qk * L_qk;
          for (index_t i = begin; i < aligned_begin; ++i) {
            p_Y[i] = p_input[i / L_key * num_buckets + p_index[i % L_qk]];
          }

          // [aligned_begin, aligned_end)
          // where aligned_begin % L_qk == 0, aligned_end % L_qk == 0
          index_t base = aligned_begin / L_key * num_buckets;
          const index_t base_end = aligned_end / L_key * num_buckets;
          index_t i = aligned_begin;
          while (base < base_end) {
            for (index_t q = 0, j = 0; q < L_query; ++q) {
              for (index_t k = 0; k < L_key; ++k) {
                p_Y[i++] = p_input[base + p_index[j++]];
              }
              base += num_buckets;
            }
          }

          for (index_t i = aligned_end; i < end; ++i) {
            p_Y[i] = p_input[i / L_key * num_buckets + p_index[i % L_qk]];
          }
        });
      });
  return Y;
}

template <typename scalar_t>
inline scalar_t cpuAtomicAdd(scalar_t *address, const scalar_t val) {
#pragma omp critical
  *address += val;
  return *address;
}

void rpe_index_backward_cpu(torch::Tensor grad_input, torch::Tensor grad_output,
                            torch::Tensor index) {
  /*
  - Inputs
      grad_output: float32 (B, H, L_query, L_key)
      index: index_t (L_query, L_key)
  - Outputs
      grad_input: float32 (B, H, L_query, num_buckets)
   */
  AT_ASSERTM(grad_input.device().is_cpu(), "grad_input must be a CPU tensor");
  AT_ASSERTM(grad_output.device().is_cpu(), "grad_output must be a CPU tensor");
  AT_ASSERTM(index.device().is_cpu(), "grad_index must be a CPU tensor");
  AT_ASSERTM(grad_input.ndimension() == 4, "input must be a 4D tensor");
  AT_ASSERTM(grad_output.ndimension() == 4, "input must be a 4D tensor");
  AT_ASSERTM(index.ndimension() == 2, "index must be a 2D tensor");
  AT_ASSERTM(index.scalar_type() == at::kInt, "index must be Int type");

  const index_t num_buckets = grad_input.size(3);
  const index_t L_query = index.size(0);
  const index_t L_key = index.size(1);
  const index_t L_qk = L_query * L_key;

  auto grad_input_ = grad_input.contiguous();
  auto grad_output_ = grad_output.contiguous();
  auto index_ = index.contiguous();

  const index_t grain_size = 3000;
  const index_t numel = grad_output.numel();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_input.scalar_type(), "rpe_index_backward_atomic_cpu", [&] {
        scalar_t *p_grad_input = grad_input_.data_ptr<scalar_t>();
        const index_t *p_index = index_.data_ptr<index_t>();
        const scalar_t *p_grad_output = grad_output_.data_ptr<scalar_t>();
        at::parallel_for(0, numel, grain_size, [&](index_t begin, index_t end) {
          for (index_t i = begin; i < end; ++i) {
            const index_t input_i = i / L_key * num_buckets + p_index[i % L_qk];
            const scalar_t v = p_grad_output[i];
            cpuAtomicAdd(p_grad_input + input_i, v);
          }
        });
      });
}

std::string version() {
  return "1.2.0";
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("version", &version, "The version of the package `rpe_index_cpp`");
  m.def("forward_cpu", &rpe_index_forward_cpu, "2D RPE Index Forward (CPU)");
  m.def("backward_cpu", &rpe_index_backward_cpu, "2D RPE Index Backward (CPU)");

#if defined(WITH_CUDA)
  at::Tensor rpe_index_forward_gpu(torch::Tensor input, torch::Tensor index);
  void rpe_index_backward_gpu(torch::Tensor grad_input,
                              torch::Tensor grad_output, torch::Tensor index);
  m.def("forward_gpu", &rpe_index_forward_gpu, "2D RPE Index Forward (GPU)");
  m.def("backward_gpu", &rpe_index_backward_gpu, "2D RPE Index Backward (GPU)");
#endif
}
