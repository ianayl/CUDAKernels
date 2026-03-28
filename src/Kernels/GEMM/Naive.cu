#include <memory>

#include <Kernels/GEMM/Naive.cuh>

__global__
void GEMM::Impl::NaiveGEMMKernel(float *A, float *B, size_t m, size_t n,
                                 size_t k, float *C) {
  const uint row = blockIdx.x * blockDim.x + threadIdx.x;
  const uint col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= k) return;

  float tmp = 0;
  for (uint i = 0; i < n; i++) {
      tmp += A[row * n + i] * B[i * k + col];
  }
  C[row * k + col] = tmp;
}

Status GEMM::Naive::gemm(float *A, float *B, size_t m, size_t n, size_t k,
                         float *C, const Config& conf) {
  // GEMM kernel creation / parameter handling here
  // TODO: use cudaOccupancyMaxwhatever to figure out warp and block sizes instead

  dim3 blockDim(32, 32);
  dim3 gridDim((m + 32 - 1) / 32, (n + 32 - 1) / 32);
  return launch(conf, GEMM::Impl::NaiveGEMMKernel, gridDim, blockDim,
                A, B, m, n, k, C);
}
