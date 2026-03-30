#include <Kernels/GEMM/Naive.cuh>

#include <memory>
#include <iostream>
#include <cassert>

template<typename T>
__global__
void GEMM::Impl::NaiveGEMMKernel(T* A, T* B, size_t m, size_t n,
                                 size_t k, T* C) {
  const uint row = blockIdx.x * blockDim.x + threadIdx.x;
  const uint col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= k) return;

  T tmp = 0;
  for (uint i = 0; i < n; i++) {
      tmp += A[row * n + i] * B[i * k + col];
  }
  C[row * k + col] = tmp;
}

template<typename T>
Status GEMM::Naive::gemm(Tensor<T>& A, Tensor<T>& B, Tensor<T>& C,
                         const Config& conf) {
  // TODO: check that *A, *B, *C isn't nullptr

  assert(A.dim_x == B.dim_y && "Cannot perform GEMM: A.x != B.y");
  assert(C.dim_y == A.dim_y && C.dim_x == B.dim_x &&
         "Cannot perform GEMM: C has incorrect size");

  const size_t m = A.dim_y;
  const size_t n = A.dim_x;
  const size_t k = B.dim_x;

  int minGridSize, maxBlockSize;
  // TODO: maybe this occupancy code can be wrapped with error checking.
  // similar to launch, a cudaCheckErrors can be placed in the wrapped
  // function that uses a lookup table for function pointers.
  // ALTERNATIVELY: wrap conf in runtime information set by the kernel class
  // that provides exact kernel information for better logs.
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize,
                                     GEMM::Impl::NaiveGEMMKernel<T>);
  cudaCheckErrors("GEMM::Naive::gemm: failed to obtain optimal kernel size.");

  constexpr int blockDimX = 32;
  const int blockDimY = maxBlockSize / blockDimX;
  dim3 blockDim(blockDimX, blockDimY);
  dim3 gridDim((m + blockDimX - 1) / blockDimX, (k + blockDimY - 1) / blockDimY);

  // TODO: wrap this in verbose
  std::cerr <<  "minGridSize: "  << minGridSize
            << " maxBlockSize: " << maxBlockSize << "\n";
  std::cerr <<  "blockDim.x: " << blockDim.x
            << " blockDim.y: " << blockDim.y << "\n";

  return launch(conf, GEMM::Impl::NaiveGEMMKernel<T>, gridDim, blockDim,
                *A, *B, m, n, k, *C);
}


template Status GEMM::Naive::gemm(Tensor<float>&, Tensor<float>&,
                                  Tensor<float>&, const Config&);
template
__global__ void GEMM::Impl::NaiveGEMMKernel(float*, float*, size_t, size_t,
                                            size_t, float*);
