#pragma once

#ifndef KERNELS_GEMM_NAIVE_CUH
#define KERNELS_GEMM_NAIVE_CUH

#include <Kernels/GEMM/Impl/GEMMLaunch.cuh>

namespace GEMM {
  class Naive : public Impl::GEMMLaunch<Naive> {
  public:
    template<typename T>
    static Status gemm(Tensor<T>& A, Tensor<T>& B, Tensor<T>& C,
                       const Config& conf = DefaultKernelConfig);
  };

  namespace Impl {
    template<typename T>
    __global__ void NaiveGEMMKernel(T* A, T* B, size_t m, size_t n,
                                    size_t k, T* C);
  }
}

#endif // KERNELS_GEMM_NAIVE_CUH
