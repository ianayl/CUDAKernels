#pragma once

#ifndef KERNELS_GEMM_NAIVE_CUH
#define KERNELS_GEMM_NAIVE_CUH

#include <Kernels/GEMM/Impl/GEMMLaunch.cuh>

namespace GEMM {
  class Naive : public Impl::GEMMLaunch<Naive> {
  public:
    static Status gemm(float *A, float *B, size_t m, size_t n, size_t k,
                       float *C, const Config& conf = DefaultKernelConfig);
  };

  namespace Impl {
    __global__ void NaiveGEMMKernel(float *A, float *B, size_t m, size_t n,
                                    size_t k, float *C);
  }
}

#endif // KERNELS_GEMM_NAIVE_CUH
