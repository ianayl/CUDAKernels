#pragma once

#ifndef GEMM_IMPL_CUH
#define GEMM_IMPL_CUH

#include <Kernels/Impl/KernelLaunch.cuh>
#include <Tensor.cuh>

namespace GEMM::Impl {

  template <typename Derived>
  class GEMMLaunch : protected Kernels::Impl::KernelLaunch {
  public:
    // TODO you're missing alpha, adding C, etc.
    // No implementation: should be implemented by child class.
    template<typename T>
    static Status gemm(Tensor<T>& A, Tensor<T>& B, Tensor<T>& C,
                       const Config& conf = DefaultKernelConfig);
  };

}

#endif // GEMM_IMPL_CUH
