#pragma once

#ifndef GEMM_IMPL_CUH
#define GEMM_IMPL_CUH

#include <Kernels/Impl/KernelLaunch.cuh>

namespace GEMM::Impl {

	template <typename Derived>
	class GEMMLaunch : protected Kernels::Impl::KernelLaunch {
	public:
		// No implementation: should be implemented by child class.
		static inline Status gemm(const Config& conf, float *A, float *B, size_t m,
															size_t n, size_t k, float *C) {
			return Derived::gemmImpl(conf, A, B, m, n, k, C);
		}

		static inline Status gemm(float *A, float *B, size_t m, size_t n, size_t k,
															float *C) {
			return Derived::gemmImpl(DefaultKernelConfig, A, B, m, n, k, C);
		}
	};

}

#endif // GEMM_IMPL_CUH
