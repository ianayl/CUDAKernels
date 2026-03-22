#pragma once

#ifndef KERNELS_GEMM_NAIVE_CUH
#define KERNELS_GEMM_NAIVE_CUH

#include <Kernels/GEMM/Impl/GEMMLaunch.cuh>

namespace GEMM {
	class Naive : public Impl::GEMMLaunch<Naive> { 
	public:
		static Status gemmImpl(const Config& conf, float *A, float *B, size_t m,
													 size_t n, size_t k, float *C);
	};

	namespace Impl {
		struct NaiveGEMM {
			__device__ __forceinline__
			static void kernel(float *A, float *B, size_t m, size_t n, size_t k,
											   float *C);
		};
	}
}

#endif // KERNELS_GEMM_NAIVE_CUH
