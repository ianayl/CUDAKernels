#include <memory>

#include <Kernels/GEMM/Naive.cuh>
#include <stdio.h>

__device__ __forceinline__
void GEMM::Impl::NaiveGEMM::kernel(float *A, float *B, size_t m, size_t n,
																	 size_t k, float *C) {
	printf("Hello from the GPU!\n");
}


Status GEMM::Naive::gemmImpl(const Config& conf, float *A, float *B, size_t m,
														 size_t n, size_t k, float *C) {
	// GEMM kernel creation / parameter handling here
	// TODO: use cudaOccupancyMaxwhatever to figure out warp and block sizes instead

	dim3 gridDim(1);
	dim3 blockDim(1);
	return launch<GEMM::Impl::NaiveGEMM>(conf, gridDim, blockDim, A, B, m, n, k, C);

	// std::unique_ptr<Timer> t;
	// if (conf.timer) t = std::make_unique<Timer>();
	// // GEMM Kernel execution here
	// // TODO: just implement the kernel in this file, not much room to play with warps and scheduling and whatnot
	// if (conf.timer) t->stop();
	// 
	// Status res{};
	// if (conf.timer) {
	// 	res.timed = true;
	// 	res.timeNs = t->getNs();
	// }
	// return res;
}
