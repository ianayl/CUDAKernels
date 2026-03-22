#include <Kernels/Impl/KernelLaunch.cuh>

// template<typename Kernel, typename... Args>
// Status Kernels::Impl::KernelLaunch::launch(const Config& conf, const dim3& gridDim,
// 																					 const dim3& blockDim, Args&&... args) {
// 	// ACTUALLY: handle this outside this function man
// 
// 	// kernel configuration handling here
// 	// TODO: use cudaOccupancyMaxwhatever to figure out warp and block sizes instead
// 
// 	// int minGridSize, blockSize;
// 	// cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);
// 	// int gridSize = (N + blockSize - 1) / blockSize;
// 	// gridSize = std::min(gridSize, minGridSize);
// 	
// 	std::unique_ptr<Timer> t;
// 	if (conf.timer) t = std::make_unique<Timer>();
// 	//Kernel::kernel<<<gridDim, blockDim>>>(std::forward<Args>(args)...);
// 	launchKernel<Kernel><<<gridDim, blockDim>>>(std::forward<Args>(args)...);
// 	if (conf.timer) t->stop();
// 	
// 	Status res{};
// 	if (conf.timer) {
// 		res.timed = true;
// 		res.timeNs = t->getNs();
// 	}
// 	return res;
// }
