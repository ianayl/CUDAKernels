#pragma once

#ifndef IMPL_KERNELLAUNCH_CUH
#define IMPL_KERNELLAUNCH_CUH

#include <Config.hpp>
#include <Status.cuh>
#include <Timer.hpp>
#include <Utils.cuh>

#include <cuda.h>
#include <memory>

template<typename Kernel, typename... Args>
__global__ void launchKernel(Args... args) {
	Kernel::kernel(std::forward<Args>(args)...);
}

namespace Kernels::Impl {

	class KernelLaunch {
	protected:
		// functions are passed in as typename's, where they are actually classes
	  // we call upon static class functions to run them
		// - grid/whatever calculation
		// - kernel
		template<typename Kernel, typename... Args>
		static inline Status launch(const Config& conf, const dim3& gridDim,
																const dim3& blockDim, Args&&... args) {
			std::unique_ptr<Timer> t;
			if (conf.timer) t = std::make_unique<Timer>();
			//Kernel::kernel<<<gridDim, blockDim>>>(std::forward<Args>(args)...);
			launchKernel<Kernel><<<gridDim, blockDim>>>(std::forward<Args>(args)...);
			if (conf.timer) t->stop();
			
			Status res{};
			if (conf.timer) {
				res.timed = true;
				res.timeNs = t->getNs();
			}
			return res;
		}

	};

}

#endif // IMPL_KERNELLAUNCH_CUH
