#pragma once

#ifndef IMPL_KERNELLAUNCH_CUH
#define IMPL_KERNELLAUNCH_CUH

#include <Config.hpp>
#include <Status.cuh>
#include <Timer.hpp>
#include <Utils.cuh>

#include <cuda.h>
#include <memory>

namespace Kernels::Impl {

  class KernelLaunch {
  protected:
    template<typename F, typename... Args>
    static inline Status launch(const Config& conf, const F kernel,
                                const dim3& gridDim, const dim3& blockDim,
                                Args&&... args) {
      std::unique_ptr<Timer> t;
      if (conf.timer) t = std::make_unique<Timer>();
      kernel<<<gridDim, blockDim>>>(std::forward<Args>(args)...);
      cudaCheckErrors("Kernel invocation failed");
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
