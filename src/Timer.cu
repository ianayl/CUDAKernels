#include <Timer.hpp>

Timer::Timer() {
#ifdef __CUDACC__
  cudaDeviceSynchronize();
#endif
  start = Clock::now();
}

Timer::Clock::duration Timer::stop() {
#ifdef __CUDACC__
  cudaDeviceSynchronize();
#endif
  end = Clock::now();
  return end - start;
}

Timer::Clock::duration Timer::get() {
  return end - start;
}

// TODO: this is prone to overflow: I might want to dynamically adjust
// unit type / granularity as numbers get larger and larger
int64_t Timer::getNs() {
  auto wall_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
    end - start
  );
  return wall_time.count();
}
