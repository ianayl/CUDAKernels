#pragma once

#ifndef STATUS_CUH
#define STATUS_CUH

#include <cuda.h>
#include <chrono>

struct Status {
	cudaError_t err = cudaErrorUnknown;
	bool timed = false;
	int timeNs;

	constexpr explicit operator bool() const noexcept {
		return err == cudaSuccess;
	}
};

#endif // STATUS_CUH
