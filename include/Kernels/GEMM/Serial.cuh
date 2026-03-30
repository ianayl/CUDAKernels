#pragma once

#ifndef KERNELS_GEMM_SERIAL_HPP
#define KERNELS_GEMM_SERIAL_HPP

#include <Status.cuh>
#include <Config.hpp>
#include <Tensor.cuh>

#include <stddef.h>

namespace GEMM::Serial {
  template<typename T>
  Status gemm(Tensor<T>& A, Tensor<T>& B, Tensor<T>& C,
              const Config& conf = DefaultKernelConfig);

  // // Bad:
  // inline void gemm(float *A, float *B, size_t m, size_t n, size_t k, float *C) {
  //   for (int row = 0; row < m; row++) {
  //     for (int col = 0; col < k; col++) {
  //       for (int i = 0; i < n; i++)
  //         C[row * k + col] += A[row * n + i] * B[i * k + col];
  //     }
  //   }
  // }
}
#endif // KERNELS_GEMM_SERIAL_HPP
