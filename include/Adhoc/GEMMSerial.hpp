#pragma once

#ifndef GEMM_SERIAL_HPP
#define GEMM_SERIAL_HPP

#include <stddef.h>

// TODO you're missing alpha, adding C, etc.
inline void GEMMSerial(float *A, float *B, size_t m, size_t n, size_t k, float *C) {
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < k; col++) {
      for (int i = 0; i < n; i++)
        C[row * k + col] += A[row * n + i] * B[i * k + col];
    }
  }
}

#endif // GEMM_SERIAL_HPP
