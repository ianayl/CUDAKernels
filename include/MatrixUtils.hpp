#pragma once

#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP

#include <cstddef>
#include <iostream>

inline void PrintMatrix(float *A, size_t m, size_t n) {
  for (size_t row = 0; row < m; row++) {
    for (size_t col = 0; col < n; col++)
      std::cout << A[row * n + col] << " ";
    std::cout << "\n";
  }
}

inline bool CheckMatrixEqual(float *A, float *B, size_t m, size_t n, bool printMatrix = false) {
  for (size_t row = 0; row < m; row++) {
    for (size_t col = 0; col < n; col++) {
      if (A[row * n + col] != B[row * n + col]) {
        std::cout << "CheckMatrixEqual: Matrixes do not match!\n";
        if (printMatrix) {
          std::cout << "== Matrix A: ==\n";
          PrintMatrix(A, m, n);
          std::cout << "== Matrix B: ==\n";
          PrintMatrix(B, m, n);
        }
        return false;
      }
    }
  }
  return true;
}

#endif // MATRIX_UTILS_HPP
