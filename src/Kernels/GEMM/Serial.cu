#include <Kernels/GEMM/Serial.cuh>
#include <Tensor.cuh>
#include <Timer.hpp>

#include <memory>
#include <cassert>

template<typename T>
Status GEMM::Serial::gemm(Tensor<T>& A, Tensor<T>& B, Tensor<T>& C,
                          const Config& conf) {
  assert(A.dim_x == B.dim_y && "Cannot perform GEMM: A.x != B.y");
  assert(C.dim_y == A.dim_y && C.dim_x == B.dim_x &&
         "Cannot perform GEMM: C has incorrect size");

  const size_t m = A.dim_y;
  const size_t n = A.dim_x;
  const size_t k = B.dim_x;

  // TODO: I should wrap the timer/conf code somehow
  std::unique_ptr<Timer> t;
  if (conf.timer) t = std::make_unique<Timer>();

  for (size_t row = 0; row < m; row++) {
    for (size_t i = 0; i < n; i++) {
      T A_row_i = A[row * n + i];
      for (size_t col = 0; col < k; col++) {
        C[row * k + col] +=  A_row_i * B[i * k + col];
      }
    }
  }

  if (conf.timer) t->stop();

  Status res{};
  if (conf.timer) {
    res.timed = true;
    res.timeNs = t->getNs();
  }
  return res;
}

// Explicit instantiations:
template Status GEMM::Serial::gemm(Tensor<float>&, Tensor<float>&,
                                   Tensor<float>&, const Config&);

