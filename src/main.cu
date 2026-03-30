#include <Kernels/GEMM/Naive.cuh>
#include <Kernels/GEMM/Serial.cuh>
#include <Tensor.cuh>

#include <iostream>

int main() {
  constexpr size_t M = 3;
  constexpr size_t N = 2;
  constexpr size_t K = 3;

  Tensor<float> A{M, N, {
    7, 5,
    3, 0,
    2, 1
  }};
  A.push();
  Tensor<float> B{N, K, {
    3, 2, 5,
    4, 7, 8
  }};
  B.push();
  Tensor<float> C{M, K, 0};

  auto status = GEMM::Serial::gemm(A, B, C);

  std::cout << "-- Serial GEMM --" << std::endl;
  C.print();
  std::cout << "Elapsed time: " << status.timeNs << "ns\n";

  // --- GPU setup ---
  Tensor<float> C_Naive{M, K};
  C_Naive.push();

  // --- Naive GEMM ---
  status = GEMM::Naive::gemm(A, B, C_Naive);

  C_Naive.pull();
  std::cout << "-- Naive GEMM --" << std::endl;
  if (C == C_Naive) {
    std::cout << "Results match serial GEMM.\n";
  } else {
    std::cout << "ERROR: Results do not match serial GEMM.\n";
    C_Naive.print();
  }
  std::cout << "Elapsed time: " << status.timeNs << "ns\n";
}
