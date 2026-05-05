#include <Kernels/GEMM/Naive.cuh>
#include <Kernels/GEMM/Serial.cuh>
#include <Tensor.cuh>

#include <iostream>

int main() {
  constexpr size_t M = 3000;
  constexpr size_t N = 2000;
  constexpr size_t K = 3000;

  // Tensor<float> A{M, N, {
  //   7, 5,
  //   3, 0,
  //   2, 1
  // }};
  // A.push();
  // Tensor<float> B{N, K, {
  //   3, 2, 5,
  //   4, 7, 8
  // }};
  Tensor<float> A = Tensor<float>::random(M, N);
  A.push();
  Tensor<float> B = Tensor<float>::random(N, K);
  B.push();
  Tensor<float> C{M, K, 0};

  std::cout << "-- Serial GEMM --" << std::endl;
  auto status = GEMM::Serial::gemm(A, B, C);

  //C.print();
  std::cout << "Elapsed time: " << status.timeNs << "ns\n";

  // -- GPU setup --
  Tensor<float> C_Naive{M, K};
  C_Naive.push();

  std::cout << "-- Naive GEMM --" << std::endl;
  status = GEMM::Naive::gemm(A, B, C_Naive);

  C_Naive.pull();
  if (C == C_Naive) {
    std::cout << "Results match serial GEMM.\n";
  } else {
    std::cout << "ERROR: Results do not match serial GEMM.\n";
    C_Naive.print();
  }
  std::cout << "Elapsed time: " << status.timeNs << "ns\n";
}
