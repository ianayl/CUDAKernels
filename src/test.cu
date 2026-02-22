#include <Kernels/GEMMSerial.hpp>
#include <Kernels/GEMMNaive.cuh>
#include <Kernels/GEMMAligned.cuh>
#include <Timer.hpp>
#include <MatrixUtils.hpp>

#include <vector>

int main() {
  constexpr size_t M = 3;
  constexpr size_t N = 2;
  constexpr size_t K = 3;

  std::vector<float> hA = {
    3, 2, 5,
    4, 7, 8
  };
  std::vector<float> hB = {
    7, 5,
    3, 0,
    2, 1
  };
  std::vector<float> hC_Serial(M*K, 0);

  Timer t_serial{};
  GEMMSerial(hA.data(), hB.data(), M, N, K, hC_Serial.data());
  t_serial.stop();

  std::cout << "-- Serial GEMM --" << std::endl;
  PrintMatrix(hC_Serial.data(), M, K);
  std::cout << "Elapsed time: " << t_serial.getNs() << "ns\n";

  // --- GPU setup ---
  float *dA, *dB, *dC;
  cudaMalloc(&dA, sizeof(float)*M*N);
  cudaMalloc(&dB, sizeof(float)*N*K);
  cudaMalloc(&dC, sizeof(float)*M*K);

  cudaMemcpy(dA, hA.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), sizeof(float)*N*K, cudaMemcpyHostToDevice);

  // --- Naive GEMM ---
  std::vector<float> hC_Naive(M*K, 0);
  cudaMemcpy(dC, hC_Naive.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice);

  dim3 blockDim(32, 32);
  dim3 gridDim((M + 32 - 1) / 32, (K + 32 - 1) / 32);
  Timer t_naive{};
  GEMMNaive<<<gridDim, blockDim>>>(dA, dB, M, N, K, dC);
  t_naive.stop();

  cudaMemcpy(hC_Naive.data(), dC, sizeof(float)*M*K, cudaMemcpyDeviceToHost);
  
  std::cout << "-- Naive GEMM --" << std::endl;
  if (CheckMatrixEqual(hC_Serial.data(), hC_Naive.data(), M, K)) {
    std::cout << "Results match serial GEMM.\n";
  } else {
    std::cout << "ERROR: Results do not match serial GEMM.\n";
  }
  //PrintMatrix(hC_Naive.data(), M, K);
  std::cout << "Elapsed time: " << t_naive.getNs() << "ns\n";

  // --- Aligned GEMM ---
  std::vector<float> hC_Aligned(M*K, 0);
  cudaMemcpy(dC, hC_Aligned.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice);

  // TODO make helper wrappers to auto generate the dims and launch kernel?
  dim3 gridDimAligned((K + 32 - 1) / 32, (M + 32 - 1) / 32);
  Timer t_aligned{};
  GEMMAligned<<<gridDimAligned, blockDim>>>(dA, dB, M, N, K, dC);
  t_aligned.stop();

  cudaMemcpy(hC_Aligned.data(), dC, sizeof(float)*M*K, cudaMemcpyDeviceToHost);
  
  std::cout << "-- Aligned GEMM --" << std::endl;
  if (CheckMatrixEqual(hC_Serial.data(), hC_Aligned.data(), M, K)) {
    std::cout << "Results match serial GEMM.\n";
  } else {
    std::cout << "ERROR: Results do not match serial GEMM.\n";
  }
  //PrintMatrix(hC_Naive.data(), M, K);
  std::cout << "Elapsed time: " << t_aligned.getNs() << "ns\n";
}
