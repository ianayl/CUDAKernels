#include <vector>
#include <iostream>
#include <stdio.h>

#include <cuda.h>

constexpr int N = 9;

__global__ void prefixSumNaive(int *a, int *b, int n) {
    __shared__ int res[N];
    int i = threadIdx.x;
    if (i < n) res[i] = a[i];

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (i >= stride) res[i] += res[i-stride];
    }
    
    b[i] = res[i];
}

int main() {
    std::vector<int> hA(N);
    std::vector<int> hB(N);
    for (int i = 0; i < N; i++) hA[i] = i+1;

    for (const int& n : hA)
        std::cout << n << " ";
    std::cout << "\n";
    for (const int& n : hB)
        std::cout << n << " ";
    std::cout << "\n";

    int *dA, *dB;
    cudaMalloc(&dA, sizeof(int)*N);
    cudaMalloc(&dB, sizeof(int)*N);
    cudaMemcpy(dA, hA.data(), sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), sizeof(int)*N, cudaMemcpyHostToDevice);

    prefixSumNaive<<<1, N>>>(dA, dB, N);

    cudaMemcpy(hB.data(), dB, sizeof(int)*N, cudaMemcpyDeviceToHost);

    for (int i = 0, accum = 0; i < N; i++) {
        accum += i+1;
        std::cout << accum << " ";
    }
    std::cout << "\n";

    for (const int& n : hB)
        std::cout << n << " ";
    std::cout << "\n";

}
