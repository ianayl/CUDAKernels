#include <vector>
#include <iostream>
#include <stdio.h>

#include <cuda.h>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

constexpr int N = 18;
constexpr int BLOCKSIZE = 16;

// BAD, don't do tid = threadIdx.x*2 unless you want extra branch instructions in the for loop checking if id is within bounds
__global__ void reductionNaive(int *a, int *b, int n) {
    __shared__ int sdata[BLOCKSIZE*2];
    const int tid = threadIdx.x * 2; 
    int i = 2*(blockDim.x * blockIdx.x + threadIdx.x);
    if (i >= n) return;

    sdata[tid] = a[i];
    sdata[tid + 1] = a[i + 1];
    __syncthreads();

    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (tid % (2*stride) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    //b[i] = 8;
    b[i] = sdata[tid];
    b[i+1] = sdata[tid+1];
    // b[i] = sdata[tid];
    // b[i+1] = sdata[tid + 1];
}

// BAD, don't do tid = threadIdx.x*2 unless you want extra branch instructions in the for loop checking if id is within bounds
__global__ void reductionNaive1(int *a, int *b, int n) {
    __shared__ int sdata[BLOCKSIZE * 2];
    const int tid = threadIdx.x * 2; 
    int i = 2*(blockDim.x * blockIdx.x + threadIdx.x);
    if (i >= n) return;

    sdata[tid] = a[i];
    sdata[tid + 1] = a[i + 1];
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int id = tid * stride;
        if (id < blockDim.x) {
            sdata[id] += sdata[id + stride];
        }
        __syncthreads();
    }

    b[i] = sdata[0];
    // b[i] = sdata[tid];
    // b[i+1] = sdata[tid + 1];
}


__global__ void reductionCoalesed(int *dA, int *dB, int n) {
    __shared__ int sdata[BLOCKSIZE*2];
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int tid = threadIdx.x;
    if (i >= n) return;

    sdata[tid] = dA[i];
    __syncthreads();



    //dB[i] = sdata[tid];
}

int main() {
    std::vector<int> hA(N);
    std::vector<int> hB(N);
    int solution = 0;
    for (int i = 0; i < N; i++) {
        hA[i] = i+1;
        solution += i+1;
    }

    for (const int& n : hA)
        std::cout << n << " ";
    std::cout << "\n";
    // for (const int& n : hB)
    //     std::cout << n << " ";
    // std::cout << "\n";

    int *dA, *dB;
    cudaMalloc(&dA, sizeof(int)*N);
    cudaMalloc(&dB, sizeof(int)*N);
    cudaMemcpy(dA, hA.data(), sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), sizeof(int)*N, cudaMemcpyHostToDevice);

    //reductionNaive<<<(N/2 + BLOCKSIZE - 1)/ BLOCKSIZE, BLOCKSIZE>>>(dA, dB, N);
    //reductionNaive1<<<(N/2 + BLOCKSIZE - 1)/ BLOCKSIZE, BLOCKSIZE>>>(dA, dB, N);
    reductionCoalesed<<<(N/2 + BLOCKSIZE - 1)/ BLOCKSIZE, BLOCKSIZE>>>(dA, dB, N);

    cudaMemcpy(hB.data(), dB, sizeof(int)*N, cudaMemcpyDeviceToHost);

    std::cout << "solution: " << solution << std::endl;
    for (const int& n : hB)
        std::cout << n << " ";
    std::cout << "\n";

}
