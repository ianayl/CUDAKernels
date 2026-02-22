#include <Kernels/GEMMAligned.cuh>

__global__ void GEMMNaive(float *A, float *B, size_t m, size_t n, size_t k, float *C) {
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= k) return;
    
    float tmp = 0;
    for (uint i = 0; i < n; i++) {
        tmp += A[row * n + i] * B[i * k + col];
    }
    C[row * k + col] = tmp;
    // for i : length of n
    // C[row * K + col] += A[row * N + i] + B[i * K + col]
}
