#include <Kernels/GEMMAligned.cuh>

__global__ void GEMMAligned(float *A, float *B, size_t m, size_t n, size_t k, float *C) {
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= k) return;
    
    float tmp = 0;
    for (uint i = 0; i < n; i++) {
        tmp += A[row * n + i] * B[i * k + col];
    }
    C[row * k + col] = tmp;
}
