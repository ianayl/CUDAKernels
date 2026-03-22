#include <vector>
#include <iostream>
#include <stdio.h>

#include <cuda.h>

static constexpr size_t N = 7;

// I stole this
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

__global__ void vec_add(float *x_d, float *y_d, float *res_d, size_t n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        res_d[i] = x_d[i] + y_d[i];
}

int main() {
    std::cout << "Hello, world!\n";

    std::vector<float> a_h = {1, 2, 3, 4, 5, 6, 7};
    std::vector<float> b_h = {7, 6, 5, 4, 3, 2, 1};
    std::vector<float> c_h(N, 0);

    float *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, sizeof(float) * N);
    cudaMalloc(&b_d, sizeof(float) * N);
    cudaMalloc(&c_d, sizeof(float) * N);

    cudaMemcpy(a_d, a_h.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_h.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    vec_add<<<1, 64>>>(a_d, b_d, c_d, N);
    cudaCheckErrors("kernel call\n");
    cudaDeviceSynchronize();

    cudaMemcpy(c_h.data(), c_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (float f : c_h)
        std::cout << f << " ";
    std::cout << "\n";
}
