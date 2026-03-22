#pragma once

#ifndef GEMM_NAIVE_CUH
#define GEMM_NAIVE_CUH

__global__ void GEMMNaive(float *A, float *B, size_t m, size_t n, size_t k, float *C);

#endif // GEMM_NAIVE_CUH
