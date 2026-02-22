#pragma once

#ifndef GEMM_ALIGNED_CUH
#define GEMM_ALIGNED_CUH

__global__ void GEMMAligned(float *A, float *B, size_t m, size_t n, size_t k, float *C);

#endif // GEMM_ALIGNED_CUH
