#pragma once
#include <cuda_runtime.h>

// One thread computes one CSR row.
// All pointers refer to memory already allocated on the GPU.
void spmv_naive(int numRows,
                const int* rowIdx,
                const int* colIdx,
                const double* vals,
                const double* x,
                double* y);

void spmv_warp(int numRows,
               const int* rowIdx,
               const int* colIdx,
               const double* vals,
               const double* x,
               double* y);

void spmv_warp_dot(
    int numRows,
    const int* rowIdx,
    const int* colIdx,
    const double* vals,
    const double* x,
    double* y,
    double* dot,
    cudaStream_t stream = 0);
