/*
 * CUDA SpMV kernels for the ECE379K sparse solver project.
 *
 * Organization intentionally follows the Lab 2 SGEMM ladder:
 * one kernel and one launcher per implementation.
 */

#include <cuda_runtime.h>

#include "spmv_kernels.cuh"

// ---------------------------------------------------------------------
// Naive CSR SpMV
// ---------------------------------------------------------------------

__global__ void spmv_naive_kernel(
    int numRows,
    const int* rowIdx,
    const int* colIdx,
    const double* vals,
    const double* x,
    double* y)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows) {
        double sum = 0.0;

        for (int k = rowIdx[row]; k < rowIdx[row + 1]; ++k) {
            sum += vals[k] * x[colIdx[k]];
        }

        y[row] = sum;
    }
}

void spmv_naive(
    int numRows,
    const int* rowIdx,
    const int* colIdx,
    const double* vals,
    const double* x,
    double* y)
{
    constexpr int threads = 256;
    const int blocks = (numRows + threads - 1) / threads;

    spmv_naive_kernel<<<blocks, threads>>>(
        numRows,
        rowIdx,
        colIdx,
        vals,
        x,
        y);
}

// ---------------------------------------------------------------------
// Warp-per-row CSR SpMV
// ---------------------------------------------------------------------

__global__ void spmv_warp_kernel(
    int numRows,
    const int* rowIdx,
    const int* colIdx,
    const double* vals,
    const double* x,
    double* y)
{
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = thread / 32;
    const int lane = thread % 32;
    const int row = warp;

    if (row < numRows) {
        double sum = 0.0;

        const int start = rowIdx[row];
        const int end = rowIdx[row + 1];

        for (int k = start + lane; k < end; k += 32) {
            sum += vals[k] * x[colIdx[k]];
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }

        if (lane == 0) {
            y[row] = sum;
        }
    }
}

void spmv_warp(
    int numRows,
    const int* rowIdx,
    const int* colIdx,
    const double* vals,
    const double* x,
    double* y)
{
    constexpr int threads = 256;
    constexpr int warps_per_block = threads / 32;

    const int blocks =
        (numRows + warps_per_block - 1) / warps_per_block;

    spmv_warp_kernel<<<blocks, threads>>>(
        numRows,
        rowIdx,
        colIdx,
        vals,
        x,
        y);
}

// ---------------------------------------------------------------------
// Warp-per-row CSR SpMV + dot(x, y)
//
// Used by CG for:
//
//     Ap = A * p
//     pAp = p^T * Ap
//
// The SpMV result is already available in lane 0 of each warp, so the
// dot-product contribution can be formed without launching a second
// kernel or rereading the entire p and Ap vectors.
// ---------------------------------------------------------------------

__global__ void spmv_warp_dot_kernel(
    int numRows,
    const int* rowIdx,
    const int* colIdx,
    const double* vals,
    const double* x,
    double* y,
    double* dot)
{
    constexpr int warp_size = 32;
    constexpr int warps_per_block = 256 / warp_size;
    constexpr unsigned int full_mask = 0xffffffffu;

    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp = thread / warp_size;
    const int lane = thread % warp_size;
    const int warp_in_block = threadIdx.x / warp_size;
    const int row = global_warp;

    double sum = 0.0;

    if (row < numRows) {
        const int start = rowIdx[row];
        const int end = rowIdx[row + 1];

        for (int k = start + lane; k < end; k += warp_size) {
            sum += vals[k] * x[colIdx[k]];
        }
    }

    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(full_mask, sum, offset);
    }

    __shared__ double warp_dot_sums[warps_per_block];

    if (lane == 0) {
        if (row < numRows) {
            y[row] = sum;
            warp_dot_sums[warp_in_block] = x[row] * sum;
        } else {
            warp_dot_sums[warp_in_block] = 0.0;
        }
    }

    __syncthreads();

    if (warp_in_block == 0) {
        double block_sum =
            (lane < warps_per_block)
                ? warp_dot_sums[lane]
                : 0.0;

        for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
            block_sum +=
                __shfl_down_sync(full_mask, block_sum, offset);
        }

        if (lane == 0) {
            atomicAdd(dot, block_sum);
        }
    }
}

void spmv_warp_dot(
    int numRows,
    const int* rowIdx,
    const int* colIdx,
    const double* vals,
    const double* x,
    double* y,
    double* dot,
    cudaStream_t stream)
{
    constexpr int threads = 256;
    constexpr int warps_per_block = threads / 32;

    const int blocks =
        (numRows + warps_per_block - 1) / warps_per_block;

    spmv_warp_dot_kernel<<<blocks, threads, 0, stream>>>(
        numRows,
        rowIdx,
        colIdx,
        vals,
        x,
        y,
        dot);
}