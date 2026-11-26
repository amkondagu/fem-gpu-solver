/*
 * gpu_cg.cu -- GPU-resident Jacobi-preconditioned Conjugate Gradient benchmark.
 *
 * Uses a CUDA Graph to submit one full convergence-check interval
 * of PCG iterations with a single graph launch.
 *
 * Usage:
 *   ./gpu_cg matrix.mtx output.csv
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "read_mtx.h"
#include "rcm.h"
#include "spmv.h"
#include "spmv_kernels.cuh"

#define CUDA_CHECK(x)                                                      \
    do {                                                                   \
        cudaError_t e_ = (x);                                              \
        if (e_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s:%d: %s\n  -> %s\n", __FILE__,         \
                         __LINE__, #x, cudaGetErrorString(e_));             \
            std::exit(2);                                                  \
        }                                                                  \
    } while (0)

namespace {

constexpr int kThreads = 256;
constexpr int kMaxDotBlocks = 1024;
constexpr int kMaxIterations = 50000;
constexpr double kRtol = 1e-8;
constexpr double kAtol = 0.0;
constexpr int kWarmups = 1;
constexpr int kRepeats = 5;
constexpr int kCheckInterval = 50;

constexpr int kRunning = 0;
constexpr int kNumericalFailure = -1;
constexpr int kNonpositiveCurvature = -2;

static_assert(kCheckInterval % 2 == 0,
              "Graph interval must be even so rho pointer roles remain unchanged");

struct GpuCgResult {
    int iterations = 0;
    double residual_norm = std::numeric_limits<double>::infinity();
    double relative_residual = std::numeric_limits<double>::infinity();
    double solve_ms = 0.0;
    std::string status = "numerical_failure";
    std::vector<double> x;
};

__global__ void dot_kernel(
    const double* a,
    const double* b,
    int n,
    double* out)
{
    constexpr unsigned int full_mask = 0xffffffffu;
    constexpr int warp_size = 32;

    double local = 0.0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        local += a[i] * b[i];
    }

    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        local += __shfl_down_sync(full_mask, local, offset);
    }

    __shared__ double warp_sums[kThreads / warp_size];

    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;

    if (lane == 0) {
        warp_sums[warp] = local;
    }

    __syncthreads();

    if (warp == 0) {
        local = (lane < kThreads / warp_size) ? warp_sums[lane] : 0.0;

        for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
            local += __shfl_down_sync(full_mask, local, offset);
        }

        if (lane == 0) {
            atomicAdd(out, local);
        }
    }
}

__global__ void initialize_pcg_kernel(
    int n,
    const double* r,
    const double* inverse_diagonal,
    double* z,
    double* p)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        const double zi = inverse_diagonal[i] * r[i];
        z[i] = zi;
        p[i] = zi;
    }
}

__global__ void update_x_r_z_kernel(
    int n,
    double* x,
    double* r,
    double* z,
    const double* p,
    const double* Ap,
    const double* inverse_diagonal,
    const double* rho,
    const double* pAp,
    double* rho_new,
    double* rr_new,
    int* status)
{
    constexpr unsigned int full_mask = 0xffffffffu;
    constexpr int warp_size = 32;

    if (*status != kRunning) {
        return;
    }

    const double rho_value = *rho;
    const double denom = *pAp;

    if (!isfinite(rho_value) || rho_value <= 0.0 ||
        !isfinite(denom) || denom <= 0.0) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *status = (isfinite(denom) && denom <= 0.0)
                ? kNonpositiveCurvature
                : kNumericalFailure;
        }
        return;
    }

    const double alpha = rho_value / denom;

    if (!isfinite(alpha)) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *status = kNumericalFailure;
        }
        return;
    }

    double local_rho = 0.0;
    double local_rr = 0.0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        x[i] += alpha * p[i];

        const double new_r = r[i] - alpha * Ap[i];
        const double new_z = inverse_diagonal[i] * new_r;

        r[i] = new_r;
        z[i] = new_z;

        local_rho += new_r * new_z;
        local_rr += new_r * new_r;
    }

    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        local_rho += __shfl_down_sync(full_mask, local_rho, offset);
        local_rr += __shfl_down_sync(full_mask, local_rr, offset);
    }

    __shared__ double warp_rho_sums[kThreads / warp_size];
    __shared__ double warp_rr_sums[kThreads / warp_size];

    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;

    if (lane == 0) {
        warp_rho_sums[warp] = local_rho;
        warp_rr_sums[warp] = local_rr;
    }

    __syncthreads();

    if (warp == 0) {
        local_rho = (lane < kThreads / warp_size) ? warp_rho_sums[lane] : 0.0;
        local_rr = (lane < kThreads / warp_size) ? warp_rr_sums[lane] : 0.0;

        for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
            local_rho += __shfl_down_sync(full_mask, local_rho, offset);
            local_rr += __shfl_down_sync(full_mask, local_rr, offset);
        }

        if (lane == 0) {
            atomicAdd(rho_new, local_rho);
            atomicAdd(rr_new, local_rr);
        }
    }
}

__global__ void update_p_kernel(
    int n,
    const double* z,
    double* p,
    const double* rho,
    const double* rho_new,
    const double* rr_new,
    double tolerance_sq,
    int* status,
    int iteration)
{
    if (*status != kRunning) {
        return;
    }

    const double old_rho = *rho;
    const double new_rho = *rho_new;
    const double new_rr = *rr_new;

    if (!isfinite(new_rr) || new_rr < 0.0) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *status = kNumericalFailure;
        }
        return;
    }

    if (new_rr <= tolerance_sq) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *status = iteration;
        }
        return;
    }

    if (!isfinite(old_rho) || old_rho <= 0.0 ||
        !isfinite(new_rho) || new_rho <= 0.0) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *status = kNumericalFailure;
        }
        return;
    }

    const double beta = new_rho / old_rho;

    if (!isfinite(beta)) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *status = kNumericalFailure;
        }
        return;
    }

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        p[i] = z[i] + beta * p[i];
    }
}

__global__ void residual_kernel(
    int n,
    const double* b,
    const double* Ax,
    double* r)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        r[i] = b[i] - Ax[i];
    }
}

__global__ void restart_pcg_kernel(
    int n,
    const double* r,
    const double* inverse_diagonal,
    double* z,
    double* p,
    double* rho)
{
    constexpr unsigned int full_mask = 0xffffffffu;
    constexpr int warp_size = 32;

    double local_rho = 0.0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        const double zi = inverse_diagonal[i] * r[i];
        z[i] = zi;
        p[i] = zi;
        local_rho += r[i] * zi;
    }

    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        local_rho += __shfl_down_sync(full_mask, local_rho, offset);
    }

    __shared__ double warp_sums[kThreads / warp_size];

    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;

    if (lane == 0) {
        warp_sums[warp] = local_rho;
    }

    __syncthreads();

    if (warp == 0) {
        local_rho = (lane < kThreads / warp_size) ? warp_sums[lane] : 0.0;

        for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
            local_rho += __shfl_down_sync(full_mask, local_rho, offset);
        }

        if (lane == 0) {
            atomicAdd(rho, local_rho);
        }
    }
}

int blocks_for(int n)
{
    return std::max(
        1,
        std::min(kMaxDotBlocks, (n + kThreads - 1) / kThreads));
}

void device_dot(
    const double* a,
    const double* b,
    int n,
    double* out)
{
    CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(double)));

    dot_kernel<<<blocks_for(n), kThreads>>>(a, b, n, out);

    CUDA_CHECK(cudaGetLastError());
}

double stable_norm(const std::vector<double>& values)
{
    long double scale = 0.0L;
    long double sum = 1.0L;

    for (double v : values) {
        const long double mag = std::abs(static_cast<long double>(v));

        if (mag == 0.0L) {
            continue;
        }

        if (mag > scale) {
            const long double ratio = scale / mag;
            sum = 1.0L + sum * ratio * ratio;
            scale = mag;
        } else {
            const long double ratio = mag / scale;
            sum += ratio * ratio;
        }
    }

    return scale == 0.0L
        ? 0.0
        : static_cast<double>(scale * std::sqrt(sum));
}

double cpu_relative_residual(
    const CSR& A,
    const std::vector<double>& b,
    const std::vector<double>& x)
{
    std::vector<double> Ax;
    spmv_serial(A, x, Ax);

    std::vector<double> r(b.size());

    for (std::size_t i = 0; i < b.size(); ++i) {
        r[i] = b[i] - Ax[i];
    }

    const double bnorm = stable_norm(b);
    const double rnorm = stable_norm(r);

    return bnorm > 0.0 ? rnorm / bnorm : rnorm;
}

std::vector<double> build_jacobi_inverse(const CSR& A)
{
    std::vector<double> inverse_diagonal(
        static_cast<std::size_t>(A.numRows), 0.0);

    for (int row = 0; row < A.numRows; ++row) {
        long double diagonal = 0.0L;

        for (int k = A.rowIdx[row]; k < A.rowIdx[row + 1]; ++k) {
            if (A.colIdx[k] == row) {
                diagonal += static_cast<long double>(A.vals[k]);
            }
        }

        if (!std::isfinite(diagonal) || diagonal <= 0.0L) {
            throw std::runtime_error(
                "Jacobi-PCG requires a finite positive diagonal");
        }

        const long double inverse = 1.0L / diagonal;

        if (!std::isfinite(inverse)) {
            throw std::runtime_error(
                "Jacobi inverse diagonal is not finite");
        }

        inverse_diagonal[static_cast<std::size_t>(row)] =
            static_cast<double>(inverse);
    }

    return inverse_diagonal;
}

struct DeviceProblem {
    int n = 0;
    int nnz = 0;

    int* rowIdx = nullptr;
    int* colIdx = nullptr;
    double* vals = nullptr;

    double* b = nullptr;
    double* x = nullptr;
    double* r = nullptr;
    double* z = nullptr;
    double* p = nullptr;
    double* Ap = nullptr;
    double* inverse_diagonal = nullptr;

    double* rho = nullptr;
    double* rho_new = nullptr;
    double* rr_new = nullptr;
    double* pAp = nullptr;

    int* status = nullptr;

    cudaStream_t graph_stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    bool graph_ready = false;

    double setup_ms = 0.0;

    DeviceProblem(
        const CSR& A,
        const std::vector<double>& rhs)
    {
        n = A.numRows;
        nnz = static_cast<int>(A.vals.size());

        const auto start = std::chrono::steady_clock::now();

        const std::vector<double> h_inverse_diagonal =
            build_jacobi_inverse(A);

        CUDA_CHECK(cudaMalloc(
            &rowIdx,
            A.rowIdx.size() * sizeof(int)));

        CUDA_CHECK(cudaMalloc(
            &colIdx,
            A.colIdx.size() * sizeof(int)));

        CUDA_CHECK(cudaMalloc(
            &vals,
            A.vals.size() * sizeof(double)));

        CUDA_CHECK(cudaMalloc(
            &b,
            static_cast<std::size_t>(n) * sizeof(double)));

        CUDA_CHECK(cudaMalloc(
            &x,
            static_cast<std::size_t>(n) * sizeof(double)));

        CUDA_CHECK(cudaMalloc(
            &r,
            static_cast<std::size_t>(n) * sizeof(double)));

        CUDA_CHECK(cudaMalloc(
            &z,
            static_cast<std::size_t>(n) * sizeof(double)));

        CUDA_CHECK(cudaMalloc(
            &p,
            static_cast<std::size_t>(n) * sizeof(double)));

        CUDA_CHECK(cudaMalloc(
            &Ap,
            static_cast<std::size_t>(n) * sizeof(double)));

        CUDA_CHECK(cudaMalloc(
            &inverse_diagonal,
            static_cast<std::size_t>(n) * sizeof(double)));

        CUDA_CHECK(cudaMalloc(&rho, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&rho_new, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&rr_new, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&pAp, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&status, sizeof(int)));

        CUDA_CHECK(cudaMemcpy(
            rowIdx,
            A.rowIdx.data(),
            A.rowIdx.size() * sizeof(int),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(
            colIdx,
            A.colIdx.data(),
            A.colIdx.size() * sizeof(int),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(
            vals,
            A.vals.data(),
            A.vals.size() * sizeof(double),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(
            b,
            rhs.data(),
            static_cast<std::size_t>(n) * sizeof(double),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(
            inverse_diagonal,
            h_inverse_diagonal.data(),
            static_cast<std::size_t>(n) * sizeof(double),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaStreamCreate(&graph_stream));
        CUDA_CHECK(cudaDeviceSynchronize());

        setup_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    }

    DeviceProblem(const DeviceProblem&) = delete;
    DeviceProblem& operator=(const DeviceProblem&) = delete;

    ~DeviceProblem()
    {
        if (graph_exec) {
            cudaGraphExecDestroy(graph_exec);
        }

        if (graph) {
            cudaGraphDestroy(graph);
        }

        if (graph_stream) {
            cudaStreamDestroy(graph_stream);
        }

        cudaFree(rowIdx);
        cudaFree(colIdx);
        cudaFree(vals);
        cudaFree(b);
        cudaFree(x);
        cudaFree(r);
        cudaFree(z);
        cudaFree(p);
        cudaFree(Ap);
        cudaFree(inverse_diagonal);
        cudaFree(rho);
        cudaFree(rho_new);
        cudaFree(rr_new);
        cudaFree(pAp);
        cudaFree(status);
    }

    void reset(int blocks)
    {
        const std::size_t bytes =
            static_cast<std::size_t>(n) * sizeof(double);

        CUDA_CHECK(cudaMemsetAsync(x, 0, bytes));

        CUDA_CHECK(cudaMemcpyAsync(
            r,
            b,
            bytes,
            cudaMemcpyDeviceToDevice));

        initialize_pcg_kernel<<<blocks, kThreads>>>(
            n, r, inverse_diagonal, z, p);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemsetAsync(
            status,
            0,
            sizeof(int)));
    }

    void build_graph(
        double tolerance_sq,
        int blocks)
    {
        if (graph_ready) {
            return;
        }

        CUDA_CHECK(cudaStreamBeginCapture(
            graph_stream,
            cudaStreamCaptureModeGlobal));

        for (int step = 0; step < kCheckInterval; ++step) {
            CUDA_CHECK(cudaMemsetAsync(
                pAp,
                0,
                sizeof(double),
                graph_stream));

            spmv_warp_dot(
                n, rowIdx, colIdx, vals, p, Ap, pAp, graph_stream);

            CUDA_CHECK(cudaMemsetAsync(
                rho_new,
                0,
                sizeof(double),
                graph_stream));

            CUDA_CHECK(cudaMemsetAsync(
                rr_new,
                0,
                sizeof(double),
                graph_stream));

            update_x_r_z_kernel<<<blocks, kThreads, 0, graph_stream>>>(
                n, x, r, z, p, Ap, inverse_diagonal,
                rho, pAp, rho_new, rr_new, status);

            update_p_kernel<<<blocks, kThreads, 0, graph_stream>>>(
                n, z, p, rho, rho_new, rr_new,
                tolerance_sq, status, step + 1);

            std::swap(rho, rho_new);
        }

        CUDA_CHECK(cudaStreamEndCapture(
            graph_stream,
            &graph));

        CUDA_CHECK(cudaGraphInstantiate(
            &graph_exec,
            graph,
            nullptr,
            nullptr,
            0));

        graph_ready = true;
    }
};

GpuCgResult solve_gpu(
    DeviceProblem& d,
    const CSR& A,
    const std::vector<double>& rhs)
{
    GpuCgResult result;

    const int n = d.n;
    const int blocks = blocks_for(n);

    const double bnorm = stable_norm(rhs);
    const double tolerance = std::max(kAtol, kRtol * bnorm);
    const double tolerance_sq = tolerance * tolerance;

    const auto start = std::chrono::steady_clock::now();

    d.reset(blocks);

    device_dot(d.r, d.r, n, d.rr_new);
    device_dot(d.r, d.z, n, d.rho);

    double h_rr = 0.0;
    double h_rho = 0.0;

    CUDA_CHECK(cudaMemcpyAsync(
        &h_rr,
        d.rr_new,
        sizeof(double),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpyAsync(
        &h_rho,
        d.rho,
        sizeof(double),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaStreamSynchronize(0));

    if (!std::isfinite(h_rr) || h_rr < 0.0) {
        result.status = "numerical_failure";
    } else if (std::sqrt(h_rr) <= tolerance) {
        result.status = "converged";
    } else if (!std::isfinite(h_rho) || h_rho <= 0.0) {
        result.status = "numerical_failure";
    } else {
        result.status = "max_iterations";

        d.build_graph(tolerance_sq, blocks);

        int completed = 0;

        while (completed < kMaxIterations) {
            CUDA_CHECK(cudaGraphLaunch(
                d.graph_exec,
                d.graph_stream));

            int h_status = kRunning;

            CUDA_CHECK(cudaMemcpyAsync(
                &h_status,
                d.status,
                sizeof(int),
                cudaMemcpyDeviceToHost,
                d.graph_stream));

            CUDA_CHECK(cudaStreamSynchronize(d.graph_stream));

            if (h_status == kNonpositiveCurvature) {
                completed += kCheckInterval;
                result.iterations = completed;
                result.status = "nonpositive_curvature";
                break;
            }

            if (h_status == kNumericalFailure) {
                completed += kCheckInterval;
                result.iterations = completed;
                result.status = "numerical_failure";
                break;
            }

            if (h_status > 0) {
                result.iterations = completed + h_status;

                spmv_warp(
                    n,
                    d.rowIdx,
                    d.colIdx,
                    d.vals,
                    d.x,
                    d.Ap);

                CUDA_CHECK(cudaGetLastError());

                residual_kernel<<<blocks, kThreads>>>(
                    n,
                    d.b,
                    d.Ap,
                    d.r);

                CUDA_CHECK(cudaGetLastError());

                device_dot(d.r, d.r, n, d.rr_new);

                double h_true_rr = 0.0;

                CUDA_CHECK(cudaMemcpyAsync(
                    &h_true_rr,
                    d.rr_new,
                    sizeof(double),
                    cudaMemcpyDeviceToHost));

                CUDA_CHECK(cudaStreamSynchronize(0));

                if (!std::isfinite(h_true_rr) || h_true_rr < 0.0) {
                    result.status = "numerical_failure";
                    break;
                }

                if (std::sqrt(h_true_rr) <= tolerance) {
                    result.status = "converged";
                    break;
                }

                CUDA_CHECK(cudaMemsetAsync(
                    d.rho,
                    0,
                    sizeof(double)));

                restart_pcg_kernel<<<blocks, kThreads>>>(
                    n, d.r, d.inverse_diagonal, d.z, d.p, d.rho);

                CUDA_CHECK(cudaGetLastError());

                CUDA_CHECK(cudaMemsetAsync(
                    d.status,
                    0,
                    sizeof(int)));

                CUDA_CHECK(cudaStreamSynchronize(0));

                completed += h_status;
                continue;
            }

            completed += kCheckInterval;
            result.iterations = completed;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    result.solve_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();

    result.x.resize(static_cast<std::size_t>(n));

    CUDA_CHECK(cudaMemcpy(
        result.x.data(),
        d.x,
        static_cast<std::size_t>(n) * sizeof(double),
        cudaMemcpyDeviceToHost));

    result.relative_residual =
        cpu_relative_residual(A, rhs, result.x);

    result.residual_norm =
        result.relative_residual * bnorm;

    if (result.status == "converged" &&
        (!std::isfinite(result.relative_residual) ||
         result.relative_residual > kRtol)) {
        result.status = "numerical_failure";
    }

    return result;
}

void write_header(std::ofstream& out)
{
    out
        << "matrix,n,nnz,ordering,backend,"
           "repeat,solve_ms,iterations,"
           "relative_residual,status,correct,"
           "reordering_ms,rhs_permutation_ms,"
           "gpu_setup_ms,solve_plus_setup_ms,"
           "end_to_end_ms,rtol,atol,"
           "max_iterations\n";
}

void run_ordering(
    const std::string& matrix_name,
    const char* ordering,
    const CSR& A,
    const std::vector<double>& rhs,
    double reordering_ms,
    double rhs_permutation_ms,
    std::ofstream& out)
{
    DeviceProblem device(A, rhs);

    std::printf(
        "%s / %s: n=%d nnz=%zu setup=%.3f ms\n",
        matrix_name.c_str(),
        ordering,
        A.numRows,
        A.vals.size(),
        device.setup_ms);

    for (int w = 0; w < kWarmups; ++w) {
        const GpuCgResult warm =
            solve_gpu(device, A, rhs);

        std::printf(
            "  warmup: %s, %d iterations, rel=%.3e\n",
            warm.status.c_str(),
            warm.iterations,
            warm.relative_residual);
    }

    for (int repeat = 0; repeat < kRepeats; ++repeat) {
        const GpuCgResult result =
            solve_gpu(device, A, rhs);

        const bool correct =
            result.status == "converged" &&
            std::isfinite(result.relative_residual) &&
            result.relative_residual <= kRtol;

        const double solve_plus_setup =
            result.solve_ms + device.setup_ms;

        const double end_to_end =
            solve_plus_setup +
            reordering_ms +
            rhs_permutation_ms;

        out
            << '"' << matrix_name << '"'
            << ',' << A.numRows
            << ',' << A.vals.size()
            << ',' << ordering
            << ',' << "gpu_warp_jacobi_pcg"
            << ',' << repeat
            << ',' << result.solve_ms
            << ',' << result.iterations
            << ',' << result.relative_residual
            << ',' << result.status
            << ',' << (correct ? 1 : 0)
            << ',' << reordering_ms
            << ',' << rhs_permutation_ms
            << ',' << device.setup_ms
            << ',' << solve_plus_setup
            << ',' << end_to_end
            << ',' << kRtol
            << ',' << kAtol
            << ',' << kMaxIterations
            << '\n';

        out.flush();

        std::printf(
            "  repeat %d: %.3f ms, "
            "%d iterations, rel=%.3e, %s%s\n",
            repeat,
            result.solve_ms,
            result.iterations,
            result.relative_residual,
            result.status.c_str(),
            correct ? " [OK]" : " [FAIL]");
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::fprintf(
            stderr,
            "Usage: %s INPUT.mtx OUTPUT.csv\n",
            argv[0]);

        return 1;
    }

    try {
        CUDA_CHECK(cudaFree(0));

        const std::string input = argv[1];
        const std::string output = argv[2];

        const std::string matrix_name =
            std::filesystem::path(input).stem().string();

        const CSR original =
            convert_COO_CSR(read_mtx(input));

        if (original.numRows <= 0 ||
            original.numRows != original.numCols) {
            throw std::runtime_error(
                "PCG requires a nonempty square matrix");
        }

        const std::vector<double> rhs(
            static_cast<std::size_t>(original.numRows),
            1.0);

        const auto rcm_start =
            std::chrono::steady_clock::now();

        const auto permutation =
            compute_rcm(original);

        const CSR reordered =
            apply_rcm(original, permutation);

        const double rcm_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - rcm_start).count();

        const auto rhs_start =
            std::chrono::steady_clock::now();

        const std::vector<double> reordered_rhs =
            permute_vector(rhs, permutation);

        const double rhs_permutation_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - rhs_start).count();

        std::ofstream out(output);

        if (!out) {
            throw std::runtime_error(
                "Could not open output CSV: " + output);
        }

        write_header(out);

        std::printf(
            "%s: bandwidth %d -> %d; RCM %.3f ms\n",
            matrix_name.c_str(),
            matrix_bandwidth(original),
            matrix_bandwidth(reordered),
            rcm_ms);

        run_ordering(
            matrix_name,
            "original",
            original,
            rhs,
            0.0,
            0.0,
            out);

        run_ordering(
            matrix_name,
            "reordered",
            reordered,
            reordered_rhs,
            rcm_ms,
            rhs_permutation_ms,
            out);

        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}