/*
 * spmv_gpu.cu -- CUDA SpMV correctness/timing harness.
 *
 * Usage:
 *
 *     ./spmv_gpu naive matrix.mtx
 *     ./spmv_gpu naive matrix.mtx --once
 *
 * Timed mode follows the Lab 2 measurement structure:
 *
 *   1. compare one GPU launch against the existing CPU serial SpMV
 *   2. run 3 warm-up launches
 *   3. time launches individually with CUDA events
 *   4. report the median
 *
 * --once performs exactly one kernel launch and exits.  This is useful
 * later with compute-sanitizer or Nsight Compute.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "read_mtx.h"
#include "spmv.h"
#include "spmv_kernels.cuh"

#define CUDA_CHECK(x)                                                     \
    do {                                                                  \
        cudaError_t e_ = (x);                                             \
        if (e_ != cudaSuccess) {                                          \
            std::fprintf(stderr, "%s:%d: %s\n  -> %s\n", __FILE__,        \
                         __LINE__, #x, cudaGetErrorString(e_));            \
            std::exit(2);                                                 \
        }                                                                 \
    } while (0)

#define CUSPARSE_CHECK(x)                                                \
    do {                                                                 \
        cusparseStatus_t s_ = (x);                                       \
        if (s_ != CUSPARSE_STATUS_SUCCESS) {                             \
            std::fprintf(stderr, "%s:%d: cuSPARSE error in %s\n",        \
                         __FILE__, __LINE__, #x);                         \
            std::exit(2);                                                \
        }                                                                \
    } while (0)

namespace {

struct Device {
    std::string name;
    int major = 0;
    int minor = 0;
    int sms = 0;
    double gib = 0.0;
};

Device describe_device()
{
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));

    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

    Device d;
    d.name = p.name;
    d.major = p.major;
    d.minor = p.minor;
    d.sms = p.multiProcessorCount;
    d.gib = static_cast<double>(p.totalGlobalMem) / (1ull << 30);

    return d;
}

void print_header(const Device& d, const CSR& matrix)
{
    std::printf("# device %s  CC %d.%d  %d SMs  %.1f GiB\n",
                d.name.c_str(), d.major, d.minor, d.sms, d.gib);

    std::printf("# rows=%d cols=%d nnz=%zu\n",
                matrix.numRows,
                matrix.numCols,
                matrix.vals.size());

    if (matrix.numRows > 0) {
        const double avg =
            static_cast<double>(matrix.vals.size()) / matrix.numRows;

        std::printf("# average nnz/row %.2f\n", avg);
    }
}

// Device-resident copy of one CSR SpMV problem.
struct Problem {
    int numRows = 0;
    int numCols = 0;
    int nnz = 0;

    int* rowIdx = nullptr;
    int* colIdx = nullptr;
    double* vals = nullptr;
    double* x = nullptr;
    double* y = nullptr;

    std::vector<double> h_x;

    cusparseHandle_t cusparse = nullptr;
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr;
    cusparseDnVecDescr_t vecY = nullptr;

    void* spmvBuffer = nullptr;
    size_t spmvBufferSize = 0;

    explicit Problem(const CSR& matrix)
    {
        numRows = matrix.numRows;
        numCols = matrix.numCols;
        nnz = static_cast<int>(matrix.vals.size());

        h_x.resize(numCols);

        // Fixed seed so every run sees the same x.
        std::mt19937 rng(379);
        std::uniform_real_distribution<double> u(-1.0, 1.0);

        for (double& value : h_x) {
            value = u(rng);
        }

        CUDA_CHECK(cudaMalloc(&rowIdx,
            matrix.rowIdx.size() * sizeof(int)));

        CUDA_CHECK(cudaMalloc(&colIdx,
            matrix.colIdx.size() * sizeof(int)));

        CUDA_CHECK(cudaMalloc(&vals,
            matrix.vals.size() * sizeof(double)));

        CUDA_CHECK(cudaMalloc(&x,
            h_x.size() * sizeof(double)));

        CUDA_CHECK(cudaMalloc(&y,
            static_cast<std::size_t>(numRows) * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(
            rowIdx,
            matrix.rowIdx.data(),
            matrix.rowIdx.size() * sizeof(int),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(
            colIdx,
            matrix.colIdx.data(),
            matrix.colIdx.size() * sizeof(int),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(
            vals,
            matrix.vals.data(),
            matrix.vals.size() * sizeof(double),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(
            x,
            h_x.data(),
            h_x.size() * sizeof(double),
            cudaMemcpyHostToDevice));
        
        CUSPARSE_CHECK(cusparseCreate(&cusparse));

        CUSPARSE_CHECK(cusparseCreateCsr(
            &matA,
            numRows,
            numCols,
            nnz,
            rowIdx,
            colIdx,
            vals,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F
        ));

        CUSPARSE_CHECK(cusparseCreateDnVec(
            &vecX,
            numCols,
            x,
            CUDA_R_64F
        ));

        CUSPARSE_CHECK(cusparseCreateDnVec(
            &vecY,
            numRows,
            y,
            CUDA_R_64F
        ));

        double alpha = 1.0;
        double beta = 0.0;

        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            vecX,
            &beta,
            vecY,
            CUDA_R_64F,
            CUSPARSE_SPMV_CSR_ALG1,
            &spmvBufferSize
        ));

        CUDA_CHECK(cudaMalloc(&spmvBuffer, spmvBufferSize));
    }

    Problem(const Problem&) = delete;
    Problem& operator=(const Problem&) = delete;

    ~Problem()
    {
        if (spmvBuffer) {
            cudaFree(spmvBuffer);
        }

        if (vecY) {
            cusparseDestroyDnVec(vecY);
        }

        if (vecX) {
            cusparseDestroyDnVec(vecX);
        }

        if (matA) {
            cusparseDestroySpMat(matA);
        }

        if (cusparse) {
            cusparseDestroy(cusparse);
        }

        cudaFree(rowIdx);
        cudaFree(colIdx);
        cudaFree(vals);
        cudaFree(x);
        cudaFree(y);
    }

    std::size_t y_bytes() const
    {
        return static_cast<std::size_t>(numRows) * sizeof(double);
    }
};

void spmv_cusparse(Problem& p)
{
    double alpha = 1.0;
    double beta = 0.0;

    CUSPARSE_CHECK(cusparseSpMV(
        p.cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        p.matA,
        p.vecX,
        &beta,
        p.vecY,
        CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG1,
        p.spmvBuffer
    ));
}

using Launcher = std::function<void(Problem&)>;

struct Rung {
    std::string name;
    Launcher fn;
};

std::vector<Rung> implemented()
{
    return {
        {
            "naive",
            [](Problem& p) {
                spmv_naive(
                    p.numRows,
                    p.rowIdx,
                    p.colIdx,
                    p.vals,
                    p.x,
                    p.y
                );
            }
        },

        {
            "warp",
            [](Problem& p) {
                spmv_warp(
                    p.numRows,
                    p.rowIdx,
                    p.colIdx,
                    p.vals,
                    p.x,
                    p.y
                );
            }
        },

        {
            "cusparse",
            [](Problem& p) {
                spmv_cusparse(p);
            }
        }
    };
}

struct CheckResult {
    double worst_abs = 0.0;
    double worst_rel = 0.0;
    std::size_t at = 0;
    bool ok = true;
};

// Run one GPU launch and compare against the existing serial CPU SpMV.
CheckResult check_rung(const CSR& matrix, Problem& p, const Rung& r)
{
    std::vector<double> want;
    spmv_serial(matrix, p.h_x, want);

    r.fn(p);

    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) {
        e = cudaDeviceSynchronize();
    }

    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n",
                     r.name.c_str(),
                     cudaGetErrorString(e));
        std::exit(2);
    }

    std::vector<double> got(p.numRows);

    CUDA_CHECK(cudaMemcpy(
        got.data(),
        p.y,
        p.y_bytes(),
        cudaMemcpyDeviceToHost));

    CheckResult result;

    for (std::size_t i = 0; i < got.size(); i++) {
        const double abs_err = std::fabs(got[i] - want[i]);
        const double scale = std::max(1.0, std::fabs(want[i]));
        const double rel_err = abs_err / scale;

        if (abs_err > result.worst_abs) {
            result.worst_abs = abs_err;
            result.worst_rel = rel_err;
            result.at = i;
        }

        if (!std::isfinite(got[i]) || rel_err > 1e-10) {
            result.ok = false;
        }
    }

    if (!result.ok) {
        std::fprintf(stderr,
                     "%s: worst mismatch at y[%zu]: "
                     "abs_err=%.3e rel_err=%.3e\n",
                     r.name.c_str(),
                     result.at,
                     result.worst_abs,
                     result.worst_rel);
    }

    return result;
}

// Median milliseconds per kernel launch.
// This mirrors the Lab 2 timing policy.
double time_rung(Problem& p, const Rung& r)
{
    for (int w = 0; w < 3; w++) {
        r.fn(p);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> ms;
    double total = 0.0;

    while (ms.size() < 25 && (ms.size() < 5 || total < 1000.0)) {
        CUDA_CHECK(cudaEventRecord(start));

        r.fn(p);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float t = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));

        ms.push_back(t);
        total += t;
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(ms.begin(), ms.end());

    return ms[ms.size() / 2];
}

bool run_rung(const CSR& matrix,
              Problem& p,
              const Device& dev,
              const Rung& rung)
{
    print_header(dev, matrix);

    const CheckResult result = check_rung(matrix, p, rung);

    if (!result.ok) {
        std::printf("%-10s  FAIL  max_abs %.3e  max_rel %.3e\n",
                    rung.name.c_str(),
                    result.worst_abs,
                    result.worst_rel);

        return false;
    }

    const double ms = time_rung(p, rung);

    const double gflops =
        (2.0 * static_cast<double>(matrix.vals.size())) / ms * 1e-6;

    std::printf("%-10s %10.6f ms %10.3f GFLOP/s  "
                "max_abs %.3e  OK\n",
                rung.name.c_str(),
                ms,
                gflops,
                result.worst_abs);

    return true;
}

int usage()
{
    std::fprintf(stderr,
                 "usage: spmv_gpu RUNG matrix.mtx [--once]\n"
                 "RUNG: naive | warp | cusparse | all\n");

    return 1;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        return usage();
    }

    const std::string which = argv[1];
    const std::string matrix_file = argv[2];

    bool once = false;

    for (int a = 3; a < argc; a++) {
        if (std::strcmp(argv[a], "--once") == 0) {
            once = true;
        }
    }

    COO coo = read_mtx(matrix_file);
    CSR matrix = convert_COO_CSR(coo);

    if (matrix.numRows <= 0 || matrix.numCols <= 0) {
        std::fprintf(stderr, "matrix has invalid dimensions\n");
        return 1;
    }

    Problem p(matrix);
    const Device dev = describe_device();

    std::vector<Rung> rungs = implemented();

    if (which == "all") {
        bool ok = true;

        for (const Rung& r : rungs) {
            if (once) {
                r.fn(p);

                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            else {
                ok = run_rung(matrix, p, dev, r) && ok;
            }
        }

        return ok ? 0 : 1;
    }

    Rung rung;
    bool found = false;

    for (const Rung& r : rungs) {
        if (r.name == which) {
            rung = r;
            found = true;
            break;
        }
    }

    if (!found) {
        std::fprintf(stderr, "rung '%s' is not implemented\n",
                     which.c_str());
        return usage();
    }

    if (once) {
        rung.fn(p);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        return 0;
    }

    return run_rung(matrix, p, dev, rung) ? 0 : 1;
}