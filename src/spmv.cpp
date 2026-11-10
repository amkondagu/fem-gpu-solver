#include "spmv.h"

#include <omp.h>
#include <stdexcept>

namespace {

void prepare_output(const CSR& matrix, const std::vector<double>& x,
                    std::vector<double>& y) {
    if (x.size() != static_cast<std::size_t>(matrix.numCols)) {
        throw std::invalid_argument("SpMV input vector size does not match matrix");
    }
    if (&x == &y) {
        throw std::invalid_argument("SpMV input and output must be distinct vectors");
    }
    y.resize(matrix.numRows);
}

double multiply_row(const CSR& matrix, const std::vector<double>& x, int row) {
    double sum = 0.0;
    for (int k = matrix.rowIdx[row]; k < matrix.rowIdx[row + 1]; ++k) {
        sum += matrix.vals[k] * x[matrix.colIdx[k]];
    }
    return sum;
}

omp_sched_t omp_schedule(SpmvSchedule schedule) {
    switch (schedule) {
    case SpmvSchedule::Static:
        return omp_sched_static;
    case SpmvSchedule::Dynamic:
        return omp_sched_dynamic;
    case SpmvSchedule::Guided:
        return omp_sched_guided;
    }
    throw std::invalid_argument("Unknown SpMV schedule");
}

}  // namespace

void spmv_serial(const CSR& matrix, const std::vector<double>& x,
                 std::vector<double>& y) {
    prepare_output(matrix, x, y);
    for (int row = 0; row < matrix.numRows; ++row) {
        y[row] = multiply_row(matrix, x, row);
    }
}

void spmv_openmp(const CSR& matrix, const std::vector<double>& x,
                 std::vector<double>& y, int threads, SpmvSchedule schedule) {
    if (threads <= 0) {
        throw std::invalid_argument("SpMV thread count must be positive");
    }
    const auto kind = omp_schedule(schedule);
    prepare_output(matrix, x, y);

    omp_sched_t previous_kind;
    int previous_chunk;
    omp_get_schedule(&previous_kind, &previous_chunk);
    omp_set_schedule(kind, 0);
    #pragma omp parallel for num_threads(threads) schedule(runtime)
    for (int row = 0; row < matrix.numRows; ++row) {
        y[row] = multiply_row(matrix, x, row);
    }
    omp_set_schedule(previous_kind, previous_chunk);
}
