#ifndef SPMV_H
#define SPMV_H

#include "read_mtx.h"

enum class SpmvSchedule {
    Static,
    Dynamic,
    Guided
};

// Multiply a valid CSR matrix by x. Input and output must be distinct vectors.
void spmv_serial(const CSR& matrix, const std::vector<double>& x,
                 std::vector<double>& y);

void spmv_openmp(const CSR& matrix, const std::vector<double>& x,
                 std::vector<double>& y, int threads, SpmvSchedule schedule);

#endif
