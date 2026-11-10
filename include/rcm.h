
#ifndef RCM_H
#define RCM_H

#include "read_mtx.h"
#include <vector>

// Compute Reverse Cuthill-McKee ordering.
//
// permutation[new_index] = old_index
//
// Uses the undirected graph of the matrix's
// sparsity pattern.
std::vector<int> compute_rcm(const CSR& matrix);

// Apply a symmetric row/column permutation.
//
// Result B satisfies B = P * A * P^T.
//
// permutation[new_index] = old_index
CSR apply_rcm(
    const CSR& matrix,
    const std::vector<int>& permutation
);

// Reorder a vector into RCM ordering.
//
// result[i] = original[permutation[i]]
std::vector<double> permute_vector(
    const std::vector<double>& original,
    const std::vector<int>& permutation
);

// Convert a vector from RCM ordering back
// into the original ordering.
std::vector<double> unpermute_vector(
    const std::vector<double>& reordered,
    const std::vector<int>& permutation
);

// Maximum distance of a nonzero entry
// from the main diagonal.
int matrix_bandwidth(const CSR& matrix);

#endif
