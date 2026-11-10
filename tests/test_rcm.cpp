
#include "rcm.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

// Read an individual matrix entry.
// Sum duplicates if they exist.
double get_entry(const CSR& A, int row, int col) {
    double value = 0.0;

    for (int k = A.rowIdx[row];
         k < A.rowIdx[row + 1]; ++k) {
        if (A.colIdx[k] == col) {
            value += A.vals[k];
        }
    }

    return value;
}

void check_permutation(
    const std::vector<int>& permutation,
    int n
) {
    assert(static_cast<int>(permutation.size()) == n);

    auto sorted = permutation;
    std::sort(sorted.begin(), sorted.end());

    for (int i = 0; i < n; ++i) {
        assert(sorted[i] == i);
    }
}

void check_reordered_matrix(
    const CSR& A,
    const CSR& B,
    const std::vector<int>& permutation
) {
    assert(A.numRows == B.numRows);
    assert(A.numCols == B.numCols);
    assert(A.vals.size() == B.vals.size());

    const int n = A.numRows;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {

            double expected = get_entry(
                A, permutation[i], permutation[j]
            );

            double actual = get_entry(B, i, j);

            assert(std::abs(expected - actual) < 1e-10);
        }
    }
}

void test_tridiagonal() {
    CSR A;

    A.numRows = 4;
    A.numCols = 4;

    A.rowIdx = {0, 2, 5, 8, 10};
    A.colIdx = {
        0, 1,
        0, 1, 2,
        1, 2, 3,
        2, 3
    };
    A.vals = {
        2, -1,
        -1, 2, -1,
        -1, 2, -1,
        -1, 2
    };

    auto permutation = compute_rcm(A);

    check_permutation(permutation, 4);

    CSR B = apply_rcm(A, permutation);

    check_reordered_matrix(A, B, permutation);

    assert(matrix_bandwidth(A) == 1);
    assert(matrix_bandwidth(B) == 1);

    std::cout << "Tridiagonal test passed\n";
}

void test_disconnected() {
    // Two disconnected pairs of vertices.
    CSR A;

    A.numRows = 4;
    A.numCols = 4;

    A.rowIdx = {0, 1, 2, 3, 4};
    A.colIdx = {1, 0, 3, 2};
    A.vals = {1, 1, 1, 1};

    auto permutation = compute_rcm(A);

    check_permutation(permutation, 4);

    CSR B = apply_rcm(A, permutation);

    check_reordered_matrix(A, B, permutation);

    std::cout << "Disconnected graph test passed\n";
}

void test_isolated_vertex() {
    CSR A;

    A.numRows = 3;
    A.numCols = 3;

    A.rowIdx = {0, 1, 2, 2};
    A.colIdx = {1, 0};
    A.vals = {1, 1};

    auto permutation = compute_rcm(A);

    check_permutation(permutation, 3);

    CSR B = apply_rcm(A, permutation);

    check_reordered_matrix(A, B, permutation);

    std::cout << "Isolated vertex test passed\n";
}

void test_nonsymmetric() {
    CSR A;

    A.numRows = 3;
    A.numCols = 3;

    A.rowIdx = {0, 1, 3, 4};
    A.colIdx = {1, 0, 2, 1};
    A.vals = {2, 3, 4, 5};

    auto permutation = compute_rcm(A);

    check_permutation(permutation, 3);

    CSR B = apply_rcm(A, permutation);

    check_reordered_matrix(A, B, permutation);

    std::cout << "Nonsymmetric test passed\n";
}

void test_vectors() {
    std::vector<int> permutation = {2, 0, 1};
    std::vector<double> original = {10, 20, 30};

    auto reordered = permute_vector(
        original, permutation
    );

    assert((reordered ==
        std::vector<double>{30, 10, 20}));

    auto restored = unpermute_vector(
        reordered, permutation
    );

    assert(restored == original);

    std::cout << "Vector permutation test passed\n";
}

void test_invalid_permutation() {
    CSR A;

    A.numRows = 2;
    A.numCols = 2;
    A.rowIdx = {0, 1, 2};
    A.colIdx = {0, 1};
    A.vals = {1, 1};

    bool threw = false;

    try {
        apply_rcm(A, {0, 0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);

    std::cout << "Invalid permutation test passed\n";
}

int main() {
    test_tridiagonal();
    test_disconnected();
    test_isolated_vertex();
    test_nonsymmetric();
    test_vectors();
    test_invalid_permutation();

    std::cout << "\nAll RCM tests passed!\n";

    return 0;
}
