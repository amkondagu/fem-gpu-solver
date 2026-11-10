
#include "rcm.h"

#include <algorithm>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

// Validate the CSR representation before accessing it.
void validate_csr(const CSR& A) {
    if (A.numRows < 0 || A.numCols < 0) {
        throw std::invalid_argument("Negative matrix dimensions");
    }

    if (A.rowIdx.size() !=
        static_cast<size_t>(A.numRows) + 1) {
        throw std::invalid_argument("Invalid CSR rowIdx size");
    }

    if (A.colIdx.size() != A.vals.size()) {
        throw std::invalid_argument("CSR array sizes differ");
    }

    if (A.rowIdx[0] != 0 ||
        A.rowIdx.back() != static_cast<int>(A.vals.size())) {
        throw std::invalid_argument("Invalid CSR row offsets");
    }

    for (int i = 0; i < A.numRows; ++i) {
        if (A.rowIdx[i] < 0 ||
            A.rowIdx[i] > A.rowIdx[i + 1]) {
            throw std::invalid_argument("Invalid CSR row ordering");
        }
    }

    for (int col : A.colIdx) {
        if (col < 0 || col >= A.numCols) {
            throw std::invalid_argument("Invalid CSR column index");
        }
    }
}

// Validate a permutation and construct old_to_new.
std::vector<int> inverse_permutation(
    const std::vector<int>& permutation,
    int n
) {
    if (static_cast<int>(permutation.size()) != n) {
        throw std::invalid_argument("Wrong permutation size");
    }

    std::vector<int> inverse(n, -1);

    for (int newIndex = 0; newIndex < n; ++newIndex) {
        int oldIndex = permutation[newIndex];

        if (oldIndex < 0 || oldIndex >= n ||
            inverse[oldIndex] != -1) {
            throw std::invalid_argument("Invalid permutation");
        }

        inverse[oldIndex] = newIndex;
    }

    return inverse;
}

} // namespace

// -----------------------------------------------------
// Compute Reverse Cuthill-McKee ordering
// -----------------------------------------------------

std::vector<int> compute_rcm(const CSR& A) {
    validate_csr(A);

    if (A.numRows != A.numCols) {
        throw std::invalid_argument(
            "RCM requires a square matrix"
        );
    }

    const int n = A.numRows;

    // Step 1: Construct an undirected graph.
    // A nonzero at (i,j) creates an edge i <-> j.
    // Ignore diagonal entries.

    std::vector<std::vector<int>> adj(n);

    for (int i = 0; i < n; ++i) {
        for (int k = A.rowIdx[i];
             k < A.rowIdx[i + 1]; ++k) {

            int j = A.colIdx[k];

            if (i == j) {
                continue;
            }

            adj[i].push_back(j);
            adj[j].push_back(i);
        }
    }

    // Step 2: Remove duplicate edges and
    // calculate each vertex's degree.

    std::vector<int> degree(n);

    for (int i = 0; i < n; ++i) {
        auto& neighbors = adj[i];

        std::sort(neighbors.begin(), neighbors.end());

        neighbors.erase(
            std::unique(
                neighbors.begin(),
                neighbors.end()
            ),
            neighbors.end()
        );

        degree[i] = static_cast<int>(neighbors.size());
    }

    // Step 3: BFS from minimum-degree vertices.
    // Process each connected component.

    std::vector<bool> visited(n, false);
    std::vector<int> order;
    order.reserve(n);

    while (static_cast<int>(order.size()) < n) {

        int start = -1;

        // Find the unvisited vertex with
        // the smallest degree.
        for (int i = 0; i < n; ++i) {
            if (!visited[i] &&
                (start == -1 ||
                 degree[i] < degree[start])) {
                start = i;
            }
        }

        std::queue<int> q;
        q.push(start);
        visited[start] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            order.push_back(u);

            // Visit neighbors in increasing
            // degree order.
            std::vector<int> neighbors = adj[u];

            std::sort(
                neighbors.begin(),
                neighbors.end(),
                [&](int a, int b) {
                    if (degree[a] != degree[b]) {
                        return degree[a] < degree[b];
                    }
                    return a < b;
                }
            );

            for (int v : neighbors) {
                if (!visited[v]) {
                    visited[v] = true;
                    q.push(v);
                }
            }
        }
    }

    // Step 4: Reverse the Cuthill-McKee order.
    std::reverse(order.begin(), order.end());

    return order;
}

// -----------------------------------------------------
// Apply RCM permutation to CSR matrix
// -----------------------------------------------------

CSR apply_rcm(
    const CSR& A,
    const std::vector<int>& permutation
) {
    validate_csr(A);

    if (A.numRows != A.numCols) {
        throw std::invalid_argument(
            "Matrix must be square"
        );
    }

    const int n = A.numRows;

    // old_to_new[old_index] = new_index
    std::vector<int> old_to_new =
        inverse_permutation(permutation, n);

    CSR B;
    B.numRows = n;
    B.numCols = n;

    B.rowIdx.resize(n + 1, 0);
    B.colIdx.reserve(A.colIdx.size());
    B.vals.reserve(A.vals.size());

    // Reorder rows and columns.
    for (int newRow = 0; newRow < n; ++newRow) {

        int oldRow = permutation[newRow];

        std::vector<std::pair<int, double>> entries;

        for (int k = A.rowIdx[oldRow];
             k < A.rowIdx[oldRow + 1]; ++k) {

            int oldCol = A.colIdx[k];
            int newCol = old_to_new[oldCol];

            entries.emplace_back(newCol, A.vals[k]);
        }

        // Keep CSR columns sorted within each row.
        std::sort(
            entries.begin(),
            entries.end(),
            [](const auto& a, const auto& b) {
                return a.first < b.first;
            }
        );

        for (const auto& entry : entries) {
            B.colIdx.push_back(entry.first);
            B.vals.push_back(entry.second);
        }

        B.rowIdx[newRow + 1] =
            static_cast<int>(B.vals.size());
    }

    return B;
}

// -----------------------------------------------------
// Permute a vector into RCM ordering
// -----------------------------------------------------

std::vector<double> permute_vector(
    const std::vector<double>& original,
    const std::vector<int>& permutation
) {
    const int n = static_cast<int>(original.size());
    inverse_permutation(permutation, n);

    std::vector<double> result(n);

    for (int i = 0; i < n; ++i) {
        result[i] = original[permutation[i]];
    }

    return result;
}

// -----------------------------------------------------
// Convert a vector back to original ordering
// -----------------------------------------------------

std::vector<double> unpermute_vector(
    const std::vector<double>& reordered,
    const std::vector<int>& permutation
) {
    const int n = static_cast<int>(reordered.size());
    inverse_permutation(permutation, n);

    std::vector<double> result(n);

    for (int i = 0; i < n; ++i) {
        result[permutation[i]] = reordered[i];
    }

    return result;
}

// -----------------------------------------------------
// Calculate matrix bandwidth
// -----------------------------------------------------

int matrix_bandwidth(const CSR& A) {
    validate_csr(A);

    int bandwidth = 0;

    for (int i = 0; i < A.numRows; ++i) {
        for (int k = A.rowIdx[i];
             k < A.rowIdx[i + 1]; ++k) {

            if (A.vals[k] == 0.0) {
                continue;
            }

            int distance = i - A.colIdx[k];

            if (distance < 0) {
                distance = -distance;
            }

            bandwidth = std::max(bandwidth, distance);
        }
    }

    return bandwidth;
}
