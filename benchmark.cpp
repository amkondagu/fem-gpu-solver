// Include the solver directly to keep this deliverable to three code files.
// Compile benchmark.cpp once; do not also link a separate solver.cpp object.
#include "solver.cpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace fem {

struct BackendCase {
    std::string name;
    std::string schedule = "none";
    int threads = 1;
    Spmv multiply;
};

struct BenchmarkOptions {
    int repeats = 3;
    int warmups = 1;
    CgOptions solver;
};

namespace {
using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string csv_quote(const std::string& text) {
    std::string result = "\"";
    for (char c : text) { result += c; if (c == '"') result += '"'; }
    return result + '"';
}
}  // namespace

// Matrices, serial/OpenMP kernels and the reordered matrix come from teammates.
// Each callback must already use the thread count/schedule supplied in its label.
// new_to_old[i] identifies the original index stored at reordered index i.
// reordering_ms is the caller's measured generation + CSR application cost;
// -1 means unknown. This function does not read or reorder a CSR matrix.
bool run_benchmark(const std::string& matrix_name, const CSR& original,
                   const Vector& rhs, const Spmv& serial_multiply,
                   const std::vector<BackendCase>& backends,
                   const std::string& csv_path,
                   const BenchmarkOptions& options = {},
                   const CSR* reordered = nullptr,
                   const std::vector<int>& new_to_old = {},
                   double reordering_ms = -1.0) {
    if (!serial_multiply || options.repeats < 1 || options.warmups < 0 ||
        !std::isfinite(reordering_ms) || (reordering_ms < 0 && reordering_ms != -1.0) ||
        original.numRows != original.numCols ||
        rhs.size() != static_cast<std::size_t>(original.numRows))
        throw std::invalid_argument("Invalid benchmark input, callback, repeats or warmups");
    std::vector<BackendCase> cases{{"serial", "none", 1, serial_multiply}};
    for (const auto& backend : backends) {
        if (!backend.multiply || backend.name.empty() || backend.name == "serial" || backend.threads < 1)
            throw std::invalid_argument("Additional backend needs a non-serial name, callback and positive threads");
        cases.push_back(backend);
    }

    Vector permuted_rhs;
    double rhs_permutation_ms = 0;
    if (reordered) {
        if (reordered->numRows != original.numRows || reordered->numCols != original.numCols ||
            new_to_old.size() != rhs.size())
            throw std::invalid_argument("Reordered matrix/permutation dimensions do not match");
        std::vector<bool> seen(rhs.size());
        for (int index : new_to_old) {
            if (index < 0 || index >= original.numRows || seen[index])
                throw std::invalid_argument("new_to_old must be a zero-based permutation");
            seen[index] = true;
        }
        const auto start = Clock::now();
        permuted_rhs.resize(rhs.size());
        for (std::size_t i = 0; i < rhs.size(); ++i) permuted_rhs[i] = rhs[new_to_old[i]];
        rhs_permutation_ms = milliseconds(start);
    } else if (!new_to_old.empty() || reordering_ms != -1.0) {
        throw std::invalid_argument("Permutation/cost requires a reordered matrix");
    }

    std::ofstream csv(csv_path);
    if (!csv) throw std::runtime_error("Cannot create benchmark CSV: " + csv_path);
    csv << std::setprecision(17)
        << "matrix,n,nnz,ordering,backend,schedule,threads,repeat,solve_ms,iterations,"
           "relative_residual,status,correct,reordering_ms,rhs_permutation_ms,rtol,atol,max_iterations\n";
    const double rhs_norm = vector_norm(rhs);
    const double threshold = std::max(options.solver.atol, options.solver.rtol * rhs_norm);
    auto solver_options = options.solver;
    solver_options.record_history = false;
    bool all_correct = true;

    for (int ordering = 0; ordering < (reordered ? 2 : 1); ++ordering) {
        const CSR& matrix = ordering ? *reordered : original;
        const Vector& b = ordering ? permuted_rhs : rhs;
        for (const auto& backend : cases) {
            for (int warmup = 0; warmup < options.warmups; ++warmup) {
                auto result = conjugate_gradient(matrix, b, backend.multiply, solver_options, serial_multiply);
                all_correct = all_correct && result.status == "converged";
            }
            for (int repeat = 0; repeat < options.repeats; ++repeat) {
                const auto start = Clock::now();
                const auto result = conjugate_gradient(matrix, b, backend.multiply, solver_options, serial_multiply);
                const double solve_ms = milliseconds(start);

                // Restore the solution, then check against the ORIGINAL system.
                Vector restored = result.x;
                if (ordering)
                    for (std::size_t i = 0; i < rhs.size(); ++i) restored[new_to_old[i]] = result.x[i];
                const bool finite = std::all_of(restored.begin(), restored.end(),
                                                [](double value) { return std::isfinite(value); });
                double residual = std::numeric_limits<double>::infinity();
                if (finite) {
                    try { residual = true_residual_norm(original, rhs, restored, serial_multiply); }
                    catch (const std::runtime_error&) { /* Preserve a failed verification row. */ }
                }
                const bool correct = result.status == "converged" && std::isfinite(residual) && residual <= threshold;
                all_correct = all_correct && correct;
                csv << csv_quote(matrix_name) << ',' << original.numRows << ',' << original.vals.size() << ','
                    << (ordering ? "reordered" : "original") << ',' << csv_quote(backend.name) << ','
                    << csv_quote(backend.schedule) << ',' << backend.threads << ',' << repeat << ','
                    << solve_ms << ',' << result.iterations << ',' << (rhs_norm ? residual / rhs_norm : residual)
                    << ',' << result.status << ',' << correct << ',';
                if (!ordering) csv << 0;
                else if (reordering_ms >= 0) csv << reordering_ms;
                csv << ',' << (ordering ? rhs_permutation_ms : 0) << ',' << solver_options.rtol << ','
                    << solver_options.atol << ',' << solver_options.max_iterations << '\n';
            }
        }
    }
    csv.flush();
    if (!csv) throw std::runtime_error("Failed writing benchmark CSV");
    return all_correct;
}

}  // namespace fem

#ifdef FEM_DEMO
// A fixed two-by-two teaching example, not a general CSR SpMV implementation.
int main(int argc, char** argv) {
    try {
        const CSR matrix{2, 2, {0, 2, 4}, {0, 1, 0, 1}, {4, 1, 1, 3}};
        const fem::Spmv multiply = [](const CSR&, const fem::Vector& x, fem::Vector& y) {
            y = {4 * x[0] + x[1], x[0] + 3 * x[1]};
        };
        const std::string output = argc > 1 ? argv[1] : "/tmp/fem_demo.csv";
        const bool correct = fem::run_benchmark("two_by_two", matrix, {1, 2}, multiply, {}, output);
        const auto solution = fem::conjugate_gradient(matrix, {1, 2}, multiply);
        std::cout << "x = [" << solution.x[0] << ", " << solution.x[1] << "]  iterations = "
                  << solution.iterations << "  relative residual = " << solution.relative_residual << '\n';
        std::cout << (correct ? "Checks passed; wrote " : "Checks failed; wrote ") << output << '\n';
        return correct ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
