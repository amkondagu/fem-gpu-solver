// benchmark.cpp already includes solver.cpp. Compile both through this file.
#include "benchmark.cpp"
#include "rcm.h"
#include "spmv.h"

#include <filesystem>
#include <omp.h>

namespace {

fem::BackendCase openmp_case(const char* label, int threads, SpmvSchedule schedule) {
    return {"openmp", label, threads,
            [threads, schedule](const CSR& matrix, const fem::Vector& x, fem::Vector& y) {
                spmv_openmp(matrix, x, y, threads, schedule);
            }};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " INPUT.mtx OUTPUT.csv\n";
        return 1;
    }

    try {
        const CSR original = convert_COO_CSR(read_mtx(argv[1]));
        const auto start = std::chrono::steady_clock::now();
        const auto permutation = compute_rcm(original);
        const CSR reordered = apply_rcm(original, permutation);
        const double rcm_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();

        const fem::Vector rhs(original.numRows, 1.0);
        const std::vector<fem::BackendCase> cases = {
            openmp_case("static", 4, SpmvSchedule::Static),
            openmp_case("static", 16, SpmvSchedule::Static),
            openmp_case("static", 32, SpmvSchedule::Static),
            openmp_case("dynamic", 16, SpmvSchedule::Dynamic),
            openmp_case("guided", 16, SpmvSchedule::Guided)
        };
        fem::BenchmarkOptions options;
        options.repeats = 5;
        options.warmups = 1;
        omp_set_dynamic(0);

        const auto matrix_name = std::filesystem::path(argv[1]).stem().string();
        std::cout << matrix_name << " (" << original.numRows << " rows, "
                  << original.vals.size() << " nonzeros)\n"
                  << "RCM bandwidth: " << matrix_bandwidth(original) << " -> "
                  << matrix_bandwidth(reordered) << "; preprocessing: " << rcm_ms
                  << " ms\n" << std::flush;
        const bool correct = fem::run_benchmark(matrix_name, original, rhs, spmv_serial,
                                                cases, argv[2], options, &reordered,
                                                permutation, rcm_ms);
        std::cout << (correct ? "All checks passed; wrote " : "Some checks failed; wrote ")
                  << argv[2] << '\n';
        return correct ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
