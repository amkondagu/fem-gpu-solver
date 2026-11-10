// Included by benchmark.cpp to keep the deliverable to three source files.
// The caller supplies a validated SPD matrix and the team's SpMV operation.
#ifndef FEM_SOLVER_CPP_INCLUDED
#define FEM_SOLVER_CPP_INCLUDED

#include "read_mtx.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fem {
using Vector = std::vector<double>;
using Spmv = std::function<void(const CSR&, const Vector&, Vector&)>;

struct CgOptions {
    int max_iterations = 1000;
    double rtol = 1e-8;
    double atol = 0.0;
    bool record_history = false;
};

struct CgResult {
    Vector x;
    int iterations = 0;
    double residual_norm = 0.0;
    double relative_residual = 0.0;
    std::string status;
    std::vector<double> residual_history;
};

namespace detail {
bool finite_vector(const Vector& values) {
    return std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); });
}

long double norm(const Vector& values) {
    // Scale before squaring so large/small finite vectors do not overflow/underflow.
    long double scale = 0, sum = 1;
    for (double value : values) {
        const long double magnitude = std::abs(static_cast<long double>(value));
        if (!std::isfinite(magnitude)) return magnitude;
        if (magnitude == 0) continue;
        if (magnitude > scale) {
            const long double ratio = scale / magnitude;
            sum = 1 + sum * ratio * ratio;
            scale = magnitude;
        } else {
            const long double ratio = magnitude / scale;
            sum += ratio * ratio;
        }
    }
    return scale == 0 ? 0 : scale * std::sqrt(sum);
}

long double normalized_dot(const Vector& a, const Vector& b,
                           long double a_norm, long double b_norm) {
    long double sum = 0;
    for (std::size_t i = 0; i < a.size(); ++i) sum += (a[i] / a_norm) * (b[i] / b_norm);
    return sum;
}

void check_vectors(const CSR& a, const Vector& b, const Vector& x) {
    if (a.numRows < 0 || a.numCols < 0 || b.size() != static_cast<std::size_t>(a.numRows) ||
        x.size() != static_cast<std::size_t>(a.numCols) || !finite_vector(b) || !finite_vector(x))
        throw std::invalid_argument("Invalid vector dimensions or nonfinite vector values");
}

bool residual(const CSR& a, const Vector& b, const Vector& x, const Spmv& multiply, Vector& r) {
    if (!finite_vector(x)) return false;
    Vector ax;
    multiply(a, x, ax);
    if (ax.size() != b.size() || !finite_vector(ax)) return false;
    r.resize(b.size());
    for (std::size_t i = 0; i < b.size(); ++i)
        r[i] = static_cast<double>(static_cast<long double>(b[i]) - ax[i]);
    return finite_vector(r);
}
}  // namespace detail

double vector_norm(const Vector& values) { return static_cast<double>(detail::norm(values)); }

double true_residual_norm(const CSR& a, const Vector& b, const Vector& x, const Spmv& multiply) {
    detail::check_vectors(a, b, x);
    if (!multiply) throw std::invalid_argument("A supplied SpMV operation is required");
    Vector r;
    if (!detail::residual(a, b, x, multiply, r))
        throw std::runtime_error("SpMV returned an invalid vector while checking the residual");
    return vector_norm(r);
}

CgResult conjugate_gradient(const CSR& a, const Vector& b, const Spmv& multiply,
                            const CgOptions& options = {}, const Spmv& verify_multiply = {},
                            const Vector& initial_guess = {}) {
    if (a.numRows != a.numCols || a.numRows < 0 || !multiply || options.max_iterations < 0 ||
        !std::isfinite(options.rtol) || options.rtol < 0 ||
        !std::isfinite(options.atol) || options.atol < 0)
        throw std::invalid_argument("CG needs a square SPD matrix, SpMV callback and valid tolerances");
    CgResult result;
    result.x = initial_guess.empty() ? Vector(static_cast<std::size_t>(a.numCols), 0) : initial_guess;
    detail::check_vectors(a, b, result.x);
    const Spmv& verify = verify_multiply ? verify_multiply : multiply;
    const long double b_norm = detail::norm(b);
    const long double tolerance = std::max(static_cast<long double>(options.atol), options.rtol * b_norm);

    // Every exit recomputes b-A*x through the supplied verification operation.
    auto finish = [&](const char* status) {
        Vector actual;
        const long double actual_norm = detail::residual(a, b, result.x, verify, actual)
            ? detail::norm(actual) : std::numeric_limits<long double>::infinity();
        result.residual_norm = static_cast<double>(actual_norm);
        result.relative_residual = static_cast<double>(b_norm > 0 ? actual_norm / b_norm : actual_norm);
        result.status = std::isfinite(actual_norm) && actual_norm <= tolerance ? "converged" :
            (std::string(status) == "converged" ? "numerical_failure" : status);
        if (options.record_history) {
            if (result.residual_history.empty()) result.residual_history.push_back(result.residual_norm);
            else result.residual_history.back() = result.residual_norm;
        }
        return std::move(result);
    };

    // Start with r=b-A*x and p=r; x defaults to zero.
    Vector r;
    if (!detail::residual(a, b, result.x, verify, r)) return finish("numerical_failure");
    long double r_norm = detail::norm(r);
    if (options.record_history) result.residual_history.push_back(static_cast<double>(r_norm));
    if (!std::isfinite(r_norm)) return finish("numerical_failure");
    if (r_norm <= tolerance) return finish("converged");
    Vector p = r, ap;

    for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
        multiply(a, p, ap);
        if (ap.size() != p.size() || !detail::finite_vector(ap)) return finish("numerical_failure");
        const long double p_norm = detail::norm(p), ap_norm = detail::norm(ap);
        if (!std::isfinite(p_norm) || !std::isfinite(ap_norm)) return finish("numerical_failure");
        const long double curvature = p_norm > 0 && ap_norm > 0
            ? detail::normalized_dot(p, ap, p_norm, ap_norm) : 0;
        if (!std::isfinite(curvature)) return finish("numerical_failure");
        if (curvature <= 0) return finish("nonpositive_curvature");

        // Equivalent to alpha=(r dot r)/(p dot A*p), without squaring large norms.
        const long double alpha = (r_norm / p_norm) * (r_norm / ap_norm) / curvature;
        if (!std::isfinite(alpha)) return finish("numerical_failure");
        for (std::size_t i = 0; i < r.size(); ++i) {
            result.x[i] = static_cast<double>(result.x[i] + alpha * p[i]);
            r[i] = static_cast<double>(r[i] - alpha * ap[i]);
        }
        result.iterations = iteration + 1;
        const long double previous_norm = r_norm;
        r_norm = detail::norm(r);
        if (options.record_history) result.residual_history.push_back(static_cast<double>(r_norm));
        if (!detail::finite_vector(result.x) || !std::isfinite(r_norm)) return finish("numerical_failure");

        if (r_norm <= tolerance) {
            Vector actual;
            if (!detail::residual(a, b, result.x, verify, actual)) return finish("numerical_failure");
            const long double actual_norm = detail::norm(actual);
            if (!std::isfinite(actual_norm)) return finish("numerical_failure");
            if (actual_norm <= tolerance) return finish("converged");
            // Roundoff drift: restart from the true residual within the same budget.
            r = std::move(actual);
            p = r;
            r_norm = actual_norm;
            if (options.record_history) result.residual_history.back() = static_cast<double>(r_norm);
            continue;
        }

        // beta=(r_new dot r_new)/(r_old dot r_old); update the search direction.
        const long double ratio = r_norm / previous_norm, beta = ratio * ratio;
        if (!std::isfinite(beta)) return finish("numerical_failure");
        for (std::size_t i = 0; i < p.size(); ++i) p[i] = static_cast<double>(r[i] + beta * p[i]);
        if (!detail::finite_vector(p)) return finish("numerical_failure");
    }
    return finish("max_iterations");
}
}  // namespace fem
#endif
