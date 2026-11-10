# Walk through your part of the FEM project

Your additions are **solver.cpp**, **benchmark.cpp**, **plot_results.py**, and this
guide. The other repository files belong to the existing project. No extra
reader, general SpMV kernel, RCM implementation, build system, logs or result
bundles are included in this version.

## 1. What each file does

| File | Your responsibility |
| --- | --- |
| `solver.cpp` | Solve `A*x=b` with Conjugate Gradient and check convergence |
| `benchmark.cpp` | Run supplied serial/OpenMP operations, time solves, compare original/reordered systems, write CSV |
| `plot_results.py` | Read that CSV and plot runtime, serial speedup and reordering effects |

The flow is:

```text
Team supplies matrix + SpMV operations
                ↓
benchmark.cpp calls solver.cpp
                ↓
CG produces x and a convergence status
                ↓
benchmark checks the original system and writes CSV
                ↓
plot_results.py makes comparison charts
```

Emma supplies the validated matrix/CSR input; Aneesh supplies serial/OpenMP SpMV;
Nitin supplies the reordered matrix and permutation. Those operations are inputs
to your code. This follows the responsibility screenshot, whose assignments
differ from the older proposal.

## 2. Run the tiny teaching example first

From Terminal:

```sh
cd "$HOME/Downloads/FEM Solver Project/fem-gpu-solver"
c++ -std=c++17 -O2 -DFEM_DEMO benchmark.cpp -o /tmp/fem_demo
/tmp/fem_demo /tmp/fem_demo.csv
python3 plot_results.py /tmp/fem_demo.csv --output-dir /tmp/fem_plots
```

Plotting needs Matplotlib in the Python environment you use; if needed, install it
there with `python3 -m pip install matplotlib`. C++ execution does not need Python.
These commands keep the executable, CSV and plots in `/tmp`, outside the project.

The demo solves:

```text
A = [4 1]    b = [1]
    [1 3]        [2]

x = [1/11] ≈ [0.0909091]
    [7/11]   [0.636364 ]
```

You should see convergence in two iterations and a very small relative residual.
The demo's multiply callback is a fixed two-by-two formula under `#ifdef
FEM_DEMO`; it is a teaching fixture, not Aneesh's general CSR SpMV implementation.
It benchmarks only this serial example. Real matrices and OpenMP configurations
enter through the integration function described below.

**Check:** Multiply the reported `x` by `A`. Do you recover `b`?

## 3. Understand the shared matrix and the callback

Open the existing `read_mtx.h`. The shared `CSR` has `numRows`, `numCols`,
`rowIdx`, `colIdx` and `vals`. For the example:

```text
rowIdx = [0, 2, 4]
colIdx = [0, 1, 0, 1]
vals   = [4, 1, 1, 3]
```

The solver does not walk those arrays to implement SpMV. Instead, `Spmv` is a
function taking `(matrix, input_vector, output_vector)` and overwriting the
output with `A*input`. A callback lets the solver use a serial or OpenMP function
without containing either implementation.

The team must supply a structurally valid, symmetric positive-definite (SPD)
matrix. The solver checks dimensions, vectors and options, but does not implement
Emma's CSR validator or prove symmetry/positive definiteness.

**Check:** What changes in CG when the multiply callback changes from serial to
OpenMP? Which mathematical operations should remain the same?

## 4. Read the CG loop in solver.cpp

Start with `CgOptions`, `CgResult`, then `conjugate_gradient`. Ignore the norm
helper on your first pass. `x` is the current guess; `r=b-A*x` is the residual;
`p` is the search direction. With a zero initial guess, `r=p=b`.

Each textbook iteration is:

```text
Ap    = A*p
alpha = (r dot r) / (p dot Ap)
x     = x + alpha*p
r_new = r - alpha*Ap
beta  = (r_new dot r_new) / (r dot r)
p     = r_new + beta*p
```

For the first example step:

```text
r = p = [1, 2]
Ap = [6, 7]
alpha = 5/20 = 1/4
x = [1/4, 1/2]
r_new = [-1/2, 1/4]
beta = 1/16
```

The second step reaches `[1/11,7/11]` in exact arithmetic. The implementation
forms equivalent ratios using scaled norms and normalized dot products to
reduce overflow/underflow. Those helpers support the same CG equations.

**Check:** Which callback call computes `Ap`, and which loop updates `x`?

## 5. Understand convergence and failure

Success means `||b-A*x|| <= max(atol, rtol*||b||)`. CG updates a recursive
residual during iteration, but recomputes `b-A*x` before claiming convergence.
For parallel solves, the benchmark supplies the team's serial operation as the
verification callback. If no separate verifier is passed, the solver reuses its
multiply callback.

If recursive and recomputed residuals disagree, CG restarts from the recomputed
residual within the same iteration budget. Status strings distinguish
`converged`, `max_iterations`, `nonpositive_curvature`, and `numerical_failure`.
Invalid dimensions/options throw an exception. Nonpositive curvature can reveal
a matrix unsuitable for CG, but its absence does not prove the matrix is SPD.

**Check:** Why is a small recursively updated residual insufficient by itself?

## 6. Read the benchmark and teammate integration

`benchmark.cpp` includes `solver.cpp` directly so your contribution needs no extra
header. Compile `benchmark.cpp` once; do not also link a separate `solver.cpp`
object. If a team driver includes `benchmark.cpp`, compile that driver rather
than also compiling either included file separately.

`run_benchmark` receives the original CSR matrix, RHS, serial callback, additional
`BackendCase` configurations, output CSV path, and `BenchmarkOptions`. It always
runs a serial baseline. Each additional case contains a name, schedule label,
thread count and multiply callback. The supplied callback must actually configure
the corresponding OpenMP schedule/thread count; the labels do not configure it.

Here is the integration shape inside the team's driver, once `A`, `b`,
`serial_callback` and `omp_static_4_callback` have been supplied by teammates:

```cpp
fem::BenchmarkOptions options;
options.repeats = 5;
options.warmups = 1;
std::vector<fem::BackendCase> cases{
    {"openmp", "static", 4, omp_static_4_callback}
};
bool correct = fem::run_benchmark(
    "matrix_name", A, b, serial_callback, cases, "/tmp/results.csv", options);
```

The names in that example represent supplied inputs; they are not missing
implementations to add to these three files. Add more configured callbacks to
study other schedules or thread counts. The existing reader prototype has its
own demo `main`; use the team's library-ready reader/driver when integrating.

Warmups are excluded, and every timed solve starts from zero. `solve_ms` covers
the CG call, including allocation and residual certification. Restoring a
reordered solution and its extra original-system check happen after the timer.
Only the supplied SpMV may be parallel; CG vector operations are serial.

**Check:** Does setting a case's thread label to 4 actually create four threads?

## 7. Compare a reordered system without implementing RCM

Pass the already reordered matrix as `&B`, the zero-based `new_to_old`
permutation, and the measured external generation-plus-CSR-application cost as
the last arguments to `run_benchmark`. Unknown cost is represented by `-1` and
written as an empty field. The benchmark additionally times RHS permutation.

The convention is `p[new_index]=old_index`:

```text
B = P*A*P^T
b_new[i] = b[p[i]]
x_old[p[i]] = x_new[i]
```

For `p=[1,0]`, `B=[[3,1],[1,4]]` and `b_new=[2,1]`. The reordered answer is
`[7/11,1/11]`; restoring its indices recovers the original answer. Your benchmark
validates permutation indices and checks the restored solution against the
original system. Nitin remains responsible for constructing and validating `B`.

**Check:** Why must the RHS and solution be mapped along with the matrix?

## 8. Read the CSV and plots

The CSV records matrix/configuration identity, repeated `solve_ms`, iterations,
relative residual, status, correctness, supplied reordering cost, measured RHS
permutation cost, and stopping settings. `run_benchmark` returns false if a
warmup or measured solve fails. Failed measured rows remain in the CSV.

`load_results` in `plot_results.py` keeps converged/correct rows. Runtime bars
show medians and observed min/max. Serial speedup uses the same matrix and
ordering; reordering speedup matches backend, schedule and thread count. Matrix
sizes/tolerances must match too. Keep different machines in separate CSVs.
The reordering plot compares solve time only; include preprocessing and any
per-solve vector mapping before claiming application-level payback.

For the tiny demo you will see runtime and serial-baseline speedup plots. A
reordering plot appears only when both original and reordered data are supplied.
These tiny timings illustrate the machinery, not meaningful performance.

Tomorrow's useful work is controlled CPU scaling/scheduling and actual team-RCM
comparisons, plus a separate CPU hotspot profile if needed to explain scaling.
There are no CUDA kernels here for NCU to profile. Generated CSVs/charts are
created only when you run the program; none are stored in this clean project.

**Check:** Why can SpMV get faster while the overall CG solve improves little?
