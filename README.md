# fem-gpu-solver
Parallel FEM solver with sparse matrix reordering and GPU acceleration for studying CPU/GPU performance.

## Current Status

A working end-to-end CPU/GPU solver pipeline is implemented. The project currently includes:

- Matrix Market input and CSR conversion
- Serial and OpenMP SpMV
- CPU Conjugate Gradient and Jacobi-PCG
- Reverse Cuthill-McKee reordering
- Naive and warp-per-row CUDA SpMV
- cuSPARSE comparison
- Full GPU-resident CG and Jacobi-PCG
- GPU kernel fusion, reduced synchronization, and CUDA Graph optimizations
- CPU and GPU benchmark drivers
- NIST and generated sparse-matrix test suites

The current final benchmark pass is comparing CPU and GPU Jacobi-PCG using a relative tolerance of `1e-8` and a maximum of 50,000 iterations.

## Project Goal

The main goal is to determine whether sparse-matrix reordering and GPU acceleration can improve the performance of iterative sparse solves representative of FEM workloads.

The project investigates:

- CPU versus GPU solver performance
- whether RCM improves sparse-matrix locality and solve time
- how different GPU SpMV implementations compare
- how GPU implementation optimizations affect full-solver runtime
- how Jacobi preconditioning affects convergence and solve time

## CPU Solver

The CPU implementation supports serial and OpenMP SpMV inside CG and PCG.

Several OpenMP thread counts and scheduling policies were benchmarked. The best CPU configuration depends on the matrix, and dynamic scheduling was generally more expensive than static scheduling for these workloads.

## Reverse Cuthill-McKee

RCM was implemented to reduce matrix bandwidth and potentially improve memory locality.

Results so far show that its benefit is matrix-dependent: it can significantly reduce bandwidth on some matrices but increase it on others, and lower bandwidth does not always produce a faster solve.

## GPU SpMV

Three GPU SpMV approaches were evaluated:

1. naive thread-per-row CUDA
2. custom warp-per-row CUDA
3. cuSPARSE

The warp-per-row implementation generally outperformed the naive kernel and was selected for the custom full GPU solver.

## Full GPU Solver

The initial GPU CG solver was limited by per-iteration synchronization and kernel-launch overhead.

The implementation was progressively optimized using device-side updates, kernel fusion, fewer CPU/GPU synchronization points, warp-level reductions, reduced memory traffic, and CUDA Graphs.

## Jacobi-PCG

Jacobi preconditioning has been implemented on both CPU and GPU while keeping the same convergence criterion as plain CG.

It has substantially reduced the iteration count on some matrices. For example, `bcsstk17` dropped from roughly 25,000 plain-CG iterations to roughly 3,000 Jacobi-PCG iterations on the GPU.

Jacobi does not solve every difficult convergence case, so the final benchmark sweep is also being used to identify matrices where stronger preconditioning would be required.

## Building and Running

The project is intended to be run on Lonestar6. CPU benchmarks require GCC/OpenMP, while GPU benchmarks require CUDA and an NVIDIA A100.

### CPU Solver

Load GCC:

```bash
module load gcc
```

Build the CPU benchmark driver:

```bash
mkdir -p build

g++ -O3 -std=c++20 -fopenmp -Iinclude \
    run_cpu_benchmarks.cpp \
    read_mtx.cpp \
    src/spmv.cpp \
    src/rcm.cpp \
    -o build/run_cpu_benchmarks
```

Run the CPU solver on one Matrix Market file:

```bash
./build/run_cpu_benchmarks INPUT.mtx OUTPUT.csv
```

For example:

```bash
./build/run_cpu_benchmarks \
    matrices/NIST_matrices/bcsstk17.mtx \
    results/bcsstk17_cpu.csv
```

The CPU benchmark compares the serial solver with several OpenMP configurations and tests both the original and RCM-reordered matrix.

### GPU Solver

Load GCC and CUDA:

```bash
module load gcc cuda
```

Build the full GPU CG/PCG solver:

```bash
make -f Makefile.gpu_cg clean
make -f Makefile.gpu_cg
```

Run it on one matrix:

```bash
./gpu_cg INPUT.mtx OUTPUT.csv
```

For example:

```bash
./gpu_cg \
    matrices/NIST_matrices/bcsstk17.mtx \
    results/bcsstk17_gpu.csv
```

The GPU solver runs the custom warp-based sparse solver on both the original and RCM-reordered matrix.

### Standalone GPU SpMV

The standalone SpMV benchmark is separate from the full solver and is used to compare the CUDA SpMV implementations.

Build it with:

```bash
make -f Makefile.spmv
```

The available implementations include the naive CUDA kernel, the custom warp-per-row kernel, and cuSPARSE.

### Batch Benchmarking

The repository also contains benchmark scripts for running multiple matrices:

```bash
./run_benchmarks.sh
./run_gpu_cg_benchmarks.sh
```

SLURM scripts under `scripts/slurm/` can be used for longer CPU and GPU benchmark sweeps on Lonestar6.

## Final Benchmark Plan

The final benchmark pass uses:

- 20 NIST matrices
- 4 generated Laplacian scaling matrices
- CPU Jacobi-PCG
- GPU Jacobi-PCG
- original and RCM-reordered cases where applicable

Final PCG results will be added after the current benchmark jobs complete.

## Remaining Work

Remaining work includes:

- reviewing and validating the existing implementation
- checking for additional useful GPU optimizations
- analyzing the final CPU/GPU benchmark results
- comparing plain CG against Jacobi-PCG
- producing final plots and tables
- completing the report and presentation