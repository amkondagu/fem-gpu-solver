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