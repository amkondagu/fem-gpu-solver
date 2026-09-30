# Project Proposal — ECE 379K

**Team name:**  Squids

**Members (name + EID):** Nitin Vunnam (ncv383), Aneesh Kondagunturi (amk5537), Emma Wang (ew24223), Aashrith Attelli (aa85938)  

**Date:** 9/21/2026

---

## 1. What we are building

We are building a program that makes large physics simulations run faster by reorganizing their data and then dividing the work between two parts of a computer: the CPU, which handles general-purpose tasks, and the GPU, which is designed to perform many similar calculations at the same time.

Large engineering simulations often divide a physical object into many small pieces and calculate how those pieces affect one another. As the simulation becomes more detailed, solving the resulting equations can take a long time. Some of that delay comes from related data being stored far apart in memory, which makes it slower for the processor to access.

Our project will test whether reorganizing the data before solving the problem improves overall performance. By December, we will have a program that solves a collection of real simulation problems on both the CPU and GPU and measures when the extra time spent reorganizing the data is worth the speedup it provides later.

We will verify that all versions produce essentially the same numerical result. For our largest practical test cases, our goal is for the GPU-accelerated version to run at least **2× faster than our multithreaded CPU baseline**. We will test at least **25 problems** and report how many repeated solves are needed, if any, before the one-time cost of reorganizing the data pays for itself.

---

## 2. Who is doing what

| Team member | Owns |
|---|---|
| **Aashrith Attelli (aa85938)** | Builds and tests the Matrix Market reader and parallel input conversion, then implements and profiles the CUDA sparse matrix-vector multiplication kernels. |
| **Nitin Vunnam (ncv383)** | Builds the serial and OpenMP CPU baselines and scheduling study, then implements the GPU dot-product and vector-update operations used by the solver. |
| **Aneesh Kondagunturi (amk5537)** | Builds and evaluates the Reverse Cuthill-McKee reordering, then integrates the GPU-resident iterative solver and measures when reordering becomes worthwhile. |
| **Emma Wang (ew24223)** | Builds the timing and correctness framework, manages the benchmark collection, and evaluates alternate GPU sparse formats and cuSPARSE. |

**What we will do together:**  
We will agree on shared input/output conventions, perform the first integration, review each other's code, analyze results, and prepare the final report and presentation.

**Dependencies, and what the waiting person does meanwhile:**  
We will place several test matrices in the repository during the first week so GPU development can begin immediately. The benchmarking framework can use the CPU baseline before the GPU version is ready, and the solver can use reference operations while custom kernels are still being completed.

**Milestones:**

| By when | What exists | Who |
|---|---|---|
| **Sept. 29** | CPU baseline, input conversion, initial reordering, benchmark harness, and shared test matrices. | All |
| **Oct. 10** | First end-to-end CPU pipeline and initial scaling measurements. | Nitin, Aashrith, Aneesh |
| **Oct. 30** | Core CUDA operations working and benchmarked. | Aashrith, Nitin |
| **Nov. 20** | GPU solver integrated and first CPU/GPU break-even results. | Emma, Aneesh |
| **Dec. 1** | Final benchmark sweep and analysis complete. | All |

---

## 3. What could go wrong

| What could go wrong | How we would notice early | What we would do instead |
|---|---|---|
| Reordering does not improve performance enough to recover its cost. | Measure reordering and solve time on several matrices by Oct. 15. | Report the measured break-even behavior, including cases where reordering does not pay off. |
| The GPU provides little speedup because problems are too small or data-transfer time dominates. | Benchmark several problem sizes as soon as the first GPU operation works and measure transfer and computation time separately. | Focus on larger problems, keep data on the GPU across solver iterations, and report when GPU execution becomes worthwhile. |
| A team member falls behind or leaves the project. | Review milestone progress weekly; two consecutive missed goals trigger a re-cut of ownership. | Redistribute the core work and remove optional format variants and comparison experiments. |

**Reduced scope:**  
If we lose significant time, we will keep the core pipeline: data loading, reordering, a multithreaded CPU baseline, GPU sparse matrix-vector multiplication, and a GPU iterative solver. We will reduce the benchmark set to 10 matrices and remove alternate sparse formats and the cuSPARSE comparison.

---

## 4. What we will need

| What | Status | Needed by |
|---|---|---|
| Personal/ECE Linux systems for development and debugging | **have** | Now |
| Access to one Lonestar6 CPU node and one NVIDIA A100 GPU node through the course TACC allocation | **need from you** | Oct. 1 |
| C++ compiler, OpenMP, ThreadSanitizer, CUDA, cuSPARSE, and GPU profiling tools | **can get** | Oct. 1 |
| 25–40 matrices from the public SuiteSparse Matrix Collection | **can get** | Sept. 29 |
| GitHub repository for shared code and tests | **have** | Now |