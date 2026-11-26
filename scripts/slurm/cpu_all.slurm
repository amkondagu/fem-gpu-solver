#!/bin/bash
#SBATCH -J cpu_pcg_final
#SBATCH -o cpu_pcg_final.%j.out
#SBATCH -e cpu_pcg_final.%j.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00

cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc

echo "Node: $(hostname)"
echo "Started: $(date)"

mkdir -p build
mkdir -p results/cpu/pcg/final

g++ -O3 -std=c++20 -fopenmp -Iinclude \
    run_cpu_benchmarks.cpp \
    read_mtx.cpp \
    src/spmv.cpp \
    src/rcm.cpp \
    -o build/run_cpu_benchmarks

for matrix in \
    bcsstk11 \
    bcsstk12 \
    bcsstk13 \
    bcsstk14 \
    bcsstk15 \
    bcsstk16 \
    bcsstk17 \
    bcsstk18 \
    bcsstm27 \
    nos3 \
    nos7 \
    s1rmq4m1 \
    s1rmt3m1 \
    s2rmq4m1 \
    s2rmt3m1 \
    s3dkq4m2 \
    s3dkt3m2 \
    s3rmq4m1 \
    s3rmt3m1 \
    s3rmt3m3
do
    echo
    echo "========================================"
    echo "CPU Jacobi-PCG: $matrix"
    echo "Started: $(date)"
    echo "========================================"

    if ./build/run_cpu_benchmarks \
        "matrices/NIST_matrices/$matrix.mtx" \
        "results/cpu/pcg/final/$matrix.csv"
    then
        echo "PASS: $matrix"
    else
        echo "FAILED OR DID NOT CONVERGE: $matrix"
    fi

    echo "Finished: $(date)"
done

echo
echo "========================================"
echo "CPU generated Laplacians"
echo "========================================"

for matrix in \
    lap2d_128x128 \
    lap2d_256x256 \
    lap2d_512x512 \
    lap2d_768x786
do
    echo
    echo "CPU Jacobi-PCG: $matrix"
    echo "Started: $(date)"

    if ./build/run_cpu_benchmarks \
        "matrices/generated_symmetric_matrices/$matrix.mtx" \
        "results/cpu/pcg/final/$matrix.csv"
    then
        echo "PASS: $matrix"
    else
        echo "FAILED OR DID NOT CONVERGE: $matrix"
    fi

    echo "Finished: $(date)"
done

echo
echo "CPU final Jacobi-PCG benchmark complete: $(date)"