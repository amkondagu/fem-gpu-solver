#!/bin/bash
#SBATCH -J gpu_pcg_final
#SBATCH -o gpu_pcg_final.%j.out
#SBATCH -e gpu_pcg_final.%j.err
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc cuda

echo "Node: $(hostname)"
echo "Started: $(date)"

mkdir -p results/gpu/pcg/final

make -f Makefile.gpu_cg clean
make -f Makefile.gpu_cg

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
    echo "GPU Jacobi-PCG: $matrix"
    echo "Started: $(date)"
    echo "========================================"

    if ./gpu_cg \
        "matrices/NIST_matrices/$matrix.mtx" \
        "results/gpu/pcg/final/$matrix.csv"
    then
        echo "PASS: $matrix"
    else
        echo "FAILED OR DID NOT CONVERGE: $matrix"
    fi

    echo "Finished: $(date)"
done

echo
echo "========================================"
echo "GPU Laplacian scaling benchmarks"
echo "========================================"

for matrix in \
    lap2d_128x128 \
    lap2d_256x256 \
    lap2d_512x512 \
    lap2d_768x768
do
    echo
    echo "========================================"
    echo "GPU Jacobi-PCG: $matrix"
    echo "Started: $(date)"
    echo "========================================"

    if ./gpu_cg \
        "matrices/generated_symmetric_matrices/$matrix.mtx" \
        "results/gpu/pcg/final/$matrix.csv"
    then
        echo "PASS: $matrix"
    else
        echo "FAILED OR DID NOT CONVERGE: $matrix"
    fi

    echo "Finished: $(date)"
done

echo
echo "GPU final Jacobi-PCG benchmark complete: $(date)"