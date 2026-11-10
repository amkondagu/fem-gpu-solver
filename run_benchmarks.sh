#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
    printf 'Usage: %s [OUTPUT_DIR]\n' "$0" >&2
    exit 1
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

bin="./build/run_cpu_benchmarks"
matrix_dir="matrices/generated_symmetric_matrices"
result_dir="${1:-results}"

mkdir -p "$result_dir"

matrices=(
    "lap2d_128x128"
    "lap2d7pt_512x512"
    "elas2d_n64_lambda10p0"
)

for matrix in "${matrices[@]}"; do
    printf '\nBenchmarking %s\n' "$matrix"
    "$bin" "$matrix_dir/$matrix.mtx" "$result_dir/$matrix.csv"
    if [[ "${PLOT_RESULTS:-1}" == 1 ]]; then
        python3 plot_results.py "$result_dir/$matrix.csv" \
            --output-dir "$result_dir/${matrix}_plots"
    fi
done

printf '\nAll benchmarks complete. Results are in %s/\n' "$result_dir"
