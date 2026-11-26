#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
    printf 'Usage: %s [OUTPUT_DIR]\n' "$0" >&2
    exit 1
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

bin="./build/run_cpu_benchmarks"
matrix_dir="matrices/NIST_matrices"
result_dir="${1:-results/cpu/nist}"

mkdir -p "$result_dir"

for matrix_path in "$matrix_dir"/*.mtx; do
    matrix="$(basename "$matrix_path" .mtx)"

    printf '\nBenchmarking %s\n' "$matrix"

    if "$bin" "$matrix_path" "$result_dir/$matrix.csv"; then
	printf 'PASS: %s\n' "$matrix"
    else
	printf 'FAILED OR DID NOT CONVERGE: %s\n' "$matrix"
    fi

    if [[ "${PLOT_RESULTS:-1}" == 1 ]]; then
        python3 plot_results.py "$result_dir/$matrix.csv" \
            --output-dir "$result_dir/${matrix}_plots"
    fi
done

printf '\nAll benchmarks complete. Results are in %s/\n' "$result_dir"
