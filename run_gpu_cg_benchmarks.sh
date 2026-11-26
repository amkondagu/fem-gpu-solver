#!/bin/bash
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

mkdir -p results/gpu/cg

matrices=(
  bcsstk11
  bcsstk12
  bcsstk14
  bcsstk15
  bcsstk16
  bcsstk17
  nos3
  s1rmq4m1
  s1rmt3m1
  s2rmq4m1
  s2rmt3m1
)

summary="results/gpu/cg/all_gpu_cg_results.txt"
: > "$summary"

failures=0

for m in "${matrices[@]}"; do
    matrix="matrices/NIST_matrices/${m}.mtx"
    csv="results/gpu/cg/${m}.csv"
    log="results/gpu/cg/${m}.log"

    echo "===== ${m} =====" | tee -a "$summary"

    if ./gpu_cg "$matrix" "$csv" 2>&1 | tee "$log"; then
        cat "$csv" >> "$summary"
        echo "PASS: ${m}" | tee -a "$summary"
    else
        echo "FAILED: ${m}" | tee -a "$summary"
        failures=$((failures + 1))
    fi

    echo | tee -a "$summary"
done

echo "Completed GPU CG sweep with ${failures} failed matrix runs."
exit "$failures"
