#!/bin/bash
set -u

MATRIX_DIR="${1:-matrices/NIST_matrices}"
OUTPUT_DIR="${2:-results/gpu}"
KERNEL="${KERNEL:-naive}"

mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY="$OUTPUT_DIR/spmv_${KERNEL}_${TIMESTAMP}.csv"
LOG="$OUTPUT_DIR/spmv_${KERNEL}_${TIMESTAMP}.log"

echo "matrix,kernel,status,time_ms,gflops" > "$SUMMARY"
: > "$LOG"

count=0
passed=0
failed=0

while IFS= read -r matrix; do
    count=$((count + 1))
    name="${matrix#${MATRIX_DIR}/}"

    echo "==================================================" | tee -a "$LOG"
    echo "[$count] $name" | tee -a "$LOG"

    output=$(./spmv_gpu "$KERNEL" "$matrix" 2>&1)
    status=$?

    echo "$output" | tee -a "$LOG"

    if [ "$status" -eq 0 ]; then
        passed=$((passed + 1))

        # Expected final line:
        # naive        0.123456 ms      1.234 GFLOP/s ... OK
        time_ms=$(echo "$output" | awk '/ OK$/ {print $2}' | tail -n 1)
        gflops=$(echo "$output" | awk '/ OK$/ {print $4}' | tail -n 1)

        echo "\"$name\",\"$KERNEL\",PASS,\"$time_ms\",\"$gflops\"" >> "$SUMMARY"
    else
        failed=$((failed + 1))
        echo "\"$name\",\"$KERNEL\",FAIL,,\"" >> "$SUMMARY"
        echo "FAILED: $name" | tee -a "$LOG"
    fi

done < <(find "$MATRIX_DIR" -type f -name "*.mtx" | sort)

echo
echo "Finished."
echo "Total:  $count"
echo "Passed: $passed"
echo "Failed: $failed"
echo "CSV:    $SUMMARY"
echo "Log:    $LOG"

# Return a failure status if any matrix failed, while still allowing
# the script to test every matrix before exiting.
if [ "$failed" -ne 0 ]; then
    exit 1
fi
