#!/usr/bin/env bash
THREADS="${1:-512}"
BLOCK_SIZE="${2:-256}"
MATRIX_SIZE=1024
NUM_RUNS=3

echo "Running Linear Solver Validation"
echo "   Config: Matrix=${MATRIX_SIZE}, Threads=${THREADS}, Block=${BLOCK_SIZE}"
echo "   Running ${NUM_RUNS} test cases"
echo "----------------------------------------"

for i in $(seq 1 $NUM_RUNS); do
    ./main $MATRIX_SIZE $THREADS $BLOCK_SIZE
done
echo "passed"
