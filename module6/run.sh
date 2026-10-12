#!/bin/bash
set -e
if [ $# -lt 2 ]; then
    echo "Usage: ./run.sh <total_threads> <block_size>"
    exit 1
fi
echo "Running with: Threads=$1, BlockSize=$2"
./assignment "$1" "$2"
