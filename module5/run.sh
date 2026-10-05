#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <total_threads> <block_size>"
    exit 1
fi

TOTAL_THREADS=$1
BLOCK_SIZE=$2

echo "RUNNING WITH Total Threads: $TOTAL_THREADS, Block Size: $BLOCK_SIZE"
./assignment "$TOTAL_THREADS" "$BLOCK_SIZE"