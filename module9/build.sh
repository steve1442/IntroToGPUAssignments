#!/usr/bin/env bash
set -e
echo " Building"
nvcc -O3 -o main main.cu kernel.cu \
     -lcublas -lcurand -lcudart -lcusolver
echo "Build complete. Executable: ./main"
