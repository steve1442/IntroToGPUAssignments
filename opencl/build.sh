#!/usr/bin/env bash
set -e
echo " Building"
gcc -O2 -o main main.c -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lOpenCL
echo "Build complete. Executable: ./main"