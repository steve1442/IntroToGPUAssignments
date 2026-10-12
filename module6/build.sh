#!/bin/bash
set -e

nvcc -O2 assignment.cu -o assignment
if [ $? -eq 0 ]; then
    echo "Build successful."
else
    echo "Build failed."
    exit 1
fi