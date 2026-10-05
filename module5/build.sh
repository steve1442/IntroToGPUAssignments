#!/bin/bash

nvcc -o assignment assignment.cu -O2 -arch=sm_60 -std=c++14

if [ $? -eq 0 ]; then
    echo "Build successful."
else
    echo "Build failed."
    exit 1
fi