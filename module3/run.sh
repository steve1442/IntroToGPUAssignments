#!/bin/bash
set -e
echo "Running Assignment"
if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh [blocks] [threads_per_block]"
    exit 1
fi
./assignment.exe "$@"
echo "Complete"