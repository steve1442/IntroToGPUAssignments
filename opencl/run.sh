#!/usr/bin/env bash
set -e
echo "Running OpenCL Validation"
echo "   Config: N=${1}, Local=${2}, Parts=${3}"
./main "${1}" "${2}" "${3}"
echo "passed"