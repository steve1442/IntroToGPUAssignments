#include <cuda_runtime.h>

// Parallel reduction kernel to compute L2 norm of residual vector
__global__ void residual_norm_kernel(const float* d_residual,
                                     float* d_norm,
                                     int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (idx < n) {
        sum = d_residual[idx] * d_residual[idx];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *d_norm += sdata[0];
    }
}
