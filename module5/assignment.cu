#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__constant__ float constant_mem_scale;
__constant__ float constant_mem_offset;

// This code just tries different memory by scaling the values already in memory and adding to it 


// Kernel that demonstrates register, shared, global, and constant memory
__global__ void compute_kernel(const float *global_mem_array_A,
                               const float *global_mem_array_B,
                               float *global_mem_array_C,
                               int totalThreads)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory
    extern __shared__ float shared_mem_tile_base[];
    float *shared_mem_array_A = shared_mem_tile_base;
    float *shared_mem_array_B = shared_mem_tile_base + blockDim.x;

    // Register memory
    float register_temp_A = 0.0f;
    float register_temp_B = 0.0f;

    // Load data into shared memory
    if (idx < totalThreads)
    {
        register_temp_A = global_mem_array_A[idx];
        register_temp_B = global_mem_array_B[idx];
    }
    shared_mem_array_A[threadIdx.x] = register_temp_A;
    shared_mem_array_B[threadIdx.x] = register_temp_B;
    __syncthreads();

    // Compute using shared, constant, and register memory
    float register_result_val = 
    shared_mem_array_A[threadIdx.x] + shared_mem_array_B[threadIdx.x];
    register_result_val = 
    register_result_val * constant_mem_scale + constant_mem_offset;

    // Global memory
    if (idx < totalThreads)
    {
        global_mem_array_C[idx] = register_result_val;
    }
}

// Initialize host memory arrays
void init_host_mem(float *host_mem_array_A, float *host_mem_array_B,
                   float *host_mem_array_C, int element_count)
{
    for (int i = 0; i < element_count; ++i)
    {
        host_mem_array_A[i] = 9.0f;
        host_mem_array_B[i] = 1.0f;
        host_mem_array_C[i] = 0.0f;
    }
}

// Allocate device global memory
void alloc_device_mem(float **global_mem_array_A,
                      float **global_mem_array_B,
                      float **global_mem_array_C, size_t memory_size_bytes)
{
    cudaMalloc(global_mem_array_A, memory_size_bytes);
    cudaMalloc(global_mem_array_B, memory_size_bytes);
    cudaMalloc(global_mem_array_C, memory_size_bytes);
}

// Execute kernel and measure time
void time_kernel_exec(float *global_mem_array_A, float *global_mem_array_B,
                      float *global_mem_array_C, int totalThreads,
                      int blockSize)
{
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;
    size_t shared_mem_size_bytes = blockSize * sizeof(float) * 2;

    cudaEvent_t timer_start_event, timer_stop_event;
    cudaEventCreate(&timer_start_event);
    cudaEventCreate(&timer_stop_event);

    cudaEventRecord(timer_start_event);
    compute_kernel<<<numBlocks, blockSize, shared_mem_size_bytes>>>(
        global_mem_array_A, global_mem_array_B,
        global_mem_array_C, totalThreads);
    cudaDeviceSynchronize();

    cudaEventRecord(timer_stop_event);
    cudaEventSynchronize(timer_stop_event);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, timer_start_event, timer_stop_event);
    printf("Time: %f ms\n", elapsed_ms);

    cudaEventDestroy(timer_start_event);
    cudaEventDestroy(timer_stop_event);
}

// Verify results and free memory
void verify_and_free_mem(float *host_mem_array_A, float *host_mem_array_B,
                         float *host_mem_array_C,
                         float *global_mem_array_A, float *global_mem_array_B,
                         float *global_mem_array_C, int element_count)
{
    cudaMemcpy(host_mem_array_C, global_mem_array_C,
               element_count * sizeof(float), cudaMemcpyDeviceToHost);

    int verification_limit_count = (element_count < 5) ? element_count : 5;
    int verification_passed_flag = 1;
    for (int i = 0; i < verification_limit_count; ++i)
    {
        float expected_result_val = 
        (host_mem_array_A[i] + host_mem_array_B[i]) * 5.0f + 10.0f;
        if (fabsf(host_mem_array_C[i] - expected_result_val) > 1e-5)
        {
            verification_passed_flag = 0;
            break;
        }
    }
    printf("Did the copys all calculate propery?: %s\n", 
        verification_passed_flag ? "PASS" : "FAIL");

    cudaFree(global_mem_array_A);
    cudaFree(global_mem_array_B);
    cudaFree(global_mem_array_C);
    free(host_mem_array_A);
    free(host_mem_array_B);
    free(host_mem_array_C);
}

int main(int argc, char **argv)
{
    int totalThreads = 1024;
    int blockSize = 256;

    if (argc >= 2)
    {
        totalThreads = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        blockSize = atoi(argv[2]);
    }

    if (totalThreads < 64)
    {
        totalThreads = 64;
    }

    if (blockSize > 1024)
    {
        blockSize = 1024;
    }

    int numBlocks = (totalThreads + blockSize - 1) / blockSize;
    printf("configuration: %d numBlocks, %d blocksize, %d threads, %d total\n",
         numBlocks, blockSize, totalThreads, totalThreads);

    size_t memory_size_bytes = totalThreads * sizeof(float);
    float* host_mem_array_A = (float*)malloc(memory_size_bytes);
    float* host_mem_array_B = (float*)malloc(memory_size_bytes);
    float* host_mem_array_C = (float*)malloc(memory_size_bytes);
    float* global_mem_array_A = nullptr;
    float* global_mem_array_B = nullptr;
    float* global_mem_array_C = nullptr;

    init_host_mem(host_mem_array_A, host_mem_array_B,
                  host_mem_array_C, totalThreads);
    alloc_device_mem(&global_mem_array_A, &global_mem_array_B,
                     &global_mem_array_C, memory_size_bytes);

    cudaMemcpy(global_mem_array_A, host_mem_array_A,
               memory_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(global_mem_array_B, host_mem_array_B,
               memory_size_bytes, cudaMemcpyHostToDevice);

    float host_constant_scale = 5.0f;
    float host_constant_offset = 10.0f;
    cudaMemcpyToSymbol(constant_mem_scale, &host_constant_scale, 
        sizeof(float));
    cudaMemcpyToSymbol(constant_mem_offset, &host_constant_offset, 
        sizeof(float));

    time_kernel_exec(global_mem_array_A, global_mem_array_B,
                     global_mem_array_C, totalThreads, blockSize);
    verify_and_free_mem(host_mem_array_A, host_mem_array_B,
                        host_mem_array_C, global_mem_array_A,
                        global_mem_array_B, global_mem_array_C,
                        totalThreads);
    return 0;
}
