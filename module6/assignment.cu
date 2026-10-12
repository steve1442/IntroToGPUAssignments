#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define NUM_STREAMS 2
#define NUM_RUNS 3
#define PI_F 3.14

// CUDA kernel element-wise math
__global__ void compute_kernel(const float* __restrict__ A,
const float* __restrict__ B,
float* __restrict__ C,
int n) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
float a = A[idx];
float b = B[idx];
// Compute a calculation 
C[idx] = sqrtf(abs(a - b)) * tanf(a)* sinf(a) * cosf(b) * tanf(a*b);
    }
}

// Launch kernel stream measure time
static void run_andevice_time_kernel(cudaStream_t stream,
const float* device_A, const float* device_B, float* device_C,
int n, int gridevice_size, int blockSize,
float* out_time_ms) {
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
compute_kernel<<<gridevice_size, blockSize, 0, stream>>>
(device_A, device_B, device_C, n);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
cudaEventElapsedTime(out_time_ms, start, stop);

cudaEventDestroy(start);
cudaEventDestroy(stop);
}

// Main config parsing allocation time
int main(int argc, char** argv) {
    int totalThreads = (1 << 20);
    int blockSize = 256;

    if (argc >= 2) {
        totalThreads = atoi(argv[1]);
    }
    if (argc >= 3) {
        blockSize = atoi(argv[2]);
    }

    // Ensure grid covers all threads 
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;
    if (totalThreads % blockSize != 0) {
        printf(
        "Warning: Thread count %d not evenly divisible by block size %d.\n", 
        totalThreads, blockSize);
        printf("Padding to %d threads (%d blocks).\n\n",
             numBlocks * blockSize, numBlocks);
        totalThreads = numBlocks * blockSize;
    }

    const int N = totalThreads;
    const int half = (N + 1) / 2;
    const int gridevice_size = (half + blockSize - 1) / blockSize;

    printf(" Config: Threads=%d | BlockSize=%d | Runs=%d\n\n", 
        N, blockSize, NUM_RUNS);

    // Host memory allocation initialization step 
    float *host_A = (float*)malloc(N * sizeof(float));
    float *host_B = (float*)malloc(N * sizeof(float));
    float *host_C = (float*)malloc(N * sizeof(float));
    if (!host_A || !host_B || !host_C) { 
        fprintf(stderr, "failed\n"); return 1; 
    }

    for (int i = 0; i < N; ++i) {
        host_A[i] = (float)i * 0.01f;
        host_B[i] = (float)i * 0.02f + PI_F;
    }

    // Allocate device memory buffers now 
    float *device_A, *device_B, *device_C;
    cudaMalloc(&device_A, N * sizeof(float));
    cudaMalloc(&device_B, N * sizeof(float));
    cudaMalloc(&device_C, N * sizeof(float));
    cudaMemcpy(device_A, host_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA streams for work 
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) cudaStreamCreate(&streams[i]);

    // Allocate timing arrays for averages
    float seq_times[NUM_RUNS];
    float conv_times[NUM_STREAMS][NUM_RUNS];

    printf(" Running %d iterations\n\n", NUM_RUNS);

    // Loop runs mitigate GPU variance 
    for (int r = 0; r < NUM_RUNS; ++r) {
        printf(" RUN %d/%d\n", r + 1, NUM_RUNS);

        // Execute first sequential baseline run 
        cudaStream_t seqStream;
        cudaStreamCreate(&seqStream);
        float seq_time = 0.0f;
        run_andevice_time_kernel(
            seqStream, device_A, device_B, device_C, N, 
            gridevice_size, blockSize, &seq_time);
        seq_times[r] = seq_time;
        cudaStreamDestroy(seqStream);
        //  ignore the first one because it includes the load time
        if (r != 0) printf("Sequential: %f ms\n", seq_time);

        // Run concurrent execution streams events
        float max_conv_time = 0.0f;
        for (int s = 0; s < NUM_STREAMS; ++s) {
            float stream_time = 0.0f;
            run_andevice_time_kernel(streams[s], 
                device_A + (s * half),
                 device_B + (s * half), device_C + (s * half), 
                half, gridevice_size, blockSize, &stream_time);
                conv_times[s][r] = stream_time;
                // ignore the first one because it includes the load time
                if (r != 0) printf("  Stream %d: %f ms\n", s, stream_time);
                if (stream_time > max_conv_time) max_conv_time = stream_time;
            }
        if (r != 0) printf("  Concurrent (max stream): %f ms\n\n",
             max_conv_time);
    }

    // Calculate averages for accurate comparison
    float avg_seq = 0.0f, avg_conv = 0.0f;
    for (int r = 1; r < NUM_RUNS; ++r) {
        avg_seq += seq_times[r];
        avg_conv += 
        (conv_times[0][r] > conv_times[1][r]) 
        ? conv_times[0][r] : conv_times[1][r];
    }
    avg_seq /= NUM_RUNS - 1;
    avg_conv /= NUM_RUNS - 1;

    float speedup = (avg_conv > 0.0f) ? (avg_seq / avg_conv) : 0.0f;

    printf(" Final Metrics for Config (Threads=%d, BlockSize=%d):\n"
        , N, blockSize);
    printf("   Avg Sequential Time: %f ms\n", avg_seq);
    printf("   Avg Concurrent Time: %f ms\n", avg_conv);
    printf("   Seq/Conc:  %fx\n\n", speedup);

    for (int i = 0; i < NUM_STREAMS; ++i) 
        cudaStreamDestroy(streams[i]);
    cudaFree(device_A); 
    cudaFree(device_B); 
    cudaFree(device_C);
    free(host_A);
    free(host_B); 
    free(host_C);

    printf("FINISHED\n");
    return 0;
}
