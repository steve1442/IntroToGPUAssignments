// Based on the work of Andrew Krepps
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cmath>

// The "algorithm" is just adding and subtracting. And then the branching is checking if its even


// GPU Kernel with no branching
__global__ void vectorAddNoBranch(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

// GPU Kernel with branching
__global__ void vectorAddBranch(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Differentiate based on whether its even or not
        if (i % 2 == 0) {
            c[i] = a[i] + b[i];
        } else {
            c[i] = a[i] - b[i];
        }
    }
}

// CPU no branching
void vectorAddHostNoBranch(const float* a, const float* b, float* c, int N) {
    for (int i = 0; i < N; ++i) c[i] = a[i] + b[i];
}

// CPU branching
void vectorAddHostBranch(const float* a, const float* b, float* c, int N) {
    for (int i = 0; i < N; ++i) {
        if (i % 2 == 0) c[i] = a[i] + b[i];
        else c[i] = a[i] - b[i];
    }
}

int main(int argc, char* argv[]) {

    int totalThreads = (1 << 20);
    int blockSize = 256;
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

    int numBlocks = totalThreads / blockSize;

    // validate command line arguments
    if (totalThreads % blockSize != 0) {
        ++numBlocks;
        totalThreads = numBlocks * blockSize;
        printf("Warning: Total thread count is not evenly divisible by the block size\n");
        printf("The total number of threads will be rounded up to %d\n", totalThreads);
    }

    int N = numBlocks * blockSize;

    // Forces data to be over 1 mil to meet requirement
    if (N < 1000000) {
        numBlocks = (1000000 / blockSize) + 1;
        N = numBlocks * blockSize;
        printf("WARNING: Adjusted it to %d blocks in order to meet the minimum dataset size (I use 1M elements)\n", numBlocks);
    }

    std::cout << "Configuration: " << numBlocks << " blocks, " << blockSize << " threads/block.\n";
    std::cout << "Data size (N): " << N << " elements.\n\n";

    // Host allocation & initialization
    std::vectortor<float> h_a(N), h_b(N), h_c_no(N), h_c_branch(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i) * 1.0f / N;
        h_b[i] = static_cast<float>(i) * 2.0f / N;
    }

    // Device allocation for inputs and everything
    float *d_a, *d_b, *d_c_no, *d_c_branch;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c_no, N * sizeof(float));
    cudaMalloc(&d_c_branch, N * sizeof(float));
    cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Do a number of warm up runs before doing some actual ones in order to avoid the preloading issues
    const int WARMUP = 10;
    const int MEASURED = 100;

    // GPU No Branch Timing
    for (int i = 0; i < WARMUP; ++i) vectorAddNoBranch<<<numBlocks, blockSize>>>(d_a, d_b, d_c_no, N);
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < MEASURED; ++i) vectorAddNoBranch<<<numBlocks, blockSize>>>(d_a, d_b, d_c_no, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_no = 0;
    cudaEventElapsedTime(&gpu_no, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    gpu_no /= MEASURED;

    // GPU With Branch Timing
    for (int i = 0; i < WARMUP; ++i) vectorAddBranch<<<numBlocks, blockSize>>>(d_a, d_b, d_c_branch, N);
    cudaDeviceSynchronize();

    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < MEASURED; ++i) vectorAddBranch<<<numBlocks, blockSize>>>(d_a, d_b, d_c_branch, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_br = 0;
    cudaEventElapsedTime(&gpu_br, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    gpu_br /= MEASURED;

    // CPU No Branch Timing
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < MEASURED; ++i) vectorAddHostNoBranch(h_a.data(), h_b.data(), h_c_no.data(), N);
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_no = std::chrono::duration<double, std::milli>(t2 - t1).count() / MEASURED;

    // CPU With Branch Timing
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < MEASURED; ++i) vectorAddHostBranch(h_a.data(), h_b.data(), h_c_branch.data(), N);
    auto t4 = std::chrono::high_resolution_clock::now();
    double cpu_br = std::chrono::duration<double, std::milli>(t4 - t3).count() / MEASURED;

    // Output results to CSV
    std::ofstream csv("results.csv");
    csv << "Mode,Type,Blocks,ThreadsPerBlock,N,Time_ms\n";
    csv << "no_branch,gpu," << numBlocks << "," << blockSize << "," << N << "," << gpu_no << "\n";
    csv << "with_branch,gpu," << numBlocks << "," << blockSize << "," << N << "," << gpu_br << "\n";
    csv << "no_branch,cpu," << numBlocks << "," << blockSize << "," << N << "," << cpu_no << "\n";
    csv << "with_branch,cpu," << numBlocks << "," << blockSize << "," << N << "," << cpu_br << "\n";
    csv.close();

    // Save to a file so we can parse and use a script to generate a chart
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "GPU No Branch:  " << gpu_no << " ms\n";
    std::cout << "GPU With Branch:" << gpu_br << " ms\n";
    std::cout << "CPU No Branch:  " << cpu_no << " ms\n";
    std::cout << "CPU With Branch:" << cpu_br << " ms\n";

    // Cleanup
    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c_no); 
    cudaFree(d_c_branch);
    return 0;
}