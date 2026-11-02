#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusolverDn.h> // PROBLEM CHILD
#include <iostream>
#include <vector>

extern __global__ void residual_norm_kernel(const float* residual, float* norm, int N);

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: ./main <N> <threads> <block_size>\n";
        return EXIT_FAILURE;
    }
    int N = atoi(argv[1]);
    int threads = atoi(argv[2]);
    int block_size = atoi(argv[3]);

    int n_bytes = N * N * sizeof(float);
    int v_bytes = N * sizeof(float);

    float *h_A = new float[N * N];
    float *h_R = new float[N * N];
    float *h_b = new float[N];
    float *h_x = new float[N];

    float *d_A = nullptr, *d_b = nullptr, *d_x = nullptr;
    float *d_residual = nullptr, *d_norm = nullptr;

    cudaMalloc(&d_A, n_bytes);
    cudaMalloc(&d_b, v_bytes);
    cudaMalloc(&d_x, v_bytes);
    cudaMalloc(&d_residual, v_bytes);
    cudaMalloc(&d_norm, sizeof(float));
    cudaMemset(d_norm, 0, sizeof(float));

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // Generate Random Symmetric Positive-Definite Matrix A
    float *d_R = nullptr;
    cudaMalloc(&d_R, n_bytes);
    curandGenerateNormal(gen, d_R, N * N, 0.0f, 1.0f);
    cudaMemcpy(h_R, d_R, n_bytes, cudaMemcpyDeviceToHost);

    // Compute A = R^T * R + alpha*I on host to ensure SPD property
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += h_R[k * N + i] * h_R[k * N + j];
            }
            h_A[i * N + j] = sum + (i == j ? 10.0f : 0.0f);
        }
    }
    cudaMemcpy(d_A, h_A, n_bytes, cudaMemcpyHostToDevice);

    // Generate true solution x_true using cuRAND
    float *d_x_true = nullptr;
    cudaMalloc(&d_x_true, v_bytes);
    curandGenerateNormal(gen, d_x_true, N, 0.0f, 1.0f);

    // Compute b = A * x_true using cuBLAS
    float one = 1.0f, zero = 0.0f;
    cublasSgemv(cublas_handle, CUBLAS_OP_N, N, N, &one, d_A, N,
                             d_x_true, 1, &zero, d_b, 1);

    // Solve Ax = b using cuSOLVER Cholesky decomposition
    int *d_info = nullptr;
    cudaMalloc(&d_info, sizeof(int));

    int lwork = 0;
    cusolverDnSpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, &lwork);

    float *d_workspace = nullptr;
    if (lwork > 0) {
        cudaMalloc(&d_workspace, lwork * sizeof(float));
    }

    // Perform Cholesky Factorization (A = L * L^T)
    cusolverDnSpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, d_workspace, lwork, d_info);

    int h_info = 0;
    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_info != 0) {
        std::cerr << "Cholesky factorization failed with info: " << h_info << "\n";
        exit(EXIT_FAILURE);
    }

    // Copy b into x because Spotrs solves in-place on the right-hand-side vector
    cudaMemcpy(d_x, d_b, v_bytes, cudaMemcpyDeviceToDevice);

    // Perform Cholesky Solve
    cusolverDnSpotrs(cusolver_handle, CUBLAS_FILL_MODE_LOWER, N, 1, d_A, N, d_x, N, d_info);
    
    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_info != 0) {
        std::cerr << "Cholesky solve failed with info: " << h_info << "\n";
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_A, h_A, n_bytes, cudaMemcpyHostToDevice);

    // Compute residual r = Ax - b
    cublasSgemv(cublas_handle, CUBLAS_OP_N, N, N, &one, d_A, N,
                             d_x, 1, &zero, d_residual, 1);
    float neg_one = -1.0f;
    cublasSaxpy(cublas_handle, N, &neg_one, d_b, 1, d_residual, 1);

    
    // Compute L2 norm to get result data
    int blocks = (N + threads - 1) / threads;
    size_t shared_mem = threads * sizeof(float);
    residual_norm_kernel<<<blocks, threads, shared_mem>>>(d_residual, d_norm, N);
    cudaDeviceSynchronize();

    float norm = 0.0f;
    cudaMemcpy(&norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Matrix Size: " << N << "\n";
    std::cout << "Config: Threads=" << threads
              << ", Block Size Parameter=" << block_size << "\n";
    std::cout << "Residual L2 Norm: " << norm << "\n";

    // Cleanup
    cublasDestroy(cublas_handle);
    cusolverDnDestroy(cusolver_handle);
    curandDestroyGenerator(gen);
    cudaFree(d_A); 
    cudaFree(d_b); 
    cudaFree(d_x);
    cudaFree(d_residual); 
    cudaFree(d_norm);
    cudaFree(d_R); 
    cudaFree(d_x_true); 
    cudaFree(d_info);
    if (d_workspace) 
    cudaFree(d_workspace);
    
    delete[] h_A; 
    delete[] h_R; 
    delete[] h_b; 
    delete[] h_x;

    return EXIT_SUCCESS;
}