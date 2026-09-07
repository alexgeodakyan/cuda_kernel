#include "naive_gemm.cuh"
#include "cuda_utils.cuh"
#include <iostream>

// Function to launch the naive GEMM kernel. Multiplication of matrices A (M x K) and B (K x N) to produce matrix C (M x N)
void launch_naive_gemm(float *A_d, float *B_d, float *C_d, int M, int N, int K)
{
    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    gemmNaive<<<gridSize, blockSize>>>(A_d, B_d, C_d, M, N, K);

    // Check for kernel launch errors
    CUDA_ERROR_CHECK(cudaGetLastError());

    // Synchronize to ensure completion
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

// multiplies a matrix A (M x K) with a matrix B (K x N) to produce product matrix C (M x N)
void __global__ gemmNaive(float *A, float *B, float *C, int M, int N, int K)
{
    // Calculate the row and column index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform the multiplication if within bounds (of product matrix C)
    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i)
        {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}