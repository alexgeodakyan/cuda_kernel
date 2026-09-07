#include "naive_gemm.cuh"
#include "tiled_gemm.cuh"
#include "cuda_utils.cuh"
#include <random>
#include <iostream>
#include <string>
#include <cmath>

int main()
{

    std::cout << "Enter 'naive' or 'tiled' to select kernel" << std::endl;

    std::string kernelSelection;

    std::cin >> kernelSelection;

    // matrix dimensions
    // matrix A is M x K, matrix B is K x N, product matrix is M x N.
    int M = 1024;
    int N = 1024;
    int K = 1024;

    // allocate memory

    float *A_h = new float[M * K];
    float *B_h = new float[K * N];
    float *C_h = new float[M * N];
    float *C_h_ref = new float[M * N]; // cpu reference

    // create random matrices
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < M * K; ++i)
    {
        A_h[i] = dist(gen);
    }
    for (int i = 0; i < K * N; ++i)
    {
        B_h[i] = dist(gen);
    }

    // CPU multiplication for comparison:
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k)
            {
                sum += A_h[row * K + k] * B_h[k * N + col];
            }

            C_h_ref[row * N + col] = sum;
        }
    }

    float *A_d, *B_d, *C_d; // device memory pointers

    // allocate device memory
    CUDA_ERROR_CHECK(cudaMalloc((void **)&A_d, M * K * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc((void **)&B_d, K * N * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc((void **)&C_d, M * N * sizeof(float)));

    // copy data from host to device
    CUDA_ERROR_CHECK(cudaMemcpy(A_d, A_h, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(B_d, B_h, K * N * sizeof(float), cudaMemcpyHostToDevice));

    if (kernelSelection == "naive")
    {
        launch_naive_gemm(A_d, B_d, C_d, M, N, K);
    }

    else if (kernelSelection == "tiled")
    {
        launch_tiled_gemm(A_d, B_d, C_d, M, N, K);
    }
    else
    {
        std::cerr << "Unknown kernel selection: " << kernelSelection << '\n';
        CUDA_ERROR_CHECK(cudaFree(A_d));
        CUDA_ERROR_CHECK(cudaFree(B_d));
        CUDA_ERROR_CHECK(cudaFree(C_d));
        delete[] A_h;
        delete[] B_h;
        delete[] C_h;
        delete[] C_h_ref;
        return 1;
    }

    // copy the result back to host and then compare to reference
    CUDA_ERROR_CHECK(cudaMemcpy(C_h, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    bool error = false;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(C_h[i] - C_h_ref[i]) > 1e-3f) {
            error = true;
            break;
        }
    }

    if (error) {
        std::cout << "Incorrect matrix multiplication.\n";
    }
    else {
        std::cout << "Correct matrix multiplication.\n";
    }

    CUDA_ERROR_CHECK(cudaFree(A_d));
    CUDA_ERROR_CHECK(cudaFree(B_d));
    CUDA_ERROR_CHECK(cudaFree(C_d));
    delete[] A_h;
    delete[] B_h;
    delete[] C_h;
    delete[] C_h_ref;

    return error ? 1 : 0;
}