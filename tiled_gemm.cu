#include "tiled_gemm.cuh"
#include "cuda_utils.cuh"
#include <iostream>

void launch_tiled_gemm(float *A_d, float *B_d, float *C_d, int M, int N, int K)
{
    // Define grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    gemmTiled<<<gridSize, blockSize>>>(A_d, B_d, C_d, M, N, K);

    // Check for kernel launch errors
    CUDA_ERROR_CHECK(cudaGetLastError());

    // Synchronize to ensure completion
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

__global__ void gemmTiled(float *A_d, float *B_d, float *C_d, int M, int N, int K)
{
    // allocate shared memory
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    // identify global row and column of element to work on
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // loop over the phases of the multiplication of tiles of A and B
    float Cvalue = 0.0; // initialize dot product result

    // multiplication happens in ceil(Width/TILE_WIDTH) phases
    for (int phase = 0; phase < ((K + TILE_WIDTH - 1) / TILE_WIDTH); phase++)
    { // K is the common dimension between A and B and the dot product happens along that dimension
        // collaboratively load into shared memory
        // check bounds when loading
        if (row < M && phase * TILE_WIDTH + threadIdx.x < K)
            Ads[threadIdx.y][threadIdx.x] = A_d[row * K + phase * TILE_WIDTH + threadIdx.x];
        else
            Ads[threadIdx.y][threadIdx.x] = 0.0;
        if (col < N && phase * TILE_WIDTH + threadIdx.y < K)
            Bds[threadIdx.y][threadIdx.x] = B_d[col + N * (phase * TILE_WIDTH + threadIdx.y)];
        else
            Bds[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();
        // sync so all values are loaded into the tile
        // then perform the dot product for that tile
        for (int k = 0; k < TILE_WIDTH; k++) {
            Cvalue += Ads[threadIdx.y][k] * Bds[k][threadIdx.x];
        }
        __syncthreads();    // sync again so that dot product is calculated before old tile is overwritten
    }
    // write to product matrix C
    if (row < M && col < N)
        C_d[row * N + col] = Cvalue;
}