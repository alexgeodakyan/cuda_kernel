#pragma once

void launch_tiled_gemm(
    float* A_d,
    float* B_d,
    float* C_d,
    int M,
    int N,
    int K
);

__global__ void gemmTiled(
    float* A_d,
    float* B_d,
    float* C_d,
    int M,
    int N,
    int K
);

#define TILE_WIDTH 16