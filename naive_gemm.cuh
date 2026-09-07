#pragma once

void launch_naive_gemm(
    float* A_d,
    float* B_d,
    float* C_d,
    int M,
    int N,
    int K
);

void __global__ gemmNaive(
    float *A,
    float *B,
    float *C,
    int M,
    int N,
    int K
);