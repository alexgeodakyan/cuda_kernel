#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_ERROR_CHECK(call)                                      \
    do                                                             \
    {                                                              \
        cudaError_t err = (call);                                  \
        if (err != cudaSuccess)                                    \
        {                                                          \
            fprintf(stderr,                                         \
                    "CUDA Error: %s (error code: %d) at %s:%d\n",   \
                    cudaGetErrorString(err),                      \
                    err,                                           \
                    __FILE__,                                      \
                    __LINE__);                                     \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)
    