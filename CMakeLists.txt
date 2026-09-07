cmake_minimum_required(VERSION 3.18)

project(cuda_gemm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cuda_gemm
    main.cu
    naive_gemm.cu
    tiled_gemm.cu
)

target_include_directories(cuda_gemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(cuda_gemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)