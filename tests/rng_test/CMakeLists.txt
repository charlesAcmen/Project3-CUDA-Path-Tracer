cmake_minimum_required(VERSION 3.24)
project(rng_compare LANGUAGES CUDA CXX)

# Host-only CUDA program — no device code, no separable compilation.
# We just need nvcc to compile the __host__ __device__ inline functions
# from rng.h for host-side execution.
add_executable(rng_compare rng_compare.cu)

target_include_directories(rng_compare PRIVATE ${CMAKE_SOURCE_DIR}/../../src)
set_target_properties(rng_compare PROPERTIES CUDA_ARCHITECTURES native)
set_target_properties(rng_compare PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

if(MSVC)
    target_compile_options(rng_compare PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()
