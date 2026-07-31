cmake_minimum_required(VERSION 3.24)
project(refraction_test LANGUAGES CUDA CXX)

# Host-only CUDA program — no device kernels launched, no separable
# compilation.  nvcc compiles the __host__ __device__ functions from
# interactions.cu / rng.h for host-side execution.
add_executable(refraction_test refraction_test.cu)

target_include_directories(refraction_test PRIVATE
    ${CMAKE_SOURCE_DIR}/../../src
    ${CMAKE_SOURCE_DIR}/../../external/include)
set_target_properties(refraction_test PROPERTIES CUDA_ARCHITECTURES native)
set_target_properties(refraction_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

if(MSVC)
    target_compile_options(refraction_test PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()
