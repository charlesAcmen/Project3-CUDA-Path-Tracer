cmake_minimum_required(VERSION 3.24)
project(texture_test LANGUAGES CUDA CXX)

# Host-only CUDA program — no device code, no separable compilation.
# We compile the real interactions.cu so the __host__ __device__ sampling
# functions (and only host-side execution paths) are the actual production
# implementations, not copies.
add_executable(texture_test
    texture_test.cu
    ${CMAKE_SOURCE_DIR}/../../src/interactions/interactions.cu
)

target_include_directories(texture_test PRIVATE
    ${CMAKE_SOURCE_DIR}/../../src
    ${CMAKE_SOURCE_DIR}/../../external/include
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include"
)
set_target_properties(texture_test PROPERTIES CUDA_ARCHITECTURES native)
set_target_properties(texture_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

if(MSVC)
    target_compile_options(texture_test PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()
