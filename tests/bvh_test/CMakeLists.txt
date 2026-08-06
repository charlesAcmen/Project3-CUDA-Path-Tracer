cmake_minimum_required(VERSION 3.24)
project(bvh_test LANGUAGES CUDA CXX)

# C++17 matches the main renderer build (assignment sequencing etc.).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Host-only CUDA program — no device kernels launched, no separable
# compilation.  bvh_test.cu #includes src/bvh/bvh.cu so the PRODUCTION
# build code (world-space bake + SAH construction + flatten) compiles once
# together with the test — the test exercises exactly what the renderer runs.

add_executable(bvh_test bvh_test.cu)

target_include_directories(bvh_test PRIVATE
    ${CMAKE_SOURCE_DIR}/../../src
    ${CMAKE_SOURCE_DIR}/../../external/include)
set_target_properties(bvh_test PROPERTIES CUDA_ARCHITECTURES native)
set_target_properties(bvh_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

if(MSVC)
    target_compile_options(bvh_test PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()
