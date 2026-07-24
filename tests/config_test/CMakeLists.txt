cmake_minimum_required(VERSION 3.24)
project(config_test LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_executable(config_test
    config_test.cu
    ${CMAKE_SOURCE_DIR}/../../src/config.cpp
)

target_include_directories(config_test PRIVATE
    ${CMAKE_SOURCE_DIR}/../../src
    ${CMAKE_SOURCE_DIR}/../../external/include
)

target_compile_definitions(config_test PRIVATE -DNOMINMAX)
set_target_properties(config_test PROPERTIES CUDA_ARCHITECTURES native)
set_target_properties(config_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Force config.cpp to be compiled by nvcc (handles cuda_runtime.h include).
set_source_files_properties(
    ${CMAKE_SOURCE_DIR}/../../src/config.cpp
    PROPERTIES LANGUAGE CUDA
)

if(MSVC)
    target_compile_options(config_test PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()
