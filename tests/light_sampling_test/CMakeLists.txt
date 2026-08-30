cmake_minimum_required(VERSION 3.24)
project(light_sampling_test LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(light_sampling_test light_sampling_test.cu)
target_include_directories(light_sampling_test PRIVATE
    ${CMAKE_SOURCE_DIR}/../../src
    ${CMAKE_SOURCE_DIR}/../../external/include)
set_target_properties(light_sampling_test PROPERTIES CUDA_ARCHITECTURES native)
set_target_properties(light_sampling_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

if(MSVC)
    target_compile_options(light_sampling_test PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()
