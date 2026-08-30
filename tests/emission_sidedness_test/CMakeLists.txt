cmake_minimum_required(VERSION 3.24)
project(emission_sidedness_test LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(emission_sidedness_test
    emission_sidedness_test.cu
    ${CMAKE_SOURCE_DIR}/../../src/interactions/interactions.cu)
target_include_directories(emission_sidedness_test PRIVATE
    ${CMAKE_SOURCE_DIR}/../../src
    ${CMAKE_SOURCE_DIR}/../../external/include)
set_target_properties(emission_sidedness_test PROPERTIES CUDA_ARCHITECTURES native)
set_target_properties(emission_sidedness_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

if(MSVC)
    target_compile_options(emission_sidedness_test PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()
