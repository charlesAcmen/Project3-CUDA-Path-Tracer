cmake_minimum_required(VERSION 3.24)
project(loader_test LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(loader_test
    loader_test.cu
    ${CMAKE_SOURCE_DIR}/../../src/utils/utilities.cu   # buildTransformationMatrix
)

target_include_directories(loader_test PRIVATE
    ${CMAKE_SOURCE_DIR}/../../src
    ${CMAKE_SOURCE_DIR}/../../external/include
)

set_target_properties(loader_test PROPERTIES CUDA_ARCHITECTURES native)
set_target_properties(loader_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

if(MSVC)
    # /utf-8 must reach nvcc via -Xcompiler, not as a bare flag (nvcc would
    # treat a bare /utf-8 as an extra input file).
    target_compile_options(loader_test PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
        "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
endif()

# Copy test data next to the executable (config-agnostic multi-config layout).
add_custom_command(TARGET loader_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/assets $<TARGET_FILE_DIR:loader_test>/assets
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/scene_baseline.json $<TARGET_FILE_DIR:loader_test>/scene_baseline.json
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/scene_glb.json $<TARGET_FILE_DIR:loader_test>/scene_glb.json
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/scene_quad_obj.json $<TARGET_FILE_DIR:loader_test>/scene_quad_obj.json
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/scene_missing_mesh.json $<TARGET_FILE_DIR:loader_test>/scene_missing_mesh.json
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/scene_unsupported_ext.json $<TARGET_FILE_DIR:loader_test>/scene_unsupported_ext.json
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/scene_multi.json $<TARGET_FILE_DIR:loader_test>/scene_multi.json
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/scene_texture.json $<TARGET_FILE_DIR:loader_test>/scene_texture.json
)
