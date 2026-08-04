cmake_minimum_required(VERSION 3.24)
project(cis565_path_tracer LANGUAGES CUDA CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++17 for host code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Force UTF-8 encoding for all source files on Windows/MSVC.
# add_compile_options is global — applies to all targets including those in subdirectories.
if(MSVC)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
    )
endif()

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# CUDA error-checking (checkCUDAError in src/utils/utilities.h) is a debug-time
# safety net: each call does a full cudaDeviceSynchronize(), serialising the
# GPU pipeline.  Compile it in only for Debug / RelWithDebInfo; Release keeps
# kernels fully asynchronous.  Global so it also covers src/stream_compaction.
add_compile_definitions(
    $<$<CONFIG:Debug>:CIS565_ENABLE_CUDA_ERRORCHECK>
    $<$<CONFIG:RelWithDebInfo>:CIS565_ENABLE_CUDA_ERRORCHECK>
)

find_package(OpenGL REQUIRED)

if(UNIX)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(GL_LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
else(UNIX)
    set(EXTERNAL "${CMAKE_SOURCE_DIR}/external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set(GL_LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

# Global include path: all code can include files relative to src/
include_directories(${CMAKE_SOURCE_DIR}/src)

set(headers
    src/bvh/aabb.h
    src/bvh/bvh.h
    src/config/config.h
    src/image.h
    src/interactions/interactions.h
    src/intersection/intersections.h
    src/glslUtility.hpp
    src/pathtrace.h
    src/profiler/profiler.h
    src/rng/rng.h
    src/scene/scene.h
    src/scene/scene_loader.h
    src/sceneStructs.h
    src/utils/utilities.h
)

set(sources
    src/bvh/bvh.cu
    src/main.cpp
    src/stb.cpp
    src/config/config.cpp
    src/image.cpp
    src/glslUtility.cpp
    src/pathtrace.cu
    src/profiler/profiler.cu
    src/interactions/interactions.cu
    src/kernels/ray_generation.cu
    src/kernels/intersection.cu
    src/kernels/shading.cu
    src/kernels/accumulation.cu
    src/scene/scene.cpp
    src/scene/scene_loader.cpp
    src/utils/utilities.cu
)
list(SORT headers)
list(SORT sources)

source_group("Headers" FILES ${headers})
source_group("Sources" FILES ${sources})

add_subdirectory(src/ImGui)
add_subdirectory(src/stream_compaction)

# Helper function to create path tracer executable with specific configuration
# (Currently not used - all configuration is runtime-based via command-line flags)
#function(add_pathtracer_executable target_name sort_by_material_default compact_method_default)
#    add_executable(${target_name} ${sources} ${headers} ${imgui_sources} ${imgui_headers})
#    target_link_libraries(${target_name}
#        ${GL_LIBRARIES}
#        stream_compaction
#    )
#    set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#    set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES native)
#
#    # Runtime-configurable defaults (overridable at runtime via --compact=N --sort=N)
#    target_compile_definitions(${target_name} PRIVATE
#        COMPACT_METHOD_DEFAULT=${compact_method_default}
#        SORT_BY_MATERIAL_DEFAULT=${sort_by_material_default}
#    )
#    
#    target_compile_options(${target_name} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
#    target_compile_options(${target_name} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-src-in-ptx>")
#    
#    # Force UTF-8 encoding for all source files on Windows/MSVC.
#    if(MSVC)
#        target_compile_options(${target_name} PRIVATE
#            "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
#            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
#        )
#    endif()
#endfunction()

# Single executable — all tuning flags are controlled at runtime:
#   --compact=N  (0=off, 1=custom scan, 2=Thrust copy_if)
#   --sort=0/1   (disable/enable material sorting before shade)
#   --benchmark  (enable CSV profiler output)
#
# Defaults are set in pathtrace.cu and can be overridden at runtime.
add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})
target_link_libraries(${CMAKE_PROJECT_NAME}
    ${GL_LIBRARIES}
    stream_compaction
    imgui
)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>")
# -use_fast_math: approximate FP division/sqrt/trig/pow on the GPU (rcp.approx
# instead of IEEE div.rn).  Errors are ~2 ulp — invisible in a path tracer —
# and it makes the hot `1.0f / a` in the triangle test (triangle.h) cheap.
# Note: results differ from a precise-math build in the last few bits.
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>")

# Suppress common Thrust/CUDA warnings that clutter build output
# These warnings come from Thrust library internals (nextafter, uniform_real_distribution) and are safe to ignore
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE 
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--display-error-number>"
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=20011>"
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=20014>"
)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})

