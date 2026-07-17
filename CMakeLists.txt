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
    set(GL_LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/stream_compaction)

set(headers
    src/image.h
    src/interactions/interactions.h
    src/intersections.h
    src/glslUtility.hpp
    src/pathtrace.h
    src/profiler.h
    src/rng.h
    src/scene.h
    src/sceneStructs.h
    src/utilities.h
)

set(sources
    src/main.cpp
    src/stb.cpp
    src/image.cpp
    src/glslUtility.cpp
    src/pathtrace.cu
    src/profiler.cu
    src/intersections.cu
    src/interactions/interactions.cu
    src/scene.cpp
    src/utilities.cu
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

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})

