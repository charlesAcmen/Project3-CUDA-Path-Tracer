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

set(headers
    src/image.h
    src/interactions.h
    src/intersections.h
    src/glslUtility.hpp
    src/pathtrace.h
    src/profiler.h
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
    src/interactions.cu
    src/scene.cpp
    src/utilities.cu
)

set(imgui_headers
    src/ImGui/imconfig.h
    src/ImGui/imgui.h
    src/ImGui/imgui_impl_glfw.h
    src/ImGui/imgui_impl_opengl3.h
    src/ImGui/imgui_impl_opengl3_loader.h
    src/ImGui/imgui_internal.h
    src/ImGui/imstb_rectpack.h
    src/ImGui/imstb_textedit.h
    src/ImGui/imstb_truetype.h
)

set(imgui_sources
    src/ImGui/imgui.cpp
    src/ImGui/imgui_demo.cpp
    src/ImGui/imgui_draw.cpp
    src/ImGui/imgui_impl_glfw.cpp
    src/ImGui/imgui_impl_opengl3.cpp
    src/ImGui/imgui_tables.cpp
    src/ImGui/imgui_widgets.cpp
)

list(SORT headers)
list(SORT sources)
list(SORT imgui_headers)
list(SORT imgui_sources)

source_group("Headers" FILES ${headers})
source_group("Sources" FILES ${sources})
source_group("ImGui\\Headers" FILES ${imgui_headers})
source_group("ImGui\\Sources" FILES ${imgui_sources})

#add_subdirectory(src/ImGui)
add_subdirectory(stream_compaction)

# Helper function to create path tracer executable with specific configuration
function(add_pathtracer_executable target_name sort_by_material_default compact_method_default)
    add_executable(${target_name} ${sources} ${headers} ${imgui_sources} ${imgui_headers})
    target_link_libraries(${target_name}
        ${GL_LIBRARIES}
        stream_compaction
    )
    set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES native)

    # Runtime-configurable defaults (overridable at runtime via --compact=N --sort=N)
    target_compile_definitions(${target_name} PRIVATE
        COMPACT_METHOD_DEFAULT=${compact_method_default}
        SORT_BY_MATERIAL_DEFAULT=${sort_by_material_default}
    )
    
    target_compile_options(${target_name} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
    target_compile_options(${target_name} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-src-in-ptx>")
    
    # Force UTF-8 encoding for all source files on Windows/MSVC.
    if(MSVC)
        target_compile_options(${target_name} PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
        )
    endif()
endfunction()

# Create two executables: one with material sorting enabled (default), one disabled.
# The default target (cis565_path_tracer) is the sorted version -- use it for
# normal rendering.  The _unsorted variant exists for performance comparisons.
# Both use Thrust compaction by default (COMPACT_METHOD_DEFAULT=2).
# Runtime overrides: --benchmark --compact=N --sort=N
add_pathtracer_executable(${CMAKE_PROJECT_NAME}             1 2)   # sorted, Thrust compact
add_pathtracer_executable(${CMAKE_PROJECT_NAME}_unsorted    0 2)   # unsorted, Thrust compact

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})

