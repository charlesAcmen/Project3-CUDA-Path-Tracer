set(imgui_headers
    imconfig.h
    imgui.h
    imgui_impl_glfw.h
    imgui_impl_opengl3.h
    imgui_impl_opengl3_loader.h
    imgui_internal.h
    imstb_rectpack.h
    imstb_textedit.h
    imstb_truetype.h
)

set(imgui_sources
    imgui.cpp
    imgui_demo.cpp
    imgui_draw.cpp
    imgui_impl_glfw.cpp
    imgui_impl_opengl3.cpp
    imgui_tables.cpp
    imgui_widgets.cpp
)

list(SORT imgui_headers)
list(SORT imgui_sources)

source_group("Headers" FILES ${imgui_headers})
source_group("Sources" FILES ${imgui_sources})

add_library(imgui STATIC ${imgui_sources} ${imgui_headers})

# The imgui_impl_glfw.cpp needs GLFW headers; include directories are
# inherited from the parent scope's include_directories() calls.
target_include_directories(imgui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
