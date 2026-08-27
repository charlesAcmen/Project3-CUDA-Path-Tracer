#include "app/window_setup.h"

#include "app/camera_controller.h"
#include "glslUtility.hpp"

#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

AppState* s_cleanupApp = nullptr;

void initTextures(AppState& app)
{
    glGenTextures(1, &app.displayImage);
    glBindTexture(GL_TEXTURE_2D, app.displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, app.width, app.height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(AppState& app)
{
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        0.0f, 1.0f,
        1.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)app.positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(app.positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)app.texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(app.texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader()
{
    const char* attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

void deletePBO(AppState& app)
{
    if (app.pbo)
    {
        // Unregister the CUDA graphics resource (modern API) before
        // deleting the GL buffer.  The deprecated cudaGLUnregisterBufferObject
        // is gone along with cudaGLMapBufferObject / cudaGLUnmapBufferObject.
        cudaGraphicsUnregisterResource(app.cudaPboResource);
        app.cudaPboResource = nullptr;

        glBindBuffer(GL_ARRAY_BUFFER, app.pbo);
        glDeleteBuffers(1, &app.pbo);

        app.pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint& tex)
{
    glDeleteTextures(1, &tex);
    tex = (GLuint)NULL;
}

void cleanupCuda()
{
    if (!s_cleanupApp) return;
    if (s_cleanupApp->pbo)
    {
        deletePBO(*s_cleanupApp);
    }
    if (s_cleanupApp->displayImage)
    {
        deleteTexture(s_cleanupApp->displayImage);
    }
}

void initCuda(AppState& app)
{
    // On newer CUDA + driver combos the GL stack may have already implicitly
    // initialised a CUDA context (e.g. through GLFW/GLEW), causing the
    // deprecated cudaGLSetGLDevice(0) to leave a stale cudaErrorSetOnActiveProcess.
    // Drain any such error so it doesn't surface at the first checkCUDAError call.
    cudaGetLastError();

    cudaGLSetGLDevice(0);

    // Clean up on program exit
    s_cleanupApp = &app;
    atexit(cleanupCuda);
}

void initPBO(AppState& app)
{
    // set up vertex data parameter
    int num_texels = app.width * app.height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &app.pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, app.pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGraphicsGLRegisterBuffer(&app.cudaPboResource, app.pbo,
                                 cudaGraphicsMapFlagsWriteDiscard);
    cudaGetLastError(); // drain stale error, same reasoning as initCuda()
}

void errorCallback(int error, const char* description)
{
    fprintf(stderr, "%s\n", description);
}

} // namespace

bool init(AppState& app)
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    app.window = glfwCreateWindow(app.windowWidth, app.windowHeight, "CIS 565 Path Tracer", NULL, NULL);
    if (!app.window)
    {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(app.window);
    glfwSetWindowUserPointer(app.window, &app);
    glfwSetKeyCallback(app.window, keyCallback);
    glfwSetCursorPosCallback(app.window, mousePositionCallback);
    glfwSetMouseButtonCallback(app.window, mouseButtonCallback);
    glfwSetScrollCallback(app.window, scrollCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        return false;
    }
    printf("Opengl Version:%s\n", glGetString(GL_VERSION));
    //Set up ImGui

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    app.io = &ImGui::GetIO(); (void)app.io;
    ImGui::StyleColorsLight();
    ImGui_ImplGlfw_InitForOpenGL(app.window, true);
    ImGui_ImplOpenGL3_Init("#version 120");

    // Initialize other stuff
    initVAO(app);
    initTextures(app);
    initCuda(app);
    initPBO(app);
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}
