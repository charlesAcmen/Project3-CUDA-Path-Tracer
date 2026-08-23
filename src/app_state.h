#pragma once

// ====================================================================
// Application State
//
// Private state for the single-window executable.  Rendering data remains
// owned by Scene / pathtrace; this only groups the former main.cpp globals
// that coordinate input, UI, CUDA-GL interop, and the application loop.
// ====================================================================

#include "scene/scene.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "ImGui/imgui.h"

#include <cuda_gl_interop.h>

#include <cstddef>
#include <string>
#include <vector>

struct AppState
{
    std::string startTimeString;

    // Auto-save final image on completion (moved from pathtrace.cu — application-level concern)
    bool autoSave = true;

    // Checkpoint iteration counts for auto-save (set via --save-at=N1,N2,...).
    // Sorted ascending.  saveImage() is triggered when iteration reaches each value.
    // saveAtIterIdx tracks how many checkpoints have been consumed.
    std::vector<int> saveAtIterations;
    std::size_t saveAtIterIdx = 0;

    // For camera controls
    bool leftMousePressed = false;
    bool rightMousePressed = false;
    bool middleMousePressed = false;
    double lastX = 0.0;
    double lastY = 0.0;

    bool cameraChanged = true;

    // Free-fly camera state:
    //   cam.position  — authoritative; translated by WASD / middle-pan / scroll-dolly.
    //   (theta, phi)  — view orientation; changed only by left-drag, so rotating
    //                   turns the camera in place without moving cam.position.
    //   zoom          — reference distance; scales fly speed and places the derived
    //                   cam.lookAt point (zoom units ahead along the view axis).
    float zoom = 0.0f;
    float theta = 0.0f;
    float phi = 0.0f;
    glm::vec3 originalCameraPosition; // original position, restored by R (recenter)
    float originalTheta = 0.0f;
    float originalPhi = 0.0f;
    float originalZoom = 0.0f; // original orientation / reference distance
    float cameraMoveSpeed = 0.5f; // WASD fly speed (× zoom, per second) — adjustable via ImGui

    Scene* scene = nullptr;
    RenderState* renderState = nullptr;
    int iteration = 0;

    int width = 0;
    int height = 0;

    // Window dimensions — may be larger than the render resolution
    // to provide space for the ImGui overlay panel.
    int windowWidth = 0;
    int windowHeight = 0;

    GLuint positionLocation = 0;
    GLuint texcoordsLocation = 1;
    GLuint pbo = 0;
    GLuint displayImage = 0;

    // Modern CUDA-GL interop resource handle for the PBO.
    // Registered in initPBO(), mapped per frame in runCuda().
    cudaGraphicsResource_t cudaPboResource = nullptr;

    GLFWwindow* window = nullptr;
    ImGuiIO* io = nullptr;
    bool mouseOverImGuiWindow = false;
    bool pathtraceInitialized = false;
};
