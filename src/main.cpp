#include "cli.h"           // CliConfig, parseFlags, printStartupHelp, printStartupSummary
#include "image.h"
#include "pathtrace.h"
#include "profiler/profiler.h"
#include "scene.h"
#include "scene_loader.h"
#include "sceneStructs.h"
#include "utilities.h"
#include "window_setup.h"  // init, initTextures, initCuda, initPBO, etc.

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// ====================================================================
// Global State
// ====================================================================

std::string  startTimeString;

// Auto-save final image on completion (moved from pathtrace.cu — application-level concern)
bool g_autoSave = true;

// Checkpoint iteration counts for auto-save (set via --save-at=N1,N2,...).
// Sorted ascending.  saveImage() is triggered when iteration reaches each value.
// g_saveAtIterIdx tracks how many checkpoints have been consumed.
static std::vector<int> g_saveAtIterations;
static size_t g_saveAtIterIdx = 0;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
RenderState* renderState;
int iteration;

int width;
int height;

// Window dimensions — may be larger than the render resolution
// to provide space for the ImGui overlay panel.
int windowWidth;
int windowHeight;

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow* window;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;

// ====================================================================
// Utilities
// ====================================================================

std::string currentTimeString()
{
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

// ====================================================================
// Image Save
// ====================================================================

void saveImage()
{
    float samples = iteration;
    // output image file
    Image img(width, height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

// ====================================================================
// ImGui Panel
// ====================================================================



void RenderImGui()
{
    mouseOverImGuiWinow = io->WantCaptureMouse;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Path Tracer Analytics");

    ImGui::Text("Traced Depth %d", g_profiler().guiData().TracedDepth);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    if (g_profiler().enabled()) {
        ImGui::Separator();
        ImGui::Text("Per-Kernel Timing (last frame):");
        ImGui::Text("  ComputeIntersections:  %.3f ms", g_profiler().guiData().perKernelMs[4]);
        ImGui::Text("  ShadeMaterial:         %.3f ms", g_profiler().guiData().perKernelMs[0]);
        ImGui::Text("  GatherTerminatedPaths: %.3f ms", g_profiler().guiData().perKernelMs[1]);
        ImGui::Text("  SortByMaterial:        %.3f ms", g_profiler().guiData().perKernelMs[2]);
        ImGui::Text("  CompactPaths:          %.3f ms", g_profiler().guiData().perKernelMs[3]);
        ImGui::Text("  BloomPass:             %.3f ms", g_profiler().guiData().perKernelMs[5]);
        ImGui::Text("  PostProcessTail:       %.3f ms", g_profiler().guiData().perKernelMs[6]);
        ImGui::Text("Bounces Last Frame: %d", g_profiler().guiData().lastBounceCount);
    }

    if (renderState != nullptr) {
        ImGui::Separator();
        ImGui::Text("Camera Settings (JSON format):");
        Camera& cam = renderState->camera;
        char jsonBuf[384];
        sprintf(jsonBuf,
            "\"EYE\": [%.4f, %.4f, %.4f],\n"
            "\"LOOKAT\": [%.4f, %.4f, %.4f],\n"
            "\"UP\": [%.4f, %.4f, %.4f],\n"
            "\"FOVY\": %.2f",
            cam.position.x, cam.position.y, cam.position.z,
            cam.lookAt.x, cam.lookAt.y, cam.lookAt.z,
            cam.up.x, cam.up.y, cam.up.z,
            cam.fov.y
        );
        ImGui::InputTextMultiline("##json_cam", jsonBuf, sizeof(jsonBuf), ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 4.5f), ImGuiInputTextFlags_ReadOnly);

        ImGui::Separator();
        ImGui::Text("DOF Debug:");
        DebugConfig& dbg = renderState->debug;
        if (ImGui::SliderFloat("Focal Distance", &cam.focalDistance, 0.5f, 30.0f))
            camchanged = true;
        if (ImGui::SliderFloat("Lens Radius", &cam.lensRadius, 0.0f, 1.0f))
            camchanged = true;
        if (ImGui::Checkbox("Focal Plane Overlay", &dbg.showDOFOverlay))
            camchanged = true;
        if (dbg.showDOFOverlay) {
            if (ImGui::SliderFloat("Focal Tolerance", &dbg.focalTolerance, 0.05f, 5.0f))
                camchanged = true;
        }

        ImGui::Separator();
        ImGui::Separator();
        ImGui::Text("RNG Mode:");
        int currentRng = getRngMode();
        if (ImGui::RadioButton("LCG", &currentRng, 0))  { setRngMode(0); camchanged = true; }
        ImGui::SameLine();
        if (ImGui::RadioButton("Halton", &currentRng, 1)) { setRngMode(1); camchanged = true; }

        ImGui::Separator();
        ImGui::Text("Bloom:");
        bool bloomEnabled = getBloomEnabled();
        if (ImGui::Checkbox("Enable Bloom", &bloomEnabled))
            setBloomEnabled(bloomEnabled);

        if (bloomEnabled) {
            float threshold = getBloomThreshold();
            if (ImGui::SliderFloat("Threshold", &threshold, 0.1f, 10.0f, "%.2f"))
                setBloomThreshold(threshold);

            float intensity = getBloomIntensity();
            if (ImGui::SliderFloat("Intensity", &intensity, 0.0f, 2.0f, "%.2f"))
                setBloomIntensity(intensity);

            int radius = getBloomRadius();
            if (ImGui::SliderInt("Radius", &radius, 1, 30))
                setBloomRadius(radius);
        }

        ImGui::Separator();
        ImGui::Text("Chromatic Aberration:");
        bool caEnabled = getChromaticAberrationEnabled();
        if (ImGui::Checkbox("Enable Chromatic Aberration", &caEnabled))
            setChromaticAberrationEnabled(caEnabled);
        if (caEnabled) {
            float caIntensity = getChromaticAberrationIntensity();
            if (ImGui::SliderFloat("CA Intensity", &caIntensity, 0.0f, 0.008f, "%.5f"))
                setChromaticAberrationIntensity(caIntensity);
        }

        ImGui::Separator();
        ImGui::Text("Vignette:");
        bool vigEnabled = getVignetteEnabled();
        if (ImGui::Checkbox("Enable Vignette", &vigEnabled))
            setVignetteEnabled(vigEnabled);
        if (vigEnabled) {
            float vigIntensity = getVignetteIntensity();
            if (ImGui::SliderFloat("Vignette Intensity", &vigIntensity, 0.0f, 1.0f, "%.2f"))
                setVignetteIntensity(vigIntensity);
            float vigExponent = getVignetteExponent();
            if (ImGui::SliderFloat("Vignette Exponent", &vigExponent, 0.5f, 8.0f, "%.1f"))
                setVignetteExponent(vigExponent);
        }

        ImGui::Separator();
        {
            SceneStats stats = computeSceneStats(*scene);
            ImGui::Text("Scene: %d objects", stats.numObjects);
            ImGui::Text("  meshes: %d  spheres: %d  cubes: %d",
                        stats.numMeshes, stats.numSpheres, stats.numCubes);
            ImGui::Text("  %d triangles, %d materials",
                        stats.numTriangles, stats.numMaterials);
        }
    }
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

bool MouseOverImGuiWindow()
{
    return mouseOverImGuiWinow;
}

// ====================================================================
// Interaction Callbacks
// ====================================================================

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
            case GLFW_KEY_ESCAPE:
                saveImage();
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_S:
                saveImage();
                break;
            case GLFW_KEY_SPACE:
                camchanged = true;
                renderState = &scene->state;
                Camera& cam = renderState->camera;
                cam.lookAt = ogLookAt;
                break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (MouseOverImGuiWindow())
    {
        return;
    }

    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    zoom *= (yoffset > 0.0) ? 0.85f : 1.15f;
    zoom = std::fmax(0.1f, zoom);
    camchanged = true;
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (xpos == lastX || ypos == lastY)
    {
        return; // otherwise, clicking back into window causes re-start
    }

    if (leftMousePressed)
    {
        // compute new camera parameters
        phi -= (xpos - lastX) / width;
        theta -= (ypos - lastY) / height;
        theta = std::fmax(0.001f, std::fmin(theta, PI));
        camchanged = true;
    }
    else if (rightMousePressed)
    {
        zoom += (ypos - lastY) / height;
        zoom = std::fmax(0.1f, zoom);
        camchanged = true;
    }
    else if (middleMousePressed)
    {
        renderState = &scene->state;
        Camera& cam = renderState->camera;

        // Pan lookAt in the camera's image plane (screen space).
        //   Horizontal: along the camera's right vector
        //   Vertical:   along the camera's up vector (Y unlocked)
        cam.lookAt -= (float)(xpos - lastX) * cam.right * 0.01f;
        cam.lookAt += (float)(ypos - lastY) * cam.up    * 0.01f;
        camchanged = true;
    }

    lastX = xpos;
    lastY = ypos;
}

// ====================================================================
// Rendering Pipeline
// ====================================================================

void runCuda()
{
    if (camchanged)
    {
        iteration = 0;
        Camera& cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        cam.up = glm::cross(r, v);
        cam.right = r;

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    static bool pathtraceInitialized = false;
    if (iteration == 0)
    {
        if (pathtraceInitialized)
        {
            pathtraceFree();
        }
        pathtraceInit(scene);
        pathtraceInitialized = true;
    }

    if (iteration < renderState->iterations)
    {
        uchar4* pbo_dptr = NULL;
        iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
        g_profiler().beginFrame();
        pathtrace(pbo_dptr, frame, iteration);
        g_profiler().endFrame();

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);

        // Checkpoint auto-save: save image at specific iteration counts.
        // --save-at=50,200,1000 triggers saves at iteration 50, 200, 1000.
        // g_saveAtIterIdx tracks which checkpoints remain (list is sorted).
        while (g_saveAtIterIdx < g_saveAtIterations.size()
               && iteration >= g_saveAtIterations[g_saveAtIterIdx])
        {
            saveImage();
            g_saveAtIterIdx++;
        }
    }
    else
    {
        if (g_autoSave) {
            saveImage();
        }
        // Write CSVs and destroy CUDA events BEFORE tearing down the context.
        // The atexit handler will fire again during exit() but is a no-op
        // (vectors already cleared, events already null).
        g_profiler().shutdown();
        pathtraceFree();
        // Null out the PBO so the atexit(cleanupCuda) handler doesn't try
        // cudaGLUnregisterBufferObject after the context was destroyed.
        pbo = 0;
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

void mainLoop()
{
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        runCuda();

        std::string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());
        // Centre the rendered scene inside the larger window so ImGui has room.
        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        int vpX = (fbW - width) / 2;
        int vpY = (fbH - height) / 2;
        glViewport(vpX, vpY, width, height);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // Binding GL_PIXEL_UNPACK_BUFFER back to default
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

        // Render ImGui Stuff
        RenderImGui();

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

// ====================================================================
// Entry Point
// ====================================================================

int main(int argc, char** argv)
{
    startTimeString = currentTimeString();

    if (argc < 2)
    {
        printStartupHelp(argv[0]);
        return 1;
    }

    CliConfig cfg = parseFlags(argc, argv);
    g_autoSave = cfg.autoSave;
    g_saveAtIterations = std::move(cfg.saveAtIterations);

    if (cfg.showHelp)
    {
        printStartupHelp(argv[0]);
        return 0;
    }

    if (!cfg.hasScene)
    {
        fprintf(stderr, "Error: No scene file specified.\n\n");
        printStartupHelp(argv[0]);
        return 1;
    }

    ProfilerConfig profCfg = cfg.profCfg;
    const char* sceneFile  = cfg.sceneFile.c_str();

    // Load scene file
    scene = new Scene(SceneLoader::loadFromJSON(sceneFile));

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    if (cfg.fresnelSet) {
        renderState->fresnelMode = cfg.fresnelMode;
    }
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    // Make the window slightly larger than the render resolution so the
    // ImGui panel (anchored to the left or right) doesn't cover the image.
    // The render is displayed centered inside the window via glViewport.
    const int IMGUI_PANEL_GUESS = 320;
    windowWidth  = width + IMGUI_PANEL_GUESS;
    windowHeight = height + 80;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    ogLookAt = cam.lookAt;
    glm::vec3 v = cam.position - ogLookAt;
    zoom = glm::length(v);
    theta = (zoom > 0.0f) ? glm::acos(v.y / zoom) : 0.0f;
    phi = atan2(v.x, v.z);

    // Initialize CUDA and GL components
    // IMPORTANT: initCuda() → cudaGLSetGLDevice(0) must be called BEFORE
    // any other CUDA API calls (including cudaEventCreate in profiler init).
    init();

    // Profiler init must come AFTER initCuda() so that CUDA-GL interop is
    // properly configured before cudaEventCreate touches the CUDA runtime.
    g_profiler().init(profCfg);

    // Graceful CSV write on any exit path (Esc key, completion, etc.)
    if (profCfg.enabled) {
        atexit([]() { g_profiler().shutdown(); });
    }

    // Print concise runtime summary before rendering
    printStartupSummary(profCfg);

    // Scene complexity summary
    {
        SceneStats stats = computeSceneStats(*scene);
        printf("  Scene objects: %d  (meshes: %d  spheres: %d  cubes: %d)\n",
               stats.numObjects, stats.numMeshes,
               stats.numSpheres, stats.numCubes);
        printf("  Mesh triangles: %d\n", stats.numTriangles);
        printf("  Materials: %d\n", stats.numMaterials);
        printf("======================================================================\n");
        printf("\n");
    }

    // GLFW main loop
    mainLoop();

    return 0;
}
