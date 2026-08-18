#include "config/config.h"        // appConfig, initAppConfig, printStartupSummary
#include "utils/logger.h"        // Log::error
#include "image.h"
#include "pathtrace.h"
#include "scene/scene.h"
#include "scene/scene_loader.h"
#include "utils/utilities.h"
#include "window_setup.h"  // init, initTextures, initCuda, initPBO, etc.

#include <glm/glm.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdlib>
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

// Free-fly camera state:
//   cam.position  — authoritative; translated by WASD / middle-pan / scroll-dolly.
//   (theta, phi)  — view orientation; changed only by left-drag, so rotating
//                   turns the camera in place without moving cam.position.
//   zoom          — reference distance; scales fly speed and places the derived
//                   cam.lookAt point (zoom units ahead along the view axis).
float zoom, theta, phi;
glm::vec3 ogCameraPosition; // original position, restored by R (recenter)
float ogTheta, ogPhi, ogZoom; // original orientation / reference distance

// Camera control feel parameters (orbit + WASD fly).
static constexpr float CAMERA_MIN_THETA      = 0.001f; // orbit latitude pole guard
static constexpr float CAMERA_MIN_ZOOM       = 0.1f;   // min camera–target distance
static constexpr float CAMERA_SCROLL_ZOOM_IN = 0.97f;  // scroll-up zoom multiplier
static constexpr float CAMERA_SCROLL_ZOOM_OUT = 1.03f; // scroll-down zoom multiplier
static constexpr float CAMERA_PAN_SPEED      = 0.01f;  // middle-drag pan sensitivity
static constexpr float CAMERA_MOVE_SPEED     = 1.5f;   // WASD fly speed (× zoom, per second)
static constexpr float CAMERA_MAX_FRAME_DT   = 0.1f;   // clamp against first-frame / lag jumps

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

// Modern CUDA-GL interop resource handle for the PBO.
// Registered in initPBO() (window_setup.h), mapped per frame in runCuda().
cudaGraphicsResource_t cuda_pbo_resource;

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
    // Fetch the latest tonemapped display buffer on demand. 
    pathtraceCopyDisplayToHost();
    // No /samples (already averaged in prepareDisplayKernel)
    // and no tonemapping here (already applied by tonemapKernel).
    Image img(width, height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(x, y, pix);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << iteration << "samp";
    filename = ss.str();

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
        int currentRng = static_cast<int>(getRngMode());
        if (ImGui::RadioButton("LCG", &currentRng, 0))  { setRngMode(RngMode::LCG); camchanged = true; }
        ImGui::SameLine();
        if (ImGui::RadioButton("Halton", &currentRng, 1)) { setRngMode(RngMode::HALTON); camchanged = true; }

        ImGui::Separator();
        ImGui::Text("Compaction:");
        int curCompact = static_cast<int>(getCompactMethod());
        if (ImGui::RadioButton("Off",        &curCompact, 0)) { setCompactMethod(CompactMethod::Off);        camchanged = true; }
        if (ImGui::RadioButton("Global",     &curCompact, 1)) { setCompactMethod(CompactMethod::GlobalScan);  camchanged = true; }
        if (ImGui::RadioButton("Thrust",     &curCompact, 2)) { setCompactMethod(CompactMethod::Thrust);      camchanged = true; }
        if (ImGui::RadioButton("SharedMem",  &curCompact, 3)) { setCompactMethod(CompactMethod::SharedMem);   camchanged = true; }

        bool sortEnabled = getSortByMaterial();
        if (ImGui::Checkbox("Sort by material", &sortEnabled)) {
            setSortByMaterial(sortEnabled);
            camchanged = true;
        }

        ImGui::Separator();
        ImGui::Text("Bloom:");
        bool bloomEnabled = getBloomEnabled();
        if (ImGui::Checkbox("Enable Bloom", &bloomEnabled))
            setBloomEnabled(bloomEnabled);

        if (bloomEnabled) {
            // Slider bounds come from the same constexpr ranges that clamp
            // config.json ingestion (config.h) — one source of truth.
            float threshold = getBloomThreshold();
            if (ImGui::SliderFloat("Threshold", &threshold,
                    BloomConfig::kThresholdMin, BloomConfig::kThresholdMax, "%.2f"))
                setBloomThreshold(threshold);

            float intensity = getBloomIntensity();
            if (ImGui::SliderFloat("Intensity", &intensity,
                    BloomConfig::kIntensityMin, BloomConfig::kIntensityMax, "%.2f"))
                setBloomIntensity(intensity);

            int radius = getBloomRadius();
            if (ImGui::SliderInt("Radius", &radius,
                    BloomConfig::kRadiusMin, BloomConfig::kRadiusMax))
                setBloomRadius(radius);
        }

        ImGui::Separator();
        ImGui::Text("Chromatic Aberration:");
        bool caEnabled = getChromaticAberrationEnabled();
        if (ImGui::Checkbox("Enable Chromatic Aberration", &caEnabled))
            setChromaticAberrationEnabled(caEnabled);
        if (caEnabled) {
            float caIntensity = getChromaticAberrationIntensity();
            if (ImGui::SliderFloat("CA Intensity", &caIntensity,
                    ChromaticAberrationConfig::kIntensityMin,
                    ChromaticAberrationConfig::kIntensityMax, "%.5f"))
                setChromaticAberrationIntensity(caIntensity);
        }

        ImGui::Separator();
        ImGui::Text("Vignette:");
        bool vigEnabled = getVignetteEnabled();
        if (ImGui::Checkbox("Enable Vignette", &vigEnabled))
            setVignetteEnabled(vigEnabled);
        if (vigEnabled) {
            float vigIntensity = getVignetteIntensity();
            if (ImGui::SliderFloat("Vignette Intensity", &vigIntensity,
                    VignetteConfig::kIntensityMin, VignetteConfig::kIntensityMax, "%.2f"))
                setVignetteIntensity(vigIntensity);
            float vigExponent = getVignetteExponent();
            if (ImGui::SliderFloat("Vignette Exponent", &vigExponent,
                    VignetteConfig::kExponentMin, VignetteConfig::kExponentMax, "%.1f"))
                setVignetteExponent(vigExponent);
        }

        ImGui::Separator();
        {
            SceneStats stats = computeSceneStats(*scene);
            ImGui::Text("Scene: %d objects  (%d meshes)",
                        stats.numObjects, stats.numMeshes);
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
            case GLFW_KEY_P: // save image (S is now walk-backward)
                saveImage();
                break;
            case GLFW_KEY_R: // recenter to original position + orientation (was SPACE)
                camchanged = true;
                renderState = &scene->state;
                Camera& cam = renderState->camera;
                cam.position = ogCameraPosition;
                zoom = ogZoom;
                theta = ogTheta;
                phi = ogPhi;
                break;
        }
    }
}

// WASD / Space / Shift fly-translation.
//
// In the free-fly model cam.position is independent state, so translating it
// along the camera's own axes moves the camera through the scene while the
// orientation (theta, phi) — and thus where it points — stays fixed.
//   W/S forward/backward (cam.view), A/D left/right (cam.right),
//   Space/Shift up/down (cam.up).
void updateCameraMovement(float dt)
{
    if (io && io->WantCaptureKeyboard)
    {
        return; // ImGui text input active
    }

    Camera& cam = renderState->camera;

    glm::vec3 translate(0.0f);
    if (glfwGetKey(window, GLFW_KEY_W)             == GLFW_PRESS) translate += cam.view;
    if (glfwGetKey(window, GLFW_KEY_S)             == GLFW_PRESS) translate -= cam.view;
    if (glfwGetKey(window, GLFW_KEY_D)             == GLFW_PRESS) translate += cam.right;
    if (glfwGetKey(window, GLFW_KEY_A)             == GLFW_PRESS) translate -= cam.right;
    if (glfwGetKey(window, GLFW_KEY_SPACE)         == GLFW_PRESS) translate += cam.up;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)    == GLFW_PRESS) translate -= cam.up;

    if (translate == glm::vec3(0.0f))
    {
        return;
    }

    // Speed scales with the camera-target distance so the same key feel
    // works at both macro and micro scale (roughly zoom distance per 0.66s).
    float speed = zoom * CAMERA_MOVE_SPEED * dt;
    cam.position += glm::normalize(translate) * speed;
    camchanged = true; // resets accumulation & recomputes view/right/up/lookAt
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
    if (io && io->WantCaptureMouse) return;
    // Dolly along the view axis: scrolling changes the reference distance
    // (zoom) and moves the camera to match, so the focused point stays put —
    // "zoom in" physically flies the camera toward what it is looking at.
    Camera& cam = renderState->camera;
    const float oldZoom = zoom;
    zoom *= (yoffset > 0.0) ? CAMERA_SCROLL_ZOOM_IN : CAMERA_SCROLL_ZOOM_OUT;
    zoom = std::fmax(CAMERA_MIN_ZOOM, zoom);
    cam.position += cam.view * (oldZoom - zoom);
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
        theta = std::fmax(CAMERA_MIN_THETA, std::fmin(theta, PI));
        camchanged = true;
    }
    else if (rightMousePressed)
    {
        // Dolly along the view axis (same as scroll), driven by drag delta.
        Camera& cam = renderState->camera;
        const float oldZoom = zoom;
        zoom += (ypos - lastY) / height;
        zoom = std::fmax(CAMERA_MIN_ZOOM, zoom);
        cam.position += cam.view * (oldZoom - zoom);
        camchanged = true;
    }
    else if (middleMousePressed)
    {
        renderState = &scene->state;
        Camera& cam = renderState->camera;

        // Pan the camera in its image plane (screen space).
        //   Horizontal: along the camera's right vector
        //   Vertical:   along the camera's up vector (Y unlocked)
        cam.position -= (float)(xpos - lastX) * cam.right * CAMERA_PAN_SPEED;
        cam.position += (float)(ypos - lastY) * cam.up    * CAMERA_PAN_SPEED;
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

        // Free-fly camera: cam.position is independent state (translated by
        // WASD / middle-pan / scroll-dolly) and is NOT touched here.  (theta,
        // phi) describe the view direction, changed only by left-drag, so
        // rotating turns the camera in place.
        //   D        = (sin(phi)sin(theta), cos(theta), cos(phi)sin(theta))
        //   cam.view = -D  →  the direction the camera faces (camera → lookAt)
        glm::vec3 D(sin(phi) * sin(theta), cos(theta), cos(phi) * sin(theta));
        cam.view = -glm::normalize(D);

        // right/up MUST stay unit length: ray_generation.cu builds the pixel
        // grid as view − right·pixelLength·offset, and pixelLength is calibrated
        // for unit vectors — a short right/up would silently narrow the FOV
        // whenever the camera is pitched.  (cross(view, (0,1,0)) is unit only
        // when looking horizontally.)
        glm::vec3 u = glm::vec3(0, 1, 0);
        glm::vec3 r = glm::normalize(glm::cross(cam.view, u));
        cam.up = glm::cross(r, cam.view); // r ⟂ view, both unit → up is unit too
        cam.right = r;

        // lookAt is a derived reference point zoom units ahead along the view
        // axis (purely bookkeeping — rendering uses position/view/up/right).
        cam.lookAt = cam.position + cam.view * zoom;
        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    static bool pathtraceInitialized = false;
    if (!pathtraceInitialized)
    {
        // First frame: full one-time init — build the BVH, upload materials /
        // triangles / textures, allocate all device buffers.
        pathtraceInit(scene);
        pathtraceInitialized = true;
    }
    else if (iteration == 0)
    {
        // Camera or a runtime setting changed → restart MC accumulation.
        // The scene never changed, so only the HDR accumulation buffer is
        // zeroed — NOT the old pathtraceFree + pathtraceInit cycle, which
        // rebuilt the BVH and re-uploaded the whole scene every interactive
        // frame (the cause of the FPS drop while dragging the camera).
        pathtraceResetAccumulation();
    }

    if (iteration < renderState->iterations)
    {
        uchar4* pbo_dptr = NULL;
        iteration++;

        // ---- Map PBO for CUDA write access (modern stream-level interop) ----
        // Modern API: cudaGraphicsMapResources avoids the full-context
        // synchronisation that the deprecated cudaGLMapBufferObject performed,
        // which on WDDM could accumulate the entire frame's GPU work into a
        // single KMD submission window, intermittently triggering TDR.
        cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&pbo_dptr,
                                             NULL, cuda_pbo_resource);

        // execute the kernel
        g_profiler().beginFrame();
        pathtrace(pbo_dptr, iteration);
        g_profiler().endFrame();

        // Ensure CUDA work completes before GL touches the PBO.
        cudaDeviceSynchronize();

        // unmap buffer object
        cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

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
    double lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Frame-rate independent WASD / Space / Shift fly movement.
        double now = glfwGetTime();
        float dt = (float)std::fmin(now - lastTime, CAMERA_MAX_FRAME_DT);
        lastTime = now;
        updateCameraMovement(dt);

        runCuda();

        std::string title = "CIS565 Path Tracer | " + std::to_string(iteration) + " Iterations";
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

    // ---- Init singleton config (JSON → CLI → ready) ----
    initAppConfig(argc, argv);
    const auto& cfg = appConfig();

    g_autoSave = cfg.autoSave;
    g_saveAtIterations = cfg.saveAtIterations;  // copy (small vector)

    if (cfg.showHelp)
    {
        printStartupHelp(argv[0]);
        return 0;
    }

    if (cfg.sceneFile.empty())
    {
        Log::error("App", "No scene file specified");
        printStartupHelp(argv[0]);
        return 1;
    }

    const char* sceneFile  = cfg.sceneFile.c_str();

    // Load scene file
    scene = new Scene(SceneLoader::loadFromJSON(sceneFile));

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
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

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 v = cam.position - cam.lookAt;
    zoom = glm::length(v);
    theta = (zoom > 0.0f) ? glm::acos(v.y / zoom) : 0.0f;
    phi = atan2(v.x, v.z);

    // Remember the loaded view so R (recenter) can restore it exactly.
    ogCameraPosition = cam.position;
    ogZoom = zoom;
    ogTheta = theta;
    ogPhi = phi;

    // Initialize CUDA and GL components
    // IMPORTANT: initCuda() → cudaGLSetGLDevice(0) must be called BEFORE
    // any other CUDA API calls (including cudaEventCreate in profiler init).
    init();

    // Profiler init must come AFTER initCuda() so that CUDA-GL interop is
    // properly configured before cudaEventCreate touches the CUDA runtime.
    g_profiler().init(cfg.profCfg);

    // Graceful CSV write on any exit path (Esc key, completion, etc.)
    if (cfg.profCfg.enabled) {
        atexit([]() { g_profiler().shutdown(); });
    }

    // Print concise runtime summary before rendering
    printStartupSummary(cfg);

    // Scene complexity summary
    {
        SceneStats stats = computeSceneStats(*scene);
        Log::raw("  Objects: %d  meshes: %d  triangles: %d  materials: %d\n",
               stats.numObjects, stats.numMeshes,
               stats.numTriangles, stats.numMaterials);
        Log::raw("======================================================================\n");
        Log::raw("\n");
    }

    // GLFW main loop
    mainLoop();

    return 0;
}
