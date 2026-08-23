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
static float g_cameraMoveSpeed     = 0.5f;   // WASD fly speed (× zoom, per second) — adjustable via ImGui
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

void printStartupSummary(const AppState& app, const AppConfig& cfg)
{
    // Unpack what the body prints from the config.
    const ProfilerConfig& profCfg = cfg.profCfg;
    const RngMode         rngMode = cfg.rngMode;

    Log::raw("\n");
    Log::raw("======================================================================\n");
    Log::raw("  Startup Summary\n");
    Log::raw("======================================================================\n");
    Log::raw("  Scene: %s\n", profCfg.sceneName.c_str());
    Log::raw("  Timestamp: %s\n", app.startTimeString.c_str());
    Log::raw("  Resolution: %d x %d\n", app.width, app.height);
    if (app.renderState) {
        Log::raw("  Trace iterations (depth): %d\n", app.renderState->iterations);
    }
    Log::raw("  Profiler: %s\n", profCfg.enabled ? "ENABLED" : "disabled");
    if (profCfg.enabled) {
        Log::raw("    Warmup iters: %d\n", profCfg.warmupIters);
        Log::raw("    Verbose logging: %s\n", profCfg.verbose ? "yes" : "no");
    }
    const char* compactName = "Unknown";
    switch (profCfg.compactMethod) {
        case CompactMethod::Off:        compactName = "Disabled (no compaction)"; break;
        case CompactMethod::GlobalScan: compactName = "Global-memory scan (custom)"; break;
        case CompactMethod::Thrust:     compactName = "Thrust copy_if"; break;
        case CompactMethod::SharedMem:  compactName = "Shared-memory multi-block scan"; break;
    }
    Log::raw("  Compact method: %s\n", compactName);
    Log::raw("  Sort by material: %s\n", profCfg.sortByMaterial ? "yes" : "no");
    const char* rngName = (rngMode == RngMode::HALTON ? "Scrambled Halton" : "LCG");
    Log::raw("  RNG mode: %s\n", rngName);
    Log::raw("  BVH traversal: enabled  (max depth %d, leaf size %d)\n",
           kBvhMaxDepth, kBvhLeafSize);
    Log::raw("  Auto-save final image: %s\n", app.autoSave ? "yes" : "no");
    Log::raw("======================================================================\n");
    Log::raw("\n");
}

} // namespace

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
