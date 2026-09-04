#include "app/app_loop.h"
#include "app/app_state.h"
#include "app/camera_controller.h"
#include "config/config.h"
#include "scene/scene_loader.h"
#include "utils/logger.h"
#include "app/window_setup.h"

#include <cstdlib>
#include <ctime>
#include <string>

namespace {

std::string currentTimeString()
{
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

// ====================================================================
// Display: startup summary
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
    Log::raw("======================================================================\n");
    Log::raw("\n");
}

} // namespace

// ====================================================================
// Entry Point
// ====================================================================

int main(int argc, char** argv)
{
    // Static lifetime keeps the state valid for the CUDA/GL atexit cleanup
    // registered by window_setup.cpp.
    static AppState app;
    app.startTimeString = currentTimeString();

    if (argc < 2)
    {
        printStartupHelp(argv[0]);
        return 1;
    }

    // ---- Init singleton config (JSON → CLI → ready) ----
    initAppConfig(argc, argv);
    const auto& cfg = appConfig();

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
    app.scene = new Scene(SceneLoader::loadFromJSON(sceneFile));

    // Set up camera stuff from loaded path tracer settings
    app.iteration = 0;
    app.renderState = &app.scene->state;
    Camera& cam = app.renderState->camera;
    app.width = cam.resolution.x;
    app.height = cam.resolution.y;

    // Make the window slightly larger than the render resolution so the
    // ImGui panel (anchored to the left or right) doesn't cover the image.
    // The render is displayed centered inside the window via glViewport.
    const int IMGUI_PANEL_GUESS = 320;
    app.windowWidth  = app.width + IMGUI_PANEL_GUESS;
    app.windowHeight = app.height + 80;

    initializeCameraController(app);

    // Initialize CUDA and GL components
    // IMPORTANT: initCuda() → cudaGLSetGLDevice(0) must be called BEFORE
    // any other CUDA API calls (including cudaEventCreate in profiler init).
    init(app);

    // Profiler init must come AFTER initCuda() so that CUDA-GL interop is
    // properly configured before cudaEventCreate touches the CUDA runtime.
    g_profiler().init(cfg.profCfg);

    // Graceful CSV write on any exit path (Esc key, completion, etc.)
    if (cfg.profCfg.enabled) {
        atexit([]() { g_profiler().shutdown(); });
    }

    // Print concise runtime summary before rendering
    printStartupSummary(app, cfg);

    // Scene complexity summary
    {
        SceneStats stats = computeSceneStats(*app.scene);
        Log::raw("  Objects: %d  meshes: %d  triangles: %d  materials: %d\n",
               stats.numObjects, stats.numMeshes,
               stats.numTriangles, stats.numMaterials);
        Log::raw("======================================================================\n");
        Log::raw("\n");
    }

    // GLFW main loop
    mainLoop(app);

    return 0;
}
