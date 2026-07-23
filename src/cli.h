#pragma once

// ====================================================================
// CLI argument parsing and startup output.
// ====================================================================

#include "profiler/profiler.h"
#include "pathtrace.h"       // getCompactMethod, setCompactMethod, etc.
#include "sceneStructs.h"    // RenderState, CompactMethod, RngMode

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ---- Extern: globals defined in main.cpp ----
// printStartupSummary reads these at startup; they are set before
// any CLI function is called, so the extern is safe.
extern std::string  startTimeString;
extern int          width;
extern int          height;
extern RenderState* renderState;
extern bool         g_autoSave;

// ---- CliConfig ----
// Parsed command-line configuration returned by parseFlags().
struct CliConfig {
    std::string  sceneFile;
    ProfilerConfig profCfg;
    bool autoSave   = true;
    bool showHelp   = false;
    bool hasScene    = false;
    FresnelMode fresnelMode = FresnelMode::Schlick;
    bool fresnelSet = false;
    std::vector<int> saveAtIterations;  // auto-save at these iteration counts
};

// ---- Functions ----

void printStartupHelp(const char* exeName)
{
    printf("\n");
    printf("======================================================================\n");
    printf("  CIS 565 Path Tracer - Command Line Help\n");
    printf("======================================================================\n");
    printf("  Usage:\n");
    printf("    %s SCENEFILE.json [options]\n", exeName);
    printf("\n");
    printf("  Examples:\n");
    printf("    %s ../scenes/cornell.json\n", exeName);
    printf("    %s ../scenes/cornell.json --benchmark --compact=2 --warmup=1\n", exeName);
    printf("    %s ../scenes/cornell.json --benchmark --sort=0 --save\n", exeName);
    printf("\n");
    printf("  Options:\n");
    printf("    --benchmark    Enable profiler CSV output.\n");
    printf("    --verbose      Print per-bounce path counts to the console.\n");
    printf("    --compact=N    Compaction mode: 0=off, 1=global scan, 2=Thrust copy_if,\n");
    printf("                   3=shared-memory scan (default).\n");
    printf("    --sort=N       Material sorting: 0=off, nonzero=on (default on).\n");
    printf("    --fresnel=N    Fresnel mode: 0=Schlick (default), 1=Accurate.\n");
    printf("    --rng=N        RNG mode: 0=LCG (default), 1=scrambled Halton.\n");
    printf("    --warmup=N     Warmup iterations excluded from profiler stats.\n");
    printf("    --save         Save the final rendered image on exit.\n");
    printf("                   (default: yes)\n");
    printf("    --save-at=N1,N2,...  Auto-save at specific iteration counts\n");
    printf("                   (e.g., --save-at=50,200,1000).  Implies --save.\n");
    printf("    -h, --help     Show this help text.\n");
    printf("\n");
    printf("  Notes:\n");
    printf("    - Flags and scene file are order-independent.\n");
    printf("    - Profiler CSVs are written to profiler_output/<scene>_<timestamp>/\n");
    printf("      when --benchmark is enabled.\n");
    printf("    - Nonzero values for --sort are treated as enabled.\n");
    printf("    - Only compact values 0..3 have defined behavior.\n");
    printf("======================================================================\n");
    printf("\n");
}

// Print a concise startup summary of key runtime options and scene info.
void printStartupSummary(const ProfilerConfig& profCfg)
{
    printf("\n");
    printf("======================================================================\n");
    printf("  Startup Summary\n");
    printf("======================================================================\n");
    printf("  Scene: %s\n", profCfg.sceneName.c_str());
    printf("  Timestamp: %s\n", startTimeString.c_str());
    printf("  Resolution: %d x %d\n", width, height);
    if (renderState) {
        printf("  Trace iterations (depth): %d\n", renderState->iterations);
    }
    printf("  Profiler: %s\n", profCfg.enabled ? "ENABLED" : "disabled");
    if (profCfg.enabled) {
        printf("    Warmup iters: %d\n", profCfg.warmupIters);
        printf("    Verbose logging: %s\n", profCfg.verbose ? "yes" : "no");
    }
    const char* compactName = "Unknown";
    switch (profCfg.compactMethod) {
        case CompactMethod::Off:        compactName = "Disabled (no compaction)"; break;
        case CompactMethod::GlobalScan: compactName = "Global-memory scan (custom)"; break;
        case CompactMethod::Thrust:     compactName = "Thrust copy_if"; break;
        case CompactMethod::SharedMem:  compactName = "Shared-memory multi-block scan"; break;
    }
    printf("  Compact method: %s\n", compactName);
    printf("  Sort by material: %s\n", profCfg.sortByMaterial ? "yes" : "no");
    const char* fresnelName = (renderState->fresnelMode == 1 ? "Accurate" : "Schlick");
    printf("  Fresnel mode: %s\n", fresnelName);
    const char* rngName = (getRngMode() == 1 ? "Scrambled Halton" : "LCG");
    printf("  RNG mode: %s\n", rngName);
    printf("  Auto-save final image: %s\n", g_autoSave ? "yes" : "no");
    printf("======================================================================\n");
    printf("\n");
}

CliConfig parseFlags(int argc, char** argv)
{
    CliConfig cfg;

    // Seed ProfilerConfig with runtime defaults from pathtrace.cu, so that
    // CSV metadata matches actual behaviour even when no CLI flag is given.
    cfg.profCfg.compactMethod  = getCompactMethod();
    cfg.profCfg.sortByMaterial = getSortByMaterial();

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // ----- Named flags -----
        if (arg == "-h" || arg == "--help") {
            cfg.showHelp = true;
        } else if (arg == "--benchmark") {
            cfg.profCfg.enabled = true;
        } else if (arg == "--verbose") {
            cfg.profCfg.verbose = true;
        } else if (arg == "--save") {
            cfg.autoSave = true;
        } else if (arg.rfind("--save-at=", 0) == 0) {
            cfg.autoSave = true;  // implies --save
            std::string list = arg.substr(10);
            std::stringstream ss(list);
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (!token.empty()) {
                    cfg.saveAtIterations.push_back(std::stoi(token));
                }
            }
            std::sort(cfg.saveAtIterations.begin(), cfg.saveAtIterations.end());
        } else if (arg.rfind("--compact=", 0) == 0) {
            int v = std::stoi(arg.substr(10));
            cfg.profCfg.compactMethod = static_cast<CompactMethod>(v);
            setCompactMethod(static_cast<CompactMethod>(v));
        } else if (arg.rfind("--sort=", 0) == 0) {
            bool v = (std::stoi(arg.substr(7)) != 0);
            cfg.profCfg.sortByMaterial = v;
            setSortByMaterial(v);
        } else if (arg.rfind("--fresnel=", 0) == 0) {
            int v = std::stoi(arg.substr(10));
            cfg.fresnelMode = (v == 1) ? FresnelMode::Accurate : FresnelMode::Schlick;
            cfg.fresnelSet  = true;
        } else if (arg.rfind("--rng=", 0) == 0) {
            int v = std::stoi(arg.substr(6));
            setRngMode(static_cast<RngMode>(v));
        } else if (arg.rfind("--warmup=", 0) == 0) {
            cfg.profCfg.warmupIters = std::stoi(arg.substr(9));
        }
        // ----- Positional argument: scene file -----
        else {
            cfg.sceneFile = arg;
            cfg.hasScene  = true;
        }
    }

    // Derive a clean scene name for CSV output (strip path + extension)
    if (cfg.hasScene) {
        std::string s = cfg.sceneFile;
        size_t slash = s.find_last_of("/\\");
        if (slash != std::string::npos) s = s.substr(slash + 1);
        size_t dot = s.find_last_of('.');
        if (dot != std::string::npos) s = s.substr(0, dot);
        cfg.profCfg.sceneName = s;
    }

    return cfg;
}
