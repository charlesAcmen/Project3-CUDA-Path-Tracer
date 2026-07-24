#pragma once

// ====================================================================
// Application Configuration
//
// Pure data — no setters, no side effects.  loadAppConfig() reads the
// config file and CLI flags, merges them by priority, and returns a
// final AppConfig.  The caller (main.cpp) applies values to the
// runtime via setters in a single explicit sync point.
//
// Priority:  CLI flags  >  config.local.json  >  code defaults
// ====================================================================

#include "profiler/profiler.h"    // ProfilerConfig
#include "sceneStructs.h"         // CompactMethod, RngMode, FresnelMode

#include <string>
#include <vector>

// Post-processing config sub-structs (mirrors pathtrace.h without
// pulling in the whole header).
struct BloomConfig {
    bool  enabled   = false;
    float threshold = 1.0f;
    float intensity = 0.5f;
    int   radius    = 10;
    float sigma     = 5.0f;

    int kernelSize() const { return 2 * radius + 1; }
};

struct ChromaticAberrationConfig {
    bool  enabled   = false;
    float intensity = 0.003f;
};

struct VignetteConfig {
    bool  enabled   = false;
    float intensity = 0.5f;
    float exponent  = 2.0f;
};

// ---- Unified startup configuration ------------------------------------

struct AppConfig {
    ProfilerConfig   profCfg;
    std::string      sceneFile;

    // Runtime settings
    CompactMethod    compactMethod    = CompactMethod::SharedMem;
    bool             sortByMaterial   = false;
    RngMode          rngMode          = RngMode::LCG;

    // Post-processing
    BloomConfig              bloom;
    ChromaticAberrationConfig chromaticAberration;
    VignetteConfig           vignette;

    // Other
    bool             autoSave         = true;
    bool             showHelp         = false;
    FresnelMode      fresnelMode      = FresnelMode::Schlick;
    bool             fresnelSet       = false;
    std::vector<int> saveAtIterations;
};

AppConfig loadAppConfig(int argc, char** argv);

// ---- Display helpers --------------------------------------------------

void printStartupHelp(const char* exeName);
void printStartupSummary(const ProfilerConfig& profCfg, RngMode rngMode);
