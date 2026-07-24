#pragma once

// ====================================================================
// Application Configuration
//
// Pure data — no side effects.  Three-layer priority:
//   CLI flags  >  config.local.json  >  code defaults
//
// Usage in main.cpp:
//   1. loadConfigFile() → parse JSON from disk
//   2. mergeConfigJson() → apply JSON onto AppConfig (lowest priority)
//   3. parseCliFlags()  → apply CLI overrides (highest priority)
//   4. Apply to runtime via setters
// ====================================================================

#include "profiler/profiler.h"    // ProfilerConfig
#include "sceneStructs.h"         // CompactMethod, RngMode, FresnelMode

#include <string>
#include <vector>

#include "json.hpp"

// ---- Post-processing sub-configs ---------------------------------------

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

// ---- Unified startup configuration --------------------------------------

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
    bool             fresnelSet       = false;   // CLI --fresnel= was given
    std::vector<int> saveAtIterations;
};

// ---- Singleton + init ---------------------------------------------------

AppConfig& appConfig();                        // global runtime config
void       initAppConfig(int argc, char** argv); // load + merge + parse

// ---- Low-level helpers (used by tests directly) -------------------------

nlohmann::json loadConfigFile(const std::string& path);
void           mergeConfigJson(AppConfig& cfg, const nlohmann::json& data);
void           parseCliFlags(AppConfig& cfg, int argc, char** argv);

// ---- Display helpers ----------------------------------------------------

void printStartupHelp(const char* exeName);
void printStartupSummary(const ProfilerConfig& profCfg, RngMode rngMode);
