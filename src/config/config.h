#pragma once

// ====================================================================
// Application Configuration (singleton)
//
// Three-layer priority:  CLI flags  >  config.local.json  >  code defaults
//
// Usage in main.cpp:
//   initAppConfig(argc, argv);           // JSON + CLI → appConfig()
//   const auto& cfg = appConfig();       // read-only access
//   g_profiler().init(cfg.profCfg);      // profiler from config
//   4. Apply to runtime via setters
// ====================================================================

#include "profiler/profiler.h"    // ProfilerConfig
#include "sceneStructs.h"         // CompactMethod, RngMode, FresnelMode
#include "constants.h"            // MAX_BLOOM_RADIUS (BloomConfig::kRadiusMax)
#include "bvh/bvh.h"              // kMaxBvhStackDepth (BvhConfig depth clamp)

#include <algorithm>              // std::min / std::max (config clamping)
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

    // ---- Legal ranges (single source of truth) ----
    // Consumed by config.json ingestion (config.cpp clamps to these), the
    // ImGui sliders (main.cpp), and the pathtrace.cu setters — so JSON,
    // code defaults, and the UI can never drift apart.
    // kRadiusMax must stay <= MAX_BLOOM_RADIUS (constants.h): the device
    // bloomWeights buffer is sized from that constant.
    static constexpr float kThresholdMin = 0.1f;
    static constexpr float kThresholdMax = 10.0f;
    static constexpr float kIntensityMin = 0.0f;
    static constexpr float kIntensityMax = 2.0f;
    static constexpr int   kRadiusMin    = 1;
    static constexpr int   kRadiusMax    = MAX_BLOOM_RADIUS;

    void clamp() {
        threshold = std::min(kThresholdMax, std::max(kThresholdMin, threshold));
        intensity = std::min(kIntensityMax, std::max(kIntensityMin, intensity));
        radius    = std::min(kRadiusMax, std::max(kRadiusMin, radius));
    }
};

struct ChromaticAberrationConfig {
    bool  enabled   = false;
    float intensity = 0.003f;

    // ---- Legal range (single source of truth; see BloomConfig) ----
    static constexpr float kIntensityMin = 0.0f;
    static constexpr float kIntensityMax = 0.008f;

    void clamp() {
        intensity = std::min(kIntensityMax, std::max(kIntensityMin, intensity));
    }
};

struct VignetteConfig {
    bool  enabled   = false;
    float intensity = 0.5f;
    float exponent  = 2.0f;

    // ---- Legal ranges (single source of truth; see BloomConfig) ----
    static constexpr float kIntensityMin = 0.0f;
    static constexpr float kIntensityMax = 1.0f;
    static constexpr float kExponentMin  = 0.5f;
    static constexpr float kExponentMax  = 8.0f;

    void clamp() {
        intensity = std::min(kIntensityMax, std::max(kIntensityMin, intensity));
        exponent  = std::min(kExponentMax, std::max(kExponentMin, exponent));
    }
};

struct BvhConfig {
    bool enabled   = false;   // default OFF -> O(N) baseline
    int  maxDepth  = 24;      // clamp [1, 63]
    int  leafSize  = 4;       // clamp [1, 64]

    // ---- Legal ranges (single source of truth; see BloomConfig) ----
    // maxDepth is capped at kMaxBvhStackDepth - 1 so a built tree can never
    // overflow the traversal kernel's explicit stack (depth clamp in bvh.cu).
    static constexpr int kMaxDepthMin = 1;
    static constexpr int kMaxDepthMax = kMaxBvhStackDepth - 1;  // 63
    static constexpr int kLeafSizeMin = 1;
    static constexpr int kLeafSizeMax = 64;

    void clamp() {
        maxDepth = std::min(kMaxDepthMax, std::max(kMaxDepthMin, maxDepth));
        leafSize = std::min(kLeafSizeMax, std::max(kLeafSizeMin, leafSize));
    }
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

    // BVH acceleration (per-mesh trees, GPU iterative traversal)
    BvhConfig                bvh;

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
void printStartupSummary(const AppConfig& cfg);
