#pragma once

#include "scene.h"
#include "utilities.h"

// ====================================================================
// Organizing structures for the path tracer's GPU resources and
// runtime configuration.  These replace scattered file-scope statics
// so that the ownership of every buffer and option is explicit.
// ====================================================================

// All GPU device buffers owned by the path tracer.
struct DeviceBuffers {
    glm::vec3*              image               = nullptr;
    PathSegment*            paths               = nullptr;
    PathSegment*            pathsCompacted      = nullptr;
    Geom*                   geoms               = nullptr;
    Material*               materials           = nullptr;
    ShadeableIntersection*  intersections       = nullptr;
    int*                    sortKeys            = nullptr;
    int*                    sortIndices         = nullptr;
    ShadeableIntersection*  intersectionsSorted = nullptr;
    glm::vec3*              imageDisplay        = nullptr;  // LDR [0,1] post-processed display output

    // Bloom post-processing buffers
    glm::vec3*              bloomBufA           = nullptr;  // threshold + final blur result (HDR)
    glm::vec3*              bloomBufB           = nullptr;  // horizontal blur output (HDR ping-pong)
    float*                  bloomWeights        = nullptr;  // 1D Gaussian kernel weights (device)
};

// Bloom post-processing configuration
struct BloomConfig {
    bool  enabled   = false;     // enable bloom effect (off by default — avoid firefly amplification)
    float threshold = 1.0f;      // brightness cutoff in HDR
    float intensity = 0.5f;      // bloom blend strength
    int   radius    = 10;        // Gaussian blur radius (pixels)
    float sigma     = 5.0f;      // Gaussian sigma (auto: radius/2)

    // Returns the 1D kernel size: 2*radius + 1
    int kernelSize() const { return 2 * radius + 1; }
};

// Chromatic Aberration post-processing configuration
// 色差
struct ChromaticAberrationConfig {
    bool  enabled   = false;
    float intensity = 0.003f;     // radial shift magnitude (UV fraction)
};

// Vignette (corner darkening) post-processing configuration
struct VignetteConfig {
    bool  enabled   = false;
    float intensity = 0.5f;       // darkness at corners
    float exponent  = 2.0f;       // radial falloff power
};

// Runtime-configurable options for the path tracing pipeline.
struct PathTracerOptions {
    CompactMethod     compactMethod    = CompactMethod::SharedMem;
    bool              sortByMaterial   = false;
    RngMode           rngMode          = RngMode::LCG;
    BloomConfig       bloom;
    ChromaticAberrationConfig chromaticAberration;
    VignetteConfig    vignette;
};

void pathtraceInit(Scene* scene);
void pathtraceFree();
void pathtrace(uchar4* pbo, int frame, int iteration);

// Runtime configuration overrides — called before pathtraceInit or at runtime.
void setCompactMethod(CompactMethod method);
CompactMethod getCompactMethod();
void setSortByMaterial(bool enable);
bool getSortByMaterial();

// Bloom runtime configuration
void setBloomEnabled(bool enable);
bool getBloomEnabled();
void setBloomThreshold(float threshold);
float getBloomThreshold();
void setBloomIntensity(float intensity);
float getBloomIntensity();
void setBloomRadius(int radius);
int  getBloomRadius();

// Chromatic Aberration runtime configuration
void setChromaticAberrationEnabled(bool enable);
bool getChromaticAberrationEnabled();
void setChromaticAberrationIntensity(float intensity);
float getChromaticAberrationIntensity();

// Vignette runtime configuration
void setVignetteEnabled(bool enable);
bool getVignetteEnabled();
void setVignetteIntensity(float intensity);
float getVignetteIntensity();
void setVignetteExponent(float exponent);
float getVignetteExponent();

void setRngMode(RngMode mode);
RngMode getRngMode();
