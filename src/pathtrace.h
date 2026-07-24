#pragma once

#include "scene.h"

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

    // Mesh geometry (OBJ)
    Triangle*               deviceTriangles     = nullptr;
};

// (BloomConfig, ChromaticAberrationConfig, VignetteConfig live in config.h)

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
void setBloomSigma(float sigma);
float getBloomSigma();

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
