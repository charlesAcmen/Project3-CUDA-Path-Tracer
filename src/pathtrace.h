#pragma once

#include "config/config.h"       // BloomConfig, ChromaticAberrationConfig, VignetteConfig
#include "bvh/bvh.h"             // BvhBuffers (per-mesh BVH trees + metadata)
#include "scene/scene.h"

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

    // Per-mesh BVH: host build output + device nodes/meta.  hostTriangles
    // holds the REORDERED flat triangle array (leaf-contiguous chunks);
    // it is uploaded as deviceTriangles, the buffer the BVH traversal
    // kernel reads.
    BvhBuffers              bvh;
};

void pathtraceInit(Scene* scene);
void pathtraceFree();
void pathtrace(uchar4* pbo, int iteration);

// On-demand D2H readback of the final post-processed display buffer into state.image.
// Called by saveImage() so the saved PNG matches the on-screen preview;
// deliberately NOT copied every frame (the copy is a synchronous stall).
void pathtraceCopyDisplayToHost();

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
