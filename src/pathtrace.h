#pragma once

#include "scene.h"
#include "utilities.h"

// ====================================================================
// Organizing structures for the path tracer's GPU resources and
// runtime configuration.  These replace scattered file-scope statics
// so that the ownership of every buffer and option is explicit.
// ====================================================================

// All GPU device buffers owned by the path tracer.
// Previously 9 separate static dev_* pointers in pathtrace.cu.
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
    bool  enabled   = true;      // enable bloom effect
    float threshold = 1.0f;      // brightness cutoff in HDR
    float intensity = 0.5f;      // bloom blend strength
    int   radius    = 10;        // Gaussian blur radius (pixels)
    float sigma     = 5.0f;      // Gaussian sigma (auto: radius/2)

    // Returns the 1D kernel size: 2*radius + 1
    int kernelSize() const { return 2 * radius + 1; }
};

// Runtime-configurable options for the path tracing pipeline.
// Previously individual g_compactMethod / g_sortByMaterial statics.
// (autoSave was moved to main.cpp — it is an application-level concern.)
struct PathTracerOptions {
    int  compactMethod  = 3;     // 0=off, 1=global scan, 2=Thrust, 3=shared-mem (default)
    bool sortByMaterial = true;  // group paths by materialId before shading
    int  debugMode      = 0;     // 0=Hill ACES, 1=linear bypass, 2=Narkowicz ACES
    BloomConfig bloom;           // bloom post-processing settings
};

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

// Runtime configuration overrides (call before pathtraceInit)
void setCompactMethod(int method);
void setSortByMaterial(bool enable);
int  getCompactMethod();
bool getSortByMaterial();
void setDebugMode(int mode);
int  getDebugMode();

// Bloom runtime configuration
void setBloomEnabled(bool enable);
bool getBloomEnabled();
void setBloomThreshold(float threshold);
float getBloomThreshold();
void setBloomIntensity(float intensity);
float getBloomIntensity();
void setBloomRadius(int radius);
int  getBloomRadius();
