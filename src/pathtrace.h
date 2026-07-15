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
};

// Runtime-configurable options for the path tracing pipeline.
// Previously individual g_compactMethod / g_sortByMaterial statics.
// (autoSave was moved to main.cpp — it is an application-level concern.)
struct PathTracerOptions {
    int  compactMethod  = 3;     // 0=off, 1=global scan, 2=Thrust, 3=shared-mem (default)
    bool sortByMaterial = true;  // group paths by materialId before shading
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
