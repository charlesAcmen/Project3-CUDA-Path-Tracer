#pragma once

#include "config/config.h"       // BloomConfig, ChromaticAberrationConfig, VignetteConfig
#include "bvh/bvh.h"             // BvhBuffers (single scene-wide BVH + baked triangles)
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
    unsigned char*          pathActivityFlags   = nullptr;  // shading output consumed by mask-based compaction
    Material*               materials           = nullptr;
    HitRecord*              intersections       = nullptr;
    // Allocated together on first material-sort use.  They stay resident
    // afterwards so an ImGui toggle does not reallocate every frame.
    int*                    sortKeys            = nullptr;
    int*                    sortIndices         = nullptr;
    HitRecord*              intersectionsSorted = nullptr;
    glm::vec3*              imageDisplay        = nullptr;  // LDR [0,1] post-processed display output

    // Bloom post-processing buffers
    glm::vec3*              bloomBufA           = nullptr;  // threshold + final blur result (HDR)
    glm::vec3*              bloomBufB           = nullptr;  // horizontal blur output (HDR ping-pong)
    float*                  bloomWeights        = nullptr;  // 1D Gaussian kernel weights (device)

    // World-space mesh data (baked + REORDERED into leaf-contiguous chunks
    // by the BVH build).  Traversal reads positions only; shading expands
    // attributes and the shared surface binding only for the winning hit.
    TrianglePos*            deviceTrianglePositions   = nullptr;
    TriangleAttr*           deviceTriangleAttrs       = nullptr;
    Surface*                deviceSurfaces            = nullptr;
    SurfaceBinding*         deviceSurfaceBindings     = nullptr;

    // Direct-light sampler built from the flattened world-space triangles.
    // Kept outside BvhBuffers because it is shading metadata, not traversal
    // state; the BVH hot loop never reads these arrays.
    LightTriangle*          lightTriangles            = nullptr;
    LightAliasEntry*        lightAliasEntries         = nullptr;
    int*                    lightIndexByTriangle      = nullptr;
    int                     lightCount                = 0;

    // Texture table: every scene image concatenated into one flat texel
    // buffer, with one TextureInfo per image telling the sampler where its
    // slice starts.
    TextureTable            textures;

    // Single scene-wide BVH: host build output + device nodes.
    BvhBuffers              bvh;
};

void pathtraceInit(Scene* scene);
void pathtraceFree();
void pathtrace(uchar4* pbo, int iteration);

// Camera / runtime-settings change → restart MC accumulation by zeroing
// only the HDR accumulation buffer (g_dev.image).  Cheaper than the old
// pathtraceFree + pathtraceInit cycle, which rebuilt the BVH and re-uploaded
// the whole scene every interactive frame.
void pathtraceResetAccumulation();

// On-demand D2H readback of the final post-processed display buffer into state.image.
// Called by saveImage() so the saved PNG matches the on-screen preview;
// deliberately NOT copied every frame (the copy is a synchronous stall).
void pathtraceCopyDisplayToHost();

// Runtime configuration overrides — called before pathtraceInit or at runtime.
void setCompactMethod(CompactMethod method);
CompactMethod getCompactMethod();
void setSortByMaterial(bool enable);
bool getSortByMaterial();
void setDirectLightingEnabled(bool enable);
bool getDirectLightingEnabled();

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
