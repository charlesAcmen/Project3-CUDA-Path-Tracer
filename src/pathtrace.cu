#include "pathtrace.h"
#include "sceneStructs.h"
#include "scene.h"
#include "utilities.h"
#include "constants.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <algorithm>

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>

#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"

#include "intersections.h"   // getPointOnRay, getExactPointOnRay, concentricSampleDisk
#include "interactions/interactions.h"    // scatterRay, fresnel*, classifyRefraction
#include "profiler/profiler.h"
#include "kernel_config.h"
#include "rng/rng.h"
#include "efficient.h"       // StreamCompaction::Efficient

// Post-processing kernels (needed by pipeline/postprocess.cuh)
#include "postprocess/tonemap.cuh"
#include "postprocess/bloom.cuh"
#include "postprocess/chromatic_aberration.cuh"
#include "postprocess/vignette.cuh"

// ====================================================================
// Global State
//
// File-scope variables shared across all module .cuh files that are
// #included below.  These are in the same translation unit, so every
// included file sees them directly.
// ====================================================================

static PathTracerOptions g_opts;
static DeviceBuffers g_dev;
static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static bool s_initialized = false;

// ====================================================================
// Runtime Configuration — Getters / Setters
//
// Public API declared in pathtrace.h.  g_opts is defined above so the
// file-scope static is visible here.
// ====================================================================

void setCompactMethod(int method) {
    g_opts.compactMethod = method;
}
void setSortByMaterial(bool enable) { g_opts.sortByMaterial = enable; }
int  getCompactMethod()             { return g_opts.compactMethod; }
bool getSortByMaterial()            { return g_opts.sortByMaterial; }
void  setBloomEnabled(bool v)       { g_opts.bloom.enabled = v; }
bool  getBloomEnabled()             { return g_opts.bloom.enabled; }
void  setBloomThreshold(float v)    { g_opts.bloom.threshold = v; }
float getBloomThreshold()           { return g_opts.bloom.threshold; }
void  setBloomIntensity(float v)    { g_opts.bloom.intensity = v; }
float getBloomIntensity()           { return g_opts.bloom.intensity; }
void  setBloomRadius(int v)         { if (v != g_opts.bloom.radius) { g_opts.bloom.radius = v; g_opts.bloom.sigma = v * 0.5f; } }
int   getBloomRadius()              { return g_opts.bloom.radius; }
void  setRngMode(int mode)          { g_opts.rngMode = mode; }
int   getRngMode()                  { return g_opts.rngMode; }
void  setChromaticAberrationEnabled(bool v)  { g_opts.chromaticAberration.enabled = v; }
bool  getChromaticAberrationEnabled()        { return g_opts.chromaticAberration.enabled; }
void  setChromaticAberrationIntensity(float v) { g_opts.chromaticAberration.intensity = v; }
float getChromaticAberrationIntensity()      { return g_opts.chromaticAberration.intensity; }
void  setVignetteEnabled(bool v)             { g_opts.vignette.enabled = v; }
bool  getVignetteEnabled()                   { return g_opts.vignette.enabled; }
void  setVignetteIntensity(float v)          { g_opts.vignette.intensity = v; }
float getVignetteIntensity()                 { return g_opts.vignette.intensity; }
void  setVignetteExponent(float v)           { g_opts.vignette.exponent = v; }
float getVignetteExponent()                  { return g_opts.vignette.exponent; }

// ====================================================================
// Module Includes (kernels → pipeline)
//
// Kernels are pure GPU __global__ functions that take all data through
// parameters.  Pipeline helpers are host-side orchestration that launch
// the kernels and reference globals (g_opts, g_dev) directly.
//
// Order matters: kernels must be included before pipeline modules
// that call them.
// ====================================================================

#include "kernels/ray_generation.cuh"
#include "kernels/intersection.cuh"
#include "kernels/shading.cuh"
#include "kernels/accumulation.cuh"

#include "pipeline/compact.cuh"       // calls gatherTerminatedPaths from accumulation
#include "pipeline/sort.cuh"
#include "pipeline/postprocess.cuh"   // calls sendImageToPBO from accumulation

// ====================================================================
// Data Container & Resource Management
// ====================================================================

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;
    setCompactMethod(g_opts.compactMethod);

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    const int maxPaddedPathCount = 1 << ilog2ceil(pixelcount);

    cudaMalloc(&g_dev.image, pixelcount * sizeof(glm::vec3));
    cudaMemset(g_dev.image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&g_dev.paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&g_dev.pathsCompacted, pixelcount * sizeof(PathSegment));

    cudaMalloc(&g_dev.geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(g_dev.geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&g_dev.materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(g_dev.materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    checkCUDAError("copy geoms and materials");

    cudaMalloc(&g_dev.intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(g_dev.intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    StreamCompaction::Efficient::initCompactionWorkspace(maxPaddedPathCount);

    // Post-process display buffer: LDR [0,1] after ACES + sRGB.
    // Separate from g_dev.image so the accumulation buffer stays in raw HDR.
    cudaMalloc(&g_dev.imageDisplay, pixelcount * sizeof(glm::vec3));
    cudaMemset(g_dev.imageDisplay, 0, pixelcount * sizeof(glm::vec3));

    // Bloom ping-pong buffers (separable Gaussian blur)
    // 泛光后处理缓冲：分离高斯模糊的乒乓缓冲对
    cudaMalloc(&g_dev.bloomBufA, pixelcount * sizeof(glm::vec3));
    cudaMemset(g_dev.bloomBufA, 0, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&g_dev.bloomBufB, pixelcount * sizeof(glm::vec3));
    cudaMemset(g_dev.bloomBufB, 0, pixelcount * sizeof(glm::vec3));

    // Gaussian weight buffer (small: max 65 floats ≈ 260 bytes)
    cudaMalloc(&g_dev.bloomWeights, (2 * MAX_BLOOM_RADIUS + 1) * sizeof(float));

    // Sort buffers — always allocated (negligible overhead); sorting
    // early-returns when g_opts.sortByMaterial is false at runtime.
    cudaMalloc(&g_dev.sortKeys, pixelcount * sizeof(int));
    cudaMalloc(&g_dev.sortIndices, pixelcount * sizeof(int));
    cudaMalloc(&g_dev.intersectionsSorted, pixelcount * sizeof(ShadeableIntersection));

    s_initialized = true;

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    if (!s_initialized)
        return;

    s_initialized = false;

    cudaFree(g_dev.image);
    cudaFree(g_dev.paths);
    cudaFree(g_dev.pathsCompacted);
    cudaFree(g_dev.geoms);
    cudaFree(g_dev.materials);
    cudaFree(g_dev.intersections);
    cudaFree(g_dev.sortKeys);
    cudaFree(g_dev.sortIndices);
    cudaFree(g_dev.intersectionsSorted);
    cudaFree(g_dev.imageDisplay);  // post-process LDR display buffer
    cudaFree(g_dev.bloomBufA);     // bloom ping-pong buffer A
    cudaFree(g_dev.bloomBufB);     // bloom ping-pong buffer B
    cudaFree(g_dev.bloomWeights);  // bloom Gaussian weight buffer
    StreamCompaction::Efficient::freeCompactionWorkspace();

    checkCUDAError("pathtraceFree");
}

// ====================================================================
// Debug Helpers (called from pathtrace)
// ====================================================================

static void debugPrintBounce(int iter, int depth, int num_paths) {
    if (g_profiler().verbose()) {
        printf("  iter=%d depth=%d paths=%d\n", iter, depth, num_paths);
    }
}

// Update the ImGui trace-depth display and per-kernel timing after each frame.
static void updateGuiAfterFrame(Profiler& prof, GuiDataContainer* gui) {
    if (prof.enabled() && gui != NULL) {
        prof.updateGuiData(gui);
    }
}

// ====================================================================
// Main Path-Tracing Entry Point
//
// Called once per frame / iteration.  Pipeline:
//   1. generateRayFromCamera  — primary rays → PathSegment buffer
//   2. Bounce loop (up to traceDepth):
//        computeIntersections  — ray ↔ scene intersection
//        [sortPathsByMaterial] — group by materialId          (optional)
//        shadeMaterial         — BSDF eval, scatter / emit
//        [compactActivePaths]  — remove dead paths            (optional)
//   3. finalGather             — accumulate remaining colors
//   4. runPostProcess           — bloom → tone → CA → vignette → PBO
// ====================================================================

void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam    = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for screen-space kernels (camera rays, post-process)
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
    // 1D block for path-tracing kernels
    const int blockSize1d = 128;

    Profiler& prof = g_profiler();
    prof.beginIteration(iter);

    // ---- 1. Primary rays ------------------------------------------------
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(
        cam, iter, traceDepth, g_dev.paths, g_opts.rngMode);
    checkCUDAError("generate camera ray");

    int  depth     = 0;
    int  num_paths = pixelcount;
    bool done      = false;

    // ---- 2. Bounce loop -------------------------------------------------
    while (!done)
    {
        prof.recordBounce(depth, num_paths);

        dim3 numBlocks((num_paths + blockSize1d - 1) / blockSize1d);

        prof.gpuStart(ProfilerOp::ComputeIntersections);
        LAUNCH_KERNEL_AUTO(computeIntersections, num_paths,
            depth, num_paths, g_dev.paths,
            g_dev.geoms, hst_scene->geoms.size(), g_dev.intersections);
        prof.gpuStop(ProfilerOp::ComputeIntersections);
        checkCUDAError("trace one bounce");
        depth++;

        // GPU timer via cudaEvent: Thrust transform/sequence/sort/gather
        // are all asynchronous and return immediately; cudaEvent captures
        // the true GPU execution time.
        prof.gpuStart(ProfilerOp::SortByMaterial);
        sortPathsByMaterial(num_paths);  // no-op when sortByMaterial is false
        prof.gpuStop(ProfilerOp::SortByMaterial);

        ShadingConfig shadingCfg = {
            traceDepth, hst_scene->state.rrMinBounces,
            hst_scene->state.fresnelMode, g_opts.rngMode, cam, hst_scene->state.debug
        };
        prof.gpuStart(ProfilerOp::ShadeMaterial);
        LAUNCH_KERNEL_AUTO(shadeMaterial, num_paths,
            iter, num_paths,
            g_dev.intersections, g_dev.paths, g_dev.materials,
            shadingCfg);
        prof.gpuStop(ProfilerOp::ShadeMaterial);

        bool allDead = compactActivePaths(num_paths);
        done = allDead || (depth >= traceDepth);

        debugPrintBounce(iter, depth, num_paths);

        if (guiData != NULL)
            guiData->TracedDepth = depth;
    }

    // ---- 3. Accumulation (only needed when compaction is disabled) -------
    // When compaction is on, all terminated paths were already gathered
    // by gatherTerminatedPaths inside compactActivePaths.
    if (g_opts.compactMethod == 0)
    {
        dim3 numBlocks((pixelcount + blockSize1d - 1) / blockSize1d);
        LAUNCH_KERNEL_AUTO(finalGather, pixelcount,
            pixelcount, g_dev.image, g_dev.paths);
    }

    // ---- 4. Post-Processing → Display -----------------------------------
    runPostProcess(g_dev, cam.resolution, iter,
                   g_opts.bloom,
                   g_opts.chromaticAberration,
                   g_opts.vignette,
                   pbo);

    cudaMemcpy(hst_scene->state.image.data(), g_dev.image,
               pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");

    prof.endIteration();
    updateGuiAfterFrame(prof, guiData);
}
