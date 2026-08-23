#include "pathtrace.h"
#include "sceneStructs.h"
#include "scene/scene.h"
#include "utils/utilities.h"
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

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include "intersection/intersections.h"   // getExactPointOnRay, concentricSampleDisk
#include "interactions/interactions.h"    // scatterRay, fresnel*, classifyRefraction
#include "profiler/profiler.h"
#include "kernels/kernel_config.h"
#include "kernels/bvh_traversal.cuh"      // bvhTraverse (BVH GPU traversal)
#include "rng/rng.h"
#include "bvh/bvh.h"                      // bvh::buildSceneBvh, uploadToDevice, freeDevice
#include "efficient.h"       // StreamCompaction::Efficient

// ====================================================================
// Global State
//
// File-scope variables shared across all module .cuh files that are
// #included below.  These are in the same translation unit, so every
// included file sees them directly.
// ====================================================================

static AppConfig& g_opts = appConfig();
static DeviceBuffers g_dev;
static Scene* hst_scene = NULL;
static bool s_initialized = false;

// ====================================================================
// Runtime Configuration — Getters / Setters
//
// Public API declared in pathtrace.h.  g_opts is defined above so the
// file-scope static is visible here.
// ====================================================================

void setCompactMethod(CompactMethod method) { g_opts.compactMethod = method; }
void setSortByMaterial(bool enable) { g_opts.sortByMaterial = enable; }
CompactMethod getCompactMethod()    { return g_opts.compactMethod; }
bool getSortByMaterial()            { return g_opts.sortByMaterial; }
void  setBloomEnabled(bool v)       { g_opts.bloom.enabled = v; }
bool  getBloomEnabled()             { return g_opts.bloom.enabled; }
void  setBloomThreshold(float v)    { g_opts.bloom.threshold = v; }
float getBloomThreshold()           { return g_opts.bloom.threshold; }
void  setBloomIntensity(float v)    { g_opts.bloom.intensity = v; }
float getBloomIntensity()           { return g_opts.bloom.intensity; }
void  setBloomRadius(int v)         { 
    v = std::min(BloomConfig::kRadiusMax, std::max(BloomConfig::kRadiusMin, v));
    if (v != g_opts.bloom.radius) { 
        g_opts.bloom.radius = v; g_opts.bloom.sigma = v * 0.5f; 
    } 
}
int   getBloomRadius()              { return g_opts.bloom.radius; }
void  setRngMode(RngMode mode)      { g_opts.rngMode = mode; }
RngMode getRngMode()                { return g_opts.rngMode; }
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
#include "kernels/shading.cuh"
#include "kernels/accumulation.cuh"

#include "pipeline/compact.cuh"       // calls gatherTerminatedPaths from accumulation
#include "pipeline/sort.cuh"
#include "pipeline/postprocess.cuh"   // calls sendImageToPBO from accumulation

// ====================================================================
// Resource Management
// ====================================================================

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    const int maxPaddedPathCount = 1 << ilog2ceil(pixelcount);

    cudaMalloc(&g_dev.image, pixelcount * sizeof(glm::vec3));
    cudaMemset(g_dev.image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&g_dev.paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&g_dev.pathsCompacted, pixelcount * sizeof(PathSegment));

    // Pipeline-owned shading output: one byte per path, consumed by the
    // shared-memory mask-based compaction stage.
    cudaMalloc(&g_dev.pathActivityFlags,
               static_cast<size_t>(pixelcount) * sizeof(unsigned char));

    cudaMalloc(&g_dev.materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(g_dev.materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    checkCUDAError("copy materials");

    // ---- Mesh triangles + BVH ----
    // Always build the scene-wide BVH (cheap: scenes are a few thousand
    // triangles) and upload its REORDERED split layout.  The reorder spans
    // the combined scene tree; traversal receives positions, while shading
    // receives attributes plus shared Surface and source-binding tables.
    {
        bvh::buildSceneBvh(g_dev.bvh, scene->hostTrianglePositions,
                           scene->hostTriangleAttrs, scene->geoms);

        for (Surface& surface : g_dev.bvh.hostSurfaces)
        {
            if (surface.surfaceBindingId >= 0 &&
                surface.surfaceBindingId < (int)scene->surfaceBindings.size() &&
                scene->surfaceBindings[surface.surfaceBindingId].normal >= 0)
            {
                surface.features |= SurfaceFeatureNormalMap;
            }
        }

        const int n = (int)g_dev.bvh.hostTrianglePositions.size();
        if (n > 0)
        {
            cudaMalloc(&g_dev.deviceTrianglePositions, n * sizeof(TrianglePos));
            cudaMemcpy(g_dev.deviceTrianglePositions,
                       g_dev.bvh.hostTrianglePositions.data(),
                       n * sizeof(TrianglePos), cudaMemcpyHostToDevice);
            cudaMalloc(&g_dev.deviceTriangleAttrs, n * sizeof(TriangleAttr));
            cudaMemcpy(g_dev.deviceTriangleAttrs, g_dev.bvh.hostTriangleAttrs.data(),
                       n * sizeof(TriangleAttr), cudaMemcpyHostToDevice);
        }

        if (!g_dev.bvh.hostSurfaces.empty())
        {
            cudaMalloc(&g_dev.deviceSurfaces,
                       g_dev.bvh.hostSurfaces.size() * sizeof(Surface));
            cudaMemcpy(g_dev.deviceSurfaces, g_dev.bvh.hostSurfaces.data(),
                       g_dev.bvh.hostSurfaces.size() * sizeof(Surface),
                       cudaMemcpyHostToDevice);
        }

        if (!scene->surfaceBindings.empty())
        {
            cudaMalloc(&g_dev.deviceSurfaceBindings,
                       scene->surfaceBindings.size() * sizeof(SurfaceBinding));
            cudaMemcpy(g_dev.deviceSurfaceBindings, scene->surfaceBindings.data(),
                       scene->surfaceBindings.size() * sizeof(SurfaceBinding),
                       cudaMemcpyHostToDevice);
        }

        bvh::uploadToDevice(g_dev.bvh);   // node + meta buffers (null if no meshes)
    }

    // ---- Texture table ----
    // Concatenate every scene image into one flat pixel buffer; record each
    // image's slice (pixelOffset/width/height).
    {
        std::vector<glm::vec3>    pixels;
        std::vector<TextureInfo>  infos;
        int offset = 0;
        for (const TextureData& td : scene->textures)
        {
            infos.push_back({ offset, td.width, td.height });
            pixels.insert(pixels.end(), td.pixels.begin(), td.pixels.end());
            offset += (int)td.pixels.size();
        }
        if (!pixels.empty())
        {
            cudaMalloc(&g_dev.textures.pixels, pixels.size() * sizeof(glm::vec3));
            cudaMemcpy(g_dev.textures.pixels, pixels.data(),
                       pixels.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
            cudaMalloc(&g_dev.textures.infos, infos.size() * sizeof(TextureInfo));
            cudaMemcpy(g_dev.textures.infos, infos.data(),
                       infos.size() * sizeof(TextureInfo), cudaMemcpyHostToDevice);
            g_dev.textures.count = (int)infos.size();
        }
    }

    cudaMalloc(&g_dev.intersections, pixelcount * sizeof(HitRecord));
    cudaMemset(g_dev.intersections, 0, pixelcount * sizeof(HitRecord));

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
    cudaMalloc(&g_dev.intersectionsSorted, pixelcount * sizeof(HitRecord));

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
    cudaFree(g_dev.pathActivityFlags);
    g_dev.pathActivityFlags = nullptr;
    cudaFree(g_dev.materials);
    cudaFree(g_dev.intersections);
    cudaFree(g_dev.sortKeys);
    cudaFree(g_dev.sortIndices);
    cudaFree(g_dev.intersectionsSorted);
    cudaFree(g_dev.imageDisplay);  // post-process LDR display buffer
    cudaFree(g_dev.bloomBufA);     // bloom ping-pong buffer A
    cudaFree(g_dev.bloomBufB);     // bloom ping-pong buffer B
    cudaFree(g_dev.bloomWeights);  // bloom Gaussian weight buffer
    cudaFree(g_dev.deviceTrianglePositions);
    g_dev.deviceTrianglePositions = nullptr;
    cudaFree(g_dev.deviceTriangleAttrs);
    g_dev.deviceTriangleAttrs = nullptr;
    cudaFree(g_dev.deviceSurfaces);
    g_dev.deviceSurfaces = nullptr;
    cudaFree(g_dev.deviceSurfaceBindings);
    g_dev.deviceSurfaceBindings = nullptr;
    cudaFree(g_dev.textures.pixels);
    g_dev.textures.pixels = nullptr;
    cudaFree(g_dev.textures.infos);
    g_dev.textures.infos = nullptr;
    g_dev.textures.count  = 0;
    bvh::freeDevice(g_dev.bvh);   // BVH node + meta device buffers
    StreamCompaction::Efficient::freeCompactionWorkspace();

    checkCUDAError("pathtraceFree");
}

// ====================================================================
// Accumulation Reset (camera / settings change)
// ====================================================================

void pathtraceResetAccumulation()
{
    // Camera or a runtime setting changed → restart the Monte Carlo
    // accumulation.  The scene (materials / triangles / textures / BVH) is
    // unchanged, and every per-frame buffer (paths, intersections, display,
    // bloom) is fully overwritten by the kernels each iteration — the ONLY
    // cross-frame state is g_dev.image, so that is all that needs zeroing.
    // This is one cheap memset instead of pathtraceFree + pathtraceInit,
    // which rebuilt the BVH and re-uploaded the whole scene on every frame
    // while the camera was being dragged.
    if (!hst_scene) return;
    g_profiler().resetForNewAccumulation();
    const int pixelcount = hst_scene->state.camera.resolution.x *
                           hst_scene->state.camera.resolution.y;
    cudaMemset(g_dev.image, 0, pixelcount * sizeof(glm::vec3));
    checkCUDAError("pathtraceResetAccumulation");
}

// ====================================================================
// Host Image Readback (on-demand)
// ====================================================================

void pathtraceCopyDisplayToHost()
{
    // Copy the tonemapped display buffer (imageDisplay, LDR sRGB [0,1]) to
    // host so saveImage() writes exactly what the user sees on screen.
    // g_dev.image holds the raw HDR accumulation and is intentionally NOT
    // copied here — a raw-linear PNG would look dark and clip highlights.
    //
    // Called on demand (only when saving) rather than every frame: the D2H
    // transfer is a synchronous stall in the render loop.
    if (!hst_scene) return;
    const int pixelcount = hst_scene->state.camera.resolution.x *
                           hst_scene->state.camera.resolution.y;
    cudaMemcpy(hst_scene->state.image.data(), g_dev.imageDisplay,
               pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}

// Update the ImGui trace-depth display after each frame.
// Per-kernel timing is synced by Profiler::updateGuiData() internally.
static void updateGuiAfterFrame(Profiler& prof) {
    if (prof.enabled()) {
        prof.updateGuiData();
    }
}

// ====================================================================
// Main Path-Tracing Entry Point
//
// Called once per frame / iteration.  Pipeline:
//   1. generateRayFromCamera  — primary rays → PathSegment buffer
//   2. Bounce loop (up to traceDepth):
//        bvhTraverse          — BVH ray ↔ scene intersection
//        [sortPathsByMaterial] — group by resolved materialId (optional)
//        shadeMaterial         — BSDF eval, scatter / emit
//        [compactActivePaths]  — remove dead paths            (optional)
//   3. finalGather             — accumulate remaining colors
//   4. runPostProcess           — bloom → tone → CA → vignette → PBO
// ====================================================================

void pathtrace(uchar4* pbo, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam    = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for screen-space kernels (camera rays, post-process)
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    Profiler& prof = g_profiler();
    prof.beginIteration(iter);

    // ---- 1. Primary rays ------------------------------------------------
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(
        cam, iter, traceDepth, g_dev.paths, g_opts.rngMode);
    checkCUDAError("generate camera ray");

    int  depth     = 0;
    int  num_paths = pixelcount;
    bool done      = false;
    unsigned char* pathActivityFlags =
        (g_opts.compactMethod == CompactMethod::SharedMem)
        ? g_dev.pathActivityFlags
        : nullptr;

    // ---- 2. Bounce loop -------------------------------------------------
    while (!done)
    {
        prof.recordBounce(depth, num_paths);

        // Single world-space BVH closest-hit traversal
        prof.gpuStart(ProfilerOp::ComputeIntersections);
        LAUNCH_KERNEL_AUTO(bvhTraverse, num_paths,
            num_paths, g_dev.paths,
            g_dev.intersections,
            g_dev.deviceTrianglePositions,
            g_dev.bvh.deviceNodes);
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
            g_opts.rngMode, cam, hst_scene->state.debug
        };
        ShadingSceneView shadingScene = {
            g_dev.materials,
            g_dev.deviceTrianglePositions,
            g_dev.deviceTriangleAttrs,
            g_dev.deviceSurfaces,
            g_dev.deviceSurfaceBindings,
            g_dev.textures
        };
        ShadingBufferView shadingBuffers = {
            g_dev.intersections,
            g_dev.paths,
            pathActivityFlags
        };
        prof.gpuStart(ProfilerOp::ShadeMaterial);
        LAUNCH_KERNEL_AUTO(shadeMaterial, num_paths,
            iter, num_paths,
            shadingCfg,
            shadingScene,
            shadingBuffers);
        prof.gpuStop(ProfilerOp::ShadeMaterial);

        bool allDead = compactActivePaths(num_paths);
        done = allDead || (depth >= traceDepth);

        g_profiler().guiData().TracedDepth = depth;
    }

    // ---- 3. Accumulation (only needed when compaction is disabled) -------
    // When compaction is on, all terminated paths were already gathered
    // by gatherTerminatedPaths inside compactActivePaths.
    if (g_opts.compactMethod == CompactMethod::Off)
    {
        LAUNCH_KERNEL_AUTO(gatherTerminatedPaths, pixelcount,
            pixelcount, g_dev.image, g_dev.paths);
    }

    // ---- 4. Post-Processing → Display -----------------------------------
    runPostProcess(g_dev, cam.resolution, iter,
                   g_opts.bloom,
                   g_opts.chromaticAberration,
                   g_opts.vignette,
                   pbo);

    checkCUDAError("pathtrace");

    prof.endIteration();
    updateGuiAfterFrame(prof);
}
