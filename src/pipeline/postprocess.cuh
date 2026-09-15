#pragma once

// ====================================================================
// Post-Processing Pipeline Orchestration
//
// Chains bloom → tone mapping → chromatic aberration → vignette → PBO
// display for each frame.  All GPU kernels are defined in src/postprocess/*.cuh
// and src/kernels/accumulation.cuh; this file only orchestrates the sequence.
//
// Pipeline (HDR → LDR sRGB → OpenGL):
//   (optional) bloom thresholdExtract → blurH → blurV   [linear HDR]
//   → prepareDisplayKernel (÷iter, composite bloom)     [imageDisplay, HDR]
//   → tonemapKernel (ACES filmic + sRGB gamma)          [sRGB, in-place]
//   → (optional) chromaticAberrationKernel              [sRGB]
//   → (optional) vignetteKernel                         [sRGB, in-place]
//   → sendImageToPBO                                    [OpenGL PBO]
// ====================================================================

#include "sceneStructs.h"
#include "pathtrace.h"         // DeviceBuffers, BloomConfig, etc.
#include "postprocess/tonemap.cuh"
#include "postprocess/bloom.cuh"
#include "postprocess/chromatic_aberration.cuh"
#include "postprocess/vignette.cuh"
#include "kernels/accumulation.cuh"  // sendImageToPBO

/**
 * Run the full post-processing pipeline for one frame.
 *
 * Bloom is applied in linear HDR space before tone mapping, so the
 * threshold has a physical meaning (brightness > ~1.0 = actually bright).
 * Chromatic aberration and vignette run in display-ready sRGB space.
 *
 * Post-process kernels use 2D block configurations (8×8) — these are
 * memory-bound, so 64 threads/block is sufficient to hide latency.
 */
static void runPostProcess(
    DeviceBuffers& dev,
    glm::ivec2 resolution,
    int iter,
    const BloomConfig& bloomCfg,
    const ChromaticAberrationConfig& caCfg,
    const VignetteConfig& vignetteCfg,
    uchar4* pbo)
{
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Precomputed per-sample average:one host-side division instead of one per pixel).
    const float invIter = 1.0f / (float)iter;

    // ---- Bloom (linear HDR space) — each stage is timed independently ----
    bool bloomHasRun = (bloomCfg.enabled && bloomCfg.intensity > 0.0f);
    if (bloomHasRun)
    {
        Profiler& profiler = g_profiler();
        const std::size_t weightsTimer = profiler.cpuStart(
            ProfilerOp::BloomWeights, ProfilerScope::Frame, -1,
            bloomCfg.kernelSize());
        int kernelSize = bloomCfg.kernelSize();
        std::vector<float> weights = computeGaussianWeights(bloomCfg.radius, bloomCfg.sigma);
        cudaMemcpy(dev.bloomWeights, weights.data(),
                   kernelSize * sizeof(float), cudaMemcpyHostToDevice);
        profiler.cpuStop(weightsTimer);

        // Threshold: keep only pixels brighter than the cutoff
        const std::size_t thresholdTimer = profiler.gpuStart(
            ProfilerOp::BloomThreshold, ProfilerScope::Frame, -1,
            resolution.x * resolution.y);
        thresholdExtract<<<blocksPerGrid2d, blockSize2d>>>(
            dev.image, dev.bloomBufA, resolution, invIter, bloomCfg.threshold);
        profiler.gpuStop(thresholdTimer);

        // Horizontal separable blur (shared-memory tiled)
        {
            dim3 gridH((resolution.x + BLOOM_BLOCK_SIZE - 1) / BLOOM_BLOCK_SIZE,
                       resolution.y, 1);
            dim3 blockH(BLOOM_BLOCK_SIZE, 1, 1);
            size_t smem = (BLOOM_BLOCK_SIZE + 2 * bloomCfg.radius) * sizeof(float) * 3;
            const std::size_t blurHTimer = profiler.gpuStart(
                ProfilerOp::BloomBlurHorizontal, ProfilerScope::Frame, -1,
                resolution.x * resolution.y);
            blurHorizontal<<<gridH, blockH, smem>>>(
                dev.bloomBufA, dev.bloomBufB,
                resolution.x, resolution.y,
                dev.bloomWeights, bloomCfg.radius);
            profiler.gpuStop(blurHTimer);
        }
        checkCUDAError("bloom blurHorizontal");

        // Vertical separable blur (overwrites bloomBufA)
        {
            dim3 gridV((resolution.y + BLOOM_BLOCK_SIZE - 1) / BLOOM_BLOCK_SIZE,
                       resolution.x, 1);
            dim3 blockV(BLOOM_BLOCK_SIZE, 1, 1);
            size_t smem = (BLOOM_BLOCK_SIZE + 2 * bloomCfg.radius) * sizeof(float) * 3;
            const std::size_t blurVTimer = profiler.gpuStart(
                ProfilerOp::BloomBlurVertical, ProfilerScope::Frame, -1,
                resolution.x * resolution.y);
            blurVertical<<<gridV, blockV, smem>>>(
                dev.bloomBufB, dev.bloomBufA,
                resolution.x, resolution.y,
                dev.bloomWeights, bloomCfg.radius);
            profiler.gpuStop(blurVTimer);
        }
        checkCUDAError("bloom blurVertical");
    }

    Profiler& profiler = g_profiler();

    // ---- Prepare display buffer: average HDR, composite bloom ----
    const std::size_t prepareTimer = profiler.gpuStart(
        ProfilerOp::PrepareDisplay, ProfilerScope::Frame, -1,
        resolution.x * resolution.y);
    prepareDisplayKernel<<<blocksPerGrid2d, blockSize2d>>>(
        dev.image, dev.imageDisplay, resolution, invIter,
        bloomHasRun ? dev.bloomBufA : nullptr,
        bloomCfg.intensity);
    profiler.gpuStop(prepareTimer);
    checkCUDAError("prepareDisplayKernel");
    // ---- Tone mapping: ACES filmic + sRGB gamma (in-place) ----
    const std::size_t tonemapTimer = profiler.gpuStart(
        ProfilerOp::Tonemap, ProfilerScope::Frame, -1,
        resolution.x * resolution.y);
    tonemapKernel<<<blocksPerGrid2d, blockSize2d>>>(
        dev.imageDisplay, dev.imageDisplay, resolution);
    profiler.gpuStop(tonemapTimer);
    checkCUDAError("tonemapKernel");
    // ---- Chromatic Aberration (sRGB, after tone mapping) ----
    // Writes to bloomBufB as scratch, then copies back or chains into
    // vignette to avoid an extra D2D memcpy.
    bool caHasRun = (caCfg.enabled && caCfg.intensity > 0.0f);
    if (caHasRun)
    {
        const std::size_t caTimer = profiler.gpuStart(
            ProfilerOp::ChromaticAberration, ProfilerScope::Frame, -1,
            resolution.x * resolution.y);
        chromaticAberrationKernel<<<blocksPerGrid2d, blockSize2d>>>(
            dev.imageDisplay, dev.bloomBufB, resolution, caCfg.intensity);
        profiler.gpuStop(caTimer);
        checkCUDAError("chromaticAberrationKernel");
    }

    // ---- Vignette (final step before display) ----
    // Reads from bloomBufB if CA ran, else directly from imageDisplay.
    // Always writes the final result to imageDisplay.
    if (vignetteCfg.enabled && vignetteCfg.intensity > 0.0f)
    {
        const glm::vec3* vigSrc = caHasRun ? dev.bloomBufB : dev.imageDisplay;
        const std::size_t vignetteTimer = profiler.gpuStart(
            ProfilerOp::Vignette, ProfilerScope::Frame, -1,
            resolution.x * resolution.y);
        vignetteKernel<<<blocksPerGrid2d, blockSize2d>>>(
            vigSrc, dev.imageDisplay, resolution,
            vignetteCfg.intensity, vignetteCfg.exponent);
        profiler.gpuStop(vignetteTimer);
        checkCUDAError("vignetteKernel");
    }
    else if (caHasRun)
    {
        // CA ran but vignette is off: copy CA result back to display buffer
        const std::size_t copyTimer = profiler.gpuStart(
            ProfilerOp::PostProcessCopy, ProfilerScope::Frame, -1,
            resolution.x * resolution.y);
        cudaMemcpy(dev.imageDisplay, dev.bloomBufB,
                   resolution.x * resolution.y * sizeof(glm::vec3),
                   cudaMemcpyDeviceToDevice);
        profiler.gpuStop(copyTimer);
    }

    // ---- Display: write LDR sRGB data to OpenGL pixel buffer ----
    const std::size_t pboTimer = profiler.gpuStart(
        ProfilerOp::SendImageToPbo, ProfilerScope::Frame, -1,
        resolution.x * resolution.y);
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(
        pbo, resolution, dev.imageDisplay);
    profiler.gpuStop(pboTimer);
}
