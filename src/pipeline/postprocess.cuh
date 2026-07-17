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

    // ---- Bloom (linear HDR space) — timed as BloomPass ----
    bool bloomHasRun = (bloomCfg.enabled && bloomCfg.intensity > 0.0f);
    if (bloomHasRun)
    {
        g_profiler().gpuStart(ProfilerOp::BloomPass);

        int kernelSize = bloomCfg.kernelSize();
        std::vector<float> weights = computeGaussianWeights(bloomCfg.radius, bloomCfg.sigma);
        cudaMemcpy(dev.bloomWeights, weights.data(),
                   kernelSize * sizeof(float), cudaMemcpyHostToDevice);

        // Threshold: keep only pixels brighter than the cutoff
        thresholdExtract<<<blocksPerGrid2d, blockSize2d>>>(
            dev.image, dev.bloomBufA, resolution, iter, bloomCfg.threshold);

        // Horizontal separable blur (shared-memory tiled)
        {
            dim3 gridH((resolution.x + BLOOM_BLOCK_SIZE - 1) / BLOOM_BLOCK_SIZE,
                       resolution.y, 1);
            dim3 blockH(BLOOM_BLOCK_SIZE, 1, 1);
            size_t smem = (BLOOM_BLOCK_SIZE + 2 * bloomCfg.radius) * sizeof(float) * 3;
            blurHorizontal<<<gridH, blockH, smem>>>(
                dev.bloomBufA, dev.bloomBufB,
                resolution.x, resolution.y,
                dev.bloomWeights, bloomCfg.radius);
        }
        checkCUDAError("bloom blurHorizontal");

        // Vertical separable blur (overwrites bloomBufA)
        {
            dim3 gridV((resolution.y + BLOOM_BLOCK_SIZE - 1) / BLOOM_BLOCK_SIZE,
                       resolution.x, 1);
            dim3 blockV(BLOOM_BLOCK_SIZE, 1, 1);
            size_t smem = (BLOOM_BLOCK_SIZE + 2 * bloomCfg.radius) * sizeof(float) * 3;
            blurVertical<<<gridV, blockV, smem>>>(
                dev.bloomBufB, dev.bloomBufA,
                resolution.x, resolution.y,
                dev.bloomWeights, bloomCfg.radius);
        }
        checkCUDAError("bloom blurVertical");

        g_profiler().gpuStop(ProfilerOp::BloomPass);
    }

    // ---- Remaining post-process (prepareDisplay → tonemap → CA → vignette → PBO)
    //      timed together as PostProcessTail ----
    g_profiler().gpuStart(ProfilerOp::PostProcessTail);

    // ---- Prepare display buffer: average HDR, composite bloom ----
    prepareDisplayKernel<<<blocksPerGrid2d, blockSize2d>>>(
        dev.image, dev.imageDisplay, resolution, iter,
        bloomHasRun ? dev.bloomBufA : nullptr,
        bloomCfg.intensity);
    checkCUDAError("prepareDisplayKernel");
    // ---- Tone mapping: ACES filmic + sRGB gamma (in-place) ----
    tonemapKernel<<<blocksPerGrid2d, blockSize2d>>>(
        dev.imageDisplay, dev.imageDisplay, resolution);
    checkCUDAError("tonemapKernel");
    // ---- Chromatic Aberration (sRGB, after tone mapping) ----
    // Writes to bloomBufB as scratch, then copies back or chains into
    // vignette to avoid an extra D2D memcpy.
    bool caHasRun = (caCfg.enabled && caCfg.intensity > 0.0f);
    if (caHasRun)
    {
        chromaticAberrationKernel<<<blocksPerGrid2d, blockSize2d>>>(
            dev.imageDisplay, dev.bloomBufB, resolution, caCfg.intensity);
        checkCUDAError("chromaticAberrationKernel");
    }

    // ---- Vignette (final step before display) ----
    // Reads from bloomBufB if CA ran, else directly from imageDisplay.
    // Always writes the final result to imageDisplay.
    if (vignetteCfg.enabled && vignetteCfg.intensity > 0.0f)
    {
        const glm::vec3* vigSrc = caHasRun ? dev.bloomBufB : dev.imageDisplay;
        vignetteKernel<<<blocksPerGrid2d, blockSize2d>>>(
            vigSrc, dev.imageDisplay, resolution,
            vignetteCfg.intensity, vignetteCfg.exponent);
        checkCUDAError("vignetteKernel");
    }
    else if (caHasRun)
    {
        // CA ran but vignette is off: copy CA result back to display buffer
        cudaMemcpy(dev.imageDisplay, dev.bloomBufB,
                   resolution.x * resolution.y * sizeof(glm::vec3),
                   cudaMemcpyDeviceToDevice);
    }

    // ---- Display: write LDR sRGB data to OpenGL pixel buffer ----
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(
        pbo, resolution, 1, dev.imageDisplay);

    g_profiler().gpuStop(ProfilerOp::PostProcessTail);
}
