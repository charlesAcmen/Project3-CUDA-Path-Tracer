/*
 * bloom.cuh — Bloom Post-Processing Effect
 * ====================================================================
 * Implements bloom (glow) via:
 *   1. Threshold extraction: isolate pixels brighter than a threshold
 *      in linear HDR space.
 *   2. Separable Gaussian blur: two-pass (horizontal + vertical) with
 *      shared-memory tiling for efficient GPU execution.
 *
 * Pipeline:
 *   HDR accumulation → thresholdExtract → blurHorizontal → blurVertical
 *                     → tonemapKernel(HDR + intensity * bloom) → display
 *
 * Bloom operates in LINEAR HDR space BEFORE tone mapping so that the
 * threshold has physical meaning (brightness > 1.0 = actually bright).
 *
 * Design references:
 *   - GPU Gems 3, Chapter 28
 *   - Real-Time Rendering 4th, §10.12 Bloom
 *   - NVIDIA separable convolution samples
 * ====================================================================
 */

#pragma once

#include "glm/glm.hpp"
#include <cuda_runtime.h>
#include <vector>

// ---------------------------------------------------------------------------
// Tunable constants
// ---------------------------------------------------------------------------
#define MAX_BLOOM_RADIUS 32       // max Gaussian blur radius
#define BLOOM_BLOCK_SIZE 256      // threads per 1D block for blur kernels

// ---------------------------------------------------------------------------
// computeGaussianWeights — host-side 1D Gaussian kernel generation
//
// Computes normalized 1D Gaussian weights: G(x) = exp(-x² / (2σ²))
// Weights are L1-normalised so they sum to 1.0 (energy-preserving blur).
//
// radius: half-width of the kernel (kernel size = 2*radius + 1)
// sigma:  standard deviation; typically radius/2 or radius/3
// ---------------------------------------------------------------------------
inline std::vector<float> computeGaussianWeights(int radius, float sigma)
{
    std::vector<float> weights(2 * radius + 1);
    float sum = 0.0f;
    for (int i = -radius; i <= radius; i++)
    {
        //Gaussian function: G(x) = exp(-x² / (2σ²))
        float w = expf(-static_cast<float>(i * i) / (2.0f * sigma * sigma));
        weights[i + radius] = w;
        sum += w;
    }
    // Normalize so sum = 1.0
    for (int i = 0; i < 2 * radius + 1; i++)
    {
        weights[i] /= sum;
    }
    return weights;
}

// ---------------------------------------------------------------------------
// thresholdExtract — Isolate bright pixels
//
// For each pixel, subtracts 'threshold' from the per-sample HDR average
// and clamps negative values to zero.  Operates per-channel so that
// coloured highlights retain their hue.
//
// Input:  g_dev.image (raw HDR accumulation)
// Output: bloomBufA (bright areas only)
// ---------------------------------------------------------------------------
__global__ void thresholdExtract(
    const glm::vec3* __restrict__ inputImage,
    glm::vec3*       __restrict__ outputImage,
    glm::ivec2 resolution,
    int    iter,
    float  threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int idx = y * resolution.x + x;

        // Average HDR samples
        glm::vec3 pix = inputImage[idx] / static_cast<float>(iter);

        // Per-channel threshold: keep only above-threshold radiance
        pix = glm::max(pix - glm::vec3(threshold), glm::vec3(0.0f));

        outputImage[idx] = pix;
    }
}

// ---------------------------------------------------------------------------
// blurHorizontal — Horizontal separable Gaussian pass
//
// Each block processes one row segment.  Shared memory holds the tile
// plus 'radius' halo pixels on each side to avoid out-of-bounds global
// memory reads during convolution.
//
// Dynamic shared memory layout:
//   [ halo_left (radius) | center (blockDim.x) | halo_right (radius) ]
//
// Launch config (1D block, 2D grid):
//   block:  (BLOOM_BLOCK_SIZE, 1, 1)
//   grid:   (ceil(width/BLOOM_BLOCK_SIZE), height, 1)
//   smem:   (BLOOM_BLOCK_SIZE + 2*radius) * sizeof(float3) bytes
// ---------------------------------------------------------------------------
__global__ void blurHorizontal(
    const glm::vec3* __restrict__ src,
    glm::vec3*       __restrict__ dst,
    int width, int height,
    const float* __restrict__ weights,
    int radius)
{
    // Dynamic shared memory: store as float3(3 float=12 bytes,R,G,B) 
    // for direct register access
    extern __shared__ float sharedMem[];
    float3* tile = reinterpret_cast<float3*>(sharedMem);

    int tid   = threadIdx.x;
    int row   = blockIdx.y;
    int col   = blockIdx.x * blockDim.x + tid;
    int blockStart = blockIdx.x * blockDim.x;

    // ---- Load center pixels into shared memory ----
    if (col < width && row < height)
    {
        glm::vec3 p = src[row * width + col];
        tile[radius + tid] = make_float3(p.x, p.y, p.z);
    }
    else
    {
        tile[radius + tid] = make_float3(0.0f, 0.0f, 0.0f);
    }

    // ---- Load left halo (first 'radius' threads) ----
    if (tid < radius)
    {
        int leftCol = blockStart - radius + tid;
        if (leftCol >= 0 && row < height)
        {
            glm::vec3 p = src[row * width + leftCol];
            tile[tid] = make_float3(p.x, p.y, p.z);
        }
        else
        {
            tile[tid] = make_float3(0.0f, 0.0f, 0.0f);
        }
    }

    // ---- Load right halo (last 'radius' threads) ----
    if (tid >= blockDim.x - radius)
    {
        int offset  = tid - (blockDim.x - radius);    // 0 .. radius-1
        int rightCol = blockStart + blockDim.x + offset;
        if (rightCol < width && row < height)
        {
            glm::vec3 p = src[row * width + rightCol];
            tile[blockDim.x + radius + offset] = make_float3(p.x, p.y, p.z);
        }
        else
        {
            tile[blockDim.x + radius + offset] = make_float3(0.0f, 0.0f, 0.0f);
        }
    }

    __syncthreads();

    // ---- Convolve along x-axis ----
    if (col < width && row < height)
    {
        float3 sum = make_float3(0.0f, 0.0f, 0.0f);
        for (int k = -radius; k <= radius; k++)
        {
            float  w = weights[k + radius];
            float3 p = tile[radius + tid + k];
            sum.x += w * p.x;
            sum.y += w * p.y;
            sum.z += w * p.z;
        }
        dst[row * width + col] = glm::vec3(sum.x, sum.y, sum.z);
    }
}

// ---------------------------------------------------------------------------
// blurVertical — Vertical separable Gaussian pass
//
// Each block processes one column segment.  Shared memory holds the tile
// plus 'radius' halo pixels above and below.
//
// Dynamic shared memory layout:
//   [ halo_top (radius) | center (blockDim.x) | halo_bottom (radius) ]
//
// Launch config (1D block, 2D grid):
//   block:  (BLOOM_BLOCK_SIZE, 1, 1)
//   grid:   (ceil(height/BLOOM_BLOCK_SIZE), width, 1)
//   smem:   (BLOOM_BLOCK_SIZE + 2*radius) * sizeof(float3) bytes
//
// Note: grid is transposed relative to blurHorizontal
//       (gridDim.x covers row segments, gridDim.y = column index).
// ---------------------------------------------------------------------------
__global__ void blurVertical(
    const glm::vec3* __restrict__ src,
    glm::vec3*       __restrict__ dst,
    int width, int height,
    const float* __restrict__ weights,
    int radius)
{
    extern __shared__ float sharedMem[];
    float3* tile = reinterpret_cast<float3*>(sharedMem);

    int tid   = threadIdx.x;
    int col   = blockIdx.y;                       // global column (fixed per block)
    int row   = blockIdx.x * blockDim.x + tid;    // global row
    int blockStart = blockIdx.x * blockDim.x;     // first row in this block

    // ---- Load center pixels ----
    if (row < height && col < width)
    {
        glm::vec3 p = src[row * width + col];
        tile[radius + tid] = make_float3(p.x, p.y, p.z);
    }
    else
    {
        tile[radius + tid] = make_float3(0.0f, 0.0f, 0.0f);
    }

    // ---- Load top halo (first 'radius' threads) ----
    if (tid < radius)
    {
        int topRow = blockStart - radius + tid;
        if (topRow >= 0 && col < width)
        {
            glm::vec3 p = src[topRow * width + col];
            tile[tid] = make_float3(p.x, p.y, p.z);
        }
        else
        {
            tile[tid] = make_float3(0.0f, 0.0f, 0.0f);
        }
    }

    // ---- Load bottom halo (last 'radius' threads) ----
    if (tid >= blockDim.x - radius)
    {
        int offset = tid - (blockDim.x - radius);
        int botRow = blockStart + blockDim.x + offset;
        if (botRow < height && col < width)
        {
            glm::vec3 p = src[botRow * width + col];
            tile[blockDim.x + radius + offset] = make_float3(p.x, p.y, p.z);
        }
        else
        {
            tile[blockDim.x + radius + offset] = make_float3(0.0f, 0.0f, 0.0f);
        }
    }

    __syncthreads();

    // ---- Convolve along y-axis ----
    if (row < height && col < width)
    {
        float3 sum = make_float3(0.0f, 0.0f, 0.0f);
        for (int k = -radius; k <= radius; k++)
        {
            float  w = weights[k + radius];
            float3 p = tile[radius + tid + k];
            sum.x += w * p.x;
            sum.y += w * p.y;
            sum.z += w * p.z;
        }
        dst[row * width + col] = glm::vec3(sum.x, sum.y, sum.z);
    }
}
