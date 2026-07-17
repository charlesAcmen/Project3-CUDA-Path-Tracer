/*
 * vignette.cuh — Vignette Post-Processing Effect
 * ================================================================
 * Simulates the natural light falloff at the edges of a camera lens
 * (also known as "corner darkening" or "lens shading").  The effect
 * darkens image corners relative to the centre using a radial power
 * falloff.
 *
 * Algorithm:
 *   For each pixel, compute the normalised distance from image centre:
 *     ndist = dist / maxDist      (0 at centre, 1 at farthest corner)
 *     factor = 1 - intensity * pow(ndist, exponent)
 *     pixel *= factor
 *
 * The single-parameter exponent provides intuitive control over the
 * falloff shape:
 *   exponent < 1.0  → gradual, wide darkening
 *   exponent = 2.0  → classic "natural lens" falloff
 *   exponent > 4.0  → tight darkening confined to extreme corners
 *
 * Pipeline position: applied AFTER chromatic aberration (if enabled),
 *                    as the final post-processing step before display.
 *
 * The kernel supports both in-place (src == dst) and separate
 * source/destination buffers, allowing efficient chaining without
 * an extra copy pass.
 * ================================================================
 */

#pragma once

#include "glm/glm.hpp"
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// vignetteKernel — Radial vignette darkening
//
// Reads from srcImage, applies the vignette falloff, and writes the
// result to dstImage.  When srcImage == dstImage the operation is
// in-place (avoids an extra buffer).
//
// Launch config: 2D grid matching the image dimensions:
//   block:  (8, 8)
//   grid:   (ceil(res.x/8), ceil(res.y/8))
//
// @param srcImage   Source image buffer (read-only, sRGB [0,1])
// @param dstImage   Destination image buffer (written, may alias src)
// @param resolution Image dimensions in pixels
// @param intensity  Corner darkness  [0, 1].  0 = no effect, 1 = full black.
// @param exponent   Radial falloff power [0.5, 8.0].
//                   Controls how sharply the darkening transitions.
// ---------------------------------------------------------------------------
__global__ void vignetteKernel(
    const glm::vec3* __restrict__ srcImage,
    glm::vec3*       __restrict__ dstImage,
    glm::ivec2 resolution,
    float intensity,
    float exponent)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= resolution.x || y >= resolution.y)
        return;

    int idx = y * resolution.x + x;

    // Image centre in pixel coordinates
    float cx = static_cast<float>(resolution.x - 1) * 0.5f;
    float cy = static_cast<float>(resolution.y - 1) * 0.5f;

    // Maximum possible distance from centre (to a corner)
    float maxDist = sqrtf(cx * cx + cy * cy);

    // Radial distance for this pixel
    float dx   = static_cast<float>(x) - cx;
    float dy   = static_cast<float>(y) - cy;
    float dist = sqrtf(dx * dx + dy * dy);

    // Normalise to [0, 1]
    float ndist = fminf(dist / maxDist, 1.0f);

    // Vignette falloff factor
    float factor = 1.0f - intensity * powf(ndist, exponent);
    factor = fmaxf(factor, 0.0f);

    // Apply darkening
    glm::vec3 pix = srcImage[idx];
    dstImage[idx] = pix * factor;
}
