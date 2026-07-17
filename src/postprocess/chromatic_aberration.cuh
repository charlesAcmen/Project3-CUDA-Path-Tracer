/*
 * chromatic_aberration.cuh — Chromatic Aberration Post-Processing Effect
 * ======================================================================
 * Simulates the wavelength-dependent refractive index of camera lenses,
 * where different colour wavelengths focus at slightly different image
 * plane locations.  The result is colour fringing at high-contrast edges,
 * most noticeable near the image periphery.
 *
 * Algorithm (radial chromatic aberration):
 *   1. For each pixel, compute its vector from image centre.
 *   2. Red channel is sampled slightly OUTWARD along the radial direction.
 *   3. Green channel remains at the original position (reference).
 *   4. Blue channel is sampled slightly INWARD along the radial direction.
 *
 * Bilinear interpolation is used for all off-centre samples to avoid
 * aliasing from sub-pixel shifts.
 *
 * Pipeline position: applied AFTER ACES filmic tone-mapping + sRGB gamma,
 *                    in display-ready [0,1] colour space.
 *
 * References:
 *   - Playdead's "Inside" GDC 2016 Post-Processing presentation
 *   - Bevy Engine radial chromatic aberration WGSL shader
 *   - 3D Game Shaders For Beginners — chromatic aberration
 * ======================================================================
 */

#pragma once

#include "glm/glm.hpp"
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// sampleBilinear — Edge-clamped bilinear texture fetch
//
// Returns an interpolated colour at floating-point pixel coordinate (fx, fy)
// from a row-major glm::vec3 device array.
//
// Integer coordinates correspond to pixel centres:
//   fx = 0.0  → leftmost pixel column
//   fx = W-1  → rightmost pixel column
//
// Edge clamping is applied so that out-of-bounds coordinates read the
// nearest valid texel.
// ---------------------------------------------------------------------------
__device__ inline glm::vec3 sampleBilinear(
    const glm::vec3* __restrict__ src,
    int width, int height,
    float fx, float fy)
{
    // Clamp to valid pixel range [0, width-1] x [0, height-1]
    fx = fminf(fmaxf(fx, 0.0f), static_cast<float>(width  - 1));
    fy = fminf(fmaxf(fy, 0.0f), static_cast<float>(height - 1));

    // Integer coordinates of the top-left neighbour
    int ix = static_cast<int>(fx);
    int iy = static_cast<int>(fy);

    // Fractional weights for interpolation
    float fracX = fx - static_cast<float>(ix);
    float fracY = fy - static_cast<float>(iy);

    // Ensure ix, iy never exceed the last valid index after clamping
    // (the clamping above guarantees this, but guard against fp edge cases)
    if (ix >= width  - 1) { ix = width  - 2; fracX = 1.0f; }
    if (iy >= height - 1) { iy = height - 2; fracY = 1.0f; }
    if (ix < 0) { ix = 0; fracX = 0.0f; }
    if (iy < 0) { iy = 0; fracY = 0.0f; }

    // Four corner texels
    glm::vec3 c00 = src[iy * width + ix];
    glm::vec3 c10 = src[iy * width + ix + 1];
    glm::vec3 c01 = src[(iy + 1) * width + ix];
    glm::vec3 c11 = src[(iy + 1) * width + ix + 1];

    // Bilinear interpolation: first along X, then along Y
    glm::vec3 top = glm::mix(c00, c10, fracX);
    glm::vec3 bot = glm::mix(c01, c11, fracX);
    return glm::mix(top, bot, fracY);
}

// ---------------------------------------------------------------------------
// chromaticAberrationKernel — Radial chromatic aberration
//
// For each pixel, the red channel is shifted outward from centre and the
// blue channel is shifted inward.  Green stays in place as the reference
// luminance channel, minimising overall brightness shift.
//
// Input:  imageDisplay (tone-mapped sRGB [0,1])
// Output: bloomBufB    (temporary scratch buffer, same format)
//
// Launch config: 2D grid matching the image dimensions:
//   block:  (8, 8)
//   grid:   (ceil(res.x/8), ceil(res.y/8))
//
// @param srcImage   Source image buffer (read-only, sRGB [0,1])
// @param dstImage   Destination image buffer (written)
// @param resolution Image dimensions in pixels
// @param intensity  Radial shift magnitude in UV coordinates.
//                   Typical range: 0.001–0.01.  Higher = stronger fringing.
// ---------------------------------------------------------------------------
__global__ void chromaticAberrationKernel(
    const glm::vec3* __restrict__ srcImage,
    glm::vec3*       __restrict__ dstImage,
    glm::ivec2 resolution,
    float intensity)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= resolution.x || y >= resolution.y)
        return;

    int   idx  = y * resolution.x + x;
    float cx   = static_cast<float>(resolution.x) * 0.5f;
    float cy   = static_cast<float>(resolution.y) * 0.5f;

    float dx   = static_cast<float>(x) - cx;
    float dy   = static_cast<float>(y) - cy;
    float dist = sqrtf(dx * dx + dy * dy);

    // Near the image centre the shift is negligible and the direction
    // normalisation would amplify floating-point noise.  Pass through
    // unchanged for pixels within half a pixel of the optical centre.
    if (dist < 0.5f)
    {
        dstImage[idx] = srcImage[idx];
        return;
    }

    // Normalise to unit radial direction
    float invDist = 1.0f / dist;
    float dirX    = dx * invDist;
    float dirY    = dy * invDist;

    // Shift magnitude grows linearly with distance from centre:
    //   shift = intensity * dist
    // Result is zero shift at centre, maximum shift at corners.
    float shift = intensity * dist;

    // Red shifts outward (away from centre)
    float rX = static_cast<float>(x) + dirX * shift;
    float rY = static_cast<float>(y) + dirY * shift;

    // Blue shifts inward (toward centre)
    float bX = static_cast<float>(x) - dirX * shift;
    float bY = static_cast<float>(y) - dirY * shift;

    // Sample two channels with bilinear interpolation
    glm::vec3 rSample = sampleBilinear(srcImage, resolution.x, resolution.y, rX, rY);
    glm::vec3 bSample = sampleBilinear(srcImage, resolution.x, resolution.y, bX, bY);

    // Green channel stays at the original pixel position
    glm::vec3 original = srcImage[idx];

    dstImage[idx] = glm::vec3(rSample.x, original.y, bSample.z);
}
