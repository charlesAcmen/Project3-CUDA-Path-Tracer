#pragma once

// ====================================================================
// Primary Ray Generation Kernel
//
// Generates the initial ray for every pixel each iteration.
// Supports anti-aliasing (sub-pixel jitter) and thin-lens depth of field.
// ====================================================================

#include "sceneStructs.h"
#include "rng/rng.h"
#include "intersections.h"  // concentricSampleDisk

/**
 * Generate initial PathSegments with camera rays through each pixel.
 *
 * Each ray carries:
 *   - origin/direction  (pinhole or thin-lens)
 *   - colour = white     (identity for multiplicative attenuation)
 *   - pixelIndex         (target accumulation pixel)
 *   - remainingBounces   = traceDepth
 *
 * Antialiasing:    sub-pixel jitter via RNG dim 0–1
 * Depth of field:  thin-lens ray perturbation via RNG dim 2–3
 * Motion blur:     (not yet implemented — jitter ray "in time")
 */
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, int rngMode)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // RNG state shared across all primary-ray sampling.
        // Halton mode:  next(dim) selects a distinct prime base per dimension,
        //               so sequential calls with dim 0..3 produce independent
        //               values from the same Halton point — identical to using
        //               four separate RngState objects with the same index.
        // LCG mode:     sequential draws advance the engine, producing different
        //               values per call (unlike separate engines with the same
        //               seed, which would produce identical first draws).
        //   dim 0 (prime 2)  = AA jitter x
        //   dim 1 (prime 3)  = AA jitter y
        //   dim 2 (prime 5)  = lens aperture u  (only when DoF is active)
        //   dim 3 (prime 7)  = lens aperture v  (only when DoF is active)
        RngState rng = makeRngState(iter, index, 0, (RngMode)rngMode);

        // Anti-aliasing: stochastic sub-pixel jitter
        float jitterX = rng.next(0) - 0.5f;
        float jitterY = rng.next(1) - 0.5f;

        // Pinhole ray direction (centre-of-lens ray, undeflected)
        glm::vec3 pinholeDir = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up    * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f));

        if (cam.lensRadius > 0.0f) {
            // ---- Thin-lens depth of field ----
            // 1. Intersect pinhole ray with the focal plane
            float cosTheta = glm::dot(pinholeDir, cam.view);
            float ft = (cosTheta > EPSILON) ? (cam.focalDistance / cosTheta)
                                            : cam.focalDistance;
            glm::vec3 pFocus = cam.position + ft * pinholeDir;

            // 2. Sample a point on the lens aperture via concentric disk mapping
            float lensU = rng.next(2);  // dim 2 (prime 5): aperture u
            float lensV = rng.next(3);  // dim 3 (prime 7): aperture v
            float dx, dy;
            concentricSampleDisk(lensU, lensV, dx, dy);

            // 3. Offset ray origin within the aperture
            glm::vec3 lensOffset = cam.lensRadius * (dx * cam.right + dy * cam.up);
            segment.ray.origin = cam.position + lensOffset;

            // 4. Aim ray at the focal-plane point — all rays for this pixel converge there
            segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
        } else {
            // Pinhole camera (default, lensRadius == 0)
            segment.ray.origin = cam.position;
            segment.ray.direction = pinholeDir;
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}
