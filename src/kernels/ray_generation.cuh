#pragma once

// ====================================================================
// Primary Ray Generation Kernel
//
// Generates the initial ray for every pixel each iteration.
// Supports anti-aliasing (sub-pixel jitter) and thin-lens depth of field.
// ====================================================================

#include "sceneStructs.h"
#include "rng/rng.h"
#include "intersection/intersections.h"   // concentricSampleDisk

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, RngMode rngMode);

