#pragma once

// ====================================================================
// Accumulation & Display Kernels
//
// Contains the terminal stage of each iteration:
//   sendImageToPBO       — write LDR tonemapped pixels to OpenGL PBO
//   finalGather          — accumulate terminated path colors into HDR image
//   gatherTerminatedPaths— bank terminated path colors BEFORE compaction
//                          discards their PathSegment entries.
// ====================================================================

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Writes the accumulated HDR image to the OpenGL pixel buffer for display.
 * Tone-mapping (ACES + sRGB) is applied in a separate pass before this;
 * this kernel treats `image` as already-LDR [0,1] data, dividing by `iter`
 * to compute the per-sample average.
 */
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

/**
 * Accumulate terminated path colours into the HDR accumulation buffer.
 *
 * Only paths whose remainingBounces <= 0 are gathered — these represent
 * rays that either hit a light (carrying the final radiance for this sample)
 * or exhausted all bounces (colour already set to 0 by shadeMaterial).
 *
 * When compaction is enabled, this kernel is called BEFORE each compaction
 * pass to bank terminated paths before their PathSegment entries are
 * discarded.  When compaction is disabled, it is called once after the
 * bounce loop as a catch-all.
 *
 * NOTE: The (1/eta^2) radiance scaling was removed because it caused energy
 * loss in the glass furnace test.  See the full explanation below.
 *
 * CORRECTNESS NOTE:
 *   After shadeMaterial, paths that hit a light carry their final radiance.
 *   If compaction runs without this gather, those contributions are dropped
 *   and the accumulation buffer stays black.
 *
 *   Active paths (remainingBounces > 0) are skipped here — they continue
 *   bouncing and contribute when they eventually terminate.
 *
 * ENERGY NOTE (1/eta^2):
 *   The former (1/eta^2) radiance scaling was removed because it caused
 *   energy loss in the glass furnace test.  pathEta is only updated when
 *   refraction occurs; it is NOT updated on Fresnel reflection or TIR.
 *   When a path exhausts its bounce budget while still inside glass (e.g.
 *   after TIR), pathEta retains the glass IOR (1.5) and the 1/eta^2 factor
 *   (~0.44) incorrectly darkened those contributions, making glass balls
 *   appear gray instead of invisible in a uniform-white furnace.
 *
 *   For paths that exit the glass and then hit furnace walls the energy is
 *   already correctly accounted for by the Fresnel Russian-roulette weights
 *   in scatterRay; no additional correction is needed here.
 */
__global__ void gatherTerminatedPaths(int nPaths, glm::vec3* image, PathSegment* paths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        const PathSegment& path = paths[index];
        if (path.remainingBounces <= 0)
        {
            image[path.pixelIndex] += path.color;
        }
    }
}
