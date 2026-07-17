#pragma once

// ====================================================================
// Shading Kernel
//
// BSDF evaluation and ray scattering for each active path.
// Material sorting should be applied BEFORE this kernel to group
// same-material paths together, reducing warp divergence.
// ====================================================================

#include "sceneStructs.h"
#include "scene.h"          // ShadingConfig
#include "interactions/interactions.h"   // scatterRay
#include "rng/rng.h"
#include "constants.h"

/**
 * Russian roulette — probabilistically terminate low-throughput paths
 * without introducing bias.
 *
 * Survival probability p = max(R,G,B) clamped to [RR_P_MIN, RR_P_MAX].
 *   - max component gives conservative survival (fewer fireflies).
 *   - RR_P_MIN prevents extreme compensation (max 1/0.2 = 5×).
 *   - RR_P_MAX = 1.0 means full-throughput paths always survive.
 *
 * Unbiased: survivors have color /= p (compensation factor).
 * Terminated paths keep their (zero) color — gatherTerminatedPaths
 * collects it during the next compaction pass.
 *
 * @return  true if the path should be terminated.
 */
__device__ bool russianRouletteTerminate(
    glm::vec3& color,
    int remainingBounces,
    int traceDepth,
    int rrMinBounces,
    RngState& rng)
{
    // Only applies after rrMinBounces guaranteed bounces.
    // scatterRay already decremented remainingBounces, so the check
    // "remainingBounces >= traceDepth - rrMinBounces" correctly
    // protects the first rrMinBounces iterations.
    if (remainingBounces <= 0 ||
        remainingBounces >= traceDepth - rrMinBounces)
    {
        return false;
    }

    float p = fmaxf(fmaxf(color.r, color.g), color.b);
    p = fminf(fmaxf(p, RR_P_MIN), RR_P_MAX);

    if (rng.next(9) < p)  // dim 9 (prime 29): RR
    {
        color /= p;
        return false;  // survived
    }
    return true; // terminated
}

/**
 * BSDF evaluation and path-scattering kernel.
 *
 * For each active path:
 *   - Light source hit  → accumulate emission, terminate path.
 *   - Surface hit       → scatter ray according to material BSDF
 *                         (diffuse, glossy, specular, refractive).
 *   - Miss              → terminate with background colour (black).
 *
 * PERFORMANCE NOTES
 *   This kernel suffers from severe warp divergence when adjacent threads
 *   hit different materials (different if/else branches serialise within
 *   each warp).  Sorting paths by materialId before launching this kernel
 *   groups same-material paths together, dramatically reducing divergence.
 *
 *   Register pressure is high — ShadeableIntersection + PathSegment +
 *   Material + RNG state per thread.  High register count lowers SM
 *   occupancy.  Switching from AoS to SoA layout would reduce this.
 *
 *   Material array access is uncoalesced when materialId varies across
 *   threads in a warp — material sorting also mitigates this.
 */
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    ShadingConfig config)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment& pathSegment = pathSegments[idx];

        // Guard: skip paths already terminated in a prior bounce.
        // Without this guard, a path that hit an emissive surface would
        // re-intersect the same geometry on every remaining bounce and
        // accumulate emission repeatedly, blowing out the image.
        if (pathSegment.remainingBounces <= 0)
        {
            return;
        }

        ShadeableIntersection intersection = shadeableIntersections[idx];

        if (intersection.t > 0.0f)
        {
            // RNG for this bounce.
            //   dim 4 (prime 11) = diffuse hemisphere theta
            //   dim 5 (prime 13) = diffuse hemisphere phi
            //   dim 6 (prime 17) = specular lobe theta
            //   dim 7 (prime 19) = specular lobe phi
            //   dim 8 (prime 23) = Fresnel roulette
            //   dim 9 (prime 29) = Russian roulette
            int bounceNum = config.traceDepth - pathSegment.remainingBounces;
            RngState rngScatter = makeRngState(iter, pathSegment.pixelIndex,
                bounceNum * MAX_DRAWS_PER_BOUNCE, (RngMode)config.rngMode);

            Material material = materials[intersection.materialId];

            glm::vec3 intersectionPoint = getExactPointOnRay(pathSegment.ray, intersection.t);

            // Debug overlay: first-bounce hits on the focal plane in green.
            if (config.debug.showDOFOverlay && pathSegment.remainingBounces == config.traceDepth) {
                float hitDist = glm::dot(intersectionPoint - config.cam.position, config.cam.view);
                float focalErr = fabsf(hitDist - config.cam.focalDistance);
                if (focalErr < config.debug.focalTolerance) {
                    pathSegment.color = glm::vec3(0.0f, 1.0f, 0.0f);
                    pathSegment.remainingBounces = 0;
                    return;
                }
            }

            if (material.emittance > 0.0f)
            {
                // Light source hit: accumulate contribution and terminate
                pathSegment.color *= (material.color * material.emittance);
                pathSegment.remainingBounces = 0;
            }
            else
            {
                // Surface hit: scatter the ray according to BSDF
                scatterRay(pathSegment, intersectionPoint, intersection.surfaceNormal, material, rngScatter, config.fresnelMode);

                // Russian roulette: terminate low-throughput paths after
                // the guaranteed minimum bounce count.
                if (russianRouletteTerminate(pathSegment.color,
                    pathSegment.remainingBounces, config.traceDepth, config.rrMinBounces, rngScatter))
                {
                    pathSegment.remainingBounces = 0;
                }

                // Terminated without hitting a light → zero contribution
                if (pathSegment.remainingBounces <= 0)
                {
                    pathSegment.color = glm::vec3(0.0f);
                }
            }
        }
        else
        {
            // No intersection: background (black)
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
        }
    }
}
