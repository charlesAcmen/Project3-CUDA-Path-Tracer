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
#include "lighting/light_sampling.h"  // sampleLightSource, testShadowRay

/**
 * Russian roulette — probabilistically terminate low-throughput paths
 * without introducing bias.
 *
 * Survival probability p = max(R,G,B) clamped to [RR_P_MIN, RR_P_MAX].
 *   - max component gives conservative survival (fewer fireflies).
 *   - RR_P_MIN prevents extreme compensation (max 1/0.2 = 5x).
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

    if (rng.next(HaltonDim::PathRR) < p)  // dim 9 (prime 29): RR
    {
        color /= p;
        return false;  // survived
    }
    return true; // terminated
}

/**
 * Evaluate direct lighting at a non-emissive surface hit using
 * next-event estimation (NEE — PBRTv4 §13.4).
 *
 * At each non-emissive surface hit, we explicitly sample a random point
 * on a randomly chosen emissive geometry and evaluate its contribution
 * through the surface BSDF.  This is unbiased and dramatically reduces
 * variance compared to waiting for a random BSDF bounce to hit a light.
 *
 * Energy conservation:
 *   The NEE estimator is  L = throughput * fr * Le * V * G / p_A
 *   where:
 *     fr = albedo / PI          (Lambertian diffuse BSDF)
 *     G  = |cosθ_r| * |cosθ_l| / r²   (geometry term + solid-angle Jacobian)
 *     p_A = 1/(numLights * area)      (combined area PDF)
 *
 *   Derivation (sampling the light surface in area measure):
 *     L = fr * Le * V * |cosθ_r| / p_ω
 *     p_ω = p_A * |cosθ_l| / r²     (area → solid-angle Jacobian)
 *     L = fr * Le * V * |cosθ_r| / (p_A * |cosθ_l| / r²)
 *     L = fr * Le * V * G / p_A     ✓   (unbiased, energy-conserving)
 *
 * The indirect path continues via scatterRay below with the ORIGINAL
 * throughput (not reduced).  Both contributions accumulate to the same
 * pixel and converge correctly.
 *
 * @param pathSegment       Current path state (throughput, pixelIndex)
 * @param intersectionPoint World-space hit point
 * @param surfaceNormal     Shading normal at the hit point
 * @param materialColor     Diffuse albedo for BSDF evaluation
 * @param config            Shading configuration (light lists, geoms, etc.)
 * @param rng               RNG state (consumes LightSelect, LightSurfaceU/V)
 * @param imageAccum        HDR accumulation buffer (atomic-add target)
 */
__device__ inline void evaluateDirectLighting(
    const PathSegment& pathSegment,
    const glm::vec3& intersectionPoint,
    const glm::vec3& surfaceNormal,
    const glm::vec3& materialColor,
    const ShadingConfig& config,
    RngState& rng,
    glm::vec3* imageAccum)
{
    // ---- 1. Sample a light source ----
    // Uniform selection among emissive geometries, then
    // uniform area sampling on the selected light's surface.
    glm::vec3 lightPos, lightNormal, Le;
    float lightPdf;  // combined PDF in area measure
    int lightGeomIdx;
    sampleLightSource(config.lightInfos, config.numLights,
        config.geoms, config.totalLightArea,
        rng,
        lightPos, lightNormal, lightPdf, Le,
        lightGeomIdx);

    // ---- 2. Direction from surface toward the light sample ----
    glm::vec3 wi = lightPos - intersectionPoint;
    float dist2 = glm::dot(wi, wi);
    float dist = sqrtf(dist2);
    wi /= dist;  // normalized

    // ---- 3. Geometry term ----
    // G = |cosθ_receiver| * |cosθ_light| / r²
    //
    // |cosθ_receiver| = max(0, dot(n_x, wi))    — Lambert's law
    // |cosθ_light|    = max(0, dot(n_y, -wi))   — light orientation
    //
    // The |cosθ_light| / r² factor is the Jacobian of the
    // area → solid-angle measure conversion.
    float cosReceiver = fmaxf(0.0f, glm::dot(surfaceNormal, wi));
    float cosLight    = fmaxf(0.0f, glm::dot(lightNormal, -wi));

    if (cosReceiver > EPSILON && cosLight > EPSILON)
    {
        // ---- 4. Visibility test (shadow ray) ----
        // Skip the light geometry itself so the ray entering
        // the light volume does not self-shadow.
        Ray shadowRay;
        shadowRay.origin    = intersectionPoint + surfaceNormal * EPSILON;
        shadowRay.direction = wi;
        bool visible = testShadowRay(shadowRay, dist,
            config.geoms, config.numGeoms, lightGeomIdx);

        if (visible)
        {
            // ---- 5. BSDF (Lambertian diffuse, energy-conserving) ----
            // fr = albedo / PI
            // ∫_H² fr * cosθ dω = albedo  ✓
            glm::vec3 bsdf = materialColor * (1.0f / PI);

            // ---- 6. Geometry term with inverse-square falloff ----
            float G = cosReceiver * cosLight / dist2;

            // ---- 7. Unbiased Monte Carlo estimator ----
            //
            // L = throughput * fr * Le * V * G / p_A
            //
            // Derivation (sampling in area measure):
            //   L = fr * Le * V * |cosθ_r| / p_ω
            //   p_ω = p_A * |cosθ_l| / r²     (solid-angle Jacobian)
            //   L = fr * Le * V * |cosθ_r| / (p_A * |cosθ_l| / r²)
            //   L = fr * Le * V * G / p_A     ✓
            //
            // where p_A = 1/(numLights * selectedLight.area)
            glm::vec3 directContrib = pathSegment.color * bsdf * Le * G / lightPdf;

            // ---- 8. Accumulate directly to pixel buffer ----
            // The direct contribution is final and does NOT
            // continue bouncing.  The indirect path continues
            // via scatterRay below with unaffected throughput.
            atomicAdd(&imageAccum[pathSegment.pixelIndex].x, directContrib.x);
            atomicAdd(&imageAccum[pathSegment.pixelIndex].y, directContrib.y);
            atomicAdd(&imageAccum[pathSegment.pixelIndex].z, directContrib.z);
        }
    }
}

/**
 * BSDF evaluation and path-scattering kernel.
 *
 * For each active path:
 *   - Light source hit  → accumulate emission, terminate path.
 *   - Surface hit       → evaluate direct lighting (NEE), then scatter
 *                         the ray according to the material BSDF (diffuse,
 *                         glossy, specular, refractive), then apply
 *                         Russian roulette for early termination.
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
    ShadingConfig config,
    glm::vec3* imageAccum)
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
                // ---- Direct Lighting (Next-Event Estimation) ----
                // Explicitly sample a random point on an emissive geometry
                // and evaluate the direct contribution through the surface
                // BSDF.  Only applies to diffuse surfaces with lights present.
                if (config.numLights > 0 && material.type == MaterialType::Diffuse)
                {
                    evaluateDirectLighting(
                        pathSegment, intersectionPoint,
                        intersection.surfaceNormal, material.color,
                        config, rngScatter, imageAccum);
                }

                // ---- Indirect illumination (BSDF continuation ray) ----
                // Surface hit: scatter the ray according to the material BSDF.
                scatterRay(pathSegment, intersectionPoint,
                    intersection.surfaceNormal, material,
                    rngScatter, config.fresnelMode);

                // ---- Russian roulette ----
                // Probabilistically terminate low-throughput paths after
                // the guaranteed minimum bounce count.
                if (russianRouletteTerminate(pathSegment.color,
                    pathSegment.remainingBounces, config.traceDepth,
                    config.rrMinBounces, rngScatter))
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
