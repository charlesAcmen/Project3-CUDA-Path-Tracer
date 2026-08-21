#include "shading.cuh"

// ====================================================================
// Shading Kernel Implementation
// ====================================================================

__device__ bool russianRouletteTerminate(
    glm::vec3& throughput,
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

    float p = fmaxf(fmaxf(throughput.r, throughput.g), throughput.b);
    p = fminf(fmaxf(p, RR_P_MIN), RR_P_MAX);

    if (rng.next(HaltonDim::PathRR) < p)  // dim 9 (prime 29): RR
    {
        throughput /= p;
        return false;  // survived
    }
    return true; // terminated
}
// Helper: handle debug DOF overlay for focal plane visualization
static __device__ void handleDebugDOFOverlay(
    PathSegment& pathSegment,
    const glm::vec3& intersectionPoint,
    const ShadingConfig& config)
{
    float hitDist = glm::dot(intersectionPoint - config.cam.position, config.cam.view);
    float focalErr = fabsf(hitDist - config.cam.focalDistance);
    if (focalErr < config.debug.focalTolerance) {
        pathSegment.accumulatedRadiance = pathSegment.throughput;
        pathSegment.remainingBounces = 0;
    }
}
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    HitRecord* __restrict__ hitRecords,
    PathSegment* __restrict__ pathSegments,
    Material* __restrict__ materials,
    const TrianglePos* __restrict__ deviceTrianglePositions,
    const TriangleAttr* __restrict__ deviceTriangleAttrs,
    const Surface* __restrict__ deviceSurfaces,
    const SurfaceBinding* __restrict__ deviceSurfaceBindings,
    TextureTable textures,        // scene texture assets (pixels + slice table)
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

        const HitRecord& hit = hitRecords[idx];

        if (hit.t > 0.0f && hit.triangleIndex >= 0)
        {
            int bounceNum = config.traceDepth - pathSegment.remainingBounces;
            RngState rngScatter = makeRngState(iter, pathSegment.pixelIndex,
                bounceNum * MAX_DRAWS_PER_BOUNCE, config.rngMode);

            const Material& material = materials[hit.materialId];

            glm::vec3 intersectionPoint = getExactPointOnRay(pathSegment.ray, hit.t);

            // Debug overlay: first-bounce hits on the focal plane in green.
            if (config.debug.showDOFOverlay && pathSegment.remainingBounces == config.traceDepth) {
                handleDebugDOFOverlay(pathSegment, intersectionPoint, config);
                return;
            }

            if (material.emittance > 0.0f)
            {
                // Light source hit (JSON Emitting): Le = texture·factor·strength
                // (flat color when no emissive slot), scaled by the JSON emittance
                // knob.  Accumulate and terminate the path.
                pathSegment.accumulatedRadiance = pathSegment.throughput *
                    resolveEmissive(intersection.tex, textures,
                                    intersection.uv, material) *
                    material.emittance;
                pathSegment.remainingBounces = 0;
            }
            else
            {
                // ---- Auto-glow: additive emission ----
                // A bound emissive slot with no JSON emittance means the model
                // glows at its own glTF-defined radiance.  Emission is ADDITIVE
                // (Lo = Le + ∫BRDF·Li), so accumulate to pathSegment.accumulatedRadiance
                // instead of directly writing to image.  The surface is still shaded
                // by its BSDF.  (Terminating here would turn a mostly-black
                // emissive map on a shaded surface black.)
                if (intersection.tex.emissive >= 0)
                {
                    pathSegment.accumulatedRadiance += pathSegment.throughput *
                        resolveEmissive(intersection.tex, textures,
                                       intersection.uv, material);
                }

                // ---- Indirect illumination (BSDF continuation ray) ----
                // Surface hit: scatter the ray according to the material BSDF.
                // The hit record carries the surface normal, UV and per-triangle
                // texture binding, so the diffuse branch can sample the texture
                // table; scatterRay derives the exact hit point from hit.t.
                scatterRay(pathSegment, intersection, material,
                    rngScatter, textures);

                // ---- Russian roulette ----
                // Probabilistically terminate low-throughput paths after
                // the guaranteed minimum bounce count.
                if (russianRouletteTerminate(pathSegment.throughput,
                    pathSegment.remainingBounces, config.traceDepth,
                    config.rrMinBounces, rngScatter))
                {
                    pathSegment.remainingBounces = 0;
                }
            }
        }
        else
        {
            // No intersection: background (black), no radiance contribution
            pathSegment.remainingBounces = 0;
        }
    }
}
