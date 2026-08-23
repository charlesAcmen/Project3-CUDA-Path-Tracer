#include "shading.cuh"
#include "intersection/triangle.h"  // interpolateTriangleAttributes

// ====================================================================
// Shading Kernel Implementation
// ====================================================================

static __device__ __forceinline__ void writePathActivity(
    unsigned char* activityFlags,
    int idx,
    const PathSegment& pathSegment)
{
    if (activityFlags != nullptr)
    {
        activityFlags[idx] = pathSegment.remainingBounces > 0 ? 1 : 0;
    }
}

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
    ShadingConfig config,
    ShadingSceneView scene,
    ShadingBufferView buffers)
{
    const HitRecord* __restrict__ hitRecords = buffers.hitRecords;
    PathSegment* __restrict__ pathSegments = buffers.pathSegments;
    unsigned char* __restrict__ pathActivityFlags = buffers.pathActivityFlags;
    const Material* __restrict__ materials = scene.materials;
    const TrianglePos* __restrict__ trianglePositions = scene.trianglePositions;
    const TriangleAttr* __restrict__ triangleAttrs = scene.triangleAttrs;
    const Surface* __restrict__ surfaces = scene.surfaces;
    const SurfaceBinding* __restrict__ surfaceBindings = scene.surfaceBindings;

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
            writePathActivity(pathActivityFlags, idx, pathSegment);
            return;
        }

        const HitRecord& hit = hitRecords[idx];

        if (hit.t > 0.0f && hit.triangleIndex >= 0)
        {
            int bounceNum = config.traceDepth - pathSegment.remainingBounces;
            RngState rngScatter = makeRngState(iter, pathSegment.pixelIndex,
                bounceNum * MAX_DRAWS_PER_BOUNCE, config.rngMode);

            glm::vec3 intersectionPoint = getExactPointOnRay(pathSegment.ray, hit.t);

            // Debug overlay: first-bounce hits on the focal plane in green.
            if (config.debug.showDOFOverlay && pathSegment.remainingBounces == config.traceDepth) {
                handleDebugDOFOverlay(pathSegment, intersectionPoint, config);
                writePathActivity(pathActivityFlags, idx, pathSegment);
                return;
            }

            // Expand the complete shading state only for this final closest
            // triangle.  During BVH traversal, all other candidate hits carry
            // only t/u/v and are therefore unable to trigger these attribute
            // loads and interpolations.  The split layout keeps positions out
            // of the normal/UV/color fetch and resolves the shared surface
            // binding only for this selected triangle.  The Surface table
            // combines this geom's material id with the source binding id,
            // removing the former per-triangle device array.
            const TrianglePos& trianglePos = trianglePositions[hit.triangleIndex];
            const TriangleAttr& triangleAttr = triangleAttrs[hit.triangleIndex];
            const Surface& surfaceRef = surfaces[triangleAttr.surfaceId];
            const Material& material = materials[surfaceRef.materialId];
            // Device binding slot 0 is the default empty binding.  Source
            // binding ids are therefore shifted by one, mapping -1 to 0.
            const SurfaceBinding* surface =
                &surfaceBindings[surfaceRef.surfaceBindingId + 1];
            ShadeableIntersection intersection{};
            intersection.t          = hit.t;
            intersection.surfaceFeatures = surfaceRef.features;
            interpolateTriangleAttributes(trianglePos, triangleAttr, hit.u, hit.v,
                                          (surfaceRef.features & SurfaceFeatureNormalMap) != 0,
                                          intersection.surfaceNormal,
                                          intersection.uv,
                                          intersection.tangent,
                                          intersection.vertexColor);
            intersection.surface = surface;

            if (material.emittance > 0.0f)
            {
                // Light source hit (JSON Emitting): Le = texture·factor·strength
                // (flat color when no emissive slot), scaled by the JSON emittance
                // knob.  Accumulate and terminate the path.
                pathSegment.accumulatedRadiance = pathSegment.throughput *
                    resolveEmissive(*intersection.surface, scene.textures,
                                    intersection.uv, material) *
                    material.emittance;
                pathSegment.remainingBounces = 0;
            }
            else
            {
                // ---- Auto-glow: additive emission ----
                // A non-zero emissive factor with no JSON emittance means the
                // model glows at its own glTF/OBJ-defined radiance.  Emission is ADDITIVE
                // (Lo = Le + ∫BRDF·Li), so accumulate to pathSegment.accumulatedRadiance
                // instead of directly writing to image.  The surface is still shaded
                // by its BSDF.  (Terminating here would turn a mostly-black
                // emissive map on a shaded surface black.)
                if (intersection.surface->emissiveFactor != glm::vec3(0.0f))
                {
                    pathSegment.accumulatedRadiance += pathSegment.throughput *
                        resolveEmissive(*intersection.surface, scene.textures,
                                       intersection.uv, material);
                }

                // ---- Indirect illumination (BSDF continuation ray) ----
                // Surface hit: scatter the ray according to the material BSDF.
                // The hit record carries the surface normal, UV and per-triangle
                // texture binding, so the diffuse branch can sample the texture
                // table; scatterRay derives the exact hit point from hit.t.
                scatterRay(pathSegment, intersection, material,
                    rngScatter, scene.textures);

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

        writePathActivity(pathActivityFlags, idx, pathSegment);
    }
}
