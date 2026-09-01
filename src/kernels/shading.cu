#include "shading.cuh"
#include "intersection/triangle.h"  // interpolateTriangleAttributes
#include "lighting/light_sampling.h"  // sampleLightTriangle

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

static __device__ bool normalizedTriangleNormal(
    const TrianglePos& triangle, glm::vec3& normal)
{
    normal = glm::cross(triangle.v1 - triangle.v0,
                        triangle.v2 - triangle.v0);
    const float length2 = glm::dot(normal, normal);
    if (!(length2 > 0.0f) || !isfinite(length2)) return false;
    normal *= glm::inversesqrt(length2);
    return isfinite(normal.x) && isfinite(normal.y) && isfinite(normal.z);
}

static __device__ __forceinline__ bool finiteVec3(const glm::vec3& value)
{
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

// Read-only inputs shared across NEE's sampling, visibility, and accumulation
// stages.  References keep this a stack-only view; no scene data is copied.
struct DirectLightingContext
{
    PathSegment& pathSegment;
    const ShadeableIntersection& receiver;
    const Material& receiverMaterial;
    const ResolvedBsdf& receiverBsdf;
    const ShadingSceneView& scene;
};

// A sampled emitter point in the exact area-measure representation produced by
// the alias-table selection and barycentric sampler.  It deliberately carries
// no derived PDF or MIS weight, so measure conversion remains centralized.
struct DirectLightSample
{
    LightTriangle light;
    const TrianglePos* triangle;
    const SurfaceBinding* binding;
    const Material* material;
    glm::vec3 point;
    glm::vec2 uv;
    glm::vec3 wi;
    float distanceSquared;
};

// Continuous-BSDF and geometric terms evaluated for one DirectLightSample.
// Keeping them separate from the sample makes it impossible to accidentally
// use the shading normal for either emitter cosine or ray-origin offsetting.
struct DirectLightEvaluation
{
    BsdfEvaluation bsdf;
    glm::vec3 lightNormal;
    float receiverCosine;
    float lightCosine;
};

// Per-hit orchestration state for the non-emissive and emissive paths.  This
// groups only values that are already live in shadeMaterial; it is not a GPU
// buffer and leaves all production data layouts untouched.
struct SurfaceShadingContext
{
    PathSegment& pathSegment;
    const HitRecord& hit;
    const ShadingConfig& config;
    const ShadingSceneView& scene;
    RngState& rng;
};

// Shared input for a BSDF-hit emitter contribution.  Both terminating JSON
// emitters and additive glTF/OBJ auto-glow use this exact recovery/MIS path.
struct EmissionHitContext
{
    PathSegment& pathSegment;
    const HitRecord& hit;
    const ShadeableIntersection& intersection;
    const Material& material;
    const ShadingSceneView& scene;
};

static __device__ void accumulateDirectLighting(
    PathSegment& pathSegment,
    const ShadeableIntersection& receiver,
    const Material& receiverMaterial,
    const ResolvedBsdf& receiverBsdf,
    const ShadingSceneView& scene,
    RngState& rng);

static __device__ float emissionHitMisWeight(
    const PathSegment& pathSegment,
    const HitRecord& hit,
    const glm::vec3& lightNormal,
    const Material& lightMaterial,
    const LightSamplingView& lights);

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
            intersection.triangleIndex = hit.triangleIndex;
            intersection.surfaceFeatures = surfaceRef.features;
            intersection.hasGeometricNormal = normalizedTriangleNormal(
                trianglePos, intersection.geometricNormal);
            if (!intersection.hasGeometricNormal)
            {
                pathSegment.remainingBounces = 0;
                writePathActivity(pathActivityFlags, idx, pathSegment);
                return;
            }
            intersection.rayOriginScale = triangleRayOriginScale(trianglePos);
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
                const glm::vec3 emittedDirection = -pathSegment.ray.direction;
                const glm::vec3 Le = evaluateEmittedRadiance(
                    *intersection.surface, scene.textures, intersection.uv, material,
                    intersection.geometricNormal, emittedDirection);
                pathSegment.accumulatedRadiance += pathSegment.throughput * Le *
                    emissionHitMisWeight(pathSegment, hit,
                                         intersection.geometricNormal,
                                         material, scene.lights);
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
                    const glm::vec3 emittedDirection = -pathSegment.ray.direction;
                    const glm::vec3 Le = evaluateEmittedRadiance(
                        *intersection.surface, scene.textures, intersection.uv, material,
                        intersection.geometricNormal, emittedDirection);
                    pathSegment.accumulatedRadiance += pathSegment.throughput * Le *
                        emissionHitMisWeight(pathSegment, hit,
                                             intersection.geometricNormal,
                                             material, scene.lights);
                }

                const ResolvedBsdf resolvedBsdf = resolveBsdf(
                    intersection, material, scene.textures,
                    pathSegment.ray.direction);
                accumulateDirectLighting(pathSegment, intersection, material,
                                         resolvedBsdf, scene, rngScatter);

                // The resolved state carries this hit's normal-map and texture
                // inputs through the continuation, while scatterRay derives the
                // exact hit point from hit.t.
                scatterRay(pathSegment, intersection, material,
                    resolvedBsdf, rngScatter);

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

static __device__ float powerHeuristic(float a, float b)
{
    if (!(a > 0.0f)) return 0.0f;
    const float scale = fmaxf(a, b);
    if (!(scale > 0.0f) || !isfinite(scale)) return 0.0f;
    a /= scale;
    b /= scale;
    return (a * a) / (a * a + b * b);
}

static __device__ float lightPdfOmega(
    const LightSamplingView& lights, int triangleIndex,
    float distanceSquared, float lightCosine)
{
    if (lights.lightIndexByTriangle == nullptr || !(distanceSquared > 0.0f) ||
        !(lightCosine > 0.0f)) return 0.0f;
    const int lightIndex = lights.lightIndexByTriangle[triangleIndex];
    if (lightIndex < 0 || lightIndex >= lights.count) return 0.0f;
    const LightTriangle light = lights.triangles[lightIndex];
    if (!(light.area > 0.0f) || !(light.selectPmf > 0.0f)) return 0.0f;
    return light.selectPmf * distanceSquared / (light.area * lightCosine);
}

static __device__ void accumulateDirectLighting(
    PathSegment& pathSegment,
    const ShadeableIntersection& receiver,
    const Material& receiverMaterial,
    const ResolvedBsdf& receiverBsdf,
    const ShadingSceneView& scene,
    RngState& rng)
{
    const int lightIndex = sampleLightTriangle(scene.lights,
        rng.next(HaltonDim::LightSelection));
    if (lightIndex < 0) return;

    const LightTriangle light = scene.lights.triangles[lightIndex];
    // A zero-thickness triangle has zero solid angle to itself.  Sampling the
    // same primitive would therefore create a near-zero-distance, numerically
    // unstable shadow segment rather than a physical lighting path.
    if (light.triangleIndex == receiver.triangleIndex) return;
    const TrianglePos& lightTriangle = scene.trianglePositions[light.triangleIndex];
    const TriangleAttr& lightAttr = scene.triangleAttrs[light.triangleIndex];
    const Surface& lightSurface = scene.surfaces[lightAttr.surfaceId];
    const Material& lightMaterial = scene.materials[lightSurface.materialId];
    const SurfaceBinding& lightBinding =
        scene.surfaceBindings[lightSurface.surfaceBindingId + 1];

    const float s = sqrtf(rng.next(HaltonDim::LightSampleU));
    const float b0 = 1.0f - s;
    const float b1 = s * (1.0f - rng.next(HaltonDim::LightSampleV));
    const float b2 = 1.0f - b0 - b1;
    const glm::vec3 lightPoint = b0 * lightTriangle.v0 + b1 * lightTriangle.v1 + b2 * lightTriangle.v2;
    const glm::vec2 lightUv = b0 * lightAttr.uv0 + b1 * lightAttr.uv1 + b2 * lightAttr.uv2;
    const glm::vec3 delta = lightPoint - getExactPointOnRay(pathSegment.ray, receiver.t);
    const float distanceSquared = glm::dot(delta, delta);
    if (!(distanceSquared > 0.0f) || !isfinite(distanceSquared)) return;
    const glm::vec3 wi = delta * glm::inversesqrt(distanceSquared);
    const BsdfEvaluation bsdf = evaluateBsdf(receiverBsdf, receiverMaterial,
        pathSegment.ray.direction, wi);
    const float receiverCosine = glm::max(glm::dot(bsdf.shadingNormal, wi), 0.0f);
    if (bsdf.isDelta || !(receiverCosine > 0.0f) || !(bsdf.pdfOmega > 0.0f)) return;

    const glm::vec3& geometricNormal = receiver.geometricNormal;
    glm::vec3 lightNormal;
    if (!normalizedTriangleNormal(lightTriangle, lightNormal)) return;
    const float lightCosine = emissionCosine(lightMaterial, lightNormal, -wi);
    if (!(lightCosine > 0.0f)) return;
    const glm::vec3 receiverPoint = getExactPointOnRay(pathSegment.ray, receiver.t);
    Ray shadowRay = spawnRayFromSurface(receiverPoint, geometricNormal, wi,
                                        receiver.rayOriginScale);
    const float maxT = nextafterf(glm::dot(lightPoint - shadowRay.origin, wi), 0.0f);
    // Any-hit returns true when an occluder is found before the sampled
    // emitter.  A visible light sample is therefore the false case; the
    // previous negation accidentally accumulated blocked samples instead.
    if (!(maxT > RAY_EPSILON) || traverseBvhAnyHit(
        shadowRay, scene.bvhNodes, scene.trianglePositions, maxT,
        receiver.triangleIndex)) return;

    const float pLight = lightPdfOmega(scene.lights, light.triangleIndex,
                                       distanceSquared, lightCosine);
    if (!(pLight > 0.0f) || !isfinite(pLight)) return;
    const glm::vec3 Le = evaluateEmittedRadiance(lightBinding, scene.textures,
                                                   lightUv, lightMaterial,
                                                   lightNormal, -wi);
    const float areaPdf = light.selectPmf / light.area;
    const glm::vec3 contribution = pathSegment.throughput * Le * bsdf.value *
        (receiverCosine * lightCosine / (distanceSquared * areaPdf)) *
        powerHeuristic(pLight, bsdf.pdfOmega);
    // The BSDF owns validation of its value/PDF.  This boundary owns the
    // product of path throughput, emission, geometry and MIS, where a finite
    // factor combination can still overflow or form Inf*0.
    if (finiteVec3(contribution))
    {
        pathSegment.accumulatedRadiance += contribution;
    }
}

static __device__ float emissionHitMisWeight(
    const PathSegment& pathSegment,
    const HitRecord& hit,
    const glm::vec3& lightNormal,
    const Material& lightMaterial,
    const LightSamplingView& lights)
{
    // Primary rays and delta events have no competing continuous BSDF PDF.
    if (!(pathSegment.previousBsdfPdfOmega > 0.0f)) return 1.0f;
    const float lightCosine = emissionCosine(
        lightMaterial, lightNormal, -pathSegment.ray.direction);
    if (!(lightCosine > 0.0f)) return 1.0f;
    const float pLight = lightPdfOmega(lights, hit.triangleIndex,
                                       hit.t * hit.t, lightCosine);
    return (pLight > 0.0f && isfinite(pLight))
        ? powerHeuristic(pathSegment.previousBsdfPdfOmega, pLight) : 1.0f;
}
