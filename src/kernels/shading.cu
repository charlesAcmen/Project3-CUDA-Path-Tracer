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
    const DirectLightingContext& context,
    RngState& rng);

static __device__ float emissionHitMisWeight(
    const PathSegment& pathSegment,
    const HitRecord& hit,
    const glm::vec3& lightNormal,
    const Material& lightMaterial,
    const LightSamplingView& lights);

static __device__ bool resolveShadeableIntersection(
    const HitRecord& hit,
    const ShadingSceneView& scene,
    ShadeableIntersection& intersection,
    const Material*& material);

static __device__ void accumulateHitEmission(
    const EmissionHitContext& context);

static __device__ void shadeSurfaceHit(
    const SurfaceShadingContext& context);

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

            const SurfaceShadingContext surfaceShading{
                pathSegment, hit, config, scene, rngScatter
            };
            shadeSurfaceHit(surfaceShading);
        }
        else
        {
            // No intersection: background (black), no radiance contribution
            pathSegment.remainingBounces = 0;
        }

        writePathActivity(pathActivityFlags, idx, pathSegment);
    }
}

// Expands attributes only for the selected closest triangle.  Traversal stays
// position-only for every rejected candidate, exactly as before this split.
static __device__ bool resolveShadeableIntersection(
    const HitRecord& hit,
    const ShadingSceneView& scene,
    ShadeableIntersection& intersection,
    const Material*& material)
{
    const TrianglePos& trianglePos = scene.trianglePositions[hit.triangleIndex];
    const TriangleAttr& triangleAttr = scene.triangleAttrs[hit.triangleIndex];
    const Surface& surfaceRef = scene.surfaces[triangleAttr.surfaceId];
    material = &scene.materials[surfaceRef.materialId];
    // Device binding slot 0 is the default empty binding.  Source binding ids
    // are shifted by one, mapping -1 to 0.
    const SurfaceBinding* surface =
        &scene.surfaceBindings[surfaceRef.surfaceBindingId + 1];

    intersection.t = hit.t;
    intersection.triangleIndex = hit.triangleIndex;
    intersection.surfaceFeatures = surfaceRef.features;
    intersection.hasGeometricNormal = normalizedTriangleNormal(
        trianglePos, intersection.geometricNormal);
    if (!intersection.hasGeometricNormal) return false;

    intersection.rayOriginScale = triangleRayOriginScale(trianglePos);
    interpolateTriangleAttributes(trianglePos, triangleAttr, hit.u, hit.v,
                                  (surfaceRef.features & SurfaceFeatureNormalMap) != 0,
                                  intersection.surfaceNormal,
                                  intersection.uv,
                                  intersection.tangent,
                                  intersection.vertexColor);
    intersection.surface = surface;
    return true;
}

// Reconstructs emitted radiance for a path that reached an emitter through
// BSDF sampling.  With NEE enabled, the competing light PDF supplies the MIS
// weight; without it, BSDF sampling is the sole estimator and has weight one.
// The caller alone decides whether the hit terminates (JSON emitter) or
// continues (auto-glow).
static __device__ void accumulateHitEmission(
    const EmissionHitContext& context)
{
    const glm::vec3 emittedDirection = -context.pathSegment.ray.direction;
    const glm::vec3 Le = evaluateEmittedRadiance(
        *context.intersection.surface, context.scene.textures,
        context.intersection.uv, context.material,
        context.intersection.geometricNormal, emittedDirection);
    const float misWeight = context.directLightingEnabled
        ? emissionHitMisWeight(context.pathSegment, context.hit,
                                context.intersection.geometricNormal,
                                context.material, context.scene.lights)
        : 1.0f;
    context.pathSegment.accumulatedRadiance +=
        context.pathSegment.throughput * Le * misWeight;
}

static __device__ void shadeSurfaceHit(
    const SurfaceShadingContext& context)
{
    ShadeableIntersection intersection{};
    const Material* material = nullptr;
    if (!resolveShadeableIntersection(context.hit, context.scene, intersection, material))
    {
        context.pathSegment.remainingBounces = 0;
        return;
    }

    const EmissionHitContext emission{
        context.pathSegment, context.hit, intersection, *material, context.scene
    };
    if (material->emittance > 0.0f)
    {
        // JSON Emitting surfaces contribute and terminate.
        accumulateHitEmission(emission);
        context.pathSegment.remainingBounces = 0;
        return;
    }

    // A glTF/OBJ emissive factor is additive auto-glow: it contributes first,
    // then the same surface continues through its BSDF.
    if (intersection.surface->emissiveFactor != glm::vec3(0.0f))
    {
        accumulateHitEmission(emission);
    }

    const ResolvedBsdf resolvedBsdf = resolveBsdf(
        intersection, *material, context.scene.textures,
        context.pathSegment.ray.direction);
    const DirectLightingContext directLighting{
        context.pathSegment, intersection, *material, resolvedBsdf, context.scene
    };
    accumulateDirectLighting(directLighting, context.rng);

    // The resolved state carries this hit's normal-map and texture inputs
    // through continuation, while scatterRay derives the exact point from hit.t.
    scatterRay(context.pathSegment, intersection, *material,
        resolvedBsdf, context.rng);

    // Probabilistically terminate low-throughput paths after the guaranteed
    // minimum bounce count.  scatterRay has already decremented the count.
    if (russianRouletteTerminate(context.pathSegment.throughput,
        context.pathSegment.remainingBounces, context.config.traceDepth,
        context.config.rrMinBounces, context.rng))
    {
        context.pathSegment.remainingBounces = 0;
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

// Performs alias-table selection followed by the original square-root
// barycentric mapping.  Keep the self-light early return before U/V draws:
// the RNG stream is part of the renderer's deterministic sampling contract.
static __device__ __forceinline__ bool sampleDirectLight(
    const DirectLightingContext& context,
    int lightIndex,
    RngState& rng,
    DirectLightSample& sample)
{
    const ShadingSceneView& scene = context.scene;
    const LightTriangle light = scene.lights.triangles[lightIndex];
    // A zero-thickness triangle has zero solid angle to itself.  Sampling the
    // same primitive would therefore create a near-zero-distance, numerically
    // unstable shadow segment rather than a physical lighting path.
    if (light.triangleIndex == context.receiver.triangleIndex) return false;
    const TrianglePos& lightTriangle = scene.trianglePositions[light.triangleIndex];
    const TriangleAttr& lightAttr = scene.triangleAttrs[light.triangleIndex];
    const Surface& lightSurface = scene.surfaces[lightAttr.surfaceId];

    const float s = sqrtf(rng.next(HaltonDim::LightSampleU));
    const float b0 = 1.0f - s;
    const float b1 = s * (1.0f - rng.next(HaltonDim::LightSampleV));
    const float b2 = 1.0f - b0 - b1;
    const glm::vec3 lightPoint = b0 * lightTriangle.v0 + b1 * lightTriangle.v1 + b2 * lightTriangle.v2;
    const glm::vec2 lightUv = b0 * lightAttr.uv0 + b1 * lightAttr.uv1 + b2 * lightAttr.uv2;
    const glm::vec3 delta = lightPoint - getExactPointOnRay(
        context.pathSegment.ray, context.receiver.t);
    const float distanceSquared = glm::dot(delta, delta);
    if (!(distanceSquared > 0.0f) || !isfinite(distanceSquared)) return false;

    sample.light = light;
    sample.triangle = &lightTriangle;
    sample.binding = &scene.surfaceBindings[lightSurface.surfaceBindingId + 1];
    sample.material = &scene.materials[lightSurface.materialId];
    sample.point = lightPoint;
    sample.uv = lightUv;
    sample.wi = delta * glm::inversesqrt(distanceSquared);
    sample.distanceSquared = distanceSquared;
    return true;
}

// Evaluates only the continuous NEE terms.  Delta BSDFs retain the existing
// no-finite-area-light-sample rule, and the exact PDFs stay in solid angle.
static __device__ __forceinline__ bool evaluateDirectLight(
    const DirectLightingContext& context,
    const DirectLightSample& sample,
    DirectLightEvaluation& evaluation)
{
    const BsdfEvaluation bsdf = evaluateBsdf(context.receiverBsdf,
        context.receiverMaterial, context.pathSegment.ray.direction, sample.wi);
    const float receiverCosine = glm::max(glm::dot(bsdf.shadingNormal, sample.wi), 0.0f);
    if (bsdf.isDelta || !(receiverCosine > 0.0f) || !(bsdf.pdfOmega > 0.0f)) return false;

    glm::vec3 lightNormal;
    if (!normalizedTriangleNormal(*sample.triangle, lightNormal)) return false;
    const float lightCosine = emissionCosine(*sample.material, lightNormal, -sample.wi);
    if (!(lightCosine > 0.0f)) return false;

    evaluation.bsdf = bsdf;
    evaluation.lightNormal = lightNormal;
    evaluation.receiverCosine = receiverCosine;
    evaluation.lightCosine = lightCosine;
    return true;
}

// Visibility owns the finite shadow segment policy.  The ray starts from the
// receiver's geometric plane using the same scale-aware offset as continuation
// rays, then ends one ULP before the sampled emitter point.
static __device__ __forceinline__ bool directLightIsVisible(
    const DirectLightingContext& context,
    const DirectLightSample& sample)
{
    const glm::vec3 receiverPoint = getExactPointOnRay(
        context.pathSegment.ray, context.receiver.t);
    const SurfaceRaySpawnContext raySpawn = makeSurfaceRaySpawnContext(
        receiverPoint, context.receiver.geometricNormal,
        context.receiver.rayOriginScale);
    const Ray shadowRay = spawnShadowRay(raySpawn, sample.wi);
    const float maxT = nextafterf(glm::dot(sample.point - shadowRay.origin, sample.wi), 0.0f);
    // Any-hit returns true when an occluder is found before the sampled
    // emitter.  A visible light sample is therefore the false case.
    return maxT > RAY_EPSILON && !traverseBvhAnyHit(
        shadowRay, context.scene.bvhNodes, context.scene.trianglePositions, maxT,
        context.receiver.triangleIndex);
}

// This is the sole area-to-solid-angle conversion and MIS accumulation site
// for NEE.  The arithmetic expression intentionally matches the former
// monolithic implementation, including operation order and finite boundary.
static __device__ __forceinline__ void accumulateDirectLightContribution(
    const DirectLightingContext& context,
    const DirectLightSample& sample,
    const DirectLightEvaluation& evaluation)
{
    const float pLight = lightPdfOmega(context.scene.lights, sample.light.triangleIndex,
                                       sample.distanceSquared, evaluation.lightCosine);
    if (!(pLight > 0.0f) || !isfinite(pLight)) return;
    const glm::vec3 Le = evaluateEmittedRadiance(*sample.binding, context.scene.textures,
                                                   sample.uv, *sample.material,
                                                   evaluation.lightNormal, -sample.wi);
    const float areaPdf = sample.light.selectPmf / sample.light.area;
    const glm::vec3 contribution = context.pathSegment.throughput * Le * evaluation.bsdf.value *
        (evaluation.receiverCosine * evaluation.lightCosine /
         (sample.distanceSquared * areaPdf)) *
        powerHeuristic(pLight, evaluation.bsdf.pdfOmega);
    // The BSDF owns validation of its value/PDF.  This boundary owns the
    // product of path throughput, emission, geometry and MIS, where a finite
    // factor combination can still overflow or form Inf*0.
    if (finiteVec3(contribution))
    {
        context.pathSegment.accumulatedRadiance += contribution;
    }
}

static __device__ void accumulateDirectLighting(
    const DirectLightingContext& context,
    RngState& rng)
{
    const int lightIndex = sampleLightTriangle(context.scene.lights,
        rng.next(HaltonDim::LightSelection));
    if (lightIndex < 0) return;

    DirectLightSample sample{};
    if (!sampleDirectLight(context, lightIndex, rng, sample)) return;

    DirectLightEvaluation evaluation{};
    if (!evaluateDirectLight(context, sample, evaluation)) return;
    if (!directLightIsVisible(context, sample)) return;
    accumulateDirectLightContribution(context, sample, evaluation);
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
