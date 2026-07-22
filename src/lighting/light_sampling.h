#pragma once

// ====================================================================
// Direct Lighting: Light Sampling & Shadow Ray Functions
//
// Device functions for next-event estimation (explicit light sampling).
// Called from the shadeMaterial kernel to compute the direct illumination
// contribution at each non-emissive surface hit.
//
// Reference: PBRTv4 S13.4 -- Direct Lighting / Next-Event Estimation.
// ====================================================================

// ====================================================================
// Include Notes
// ====================================================================
// This file depends on sceneStructs (LightInfo, Geom, etc.), the RNG
// for Halton-dimension constants, intersection helpers (multiplyMV),
// and the sphere/box intersection tests.
//
// We do NOT include kernels/intersection.cuh because that would create
// duplicate device-function definitions across translation units under
// separable compilation.  Instead, testShadowRay calls the per-shape
// intersection functions from intersections.h directly.
// ====================================================================

#include "sceneStructs.h"
#include "rng/rng.h"
#include "intersections.h"   // multiplyMV, boxIntersectionTest, sphereIntersectionTest
#include "constants.h"

/**
 * Uniformly sample a point on a cube's surface in world space.
 *
 * Algorithm:
 *   1. Select one of the 6 faces uniformly (each face has equal probability).
 *   2. Within the chosen face, sample a uniform (u,v) in [0,1]^2, then
 *      map to the canonical cube face [-0.5, 0.5]^2 at the appropriate axis.
 *   3. Transform the object-space point to world space via the geometry's
 *      transform matrix.  The caller must transform the object-space normal
 *      via invTranspose.
 *
 * Halton dimensions (each independent random decision gets its own dim):
 *   LightSurfaceW (dim 13, prime 43) -- face selection
 *   LightSurfaceU (dim 11, prime 37) -- u coordinate on face
 *   LightSurfaceV (dim 12, prime 41) -- v coordinate on face
 *
 * @param geom         The cube geometry (provides transform matrices + scale)
 * @param rng          RNG state
 * @param outObjNormal [out] Object-space surface normal at the sampled point
 * @return             World-space sampled point on the surface
 */
__device__ inline glm::vec3 sampleCubeSurface(
    const Geom& geom,
    RngState& rng,
    glm::vec3& outObjNormal)
{
    // ---- Step 1: choose a face (0..5 uniformly) ----
    // Uses HaltonDim::LightSurfaceW (dim 13, prime 43) -- independent of u/v.
    int face = (int)(rng.next(HaltonDim::LightSurfaceW) * 6.0f);

    // ---- Step 2: sample (u,v) in [0,1] for position on the chosen face ----
    // Each coordinate gets its own Halton dimension for low-discrepancy.
    // Canonical cube spans [-0.5, 0.5] in each axis.
    float u = rng.next(HaltonDim::LightSurfaceU) - 0.5f;
    float v = rng.next(HaltonDim::LightSurfaceV) - 0.5f;

    glm::vec3 objPoint;

    // Map face index to axis-aligned face on the canonical cube [-0.5, 0.5]^3.
    // Face layout: +X, -X, +Y, -Y, +Z, -Z
    switch (face) {
        case 0: objPoint = glm::vec3( 0.5f,  v,    u); outObjNormal = glm::vec3( 1,  0,  0); break;
        case 1: objPoint = glm::vec3(-0.5f,  v,    u); outObjNormal = glm::vec3(-1,  0,  0); break;
        case 2: objPoint = glm::vec3( u,     0.5f, v); outObjNormal = glm::vec3( 0,  1,  0); break;
        case 3: objPoint = glm::vec3( u,    -0.5f, v); outObjNormal = glm::vec3( 0, -1,  0); break;
        case 4: objPoint = glm::vec3( u,     v,   0.5f); outObjNormal = glm::vec3( 0,  0,  1); break;
        case 5: objPoint = glm::vec3( u,     v,  -0.5f); outObjNormal = glm::vec3( 0,  0, -1); break;
        default: objPoint = glm::vec3(0.0f); outObjNormal = glm::vec3(0, 1, 0); break;
    }

    // Transform point to world space.  Normal is left in object space;
    // the caller (samplePointOnLight) transforms it via invTranspose.
    return multiplyMV(geom.transform, glm::vec4(objPoint, 1.0f));
}

/**
 * Uniformly sample a point on a sphere's surface in world space.
 *
 * Algorithm:
 *   1. Sample spherical coordinates: theta uniformly in [0, 2pi),
 *      phi = acos(1 - 2*v) for uniform distribution over the sphere surface.
 *   2. Map to the canonical sphere radius 0.5 in object space.
 *   3. Transform to world space via the geometry's transform matrix.
 *
 * Halton dimensions:
 *   LightSurfaceU (dim 11) -- theta (longitude)
 *   LightSurfaceV (dim 12) -- v for phi = acos(1-2v) (latitude)
 *
 * @param geom         The sphere geometry
 * @param rng          RNG state
 * @param outObjNormal [out] Object-space surface normal at the sampled point
 * @return             World-space sampled point
 */
__device__ inline glm::vec3 sampleSphereSurface(
    const Geom& geom,
    RngState& rng,
    glm::vec3& outObjNormal)
{
    // Uniform spherical sampling.
    // theta ~ U[0, 2pi), v ~ U[0,1) -> phi = acos(1 - 2*v)
    // This gives uniform density over the sphere surface area.
    float theta = rng.next(HaltonDim::LightSurfaceU) * TWO_PI;
    float v     = rng.next(HaltonDim::LightSurfaceV);
    float phi   = acosf(1.0f - 2.0f * v);

    float r = 0.5f;  // canonical sphere radius in object space
    float sinPhi = sinf(phi);

    // Object-space point on sphere surface:
    // Back-face culling is not needed -- the light emits in all directions
    // and the geometry term (cosLight) handles the sign.
    glm::vec3 objPoint(
        r * sinPhi * cosf(theta),
        r * cosf(phi),
        r * sinPhi * sinf(theta));

    // For a sphere centered at origin, the normal in object space
    // points in the same direction as the position vector.
    outObjNormal = glm::normalize(objPoint);

    return multiplyMV(geom.transform, glm::vec4(objPoint, 1.0f));
}

/**
 * Compute the world-space surface area of a geometry.
 *
 * Used during initialization to build the LightInfo array with correct
 * area values for PDF computation.
 *
 * Formulas:
 *   Sphere: world radius = 0.5 * (volume-equivalent scale factor)
 *           area = 4 * PI * world_radius^2
 *   Cube:   canonical cube [-0.5, 0.5]^3 -> each face area = s_i * s_j
 *           total = 2*(sx*sy + sx*sz + sy*sz)
 *
 * LIMITATION (sphere): For non-uniformly scaled spheres (ellipsoids), the
 * equal-volume-sphere method gives an approximate surface area that is
 * NOT the true ellipsoid surface area.  Additionally, sampleSphereSurface
 * samples the canonical sphere uniformly and then applies the transform,
 * which does NOT produce a uniform distribution over an ellipsoid surface.
 * For correct ellipsoid lighting, use uniform scale (sx = sy = sz) or
 * implement ellipsoid-specific sampling with Jacobian correction.
 *
 * @param geom  Geometry (type + scale determine area)
 * @return      World-space surface area (always > 0)
 */
__host__ __device__ inline float computeGeomSurfaceArea(const Geom& geom)
{
    if (geom.type == SPHERE)
    {
        // Volume-equivalent radius for potentially non-uniform scaling.
        // The canonical sphere has radius 0.5, giving volume = (4/3)pi(0.5)^3.
        // After scaling by (sx, sy, sz), the ellipsoid volume scales by
        // |sx * sy * sz|.  We find the uniform-scaling sphere with the
        // same volume:
        //   r_eq = 0.5 * cbrt(sx * sy * sz)
        float r = 0.5f * cbrtf(geom.scale.x * geom.scale.y * geom.scale.z);
        return 4.0f * PI * r * r;
    }
    else if (geom.type == CUBE)
    {
        // Object-space cube [-0.5, 0.5]^3 scaled by (sx, sy, sz).
        // Each dimension contributes a pair of faces with area
        // = product of the two remaining scale factors.
        float sx = geom.scale.x;
        float sy = geom.scale.y;
        float sz = geom.scale.z;
        return 2.0f * (sx * sy + sx * sz + sy * sz);
    }
    return 1.0f;  // fallback (should not reach)
}

/**
 * Sample a uniform point on an emissive geometry's surface, returning
 * the world-space position, the outward-facing normal, and the area PDF.
 *
 * Dispatches to the appropriate geometry-specific sampler (sphere vs cube).
 *
 * @param geom      The geometry to sample
 * @param rng       RNG state
 * @param outPos    [out] World-space sampled point on surface
 * @param outNormal [out] World-space surface normal at the sampled point
 * @param outPdf    [out] PDF in area measure = 1/surfaceArea
 */
__device__ inline void samplePointOnLight(
    const Geom& geom,
    RngState& rng,
    glm::vec3& outPos,
    glm::vec3& outNormal,
    float& outPdfArea)
{
    glm::vec3 objNormal;  // object-space normal (for invTranspose transform)

    if (geom.type == CUBE)
    {
        outPos = sampleCubeSurface(geom, rng, objNormal);
    }
    else  // SPHERE
    {
        outPos = sampleSphereSurface(geom, rng, objNormal);
    }

    // Transform the object-space normal to world space.
    // Using the inverse-transpose of the transform matrix ensures that
    // non-uniform scaling does not break the normal direction.
    outNormal = glm::normalize(
        multiplyMV(geom.invTranspose, glm::vec4(objNormal, 0.0f)));

    // PDF in area measure: p_A = 1 / surfaceArea
    outPdfArea = 1.0f / computeGeomSurfaceArea(geom);
}

/**
 * Sample a light source: uniform selection of one emissive geometry,
 * then uniform area sampling on its surface.
 *
 * The combined PDF in area measure is:
 *   p_A = 1 / (numLights * selectedLight.area)
 *
 * @param lightInfos  Device array of LightInfo for all emissive geometries
 * @param numLights   Number of entries in lightInfos
 * @param geoms       Device array of all scene geometries (for transforms)
 * @param totalArea   Sum of all light surface areas (for PDF)
 * @param rng         RNG state (uses HaltonDim::LightSelect for selection)
 * @param outPos      [out] World-space sampled point
 * @param outNormal   [out] World-space surface normal at sampled point
 * @param outPdfArea  [out] Combined PDF = 1/(numLights * selectedLight.area)
 * @param outEmission [out] Emitted radiance Le at the sampled point
 * @param outGeomIdx  [out] Index of the selected light geometry (for shadow ray skip)
 */
__device__ inline void sampleLightSource(
    const LightInfo* lightInfos,
    int numLights,
    const Geom* geoms,
    float totalArea,
    RngState& rng,
    glm::vec3& outPos,
    glm::vec3& outNormal,
    float& outPdfArea,
    glm::vec3& outEmission,
    int& outGeomIdx)
{
    // ---- 1. Uniformly select a light source ----
    // Each light has equal probability: p_select = 1 / numLights.
    float sel = rng.next(HaltonDim::LightSelect);
    int lightIdx = (int)(sel * (float)numLights);
    lightIdx = min(lightIdx, numLights - 1);  // clamp for float precision

    const LightInfo& li = lightInfos[lightIdx];
    const Geom& geom = geoms[li.geomIndex];

    // Return the geometry index so the caller can skip this geometry
    // in the shadow-ray test (the light should not shadow itself).
    outGeomIdx = li.geomIndex;

    // ---- 2. Sample a uniform point on the selected light's surface ----
    float areaPdf;  // 1 / light.area (filled by samplePointOnLight)
    samplePointOnLight(geom, rng, outPos, outNormal, areaPdf);

    // ---- 3. Combined PDF ----
    // p_sel * p_A = (1/numLights) * (1/area)
    // (totalArea is computed at init time but not used in uniform
    //  light selection -- each light has equal selection probability.)
    outPdfArea = (1.0f / (float)numLights) * areaPdf;

    // ---- 4. Emitted radiance (pre-multiplied at init time) ----
    outEmission = li.emittedRadiance;
}

/**
 * Shadow ray occlusion test (any-hit, early-exit).
 *
 * Traces a ray from the surface hit point toward the sampled light point.
 * Returns true if the light is VISIBLE (no occluding geometry in between).
 *
 * Unlike computeIntersections which finds the CLOSEST hit, this function
 * returns as soon as ANY intersection is found.  The ray interval is
 * (EPSILON, maxT - EPSILON) to avoid self-intersection at both ends.
 *
 * The skipGeomIndex parameter prevents the light source geometry itself
 * from casting a shadow on its own emission.  Without this, a shadow ray
 * entering the light volume through a face other than the sampled one
 * would incorrectly be flagged as occluded.
 *
 * Ray origin is already offset by surfaceNormal * EPSILON at the caller.
 * The t > EPSILON test here provides an additional numerical safety margin
 * for all surfaces (not just the originating one).  At glancing angles
 * the parametric distance back to the originating surface can exceed
 * the normal-offset distance, so both safeguards are needed.
 *
 * @param ray             Shadow ray (origin already offset by EPSILON)
 * @param maxT            Distance to the light sample point
 * @param geoms           Device array of all scene geometries
 * @param numGeoms        Total number of geometries
 * @param skipGeomIndex   Index of light geometry to skip (-1 = skip none)
 * @return                true if the light is visible (no occlusion)
 */
__device__ inline bool testShadowRay(
    const Ray& ray,
    float maxT,
    const Geom* geoms,
    int numGeoms,
    int skipGeomIndex)
{
    // O(N) linear scan with early exit.  Without a BVH this is the only
    // option; for the Cornell Box scale (< 15 geoms) it is acceptable.
    // Each test checks if the intersection lies strictly between the
    // surface (EPSILON) and the light (maxT - EPSILON).
    //
    // We call the per-shape intersection functions directly instead of
    // using intersectSingleGeom (from kernels/intersection.cuh) to avoid
    // duplicate-symbol errors under CUDA separable compilation.
    for (int i = 0; i < numGeoms; i++)
    {
        // Skip the light geometry itself -- it should not shadow its own
        // emission.  This avoids the case where the shadow ray enters
        // the light volume through a different face before reaching the
        // sampled point on the intended face.
        if (i == skipGeomIndex) continue;

        glm::vec3 tmp_point, tmp_normal;
        bool outside;
        float t;
        if (geoms[i].type == CUBE)
            t = boxIntersectionTest(geoms[i], ray, tmp_point, tmp_normal, outside);
        else if (geoms[i].type == SPHERE)
            t = sphereIntersectionTest(geoms[i], ray, tmp_point, tmp_normal, outside);
        else
            t = -1.0f;

        // EPSILON avoids self-intersection with the originating surface.
        // maxT - EPSILON avoids hitting the light emitter itself (which
        // would incorrectly shadow the light we are trying to sample).
        if (t > EPSILON && t < maxT - EPSILON)
        {
            return false;  // occluded
        }
    }
    return true;  // visible
}
