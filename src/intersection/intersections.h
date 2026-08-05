#pragma once

// ====================================================================
// Ray Utility Functions
//
// Provides getExactPointOnRay, multiplyMV, and concentricSampleDisk used by
// the kernel code.
//
// All geometries should be triangulated at load time.
// ====================================================================

#include "constants.h"
#include "sceneStructs.h"

#include <glm/glm.hpp>
/**
 * Compute a point at parameter value `t` on ray `r`.
 */
__host__ __device__ inline glm::vec3 getExactPointOnRay(Ray r, float t)
{
    return r.origin + t * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
    return glm::vec3(m * v);
}

/**
 * Transform a world-space ray into a mesh's object space via the geom's
 * inverseTransform.  Triangles are stored in object space, so the test is
 * performed in the same coordinate system as the vertices.
 *
 * The origin is a POINT (w = 1, translation applies); the direction is a
 * VECTOR (w = 0, translation ignored — rotation + scale only).
 */
__host__ __device__ inline Ray transformRayToObjectSpace(const Geom& geom, const Ray& ray)
{
    Ray objRay;
    objRay.origin    = multiplyMV(geom.inverseTransform, glm::vec4(ray.origin, 1.0f));
    objRay.direction = multiplyMV(geom.inverseTransform, glm::vec4(ray.direction, 0.0f));
    return objRay;
}

/**
 * Transform an object-space shading normal to world space via the geom's
 * inverse-transpose (the correct normal transform under non-uniform scale)
 * and normalize.  Replicates the O(N) kernel's fallback exactly: (0,1,0)
 * when the result is NaN or degenerate (len2 < RAY_EPSILON).
 *
 * Shared by both the linear-scan and BVH-traversal kernels so their outputs
 * match bit-for-bit.
 */
__host__ __device__ inline glm::vec3 recordWorldNormal(const Geom& geom, const glm::vec3& objNormal)
{
    glm::vec3 worldNormal = multiplyMV(geom.invTranspose, glm::vec4(objNormal, 0.0f));
    float wLen2 = glm::dot(worldNormal, worldNormal);
    return (isnan(wLen2) || wLen2 < RAY_EPSILON) ? glm::vec3(0.0f, 1.0f, 0.0f)
                                                 : worldNormal * glm::inversesqrt(wLen2);
}

/**
 * Maps two uniform random numbers in [0,1) to a point on the unit disk
 * using concentric mapping (Shirley's method), which preserves fractional
 * area for unbiased Monte Carlo integration over a circular aperture.
 *
 * Reference: PBRT v4 Section 8.3.2 "Concentric Mapping".
 */
__host__ __device__ inline void concentricSampleDisk(
    float u1, float u2, float& dx, float& dy)
{
    float sx = 2.0f * u1 - 1.0f;
    float sy = 2.0f * u2 - 1.0f;

    // precise equality check is okay here 
    // guarding division by zero
    if (sx == 0.0f && sy == 0.0f) {
        dx = 0.0f;
        dy = 0.0f;
        return;
    }

    float r, theta;
    if (fabsf(sx) > fabsf(sy)) {
        r = sx;
        theta = (PI / 4.0f) * (sy / sx);
    } else {
        r = sy;
        theta = (PI / 2.0f) - (PI / 4.0f) * (sx / sy);
    }
    dx = r * cosf(theta);
    dy = r * sinf(theta);
}

// (boxIntersectionTest / sphereIntersectionTest — removed, unused since
//  the switch to triangle-mesh-only geometry.  See git history if needed.)
