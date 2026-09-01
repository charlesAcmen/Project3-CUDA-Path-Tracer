#pragma once

// ====================================================================
// Ray Utility Functions
//
// Provides getExactPointOnRay and concentricSampleDisk used 
// by the kernel code.
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
    // Ray directions are guaranteed unit-length by construction:
    //   generateRayFromCamera — pinhole / DoF rays are glm::normalize(...)
    //   scatterRay            — reflect / refract / sampled dirs are unit
    // Triangle position intersection already interprets t as world distance, so
    // re-normalizing here is a redundant dot + rsqrt + 3 muls per hit.
    return r.origin + t * r.direction;
}

// A hit point can be close to the origin even when its triangle is large
// (for example, a large floor centred at (0,0,0)).  The interpolation error
// is then governed by the triangle extent rather than by |point| alone.
// Return a conservative scalar in world units for the origin offset.
__host__ __device__ inline float triangleRayOriginScale(const TrianglePos& tri)
{
    const glm::vec3 e1 = tri.v1 - tri.v0;
    const glm::vec3 e2 = tri.v2 - tri.v0;
    const glm::vec3 e3 = tri.v2 - tri.v1;
    const glm::vec3 vertexAbs = glm::max(glm::abs(tri.v0),
        glm::max(glm::abs(tri.v1), glm::abs(tri.v2)));
    const glm::vec3 edgeAbs = glm::max(glm::abs(e1),
        glm::max(glm::abs(e2), glm::abs(e3)));
    const glm::vec3 scale = glm::max(vertexAbs, edgeAbs);
    return glm::max(1.0f, glm::max(scale.x, glm::max(scale.y, scale.z)));
}

/**
 * Moves a secondary-ray origin to the requested side of a surface.
 *
 * A fixed world-space epsilon can round back to the original float coordinate
 * in large scenes.  A scale-aware scalar keeps the offset representable while
 * preserving the requested direction exactly along the geometric normal.  The
 * optional geometryScale covers large triangles whose hit point is near the
 * world origin, where point magnitude alone underestimates interpolation
 * error.
 */
__host__ __device__ inline glm::vec3 offsetRayOrigin(
    const glm::vec3& point,
    const glm::vec3& normal,
    float side,
    float geometryScale = 0.0f)
{
    const glm::vec3 magnitude = glm::max(glm::abs(point), glm::vec3(1.0f));
    const float maxMagnitude = glm::max(magnitude.x,
        glm::max(magnitude.y, magnitude.z));
    // NaN fails both comparisons and infinity is not a usable scene scale;
    // treat either as absent rather than allowing an invalid origin to enter
    // the traversal.
    const float safeGeometryScale = (geometryScale > 0.0f &&
                                     geometryScale < LARGE_T)
        ? geometryScale : 0.0f;
    const float scale = glm::max(maxMagnitude, safeGeometryScale);
    const float offset = glm::max(EPSILON,
        scale * RAY_ORIGIN_RELATIVE_EPSILON);
    return point + normal * (side * offset);
}

/**
 * Spawn a ray that leaves a surface without numerically re-entering its
 * triangle.  The geometric normal determines the side of the surface; the
 * outgoing direction is otherwise preserved exactly.  Shadow rays can use
 * this helper too, then apply their own finite tMax policy.
 */
__host__ __device__ inline Ray spawnRayFromSurface(
    const glm::vec3& point,
    const glm::vec3& geometricNormal,
    const glm::vec3& direction,
    float geometryScale = 0.0f)
{
    const float side = glm::dot(direction, geometricNormal) >= 0.0f
        ? 1.0f : -1.0f;
    return Ray{
        offsetRayOrigin(point, geometricNormal, side, geometryScale),
        direction
    };
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
