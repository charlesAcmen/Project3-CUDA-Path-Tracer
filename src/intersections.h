#pragma once

#include "constants.h"
#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>


// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}
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

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);
