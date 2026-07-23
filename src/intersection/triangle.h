#pragma once

// ====================================================================
// Triangle Intersection (Möller-Trumbore, double-sided)
//
// Double-sided ray–triangle intersection.  Removes the front-face-only
// rejection (a < 0) from the standard Möller-Trumbore algorithm so
// that back-face hits are also valid.  The face normal is flipped to
// point toward the incident ray for back-face hits.
//
// This is necessary for closed meshes: a ray that enters a refractive
// object (e.g. a glass sphere) will hit the back face when exiting —
// single-sided intersection would reject that hit, making the object
// opaque from the inside.
//
// OBJ files should still follow CCW winding (outward-facing normals)
// as the primary convention.  Double-sided handling is a safety net
// and a requirement for refractive closed meshes.
// ====================================================================

#include "sceneStructs.h"
#include "constants.h"

/**
 * Double-sided ray–triangle intersection (Möller-Trumbore).
 *
 * @param ray       Ray in object space
 * @param tri       Triangle in object space
 * @param outT      [out] Parametric distance
 * @param outNormal [out] Face normal, oriented toward the ray
 * @return          true on hit (either side)
 */
__device__ inline bool triangleIntersectionTest(
    const Ray& ray,
    const Triangle& tri,
    float& outT,
    glm::vec3& outNormal)
{
    const glm::vec3& v0 = tri.v0;
    const glm::vec3& v1 = tri.v1;
    const glm::vec3& v2 = tri.v2;

    glm::vec3 e1 = v1 - v0;
    glm::vec3 e2 = v2 - v0;

    glm::vec3 p = glm::cross(ray.direction, e2);
    float a = glm::dot(e1, p);

    if (fabsf(a) < 1e-10f)
        return false;

    float f = 1.0f / a;
    glm::vec3 s = ray.origin - v0;

    float u = f * glm::dot(s, p);
    if (u < 0.0f || u > 1.0f) return false;

    glm::vec3 q = glm::cross(s, e1);
    float v = f * glm::dot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = f * glm::dot(e2, q);
    if (t < 0.0f) return false;

    outT = t;

    outNormal = glm::normalize(glm::cross(e1, e2));
    if (a < 0.0f)
        outNormal = -outNormal;

    return true;
}
