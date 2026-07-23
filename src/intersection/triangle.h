#pragma once

// ====================================================================
// Triangle Intersection (Möller-Trumbore)莫勒-特伦博尔算法
//
// Object-space ray–triangle intersection test using GLM's
// intersectRayTriangle.  Produces a flat-shaded face normal (no
// vertex-normal interpolation).
//
// This is the only primitive intersection routine in the path tracer.
// All geometries must be triangulated at load time; the linear scan
// over triangles lives in kernels/intersection.cuh.
// ====================================================================

#include "sceneStructs.h"
#include "constants.h"

#include <glm/gtx/intersect.hpp>

/**
 * Intersect a ray against a single triangle (object-space).
 *
 * The ray must already be transformed to object space (via the
 * geometry's inverseTransform) so that it shares the coordinate
 * frame of the triangle vertices.
 *
 * @param ray       Ray in object space
 * @param tri       Triangle in object space
 * @param outT      [out] Parametric distance to intersection
 * @param outNormal [out] Face normal (object space)
 * @return          true on hit
 */
__device__ inline bool triangleIntersectionTest(
    const Ray& ray,
    const Triangle& tri,
    float& outT,
    glm::vec3& outNormal)
{
    glm::vec3 baryPos;  // x=u, y=v, z=t (barycentric + distance)
    bool hit = glm::intersectRayTriangle(
        ray.origin, ray.direction,
        tri.v0, tri.v1, tri.v2,
        baryPos);

    if (hit && baryPos.z > 0.0f)
    {
        outT = baryPos.z;
        // Flat-shaded face normal: cross product of two edges.
        outNormal = glm::normalize(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
        return true;
    }
    return false;
}
