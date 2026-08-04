#pragma once

// ====================================================================
// Axis-Aligned Bounding Box — shared host/device
//
// The primitive bounding volume used by the BVH.  Purely computational,
// header-only.  All functions are __host__ __device__ so the same code
// serves the CPU build/test harness and the GPU traversal kernel.
// ====================================================================

#include "sceneStructs.h"   // Triangle
#include "constants.h"      // RAY_EPSILON

#include "glm/glm.hpp"

#include <cfloat>   // FLT_MAX
#include <cmath>    // host fmaxf/fminf (device versions come from CUDA)

struct AABB
{
    glm::vec3 min{ FLT_MAX };
    glm::vec3 max{ -FLT_MAX };

    //expand:takes the union of the current box and the given point, triangle, or box
    __host__ __device__ inline void expand(const glm::vec3& p)
    {
        min = glm::min(min, p);//compare by component and take the minimum
        max = glm::max(max, p);//compare by component and take the maximum
    }

    __host__ __device__ inline void expand(const Triangle& t)
    {
        expand(t.v0);
        expand(t.v1);
        expand(t.v2);
    }

    __host__ __device__ inline void expand(const AABB& b)
    {
        expand(b.min);
        expand(b.max);
    }

    // Surface area of the box.  Degenerate (zero-volume or NaN) boxes
    // return 1.0: the SAH split cost divides by this value, and a
    // degenerate node must neither divide-by-zero nor dominate the cost
    // comparison.  `area != area` detects NaN without calling isnan,
    // which misbehaves under the MSVC host pass.
    __host__ __device__ inline float surfaceArea() const
    {
        const glm::vec3 e = max - min;//diagonal vector of the box
        const float area = 2.0f * (e.x * e.y + e.y * e.z + e.z * e.x);
        return (area != area || area <= RAY_EPSILON) ? 1.0f : area;
    }

    __host__ __device__ inline glm::vec3 centroid() const
    {
        return 0.5f * (min + max);
    }
};

/**
 * Ray/AABB slab test (sign-based).
 *
 * The near/far planes are swapped when a direction component is negative
 * (sign-based test).  For a zero direction component the slab becomes
 * ±inf and does not restrict the interval — no 0*inf = NaN poison.  The
 * exact-boundary case (origin exactly on a box face with a zero direction
 * component) yields a NaN slab term; fminf/fmaxf return the non-NaN
 * operand, so it degrades gracefully to "no restriction" instead of
 * making the whole test false.
 *
 * @param o       Ray origin
 * @param invDir  1/direction — precomputed once per ray by the caller
 * @param box     The box to test
 * @param tNear   Near clip (RAY_EPSILON): skips self-hits
 * @param tFar    Far clip (current best closestT): far-plane pruning
 * @return        true iff the ray interval overlaps [tNear, tFar]
 */
__host__ __device__ inline bool intersectRayAABB(
    const glm::vec3& o, const glm::vec3& invDir,
    const AABB& box, float tNear, float tFar)
{
    // x slab
    float tmin = (box.min.x - o.x) * invDir.x;
    float tmax = (box.max.x - o.x) * invDir.x;
    if (invDir.x < 0.0f) { float tmp = tmin; tmin = tmax; tmax = tmp; }
    tNear = fmaxf(tNear, tmin);
    tFar  = fminf(tFar, tmax);
    if (tNear > tFar) return false;

    // y slab
    tmin = (box.min.y - o.y) * invDir.y;
    tmax = (box.max.y - o.y) * invDir.y;
    if (invDir.y < 0.0f) { float tmp = tmin; tmin = tmax; tmax = tmp; }
    tNear = fmaxf(tNear, tmin);
    tFar  = fminf(tFar, tmax);
    if (tNear > tFar) return false;

    // z slab
    tmin = (box.min.z - o.z) * invDir.z;
    tmax = (box.max.z - o.z) * invDir.z;
    if (invDir.z < 0.0f) { float tmp = tmin; tmin = tmax; tmax = tmp; }
    tNear = fmaxf(tNear, tmin);
    tFar  = fminf(tFar, tmax);

    return tNear <= tFar;
}
