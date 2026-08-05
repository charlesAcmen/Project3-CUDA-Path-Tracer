#pragma once

// ====================================================================
// Axis-Aligned Bounding Box — shared host/device
//
// TRACED COPY of src/bvh/aabb.h — keep in sync with production.  The
// only differences are the TRACE() statements below and the removal of
// the test-only AABB::centroid() (which production no longer has); the
// test computes the box centroid itself in bvh_test.cu.  Trace levels:
// 1 = the per-ray AABB test outcome, 2 = every min/max state change.
//
// Purely computational, header-only.  All functions are __host__ __device__
// so the same code serves the CPU build/test harness and the GPU traversal
// kernel.  The __CUDA_ARCH__ guard inside TRACE (trace.h) strips the
// printf on the nvcc device pass.
// ====================================================================

#include "sceneStructs.h"   // Triangle
#include "constants.h"      // RAY_EPSILON
#include "trace.h"

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
        TRACE(2, "[AABB.expand] box min=(%.4g,%.4g,%.4g) max=(%.4g,%.4g,%.4g)  +p=(%.4g,%.4g,%.4g)\n",
              min.x,min.y,min.z, max.x,max.y,max.z, p.x,p.y,p.z);
        min = glm::min(min, p);//compare by component and take the minimum
        max = glm::max(max, p);//compare by component and take the maximum
        TRACE(2, "            -> min=(%.4g,%.4g,%.4g) max=(%.4g,%.4g,%.4g)\n",
              min.x,min.y,min.z, max.x,max.y,max.z);
    }

    __host__ __device__ inline void expand(const Triangle& t)
    {
        TRACE(2, "[AABB.expand] by triangle v0=(%.4g,%.4g,%.4g) v1=(%.4g,%.4g,%.4g) v2=(%.4g,%.4g,%.4g)\n",
              t.v0.x,t.v0.y,t.v0.z, t.v1.x,t.v1.y,t.v1.z, t.v2.x,t.v2.y,t.v2.z);
        expand(t.v0);
        expand(t.v1);
        expand(t.v2);
    }

    __host__ __device__ inline void expand(const AABB& b)
    {
        TRACE(2, "[AABB.expand] by box min=(%.4g,%.4g,%.4g) max=(%.4g,%.4g,%.4g)\n",
              b.min.x,b.min.y,b.min.z, b.max.x,b.max.y,b.max.z);
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
        const float r = (area != area || area <= RAY_EPSILON) ? 1.0f : area;
        TRACE(2, "[AABB.surfaceArea] e=(%.4g,%.4g,%.4g) area=%.4g -> return %.4g%s\n",
              e.x,e.y,e.z, area, r,
              (r == 1.0f && area != 1.0f) ? "  (degenerate guard)" : "");
        return r;
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
    TRACE(1, "  [AABB.test] o=(%.4g,%.4g,%.4g) invDir=(%.4g,%.4g,%.4g) box.min=(%.4g,%.4g,%.4g) box.max=(%.4g,%.4g,%.4g) interval=[%.4g, %.4g]\n",
          o.x,o.y,o.z, invDir.x,invDir.y,invDir.z,
          box.min.x,box.min.y,box.min.z, box.max.x,box.max.y,box.max.z, tNear, tFar);

    // x slab
    float tmin = (box.min.x - o.x) * invDir.x;
    float tmax = (box.max.x - o.x) * invDir.x;
    if (invDir.x < 0.0f) { float tmp = tmin; tmin = tmax; tmax = tmp; }
    tNear = fmaxf(tNear, tmin);
    tFar  = fminf(tFar, tmax);
    TRACE(2, "    x: tmin=%.4g tmax=%.4g%s -> near=%.4g far=%.4g%s\n",
          tmin, tmax, (invDir.x < 0.0f) ? " (swap)" : "", tNear, tFar,
          (tNear > tFar) ? "  EARLY MISS" : "");
    if (tNear > tFar) { TRACE(1, "  [AABB.test] -> MISS\n"); return false; }

    // y slab
    tmin = (box.min.y - o.y) * invDir.y;
    tmax = (box.max.y - o.y) * invDir.y;
    if (invDir.y < 0.0f) { float tmp = tmin; tmin = tmax; tmax = tmp; }
    tNear = fmaxf(tNear, tmin);
    tFar  = fminf(tFar, tmax);
    TRACE(2, "    y: tmin=%.4g tmax=%.4g%s -> near=%.4g far=%.4g%s\n",
          tmin, tmax, (invDir.y < 0.0f) ? " (swap)" : "", tNear, tFar,
          (tNear > tFar) ? "  EARLY MISS" : "");
    if (tNear > tFar) { TRACE(1, "  [AABB.test] -> MISS\n"); return false; }

    // z slab
    tmin = (box.min.z - o.z) * invDir.z;
    tmax = (box.max.z - o.z) * invDir.z;
    if (invDir.z < 0.0f) { float tmp = tmin; tmin = tmax; tmax = tmp; }
    tNear = fmaxf(tNear, tmin);
    tFar  = fminf(tFar, tmax);
    TRACE(2, "    z: tmin=%.4g tmax=%.4g%s -> near=%.4g far=%.4g%s\n",
          tmin, tmax, (invDir.z < 0.0f) ? " (swap)" : "", tNear, tFar,
          (tNear > tFar) ? "  EARLY MISS" : "");

    const bool hit = tNear <= tFar;
    TRACE(1, "  [AABB.test] -> %s\n", hit ? "HIT" : "MISS");
    return hit;
}
