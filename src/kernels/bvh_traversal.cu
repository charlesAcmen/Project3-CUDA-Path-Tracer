#include "bvh_traversal.cuh"

// ====================================================================
// BVH Traversal Kernel Implementation
//
// Single world-space BVH closest-hit traversal: one thread per active
// path, one traverseBvhClosest over the whole tree.  The ray is already
// in world space and the triangles were baked to world space by
// buildSceneBvh, so no transform and no per-geom loop.
// ====================================================================

__global__ void bvhTraverse(
    int num_paths,
    PathSegment* __restrict__ pathSegments,
    HitRecord* __restrict__ intersections,
    const TrianglePos* __restrict__ deviceTrianglePositions,
    BvhNode* __restrict__ deviceBvhNodes)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index >= num_paths) return;

    const PathSegment& pathSegment = pathSegments[path_index];

    // Start from a complete miss record so sorting never sees stale data from
    // a previous bounce.
    HitRecord record;

    // Guard: a scene with no triangles produces no tree.
    if (deviceBvhNodes == nullptr || deviceTrianglePositions == nullptr)
    {
        intersections[path_index] = record;
        return;
    }

    // ---- Single closest-hit traversal over the whole scene ----
    // Root is node 0; LARGE_T far plane (the traversal tightens it).
    const BvhHit hit = traverseBvhClosest(pathSegment.ray, deviceBvhNodes,
                                          deviceTrianglePositions, LARGE_T);

    if (!hit.hit)
    {
        intersections[path_index] = record;
    }
    else
    {
        record.t             = hit.t;
        record.u             = hit.u;
        record.v             = hit.v;
        record.triangleIndex = hit.triIndex;
        intersections[path_index] = record;
    }
}
