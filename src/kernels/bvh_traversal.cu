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
    PathSegment* pathSegments,
    ShadeableIntersection* intersections,
    Triangle* deviceTriangles,
    BvhNode* deviceBvhNodes)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index >= num_paths) return;

    const PathSegment& pathSegment = pathSegments[path_index];

    // Guard: a scene with no triangles produces no tree.
    if (deviceBvhNodes == nullptr || deviceTriangles == nullptr)
    {
        intersections[path_index].t = -1.0f;
        return;
    }

    // ---- Single closest-hit traversal over the whole scene ----
    // Root is node 0; LARGE_T far plane (the traversal tightens it).
    const BvhHit hit = traverseBvhClosest(pathSegment.ray, deviceBvhNodes,
                                          deviceTriangles, LARGE_T);

    if (!hit.hit)
    {
        intersections[path_index].t = -1.0f;
    }
    else
    {
        intersections[path_index].t            = hit.t;
        intersections[path_index].surfaceNormal = hit.normal;   // world space (baked)
        intersections[path_index].materialId    = deviceTriangles[hit.triIndex].materialId;
        intersections[path_index].uv            = hit.uv;       // interpolated texture coordinate
    }
}
