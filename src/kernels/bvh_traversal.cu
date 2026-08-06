#include "bvh_traversal.cuh"

// ====================================================================
// BVH Traversal Kernel Implementation
//
// Per-mesh BVH closest-hit traversal: one thread per active path, an outer
// loop over all geoms, writing the same ShadeableIntersection layout the
// shading kernels read.
// ====================================================================

__global__ void bvhTraverse(
    int num_paths,
    PathSegment* pathSegments,
    ShadeableIntersection* intersections,
    Triangle* deviceTriangles,
    BvhNode* deviceBvhNodes,
    BvhMeta* deviceBvhMeta)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index >= num_paths) return;

    const PathSegment& pathSegment = pathSegments[path_index];

    float t_min = LARGE_T;
    int   hit_geom_index = -1;
    glm::vec3 hit_normal;

    for (int i = 0; i < geoms_size; i++)
    {
        Geom& geom = geoms[i];

        // ---- Transform ray to object space (same as the O(N) path) ----
        const Ray objRay = transformRayToObjectSpace(geom, pathSegment.ray);

        // ---- Skip meshes with no triangles (same guard as the O(N) path) ----
        if (deviceTriangles == nullptr || geom.meshTriangleCount <= 0)
            continue;

        // ---- Skip geoms with no built subtree ----
        // Every non-empty mesh gets a subtree from buildSceneBvh, so an
        // absent one (rootNodeIndex = -1) only means "nothing was built".
        if (deviceBvhMeta == nullptr || deviceBvhNodes == nullptr ||
            deviceBvhMeta[i].rootNodeIndex < 0)
            continue;

        // ---- BVH closest-hit traversal over this mesh's subtree ----
        // Pass the current best distance (t_min) as the far plane so the
        // traversal prunes subtrees that cannot beat it.
        const BvhHit hit = traverseBvhClosest(objRay, deviceBvhNodes,
                                              deviceBvhMeta[i].rootNodeIndex,
                                              deviceTriangles, t_min);
        if (!hit.hit) continue;

        // ---- Record closest hit (world-space) ----
        t_min = hit.t;
        hit_geom_index = i;
        hit_normal = recordWorldNormal(geom, hit.normal);
    }

    if (hit_geom_index == -1)
    {
        intersections[path_index].t = -1.0f;
    }
    else
    {
        intersections[path_index].t           = t_min;
        intersections[path_index].materialId  = geoms[hit_geom_index].materialid;
        intersections[path_index].surfaceNormal = hit_normal;
    }
}
