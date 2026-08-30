#pragma once

// ====================================================================
// Bounding Volume Hierarchy — CPU build / GPU traverse
//
// Single world-space BVH over ALL mesh triangles.  buildSceneBvh (bvh.cu)
// bakes every mesh's triangles to world space (vertices via the geom
// transform, normals via the inverse-transpose) and assigns each its shared
// Surface id, then builds ONE tree over the combined array (exhaustive SAH,
// 穷举表面积启发式算法分割建树).  The traversal kernel runs one closest-hit
// query per ray — no per-mesh loop, no ray transformation.
//
// Triangle layout: leaves reference a contiguous chunk [left, right) of
// the parallel triangle arrays.  buildSceneBvh's flatten pass（展平阶段）REORDERS entries
// into leaf-contiguous runs （同一个叶子节点所包含的三角形在内存中连续存放）, so a leaf's
// chunk is a sequential memory access — cache-friendly instead of
// scattered leaf-index reads into the original scene-order array.
// ====================================================================

#include "aabb.h"
#include "constants.h"               // RAY_EPSILON, LARGE_T
#include "sceneStructs.h"            // Ray, TrianglePos, TriangleAttr, Geom
#include "intersection/triangle.h"   // intersectTrianglePositions

#include <vector>

// A node in the flattened node array.
//
// `left`/`right` are overloaded by isLeaf, so read them through the
// accessors below — the meaning at each call site is then self-evident:
//   internal node: left/right = child node indices
//   leaf:          left/right = (triangle offset, triangle count)
struct BvhNode
{
    AABB bounds;
    int  left;
    int  right;
    bool isLeaf = false;

    __host__ __device__ int childL() const        { return left; }  // internal: left child index
    __host__ __device__ int childR() const        { return right; } // internal: right child index
    __host__ __device__ int leafTriOffset() const { return left; }  // leaf: offset into the triangle array
    __host__ __device__ int leafTriCount() const  { return right; } // leaf: triangle count
};

// Host build output + device upload for the single scene-wide BVH.
// The arrays are REORDERED world-space output of the flatten pass.  Traversal
// receives positions only; shading receives attrs and the deduplicated Surface
// table.  Every position/attribute index refers to the same triangle.
struct BvhBuffers
{
    // Device-side buffers (allocated in uploadToDevice, freed in freeDevice)
    BvhNode*  deviceNodes   = nullptr;
    std::vector<BvhNode>     hostNodes;      // construction output
    std::vector<TrianglePos> hostTrianglePositions;
    std::vector<TriangleAttr> hostTriangleAttrs;
    std::vector<Surface>     hostSurfaces;
};

// Result of a closest-hit BVH traversal.
// `hit` is true only when a triangle was found with t < the caller's far
// plane (maxT); `t` is that closest distance.  `triIndex` is the index of
// the hit triangle into `tris`, and `u` / `v` are its Moller-Trumbore(莫勒-特朗博尔)
// barycentric coordinates (the third weight is 1-u-v).  The shading normal,
// UV, tangent, vertex color, and texture binding are deliberately expanded
// only once, after traversal has selected this closest triangle.  On a miss
// `hit` stays false and `t` keeps maxT.
struct BvhHit
{
    bool      hit   = false;
    float     t     = LARGE_T;
    int       triIndex = -1;
    float     u     = 0.0f;
    float     v     = 0.0f;
};

/**
 * Iterative closest-hit BVH traversal with NEAR-CHILD-FIRST ordering —
 * THE algorithm both the GPU kernel and the host test execute.
 *
 * Ordered traversal: at an internal node, the AABB entry distance of both
 * children is computed; the FARTHER child is pushed onto the stack and the
 * NEARER child is descended into immediately.  Finding close hits early
 * tightens result.t sooner, so subtrees that cannot beat it are pruned
 * earlier (far-plane pruning) — the classic BVH traversal optimization.
 *
 * The AABB test clips to [RAY_EPSILON, maxT]: the near bound skips
 * self-hits, the far bound is the caller's current best distance.
 *
 * The tree is a single scene-wide structure, so the root is always node 0;
 * `nodes == nullptr` (empty scene → no tree) is a clean miss.
 *
 * @param objRay        Ray in world space (triangles are world-space baked)
 * @param nodes         Node array (device or host); nullptr → miss
 * @param tris          Position array (leaf chunks reference into it)
 * @param maxT          Far plane: only hits with t < maxT are reported.
 * @param ignoredTriangleIndex Optional previous primitive to skip as a
 *                             numerical self-intersection guard.
 * @return              BvhHit — hit = true only if a triangle with t < maxT
 */
__host__ __device__ inline BvhHit traverseBvhClosest(
    const Ray& objRay,
    const BvhNode* nodes,
    const TrianglePos* tris,
    float maxT,
    int ignoredTriangleIndex = -1)
{
    BvhHit result;
    result.t = maxT;   // tighten this as closer hits are found

    if (nodes == nullptr) return result;   // empty scene → no tree → miss

    const glm::vec3 invDir(
        1.0f / objRay.direction.x,
        1.0f / objRay.direction.y,
        1.0f / objRay.direction.z);

    int stack[kMaxBvhStackDepth];
    int sp = 0;                             // stack pointer
    int nodeIndex = 0;                      // root is always node 0
    bool nodeBoundsKnownHit = false;

    // "current" node is examined before pushing; a LIFO pop resumes the
    // loop after a subtree finishes. A near child was just AABB-tested by
    // its parent, so it enters with nodeBoundsKnownHit set. A popped far
    // child must be tested again because a nearer hit may have reduced t.
    while (true)
    {
        const BvhNode& node = nodes[nodeIndex];

        // Near side: RAY_EPSILON, far side: current best (result.t).  Skip
        // the node and its subtree if the ray misses the AABB in that window.
        if (!nodeBoundsKnownHit &&
            !intersectRayAABB(objRay.origin, invDir, node.bounds, RAY_EPSILON, result.t))
        {
            if (sp == 0) break;
            nodeIndex = stack[--sp];        // pop
            continue;
        }
        nodeBoundsKnownHit = false;

        if (node.isLeaf)
        {
            // Sequential chunk — leaf triangles were reordered into a
            // contiguous run by the build's flatten pass.
            const int triBase  = node.leafTriOffset();
            const int triCount = node.leafTriCount();
            for (int j = 0; j < triCount; j++)
            {
                const int triangleIndex = triBase + j;
                if (triangleIndex == ignoredTriangleIndex) continue;
                float t;
                float u;
                float v;
                // The hot traversal loop needs only positions plus t/u/v.
                // Interpolating normals, UVs, colors and tangents here would
                // repeat that work for any hit later replaced by a closer one.
                if (intersectTrianglePositions(objRay, tris[triangleIndex], t, u, v))
                {
                    if (t < result.t)
                    {
                        result.t        = t;
                        result.u        = u;
                        result.v        = v;
                        result.hit      = true;
                        result.triIndex = triangleIndex;
                    }
                }
            }

            if (sp == 0) break;
            nodeIndex = stack[--sp];        // pop
            continue;
        }

        // Internal node: order children by ray-entry distance.  Descend into
        // the nearer child immediately; push the farther one for later.
        float entryL, entryR;
        const bool hitL = intersectRayAABBEntry(objRay.origin, invDir,
            nodes[node.childL()].bounds, RAY_EPSILON, result.t, entryL);
        const bool hitR = intersectRayAABBEntry(objRay.origin, invDir,
            nodes[node.childR()].bounds, RAY_EPSILON, result.t, entryR);

        if (hitL && hitR)
        {
            if (entryL < entryR)
            {
                if (sp < kMaxBvhStackDepth - 1) stack[sp++] = node.childR();
                nodeIndex = node.childL();
            }
            else
            {
                if (sp < kMaxBvhStackDepth - 1) stack[sp++] = node.childL();
                nodeIndex = node.childR();
            }
            nodeBoundsKnownHit = true;
        }
        else if (hitL)
        {
            nodeIndex = node.childL();
            nodeBoundsKnownHit = true;
        }
        else if (hitR)
        {
            nodeIndex = node.childR();
            nodeBoundsKnownHit = true;
        }
        else
        {
            if (sp == 0) break;
            nodeIndex = stack[--sp];        // both children miss → pop
        }
    }

    return result;
}

/**
 * Bounded any-hit BVH traversal for visibility rays.
 *
 * The tree and triangle positions use the same world-space layout as closest
 * traversal.  Once an intersection before maxT is found, no hit attributes or
 * farther nodes are needed, so the query returns immediately.  The optional
 * ignored triangle is a numerical self-intersection guard; all other
 * triangles remain blockers.  Callers without a previous primitive pass -1.
 */
__host__ __device__ inline bool traverseBvhAnyHit(
    const Ray& ray,
    const BvhNode* nodes,
    const TrianglePos* tris,
    float maxT,
    int ignoredTriangleIndex = -1)
{
    if (nodes == nullptr || tris == nullptr || !(maxT > RAY_EPSILON))
        return false;

    const glm::vec3 invDir(
        1.0f / ray.direction.x,
        1.0f / ray.direction.y,
        1.0f / ray.direction.z);

    int stack[kMaxBvhStackDepth];
    int sp = 0;
    int nodeIndex = 0;
    bool nodeBoundsKnownHit = false;

    while (true)
    {
        const BvhNode& node = nodes[nodeIndex];
        if (!nodeBoundsKnownHit &&
            !intersectRayAABB(ray.origin, invDir, node.bounds, RAY_EPSILON, maxT))
        {
            if (sp == 0) break;
            nodeIndex = stack[--sp];
            continue;
        }
        nodeBoundsKnownHit = false;

        if (node.isLeaf)
        {
            const int triBase = node.leafTriOffset();
            const int triCount = node.leafTriCount();
            for (int j = 0; j < triCount; ++j)
            {
                const int triangleIndex = triBase + j;
                if (triangleIndex == ignoredTriangleIndex) continue;
                float t, u, v;
                if (intersectTrianglePositions(ray, tris[triangleIndex], t, u, v) &&
                    t < maxT)
                    return true;
            }

            if (sp == 0) break;
            nodeIndex = stack[--sp];
            continue;
        }

        float entryL, entryR;
        const bool hitL = intersectRayAABBEntry(ray.origin, invDir,
            nodes[node.childL()].bounds, RAY_EPSILON, maxT, entryL);
        const bool hitR = intersectRayAABBEntry(ray.origin, invDir,
            nodes[node.childR()].bounds, RAY_EPSILON, maxT, entryR);

        if (hitL && hitR)
        {
            if (entryL < entryR)
            {
                if (sp < kMaxBvhStackDepth - 1) stack[sp++] = node.childR();
                nodeIndex = node.childL();
            }
            else
            {
                if (sp < kMaxBvhStackDepth - 1) stack[sp++] = node.childL();
                nodeIndex = node.childR();
            }
            nodeBoundsKnownHit = true;
        }
        else if (hitL)
        {
            nodeIndex = node.childL();
            nodeBoundsKnownHit = true;
        }
        else if (hitR)
        {
            nodeIndex = node.childR();
            nodeBoundsKnownHit = true;
        }
        else
        {
            if (sp == 0) break;
            nodeIndex = stack[--sp];
        }
    }

    return false;
}

// Host-side construction + GPU memory management, implemented in bvh.cu.
namespace bvh
{
    // Build ONE BVH over the whole triangle array, appending nodes to
    // out.hostNodes and the flattened (leaf-contiguous) world-space parallel
    // arrays.  The tree's root is always node 0;
    // an empty array produces no nodes at all.
    void buildMeshBvh(BvhBuffers& out,
                      const std::vector<TrianglePos>& positions,
                      const std::vector<TriangleAttr>& attrs);

    // Bake every mesh's triangles to world space, deduplicate their
    // (materialId, source-binding) pairs into Surfaces, then build the tree.
    void buildSceneBvh(BvhBuffers& out,
                       const std::vector<TrianglePos>& positions,
                       const std::vector<TriangleAttr>& attrs,
                       const std::vector<Geom>& geoms);

    // Upload the host node buffer to device memory.
    void uploadToDevice(BvhBuffers& b);
    void freeDevice(BvhBuffers& b);
} // namespace bvh
