#pragma once

// ====================================================================
// Bounding Volume Hierarchy — CPU build / GPU traverse
//
// Per-mesh BVH in object space.  Built on the host (exhaustive SAH in
// bvh.cu，穷举表面积启发式算法分割建树), 
// traversed iteratively on the GPU via an explicit stack.  
//
// Triangle layout: leaves reference a contiguous chunk [left, right) of
// the triangle array.  buildSceneBvh's flatten pass（展平阶段）REORDERS triangles
// into leaf-contiguous runs （同一个叶子节点所包含的三角形在内存中连续存放）(per mesh), so a leaf's chunk is a
// sequential memory access — cache-friendly instead of
// scattered leaf-index reads into the original scene-order array.
//
// The reorder is within each mesh only: per-mesh triangle counts and
// the mesh concatenation order are unchanged, so Geom::meshTriangleOffset
// / meshTriangleCount still slice the same triangle set.  The reordered
// buffer is the single triangle array the BVH traversal kernel reads.
// ====================================================================

#include "aabb.h"
#include "constants.h"               // RAY_EPSILON, LARGE_T
#include "sceneStructs.h"            // Ray, Triangle, Geom
#include "intersection/triangle.h"   // triangleIntersectionTest

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

// Per-geom BVH metadata.  rootNodeIndex = -1 → empty mesh (kernel skips).
struct BvhMeta
{
    int rootNodeIndex = -1;
};

// Host build output + device upload for the per-mesh BVHs.
// hostTriangles holds the REORDERED flat triangle array (per-mesh
// flatten) — uploaded to the renderer as deviceTriangles.
struct BvhBuffers
{
    // Device-side buffers (allocated in uploadToDevice, freed in freeDevice)
    BvhNode*  deviceNodes   = nullptr;
    BvhMeta*  deviceBvhMeta = nullptr;
    std::vector<BvhNode>  hostNodes;      // construction output
    std::vector<BvhMeta>  hostBvhMeta;    // per-geom metadata
    std::vector<Triangle> hostTriangles;  // reordered flat triangles
};

// Result of a closest-hit BVH traversal.
// `hit` is true only when a triangle was found with t < the caller's far
// plane (maxT); `t` is that closest distance and `normal` its object-space
// shading normal.  On a miss `hit` stays false and `t` keeps maxT.
struct BvhHit
{
    bool      hit   = false;
    float     t     = LARGE_T;
    glm::vec3 normal;
};

/**
 * Iterative closest-hit BVH traversal — THE algorithm both the GPU
 * kernel and the host test execute, so correctness is validated once.
 *
 * The AABB test clips to [RAY_EPSILON, maxT]: the near bound skips
 * self-hits, the far bound is the caller's current best distance, so
 * subtrees that cannot beat it are pruned (far-plane pruning).  Leaf
 * triangles are tested with the SAME triangleIntersectionTest as the
 * O(N) path, so for the same triangle the reported t / normal are
 * bit-identical between the two paths.
 *
 * @param objRay        Ray in the mesh's object space
 * @param nodes         Node array (device or host)
 * @param rootNodeIndex Root of the mesh's subtree (-1 → no hit)
 * @param tris          Triangle array (leaf chunks reference into it)
 * @param maxT          Far plane: only hits with t < maxT are reported.
 *                      Pass the caller's current best t to prune subtrees.
 * @return              BvhHit — hit = true only if a triangle with t < maxT
 */
__host__ __device__ inline BvhHit traverseBvhClosest(
    const Ray& objRay,
    const BvhNode* nodes,
    int rootNodeIndex,
    const Triangle* tris,
    float maxT)
{
    BvhHit result;
    result.t = maxT;   // tighten this as closer hits are found

    if (rootNodeIndex < 0) return result;   // no subtree → miss

    const glm::vec3 invDir(
        1.0f / objRay.direction.x,
        1.0f / objRay.direction.y,
        1.0f / objRay.direction.z);

    int stack[kMaxBvhStackDepth];
    int sp = 0;                             // stack pointer
    stack[sp++] = rootNodeIndex;

    while (sp > 0)
    {
        const int nodeIndex = stack[--sp];
        const BvhNode& node = nodes[nodeIndex];

        // Near side: RAY_EPSILON, far side: current best (result.t).  Skip
        // the node and its subtree if the ray misses the AABB in that window.
        if (!intersectRayAABB(objRay.origin, invDir, node.bounds, RAY_EPSILON, result.t))
            continue;

        if (node.isLeaf)
        {
            // Sequential chunk — leaf triangles were reordered into a
            // contiguous run by the build's flatten pass.
            const int triBase  = node.leafTriOffset();
            const int triCount = node.leafTriCount();
            for (int j = 0; j < triCount; j++)
            {
                float t;
                glm::vec3 triNormal;
                if (triangleIntersectionTest(objRay, tris[triBase + j], t, triNormal))
                {
                    if (t < result.t)
                    {
                        result.t      = t;
                        result.normal = triNormal;
                        result.hit    = true;
                    }
                }
            }
        }
        else
        {
            // Push both children (top of stack = right, visited first).
            // The build-time depth clamp keeps the stack within capacity;
            // the guard is defense-in-depth against a malformed tree.
            if (sp < kMaxBvhStackDepth - 1)
            {
                stack[sp++] = node.childL();
                stack[sp++] = node.childR();
            }
        }
    }

    return result;
}

// Host-side construction + GPU memory management, implemented in bvh.cu.
namespace bvh
{
    // Build one mesh's BVH, appending its nodes to out.hostNodes and its
    // reordered triangles to out.hostTriangles.  The mesh's triangle slice
    // is read from geom.meshTriangleOffset/count.  Returns the root node
    // index, or -1 for an empty mesh.
    int  buildMeshBvh(BvhBuffers& out,
                      const std::vector<Triangle>& hostTris);

    // Build every mesh's BVH into out.hostNodes/out.hostTriangles and fill
    // out.hostBvhMeta.  Empty meshes keep rootNodeIndex = -1.
    void buildSceneBvh(BvhBuffers& out,
                       const std::vector<Triangle>& hostTris,
                       const std::vector<Geom>& geoms);

    // Upload the host node + meta buffers to device memory.
    void uploadToDevice(BvhBuffers& b);
    void freeDevice(BvhBuffers& b);
} // namespace bvh
