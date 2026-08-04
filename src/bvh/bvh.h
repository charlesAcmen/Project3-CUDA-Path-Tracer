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
// / meshTriangleCount still slice the same triangle set.  The O(N)
// computeIntersections kernel is order-independent and stays correct on
// the reordered buffer, so both paths share one deviceTriangles.
// ====================================================================

#include "aabb.h"
#include "constants.h"               // RAY_EPSILON, LARGE_T
#include "sceneStructs.h"            // Ray, Triangle, Geom
#include "intersection/triangle.h"   // triangleIntersectionTest

#include <vector>

// Explicit-stack capacity (compile-time).  Build-time depth clamp to
// [1, 63] guarantees the stack can never overflow for a built tree.
constexpr int kMaxBvhStackDepth = 64;

// Internal node: bounds + children.  Leaf: bounds + triangle chunk.
// left/right are overloaded by isLeaf:
//   internal: left = child node index, right = child node index
//   leaf:     left = absolute triangle offset, right = triangle count
struct BvhNode
{
    AABB bounds;
    int  left;
    int  right;
    int  isLeaf;
};

// Per-geom BVH metadata.  rootNodeIndex = -1 → empty mesh (kernel skips).
struct BvhMeta
{
    int rootNodeIndex = -1;
};

// BVH buffers: host build output + device upload.
// hostTriangles holds the REORDERED flat triangle array (per-mesh
// flatten) — uploaded to the renderer as deviceTriangles in M4.
struct BvhBuffers
{
    // Device-side buffers (allocated in uploadToDevice, freed in freeDevice)
    BvhNode* deviceNodes = nullptr;  int numNodes = 0;
    BvhMeta* deviceBvhMeta = nullptr; int numGeoms = 0;
    std::vector<BvhNode>  hostNodes;      // construction output
    std::vector<BvhMeta>  hostBvhMeta;    // per-geom metadata 
    std::vector<Triangle> hostTriangles;  // reordered flat triangles
};

/**
 * Iterative closest-hit BVH traversal — THE algorithm both the GPU
 * kernel and the host test execute, so correctness is validated once.
 *
 * The AABB test clips to [RAY_EPSILON, closestT]: the near bound skips
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
 * @param closestT      [in] initial far plane (e.g. best t so far);
 *                      [out] closest hit distance on success
 * @param objNormal     [out] object-space shading normal of the hit
 * @return              true on hit with t < (input) closestT
 */
__host__ __device__ inline bool traverseBvhClosest(
    const Ray& objRay,
    const BvhNode* nodes,
    int rootNodeIndex,
    const Triangle* tris,
    float& closestT,
    glm::vec3& objNormal)
{
    if (rootNodeIndex < 0) return false;

    const glm::vec3 invDir(
        1.0f / objRay.direction.x,
        1.0f / objRay.direction.y,
        1.0f / objRay.direction.z);

    int stack[kMaxBvhStackDepth];
    int sp = 0;//stack pointer 
    stack[sp++] = rootNodeIndex;

    bool hit = false;

    while (sp > 0)
    {
        //pop the top node off the stack
        const int nodeIndex = stack[--sp];
        const BvhNode& node = nodes[nodeIndex];

        //near side:RAY_EPSILON, far side:closestT
        if (!intersectRayAABB(objRay.origin, invDir, node.bounds, RAY_EPSILON, closestT))
            //skip the node and its subtree if the ray misses the node's AABB
            continue;

        if (node.isLeaf)
        {
            // Sequential chunk — leaf triangles were reordered into a
            // contiguous run by the build's flatten pass.
            for (int j = 0; j < node.right; j++)
            {
                float t;
                glm::vec3 triNormal;
                if (triangleIntersectionTest(objRay, tris[node.left + j], t, triNormal))
                {
                    if (t < closestT)
                    {
                        closestT  = t;
                        objNormal = triNormal;
                        hit = true;
                    }
                }
            }
        }
        else
        {
            // Push both children (top of stack = right, visited first).
            // Build-time depth clamp keeps the stack within capacity; the
            // guard is defense-in-depth against a malformed tree.
            if (sp < kMaxBvhStackDepth - 1)
            {
                stack[sp++] = node.left;
                stack[sp++] = node.right;
            }
        }
    }

    return hit;
}

// Host-side construction + GPU memory management, implemented in bvh.cu.
namespace bvh
{
    // Build one mesh's BVH and append its nodes to `out`.  The mesh's
    // triangles are ALSO appended to `dstTris` reordered into
    // leaf-contiguous chunks, and every leaf is rewritten to point at its
    // chunk (traverseBvhClosest reads tris[node.left + j]).  Returns the
    // root node index, or -1 for an empty mesh.
    int  buildMeshBvh(std::vector<BvhNode>& out,
                      std::vector<Triangle>& dstTris,
                      const std::vector<Triangle>& srcTris,
                      int triOffset, int triCount,
                      int maxDepth, int leafSize);

    // Build every mesh's BVH, reorder its triangles into
    // out.hostTriangles, and fill out.hostBvhMeta.
    void buildSceneBvh(BvhBuffers& out,
                       const std::vector<Triangle>& hostTris,
                       const std::vector<Geom>& geoms,
                       int maxDepth, int leafSize);

    // Upload node + meta buffers to device memory.
    void uploadToDevice(BvhBuffers& b);
    void freeDevice(BvhBuffers& b);
} // namespace bvh
