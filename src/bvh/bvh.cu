// ====================================================================
// BVH Construction — host-side build + GPU memory management
//
// CPU builds the BVH (exhaustive Surface Area Heuristic), GPU traverses
// it iteratively (traverseBvhClosest in bvh.h).  Construction is pure
// host code: scenes are ≤ ~1248 triangles, so an exhaustive per-axis（每个轴向上）
// SAH（穷举SAH扫描） sweep is a few milliseconds and there is no need for a binning
// approximation（分桶近似）.
//
// Triangle layout: the build permutes a per-mesh `order`（三角形的索引而非三角形结构体本身）
// array in place so each leaf's triangles occupy a CONTIGUOUS range of it.  
// buildMeshBvh then runs a FLATTEN pass (post-order DFS，把真正的三角形数据写入连续的内存块) 
// that writes each leaf's triangles into a contiguous chunk of the reordered output buffer 
// and rewrites the leaf to point at that chunk.  Leaves therefore reference
// sequential memory runs — cache-friendly GPU access, and exactly what
// traverseBvhClosest's `tris[node.left + j]` expects.
//
// The reorder stays within each mesh (counts and concatenation order are
// unchanged), so Geom::meshTriangleOffset / meshTriangleCount still slice
// the same triangle set and the O(N) computeIntersections kernel remains
// correct on the reordered buffer.
// ====================================================================

#include "bvh/bvh.h"

#include "utils/utilities.h"   // checkCUDAError (no-op in Release)

#include <algorithm>
#include <cfloat>
#include <vector>

namespace bvh {

namespace {

// Centroid of the triangle projected onto the requested axis.
float centroidComponent(const Triangle& t, int axis)
{
    const glm::vec3 c = (t.v0 + t.v1 + t.v2) * (1.0f / 3.0f);
    return axis == 0 ? c.x : (axis == 1 ? c.y : c.z);
}

/**
 * Recursive SAH build over the contiguous range [begin, end) of the
 * mesh's `order` array (which holds absolute triangle indices).  The
 * recursion PERMUTES `order` in place, so after a node becomes a leaf,
 * its triangles occupy exactly order[begin .. begin+n) — a contiguous
 * run.  A leaf therefore stores left = triOffset + begin (offset of the
 * run into the mesh's order array) and right = count.
 *
 * Exhaustive split search: for each axis (longest extent first) sort the
 * range by centroid, compute prefix/suffix child AABBs, and score every
 * split k as
 *     cost = 1 + (areaL * nL + areaR * nR) / areaNode
 * vs. the leaf cost n.  Split iff the best SAH cost beats the leaf cost.
 */
int buildRecursive(std::vector<BvhNode>& nodes,
                   const std::vector<Triangle>& tris,
                   std::vector<int>& order,   // permutable working array
                   int begin, int end,
                   int triOffset,
                   int maxDepth, int leafSize, int depth)
{
    const int n = end - begin;

    AABB bounds;
    for (int i = begin; i < end; i++)
        bounds.expand(tris[order[i]]);

    const int nodeIndex = (int)nodes.size();
    nodes.push_back(BvhNode{});
    nodes[nodeIndex].bounds = bounds;
    nodes[nodeIndex].isLeaf = 0;

    auto makeLeaf = [&]() {
        nodes[nodeIndex].isLeaf = 1;
        nodes[nodeIndex].left  = triOffset + begin;   // run offset into the order array
        nodes[nodeIndex].right = n;
    };

    if (n <= leafSize || depth >= maxDepth)
    {
        makeLeaf();
        return nodeIndex;
    }

    // ---- SAH: find the best (axis, split) ----
    // Try the axes in order of decreasing extent first.
    const glm::vec3 ext = bounds.max - bounds.min;
    int axes[3] = { 0, 1, 2 };
    if (ext[axes[0]] < ext[axes[1]]) std::swap(axes[0], axes[1]);
    if (ext[axes[1]] < ext[axes[2]]) std::swap(axes[1], axes[2]);
    if (ext[axes[0]] < ext[axes[1]]) std::swap(axes[0], axes[1]);

    const float nodeArea = bounds.surfaceArea();   // degenerate → 1.0 guard
    const float leafCost = (float)n;

    int bestAxis = -1, bestSplit = -1;   // bestSplit = element count in the LEFT half
    float bestCost = FLT_MAX;

    std::vector<AABB> pref(n);      // pref[k]  = AABB of order[begin, begin+k)
    std::vector<AABB> suff(n + 1);  // suff[k]  = AABB of order[begin+k, end)

    for (int a = 0; a < 3; a++)
    {
        const int axis = axes[a];
        std::sort(order.begin() + begin, order.begin() + end,
            [&](int lhs, int rhs) {
                return centroidComponent(tris[lhs], axis) < centroidComponent(tris[rhs], axis);
            });

        pref[0] = AABB();
        pref[0].expand(tris[order[begin]]);
        for (int k = 1; k < n; k++)
        {
            pref[k] = pref[k - 1];
            pref[k].expand(tris[order[begin + k]]);
        }
        suff[n] = AABB();
        for (int k = n - 1; k >= 0; k--)
        {
            suff[k] = suff[k + 1];
            suff[k].expand(tris[order[begin + k]]);
        }

        for (int k = 1; k < n; k++)
        {
            const float areaL = pref[k - 1].surfaceArea();
            const float areaR = suff[k].surfaceArea();
            const float cost  = 1.0f + (areaL * k + areaR * (n - k)) / nodeArea;
            if (cost < bestCost)
            {
                bestCost  = cost;
                bestAxis  = axis;
                bestSplit = k;
            }
        }
    }

    if (bestAxis < 0 || bestCost >= leafCost)
    {
        makeLeaf();
        return nodeIndex;
    }

    // Re-sort the range along the winning axis; the split at begin+bestSplit
    // partitions it into two contiguous halves (both non-empty).
    std::sort(order.begin() + begin, order.begin() + end,
        [&](int lhs, int rhs) {
            return centroidComponent(tris[lhs], bestAxis) < centroidComponent(tris[rhs], bestAxis);
        });

    // Run the recursion to completion FIRST, then write the child indices
    // back.  The recursive calls append to `nodes` and can reallocate it,
    // so the writes must not interleave with them — storing the results in
    // temporaries makes this correct regardless of the host C++ standard's
    // assignment sequencing rules (RHS-before-LHS is only guaranteed since
    // C++17; this project's tests compile with older defaults).
    const int leftChild  = buildRecursive(nodes, tris, order, begin, begin + bestSplit, triOffset, maxDepth, leafSize, depth + 1);
    const int rightChild = buildRecursive(nodes, tris, order, begin + bestSplit, end, triOffset, maxDepth, leafSize, depth + 1);
    nodes[nodeIndex].left  = leftChild;
    nodes[nodeIndex].right = rightChild;
    return nodeIndex;
}

/**
 * Post-order DFS: writes each leaf's triangles into a contiguous chunk of
 * the reordered output buffer and rewrites the leaf's left to the chunk
 * start.  A leaf's triangles are order[orderStart .. orderStart+count)
 * — contiguous by construction of the build — so the copy is exact, not
 * a "front + count" guess.  `cursor` is the running write offset within
 * this mesh's output region [dstBase, dstBase + triCount).
 */
void flattenRecursive(std::vector<BvhNode>& nodes,
                      int nodeIndex,
                      const std::vector<int>& order,
                      const std::vector<Triangle>& srcTris,
                      std::vector<Triangle>& dstTris,
                      int triOffset,
                      int dstBase,
                      int& cursor)
{
    BvhNode& node = nodes[nodeIndex];
    if (node.isLeaf)
    {
        const int orderStart = node.left - triOffset;   // offset into the mesh's order array
        const int count      = node.right;
        for (int j = 0; j < count; j++)
            dstTris[dstBase + cursor + j] = srcTris[order[orderStart + j]];
        node.left = dstBase + cursor;   // now points into the reordered buffer
        cursor += count;
    }
    else
    {
        flattenRecursive(nodes, node.left,  order, srcTris, dstTris, triOffset, dstBase, cursor);
        flattenRecursive(nodes, node.right, order, srcTris, dstTris, triOffset, dstBase, cursor);
    }
}

} // namespace

int buildMeshBvh(std::vector<BvhNode>& out,
                 std::vector<Triangle>& dstTris,
                 const std::vector<Triangle>& srcTris,
                 int triOffset, int triCount,
                 int maxDepth, int leafSize)
{
    // Clamp so the tree can never overflow the traversal stack and the
    // leaf size stays sane.
    maxDepth = std::min(63, std::max(1, maxDepth));
    leafSize = std::min(64, std::max(1, leafSize));

    if (triCount <= 0) return -1;   // empty mesh → no root

    std::vector<int> order(triCount);
    for (int i = 0; i < triCount; i++)
        order[i] = triOffset + i;

    const int root = buildRecursive(out, srcTris, order, 0, triCount,
                                    triOffset, maxDepth, leafSize, 0);

    // Flatten: append this mesh's reordered triangles to dstTris and fix
    // the leaves to reference the contiguous chunks.
    const int dstBase = (int)dstTris.size();
    dstTris.resize(dstBase + triCount);
    int cursor = 0;
    flattenRecursive(out, root, order, srcTris, dstTris, triOffset, dstBase, cursor);

    return root;
}

void buildSceneBvh(BvhBuffers& out,
                   const std::vector<Triangle>& hostTris,
                   const std::vector<Geom>& geoms,
                   int maxDepth, int leafSize)
{
    maxDepth = std::min(63, std::max(1, maxDepth));
    leafSize = std::min(64, std::max(1, leafSize));

    out.hostNodes.clear();
    out.hostBvhMeta.assign(geoms.size(), BvhMeta{});   // default root = -1 (empty)
    out.hostTriangles.clear();
    out.hostTriangles.reserve(hostTris.size());

    for (size_t i = 0; i < geoms.size(); i++)
    {
        const int triOffset = geoms[i].meshTriangleOffset;
        const int triCount  = geoms[i].meshTriangleCount;
        if (triCount <= 0) continue;   // meta stays rootNodeIndex = -1

        const int nodeStart = (int)out.hostNodes.size();
        const int root = buildMeshBvh(out.hostNodes, out.hostTriangles, hostTris,
                                      triOffset, triCount, maxDepth, leafSize);

        out.hostBvhMeta[i].rootNodeIndex = root;
        out.hostBvhMeta[i].nodeCount     = (int)out.hostNodes.size() - nodeStart;
    }
}

void uploadToDevice(BvhBuffers& b)
{
    if (!b.hostNodes.empty())
    {
        cudaMalloc(&b.deviceNodes, b.hostNodes.size() * sizeof(BvhNode));
        cudaMemcpy(b.deviceNodes, b.hostNodes.data(),
                   b.hostNodes.size() * sizeof(BvhNode), cudaMemcpyHostToDevice);
        b.numNodes = (int)b.hostNodes.size();
        checkCUDAError("bvh upload nodes");
    }
    if (!b.hostBvhMeta.empty())
    {
        cudaMalloc(&b.deviceBvhMeta, b.hostBvhMeta.size() * sizeof(BvhMeta));
        cudaMemcpy(b.deviceBvhMeta, b.hostBvhMeta.data(),
                   b.hostBvhMeta.size() * sizeof(BvhMeta), cudaMemcpyHostToDevice);
        b.numGeoms = (int)b.hostBvhMeta.size();
        checkCUDAError("bvh upload meta");
    }
}

void freeDevice(BvhBuffers& b)
{
    if (b.deviceNodes)    { cudaFree(b.deviceNodes);    b.deviceNodes = nullptr; }
    if (b.deviceBvhMeta)  { cudaFree(b.deviceBvhMeta);  b.deviceBvhMeta = nullptr; }
    b.numNodes = 0;
    b.numGeoms = 0;
}

} // namespace bvh
