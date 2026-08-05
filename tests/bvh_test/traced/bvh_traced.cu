// ====================================================================
// BVH Construction — host-side build + GPU memory management
//
// TRACED COPY of src/bvh/bvh.cu — keep in sync with production.  The
// only difference is the TRACE() statements recording the intermediate
// state of the build: node allocation + bounds, SAH per-axis split
// search, order-array permutations, and the flatten pass.  Production
// bvh.cu has none of this.
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

#include "bvh_traced.h"
#include "trace.h"

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

// Everything the recursive build needs that does NOT change during one
// mesh's recursion: the shared buffers plus the per-mesh build settings.
// Bundling them collapses the 6+ build-time params that were threaded
// through buildRecursive / flattenRecursive into a single context object.
struct BvhBuildContext
{
    std::vector<BvhNode>*         nodes;     // construction output (hostNodes)
    const std::vector<Triangle>*  tris;      // source triangles (hostTris, read-only)
    std::vector<int>*             order;     // permutable triangle indices (per-mesh)
    std::vector<Triangle>*        dstTris;   // reordered output (hostTriangles)
    int triOffset;   // this mesh's triangle base index
    int maxDepth;
    int leafSize;
};

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
int buildRecursive(BvhBuildContext& ctx,
                   int begin, int end, int depth)
{
    // Unpack the context into local names so the body reads as before.
    std::vector<BvhNode>&        nodes     = *ctx.nodes;
    const std::vector<Triangle>& tris      = *ctx.tris;
    std::vector<int>&            order     = *ctx.order;
    const int triOffset = ctx.triOffset;
    const int maxDepth  = ctx.maxDepth;
    const int leafSize  = ctx.leafSize;

    const int n = end - begin;//number of triangles in this node's range

    AABB bounds;
    for (int i = begin; i < end; i++)
        bounds.expand(tris[order[i]]);//access the triangle by its index in the order array

    const int nodeIndex = (int)nodes.size();
    nodes.push_back(BvhNode{});
    nodes[nodeIndex].bounds = bounds;
    nodes[nodeIndex].isLeaf = 0;

    trace::indent(depth);
    TRACE(1, "[build] node #%d created — order[%d,%d) n=%d depth=%d\n", nodeIndex, begin, end, n, depth);
    trace::indent(depth);
    TRACE(1, "        bounds.min=(%.4g,%.4g,%.4g) max=(%.4g,%.4g,%.4g)  (accumulated by expand over %d tris)\n",
          bounds.min.x, bounds.min.y, bounds.min.z, bounds.max.x, bounds.max.y, bounds.max.z, n);

    auto makeLeaf = [&]() {
        //&:access n, triOffset, begin
        nodes[nodeIndex].isLeaf = 1;//mark this node as a leaf
        nodes[nodeIndex].left  = triOffset + begin;   // stores the offset into the order array as starting index of the triangles in this leaf
        nodes[nodeIndex].right = n; // stores the number of triangles in this leaf
    };

    //too few triangles or too deep → make a leaf
    if (n <= leafSize || depth >= maxDepth)
    {
        trace::indent(depth);
        TRACE(1, "        -> LEAF early (n=%d <= leafSize=%d or depth=%d >= maxDepth=%d): left=%d right=%d\n",
              n, leafSize, depth, maxDepth, triOffset + begin, n);
        makeLeaf();
        return nodeIndex;//return the index of this node in the nodes vector
    }

    // ---- SAH: find the best (axis, split) ----
    // Try the axes in order of decreasing extent first.
    const glm::vec3 ext = bounds.max - bounds.min;
    int axes[3] = { 0, 1, 2 };
    if (ext[axes[0]] < ext[axes[1]]) std::swap(axes[0], axes[1]);
    if (ext[axes[1]] < ext[axes[2]]) std::swap(axes[1], axes[2]);
    if (ext[axes[0]] < ext[axes[1]]) std::swap(axes[0], axes[1]);
    trace::indent(depth);
    TRACE(1, "        extent=(%.4g,%.4g,%.4g) -> axis order: %d, %d, %d (longest first)\n",
          ext.x, ext.y, ext.z, axes[0], axes[1], axes[2]);

    const float nodeArea = bounds.surfaceArea();   // degenerate → 1.0 guard
    const float leafCost = (float)n;    //cost of making a leaf with n triangles (1.0 + area * n / area = 1.0 + n)
    trace::indent(depth);
    TRACE(1, "        nodeArea=%.4g  leafCost=%.4g\n", nodeArea, leafCost);

    int bestAxis = -1, bestSplit = -1;   // bestSplit = element count in the LEFT half
    float bestCost = FLT_MAX;

    std::vector<AABB> pref(n);      // pref[k]  = AABB of order[begin, begin+k)
    std::vector<AABB> suff(n + 1);  // suff[k]  = AABB of order[begin+k, end)

    for (int a = 0; a < 3; a++)
    {
        const int axis = axes[a];
        //sort order array by the centroid of the triangles along the current axis,the smallest centroid first
        std::sort(order.begin() + begin, order.begin() + end,
            [&](int lhs, int rhs) {
                return centroidComponent(tris[lhs], axis) < centroidComponent(tris[rhs], axis);
            });

        TRACE(2, "        axis %d: sorted order[%d..%d) =", axis, begin, end);
        for (int i = begin; i < end; i++) TRACE(2, " %d", order[i]);
        TRACE(2, "\n");

        pref[0] = AABB();
        pref[0].expand(tris[order[begin]]);
        for (int k = 1; k < n; k++)
        {
            pref[k] = pref[k - 1];//copy the previous AABB
            pref[k].expand(tris[order[begin + k]]);//expand the AABB to include the next triangle
        }
        suff[n] = AABB();
        for (int k = n - 1; k >= 0; k--)
        {
            suff[k] = suff[k + 1];
            suff[k].expand(tris[order[begin + k]]);
        }

        int axisBestK = -1;
        float axisBestCost = FLT_MAX;
        for (int k = 1; k < n; k++)
        {
            const float areaL = pref[k - 1].surfaceArea();
            const float areaR = suff[k].surfaceArea();
            //cost = 1 + area of left child * number of triangles in left child + area of right child * number of triangles in right child
            //area of parent node
            const float cost  = 1.0f + (areaL * k + areaR * (n - k)) / nodeArea;
            if (cost < axisBestCost)
            {
                axisBestCost  = cost;
                axisBestK = k;
            }
            TRACE(2, "          split k=%d: areaL=%.4g areaR=%.4g cost=%.4g%s\n",
                  k, areaL, areaR, cost, (cost < axisBestCost) ? "  <-- best so far" : "");
        }
        trace::indent(depth);
        TRACE(1, "        axis %d: best split k=%d cost=%.4g\n", axis, axisBestK, axisBestCost);

        if (axisBestCost < bestCost)
        {
            bestCost  = axisBestCost;
            bestAxis  = axis;
            bestSplit = axisBestK;
        }
    }

    trace::indent(depth);
    TRACE(1, "        SAH decision: best axis=%d split=%d cost=%.4g %s leafCost=%.4g\n",
          bestAxis, bestSplit, bestCost,
          (bestAxis < 0 || bestCost >= leafCost) ? ">= (i.e. splitting not worth it) vs" : "<",
          leafCost);

    //making a leaf is cheaper than splitting → make a leaf
    if (bestAxis < 0 || bestCost >= leafCost)
    {
        trace::indent(depth);
        TRACE(1, "        -> LEAF (SAH): left=%d right=%d\n", triOffset + begin, n);
        makeLeaf();
        return nodeIndex;
    }

    // Re-sort the range along the winning axis; the split at begin+bestSplit
    // partitions it into two contiguous halves (both non-empty).
    std::sort(order.begin() + begin, order.begin() + end,
        [&](int lhs, int rhs) {
            return centroidComponent(tris[lhs], bestAxis) < centroidComponent(tris[rhs], bestAxis);
        });
    TRACE(2, "        re-sorted along axis %d for the final partition -> left [%d,%d) right [%d,%d)\n",
          bestAxis, begin, begin + bestSplit, begin + bestSplit, end);

    // Run the recursion to completion FIRST, then write the child indices
    // back.  The recursive calls append to `nodes` and can reallocate it,
    // so the writes must not interleave with them — storing the results in
    // temporaries makes this correct regardless of the host C++ standard's
    // assignment sequencing rules (RHS-before-LHS is only guaranteed since
    // C++17; this project's tests compile with older defaults).
    const int leftChild  = buildRecursive(ctx, begin, begin + bestSplit, depth + 1);
    const int rightChild = buildRecursive(ctx, begin + bestSplit, end, depth + 1);
    nodes[nodeIndex].left  = leftChild;
    nodes[nodeIndex].right = rightChild;
    trace::indent(depth);
    TRACE(1, "        node #%d -> INTERNAL: left=#%d right=#%d\n", nodeIndex, leftChild, rightChild);
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
void flattenRecursive(BvhBuildContext& ctx,
                      int nodeIndex,
                      int dstBase,
                      int& cursor)
{
    std::vector<BvhNode>&        nodes   = *ctx.nodes;
    const std::vector<int>&      order   = *ctx.order;
    const std::vector<Triangle>& srcTris = *ctx.tris;
    std::vector<Triangle>&       dstTris = *ctx.dstTris;
    const int triOffset = ctx.triOffset;

    BvhNode& node = nodes[nodeIndex];
    TRACE(1, "  [flatten] node #%d (isLeaf=%d)  dstBase=%d cursor=%d\n",
          nodeIndex, node.isLeaf, dstBase, cursor);
    if (node.isLeaf)
    {
        const int orderStart = node.left - triOffset;   // offset into the mesh's order array
        const int count      = node.right;
        TRACE(1, "    leaf chunk: order[%d..%d) -> copy %d tris to dst[%d..%d)\n",
              orderStart, orderStart + count, count,
              dstBase + cursor, dstBase + cursor + count);
        for (int j = 0; j < count; j++)
            dstTris[dstBase + cursor + j] = srcTris[order[orderStart + j]];
        TRACE(1, "    leaf node.left %d -> %d (absolute offset into reordered buffer); cursor %d -> %d\n",
              node.left, dstBase + cursor, cursor, cursor + count);
        node.left = dstBase + cursor;   // now points into the reordered buffer
        cursor += count;
    }
    else
    {
        flattenRecursive(ctx, node.left,  dstBase, cursor);
        flattenRecursive(ctx, node.right, dstBase, cursor);
    }
}

} // namespace:private in the current translation unit,not shown to other cpp files

int buildMeshBvh(std::vector<BvhNode>& out,//hostNodes
                 std::vector<Triangle>& dstTris,//hostTriangles
                 const std::vector<Triangle>& srcTris,//hostTris
                 int triOffset, int triCount,
                 int maxDepth, int leafSize)
{
    // Clamp so the tree can never overflow the traversal stack and the
    // leaf size stays sane.  This is the single clamp point for a direct
    // call; buildSceneBvh relies on it and does not clamp itself.
    maxDepth = std::min(63, std::max(1, maxDepth));
    leafSize = std::min(64, std::max(1, leafSize));

    TRACE(1, "[buildMeshBvh] triOffset=%d triCount=%d  (after clamp) maxDepth=%d leafSize=%d\n",
          triOffset, triCount, maxDepth, leafSize);

    if (triCount <= 0) { TRACE(1, "[buildMeshBvh] empty mesh -> root=-1\n"); return -1; }   // empty mesh → no root

    // Bundle the shared buffers + this mesh's build settings into a context
    // so the recursive build/flatten no longer thread them as parameters.
    BvhBuildContext ctx;
    ctx.nodes     = &out;
    ctx.tris      = &srcTris;
    ctx.dstTris   = &dstTris;
    ctx.triOffset = triOffset;
    ctx.maxDepth  = maxDepth;
    ctx.leafSize  = leafSize;

    std::vector<int> order(triCount);
    for (int i = 0; i < triCount; i++)
        order[i] = triOffset + i;//indices rather than triangle structs themselves
    ctx.order = &order;
    TRACE(1, "[buildMeshBvh] order[0..%d) initialized = %d..%d\n", triCount, triOffset, triOffset + triCount - 1);

    const int root = buildRecursive(ctx, 0, triCount, 0);

    // Flatten: append this mesh's reordered triangles to dstTris and fix
    // the leaves to reference the contiguous chunks.
    const int dstBase = (int)dstTris.size();
    dstTris.resize(dstBase + triCount);
    int cursor = 0;
    TRACE(1, "[buildMeshBvh] flatten pass: dstBase=%d, buffer grows to %zu tris\n", dstBase, dstTris.size());
    flattenRecursive(ctx, root, dstBase, cursor);
    TRACE(1, "[buildMeshBvh] flatten done: cursor=%d (== triCount=%d), root=#%d\n", cursor, triCount, root);

    return root;
}

void buildSceneBvh(BvhBuffers& out,
                   const std::vector<Triangle>& hostTris,
                   const std::vector<Geom>& geoms,
                   int maxDepth, int leafSize)
{
    // maxDepth/leafSize are clamped once in buildMeshBvh (per-mesh entry);
    // here they are only forwarded.

    out.hostNodes.clear();
    out.hostBvhMeta.assign(geoms.size(), BvhMeta{});   // default root = -1 (empty)
    out.hostTriangles.clear();
    out.hostTriangles.reserve(hostTris.size());// preallocate for the reordered output

    TRACE(1, "[buildSceneBvh] %zu geoms, %zu source triangles, maxDepth=%d leafSize=%d\n",
          geoms.size(), hostTris.size(), maxDepth, leafSize);

    for (size_t i = 0; i < geoms.size(); i++)
    {
        const int triOffset = geoms[i].meshTriangleOffset;
        const int triCount  = geoms[i].meshTriangleCount;
        if (triCount <= 0)   // meta stays rootNodeIndex = -1
        {
            TRACE(1, "  geom #%zu: empty (offset=%d count=%d) -> rootNodeIndex=-1\n", i, triOffset, triCount);
            continue;
        }

        const int root = buildMeshBvh(out.hostNodes, out.hostTriangles, hostTris,
                                      triOffset, triCount, maxDepth, leafSize);

        out.hostBvhMeta[i].rootNodeIndex = root;
        TRACE(1, "  geom #%zu: offset=%d count=%d -> rootNodeIndex=#%d\n", i, triOffset, triCount, root);
    }

    TRACE(1, "[buildSceneBvh] done: hostNodes=%zu hostTriangles=%zu hostBvhMeta=%zu\n",
          out.hostNodes.size(), out.hostTriangles.size(), out.hostBvhMeta.size());
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
        TRACE(1, "[uploadToDevice] uploaded %d nodes\n", b.numNodes);
    }
    if (!b.hostBvhMeta.empty())
    {
        cudaMalloc(&b.deviceBvhMeta, b.hostBvhMeta.size() * sizeof(BvhMeta));
        cudaMemcpy(b.deviceBvhMeta, b.hostBvhMeta.data(),
                   b.hostBvhMeta.size() * sizeof(BvhMeta), cudaMemcpyHostToDevice);
        b.numGeoms = (int)b.hostBvhMeta.size();
        checkCUDAError("bvh upload meta");
        TRACE(1, "[uploadToDevice] uploaded %d meta entries\n", b.numGeoms);
    }
}

void freeDevice(BvhBuffers& b)
{
    TRACE(1, "[freeDevice] freeing %d nodes, %d metas\n", b.numNodes, b.numGeoms);
    if (b.deviceNodes)    { cudaFree(b.deviceNodes);    b.deviceNodes = nullptr; }
    if (b.deviceBvhMeta)  { cudaFree(b.deviceBvhMeta);  b.deviceBvhMeta = nullptr; }
    b.numNodes = 0;
    b.numGeoms = 0;
}

} // namespace bvh
