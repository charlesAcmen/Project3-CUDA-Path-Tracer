// ====================================================================
// BVH Construction — host-side build + GPU memory management
//
// CPU builds the BVH (exhaustive Surface Area Heuristic), GPU traverses
// it iteratively (traverseBvhClosest in bvh.h).  Construction is pure
// host code: scenes are ≤ ~1248 triangles, so an exhaustive per-axis（每个轴向上）
// SAH（穷举SAH扫描） sweep is a few milliseconds and there is no need for a binning
// approximation（分桶近似）.
//
// World-space bake: buildSceneBvh transforms every mesh's triangles from
// object space to world space (vertices via the geom's `transform`, vertex
// normals via `invTranspose` — the same normal transform the old kernel
// applied per hit) and tags each with its materialId.  One tree is then
// built over the COMBINED world-space array, so the GPU kernel runs a
// single closest-hit traversal per ray with no per-mesh loop or ray
// transformation.
//
// Triangle layout: the build permutes a per-mesh `order`（三角形的索引而非三角形结构体本身）
// array in place so each leaf's triangles occupy a CONTIGUOUS range of it.  
// buildMeshBvh then runs a FLATTEN pass (post-order DFS，把真正的三角形数据写入连续的内存块) 
// that writes each leaf's triangles into a contiguous chunk of the reordered output buffer 
// and rewrites the leaf to point at that chunk.  Leaves therefore reference
// sequential memory runs — cache-friendly GPU access, and exactly what
// traverseBvhClosest's `tris[node.left + j]` expects.
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

// Everything the recursive build needs that does NOT change during the
// recursion: the shared buffers plus the build settings.  Bundling them
// collapses the build-time params that were threaded through
// buildRecursive / flattenRecursive into a single context object.
struct BvhBuildContext
{
    std::vector<BvhNode>*         nodes;     // construction output (hostNodes)
    const std::vector<Triangle>*  tris;      // source triangles (world-space bake, read-only)
    std::vector<int>*             order;     // permutable triangle indices
    std::vector<Triangle>*        dstTris;   // reordered output (hostTriangles)
    int triOffset = 0;   // base index into the triangle array (0: single tree over all triangles)
};

/**
 * Recursive SAH build over the contiguous range [begin, end) of the
 * `order` array (which holds absolute triangle indices).  The
 * recursion PERMUTES `order` in place, so after a node becomes a leaf,
 * its triangles occupy exactly order[begin .. begin+n) — a contiguous
 * run.  A leaf therefore stores left = triOffset + begin (offset of the
 * run into the triangle array) and right = count.
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

    const int n = end - begin;   // number of triangles in this node's range

    AABB bounds;
    for (int i = begin; i < end; i++)
        bounds.expand(tris[order[i]]);   // access the triangle by its index in the order array

    const int nodeIndex = (int)nodes.size();
    nodes.push_back(BvhNode{});
    nodes[nodeIndex].bounds = bounds;
    nodes[nodeIndex].isLeaf = false;

    auto makeLeaf = [&]() {
        nodes[nodeIndex].isLeaf = true;
        nodes[nodeIndex].left  = triOffset + begin;   // start of the leaf's run in the order array
        nodes[nodeIndex].right = n;                    // number of triangles in this leaf
    };

    // Too few triangles or too deep → make a leaf.
    if (n <= kBvhLeafSize || depth >= kBvhMaxDepth)
    {
        makeLeaf();
        return nodeIndex;   // return the index of this node in the nodes vector
    }

    // ---- SAH: find the best (axis, split) ----
    // Try the axes in order of decreasing extent first.
    const glm::vec3 ext = bounds.max - bounds.min;
    int axes[3] = { 0, 1, 2 };
    if (ext[axes[0]] < ext[axes[1]]) std::swap(axes[0], axes[1]);
    if (ext[axes[1]] < ext[axes[2]]) std::swap(axes[1], axes[2]);
    if (ext[axes[0]] < ext[axes[1]]) std::swap(axes[0], axes[1]);

    const float nodeArea = bounds.surfaceArea();   // degenerate → 1.0 guard
    const float leafCost = (float)n;               // cost of not splitting

    int bestAxis = -1, bestSplit = -1;   // bestSplit = element count in the LEFT half
    float bestCost = FLT_MAX;

    std::vector<AABB> pref(n);      // pref[k]  = AABB of order[begin, begin+k)
    std::vector<AABB> suff(n + 1);  // suff[k]  = AABB of order[begin+k, end)

    for (int a = 0; a < 3; a++)
    {
        const int axis = axes[a];
        // Sort the range by triangle centroid along this axis (ascending).
        std::sort(order.begin() + begin, order.begin() + end,
            [&](int lhs, int rhs) {
                return centroidComponent(tris[lhs], axis) < centroidComponent(tris[rhs], axis);
            });

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

        for (int k = 1; k < n; k++)
        {
            const float areaL = pref[k - 1].surfaceArea();
            const float areaR = suff[k].surfaceArea();
            // SAH cost of splitting at k (left has k tris, right has n-k):
            //   1 + (areaL * k + areaR * (n - k)) / nodeArea
            const float cost  = 1.0f + (areaL * k + areaR * (n - k)) / nodeArea;
            if (cost < bestCost)
            {
                bestCost  = cost;
                bestAxis  = axis;
                bestSplit = k;
            }
        }
    }

    // Making a leaf is cheaper than splitting → make a leaf.
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
    const int leftChild  = buildRecursive(ctx, begin, begin + bestSplit, depth + 1);
    const int rightChild = buildRecursive(ctx, begin + bestSplit, end, depth + 1);
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
 * the output region [dstBase, dstBase + triCount).
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
    if (node.isLeaf)
    {
        const int orderStart = node.leafTriOffset() - triOffset;   // offset into the mesh's order array
        const int count      = node.leafTriCount();
        for (int j = 0; j < count; j++)
            dstTris[dstBase + cursor + j] = srcTris[order[orderStart + j]];
        node.left = dstBase + cursor;   // now points into the reordered buffer
        cursor += count;
    }
    else
    {
        flattenRecursive(ctx, node.childL(), dstBase, cursor);
        flattenRecursive(ctx, node.childR(), dstBase, cursor);
    }
}

} // anonymous namespace (translation-unit private)

int buildMeshBvh(BvhBuffers& out,
                 const std::vector<Triangle>& hostTris,
                 const Geom& geom)
{

    const int triOffset = geom.meshTriangleOffset;
    const int triCount  = geom.meshTriangleCount;
    if (triCount <= 0) return -1;   // empty mesh → no root

    BvhBuildContext ctx;
    ctx.nodes     = &out.hostNodes;
    ctx.tris      = &hostTris;
    ctx.dstTris   = &out.hostTriangles;
    ctx.triOffset = triOffset;

    std::vector<int> order(triCount);
    for (int i = 0; i < triCount; i++)
        order[i] = triOffset + i;   // store indices, not triangle structs
    ctx.order = &order;

    const int root = buildRecursive(ctx, 0, triCount, 0);

    // Flatten: write the reordered triangles and fix the leaves to reference
    // the contiguous chunks.
    const int dstBase = (int)out.hostTriangles.size();
    out.hostTriangles.resize(dstBase + triCount);
    int cursor = 0;
    flattenRecursive(ctx, root, dstBase, cursor);

    return root;
}

void buildSceneBvh(BvhBuffers& out,
                   const std::vector<Triangle>& hostTris,
                   const std::vector<Geom>& geoms)
{
    out.hostNodes.clear();
    out.hostTriangles.clear();
    out.hostTriangles.reserve(hostTris.size());   // preallocate for the reordered output

    for (size_t i = 0; i < geoms.size(); i++)
    {
        if (geoms[i].meshTriangleCount <= 0) continue;   // meta stays root = -1
        out.hostBvhMeta[i].rootNodeIndex =
            buildMeshBvh(out, hostTris, geoms[i]);
    }
}

void uploadToDevice(BvhBuffers& b)
{
    if (!b.hostNodes.empty())
    {
        cudaMalloc(&b.deviceNodes, b.hostNodes.size() * sizeof(BvhNode));
        cudaMemcpy(b.deviceNodes, b.hostNodes.data(),
                   b.hostNodes.size() * sizeof(BvhNode), cudaMemcpyHostToDevice);
        checkCUDAError("bvh upload nodes");
    }
}

void freeDevice(BvhBuffers& b)
{
    if (b.deviceNodes)    { cudaFree(b.deviceNodes);    b.deviceNodes = nullptr; }
}

} // namespace bvh
