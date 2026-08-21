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
float centroidComponent(const TrianglePos& t, int axis)
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
    const std::vector<TrianglePos>*  positions;   // source positions (world-space bake, read-only)
    const std::vector<TriangleAttr>* attrs;       // source shading attributes, same indexing
    std::vector<int>*                order;       // permutable triangle indices
    std::vector<TrianglePos>*        dstPositions;
    std::vector<TriangleAttr>*       dstAttrs;
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
    const std::vector<TrianglePos>& positions = *ctx.positions;
    std::vector<int>&            order     = *ctx.order;
    const int triOffset = ctx.triOffset;

    const int n = end - begin;   // number of triangles in this node's range

    AABB bounds;
    for (int i = begin; i < end; i++)
        bounds.expand(positions[order[i]]);   // access the triangle by its index in the order array

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
                return centroidComponent(positions[lhs], axis) < centroidComponent(positions[rhs], axis);
            });

        pref[0] = AABB();
        pref[0].expand(positions[order[begin]]);
        for (int k = 1; k < n; k++)
        {
            pref[k] = pref[k - 1];//copy the previous AABB
            pref[k].expand(positions[order[begin + k]]);//expand the AABB to include the next triangle
        }
        suff[n] = AABB();
        for (int k = n - 1; k >= 0; k--)
        {
            suff[k] = suff[k + 1];
            suff[k].expand(positions[order[begin + k]]);
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
            return centroidComponent(positions[lhs], bestAxis) < centroidComponent(positions[rhs], bestAxis);
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
    const std::vector<TrianglePos>& srcPositions = *ctx.positions;
    const std::vector<TriangleAttr>& srcAttrs = *ctx.attrs;
    std::vector<TrianglePos>& dstPositions = *ctx.dstPositions;
    std::vector<TriangleAttr>& dstAttrs = *ctx.dstAttrs;
    const int triOffset = ctx.triOffset;

    BvhNode& node = nodes[nodeIndex];
    if (node.isLeaf)
    {
        const int orderStart = node.leafTriOffset() - triOffset;   // offset into the mesh's order array
        const int count      = node.leafTriCount();
        for (int j = 0; j < count; j++)
        {
            const int srcIndex = order[orderStart + j];
            const int dstIndex = dstBase + cursor + j;
            dstPositions[dstIndex] = srcPositions[srcIndex];
            dstAttrs[dstIndex] = srcAttrs[srcIndex];
        }
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

// Transform a triangle vertex from object to world space (point, w = 1).
glm::vec3 bakePoint(const glm::mat4& transform, const glm::vec3& p)
{
    return glm::vec3(transform * glm::vec4(p, 1.0f));
}

// Transform a vertex normal from object to world space via the
// inverse-transpose (the correct normal transform under non-uniform scale).
// Deliberately NOT re-normalized here: attribute interpolation normalizes
// the interpolated result, which reproduces the old per-hit
// inverse-transpose + normalize (recordWorldNormal) exactly by linearity.
glm::vec3 bakeNormal(const glm::mat4& invTranspose, const glm::vec3& n){
    return glm::vec3(invTranspose * glm::vec4(n, 0.0f));
}

void buildMeshBvh(BvhBuffers& out,
                  const std::vector<TrianglePos>& positions,
                  const std::vector<TriangleAttr>& attrs)
{
    const int triCount = (int)positions.size();
    if (triCount <= 0) return;   // no triangles → no nodes

    BvhBuildContext ctx;
    ctx.nodes     = &out.hostNodes;
    ctx.positions = &positions;
    ctx.attrs = &attrs;
    ctx.dstPositions = &out.hostTrianglePositions;
    ctx.dstAttrs = &out.hostTriangleAttrs;
    ctx.triOffset = 0;   // single tree over the whole array

    std::vector<int> order(triCount);
    for (int i = 0; i < triCount; i++)
        order[i] = i;   // store indices, not triangle structs
    ctx.order = &order;

    const int root = buildRecursive(ctx, 0, triCount, 0);

    // Flatten: write the reordered triangles and fix the leaves to reference
    // the contiguous chunks.
    const int dstBase = (int)out.hostTrianglePositions.size();
    out.hostTrianglePositions.resize(dstBase + triCount);
    out.hostTriangleAttrs.resize(dstBase + triCount);
    int cursor = 0;
    flattenRecursive(ctx, root, dstBase, cursor);
}

void buildSceneBvh(BvhBuffers& out,
                   const std::vector<TrianglePos>& positions,
                   const std::vector<TriangleAttr>& attrs,
                   const std::vector<Geom>& geoms)
{
    out.hostNodes.clear();
    out.hostTrianglePositions.clear();
    out.hostTriangleAttrs.clear();
    out.hostSurfaces.clear();

    // 1. Bake every mesh's triangles from object space to world space and map
    //    each (materialId, source binding) pair to one shared Surface.
    //    Vertices use the model transform; normals use the inverse-transpose
    //    (the normal transform the old kernel applied per hit).
    std::vector<TrianglePos> worldPositions;
    std::vector<TriangleAttr> worldAttrs;
    worldPositions.reserve(positions.size());
    worldAttrs.reserve(attrs.size());
    for (const Geom& g : geoms)
    {
        if (g.meshTriangleCount <= 0) continue;
        for (int i = 0; i < g.meshTriangleCount; i++)
        {
            const int srcIndex = g.meshTriangleOffset + i;
            const TrianglePos& srcPos = positions[srcIndex];
            const TriangleAttr& srcAttr = attrs[srcIndex];
            TrianglePos dstPos;
            TriangleAttr dstAttr;
            dstPos.v0 = bakePoint(g.transform, srcPos.v0);
            dstPos.v1 = bakePoint(g.transform, srcPos.v1);
            dstPos.v2 = bakePoint(g.transform, srcPos.v2);
            dstAttr.n0 = bakeNormal(g.invTranspose, srcAttr.n0);
            dstAttr.n1 = bakeNormal(g.invTranspose, srcAttr.n1);
            dstAttr.n2 = bakeNormal(g.invTranspose, srcAttr.n2);
            // UVs are texture-space coordinates — the geometry transform does
            // NOT apply to them; copy through unchanged.
            dstAttr.uv0 = srcAttr.uv0;
            dstAttr.uv1 = srcAttr.uv1;
            dstAttr.uv2 = srcAttr.uv2;
            // Vertex colors are geometry attributes, so they survive the
            // world-space bake unchanged.  Omitting this copy silently turns
            // COLOR_0 into Triangle's white defaults before GPU upload.
            dst.c0 = src.c0;
            dst.c1 = src.c1;
            dst.c2 = src.c2;
            // The compact id links this triangle to an immutable(不可写), shared
            // surface binding.  It is transform-free and survives the world
            // bake and flatten unchanged.
            dst.surfaceBindingId = src.surfaceBindingId;
            worldTris.push_back(dst);
        }
    }

    // 2. ONE tree over the combined world-space array.
    buildMeshBvh(out, worldPositions, worldAttrs);
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
