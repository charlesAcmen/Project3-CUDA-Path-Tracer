// ====================================================================
// bvh_test — host-side BVH build + traverse validation
//
// Builds per-mesh BVHs for synthetic meshes (axis-aligned cube, random
// triangle cloud, degenerate soup) across maxDepth × leafSize
// configurations and asserts traverseBvhClosest matches a brute-force
// O(N) linear scan ray-for-ray (hit flag, t, shading normal).
//
// Also exercises buildSceneBvh's flatten pass: the reordered per-mesh
// slice is the same triangle set (no cross-mesh contamination), and
// traversal on the reordered buffer still matches brute force on both
// the reordered and the original buffers.
//
// Host-only: the CUDA build/traverse functions are compiled for host
// execution (no kernels launched, no GPU required).
// ====================================================================

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>

#include "bvh/bvh.h"                 // AABB, BvhNode, traverseBvhClosest
#include "constants.h"

// Build code (SAH construction + flatten).  Compiled once here; do NOT
// add src/bvh/bvh.cu to this test's CMake sources.
#include "bvh/bvh.cu"

namespace {

// ---------------------------------------------------------------------
// Deterministic helpers
// ---------------------------------------------------------------------

// Deterministic LCG so results are reproducible across runs/machines.
struct Lcg
{
    unsigned int s;
    explicit Lcg(unsigned int seed) : s(seed) {}
    unsigned int next() { s = s * 1664525u + 1013904223u; return s; }
    float unit() { return (float)(next() & 0xFFFFFFu) / (float)0x1000000u; }
    float range(float a, float b) { return a + (b - a) * unit(); }
};

Triangle makeTriangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c)
{
    glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));
    return Triangle{ a, b, c, n, n, n };
}

// ---------------------------------------------------------------------
// Synthetic meshes
// ---------------------------------------------------------------------

// Axis-aligned cube: 8 corners, 12 triangles, outward winding.
std::vector<Triangle> makeCube(float half = 1.0f)
{
    const glm::vec3 c[8] = {
        { -half, -half, -half }, {  half, -half, -half },
        {  half,  half, -half }, { -half,  half, -half },
        { -half, -half,  half }, {  half, -half,  half },
        {  half,  half,  half }, { -half,  half,  half }
    };
    const int faces[6][4] = {
        { 0, 1, 2, 3 },  // -z
        { 5, 4, 7, 6 },  // +z
        { 4, 0, 3, 7 },  // -x
        { 1, 5, 6, 2 },  // +x
        { 4, 5, 1, 0 },  // -y
        { 3, 2, 6, 7 }   // +y
    };
    std::vector<Triangle> tris;
    for (const auto& f : faces)
    {
        tris.push_back(makeTriangle(c[f[0]], c[f[1]], c[f[2]]));
        tris.push_back(makeTriangle(c[f[0]], c[f[2]], c[f[3]]));
    }
    return tris;
}

// Random triangle soup in a box — spatially incoherent, exercises the
// SAH split on a noisier distribution than a clean cube.
std::vector<Triangle> makeRandomCloud(int n, unsigned int seed)
{
    Lcg rng(seed);
    std::vector<Triangle> tris;
    for (int i = 0; i < n; i++)
    {
        glm::vec3 a(rng.range(-3, 3), rng.range(-3, 3), rng.range(-3, 3));
        glm::vec3 b = a + glm::vec3(rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1));
        glm::vec3 c = a + glm::vec3(rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1));
        tris.push_back(makeTriangle(a, b, c));
    }
    return tris;
}

// Degenerate set: collinear (zero-area) triangles.  They never register
// a hit (Möller-Trumbore near-parallel rejection) but their zero-volume
// AABBs exercise the surfaceArea() guard and the degenerate-split path.
std::vector<Triangle> makeDegenerate(int n)
{
    const glm::vec3 normal(0.0f, 1.0f, 0.0f);
    std::vector<Triangle> tris;
    for (int i = 0; i < n; i++)
    {
        const glm::vec3 p((float)i * 0.5f, 0.0f, 0.0f);
        tris.push_back(Triangle{ p, p + glm::vec3(1, 0, 0), p + glm::vec3(2, 0, 0),
                                 normal, normal, normal });
    }
    return tris;
}

// ---------------------------------------------------------------------
// Brute force reference
// ---------------------------------------------------------------------

// Closest-hit linear scan over a triangle slice — the reference both
// the BVH traversal and the O(N) GPU kernel are defined against.
bool bruteForceClosest(const Ray& ray, const std::vector<Triangle>& tris,
                       int offset, int count, float& t, glm::vec3& nrm)
{
    t = LARGE_T;
    bool hit = false;
    for (int j = 0; j < count; j++)
    {
        float tt;
        glm::vec3 nn;
        if (triangleIntersectionTest(ray, tris[offset + j], tt, nn))
        {
            if (tt < t) { t = tt; nrm = nn; hit = true; }
        }
    }
    return hit;
}

// Deterministic ray batch: origins on a sphere around the mesh aimed at
// the centroid, random rays, and axis-aligned rays (zero direction
// components) to exercise the slab test's zero-direction handling.
std::vector<Ray> makeRayBatch(const std::vector<Triangle>& tris, unsigned int seed)
{
    AABB bounds;
    for (const auto& t : tris)
        bounds.expand(t);
    const glm::vec3 center = bounds.centroid();
    const float radius = glm::length(bounds.max - bounds.min) * 0.5f + 1.0f;

    Lcg rng(seed);
    std::vector<Ray> rays;

    for (int i = 0; i < 64; i++)   // aimed at the centroid
    {
        glm::vec3 dir(rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1));
        dir = glm::normalize(dir);
        rays.push_back(Ray{ center + dir * radius, glm::normalize(center - (center + dir * radius)) });
    }
    for (int i = 0; i < 64; i++)   // random origin + random direction
    {
        glm::vec3 origin(rng.range(-radius, radius), rng.range(-radius, radius), rng.range(-radius, radius));
        glm::vec3 dir(rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1));
        dir = glm::normalize(dir);
        rays.push_back(Ray{ origin, dir });
    }
    for (int sign = -1; sign <= 1; sign += 2)   // axis-aligned (zero in 2 comps)
    {
        rays.push_back(Ray{ center + glm::vec3(0, 0, radius * sign), glm::vec3(0, 0, (float)sign) });
        rays.push_back(Ray{ center + glm::vec3(0, radius * sign, 0), glm::vec3(0, (float)sign, 0) });
        rays.push_back(Ray{ center + glm::vec3(radius * sign, 0, 0), glm::vec3((float)sign, 0, 0) });
    }
    return rays;
}

// ---------------------------------------------------------------------
// Assertions
// ---------------------------------------------------------------------

// Build one mesh END-TO-END (tree + flatten via buildSceneBvh, the real
// production path) and verify traverseBvhClosest == brute force for every
// ray in a deterministic batch.  The brute force scans the REORDERED
// buffer — the triangle set is order-preserved, so this is equivalent to
// scanning the original slice.
bool testMeshTraversal(const char* name, const std::vector<Triangle>& tris,
                       int maxDepth, int leafSize)
{
    std::vector<Geom> geoms(1);
    geoms[0].meshTriangleOffset = 0;
    geoms[0].meshTriangleCount  = (int)tris.size();

    BvhBuffers bufs;
    bvh::buildSceneBvh(bufs, tris, geoms, maxDepth, leafSize);

    const int root = bufs.hostBvhMeta[0].rootNodeIndex;
    if (root < 0)
    {
        printf("FAIL %s: no root (depth=%d leaf=%d)\n", name, maxDepth, leafSize);
        return false;
    }

    const std::vector<Ray> rays = makeRayBatch(tris, 0x1234567u);
    for (const Ray& ray : rays)
    {
        float bt = LARGE_T; glm::vec3 bn;
        const bool bhit = bruteForceClosest(ray, bufs.hostTriangles, 0, (int)tris.size(), bt, bn);

        float vt = LARGE_T; glm::vec3 vn;
        const bool vhit = traverseBvhClosest(ray, bufs.hostNodes.data(), root,
                                             bufs.hostTriangles.data(), vt, vn);

        if (bhit != vhit)
        {
            printf("FAIL %s(depth=%d leaf=%d): hit flag brute=%d bvh=%d\n",
                   name, maxDepth, leafSize, bhit, vhit);
            return false;
        }
        if (bhit && (fabsf(bt - vt) > 1e-3f || glm::dot(bn, vn) < 0.9999f))
        {
            printf("FAIL %s(depth=%d leaf=%d): t brute=%.6f bvh=%.6f dot=%.6f\n",
                   name, maxDepth, leafSize, bt, vt, glm::dot(bn, vn));
            return false;
        }
    }
    return true;
}

// Structural validation of buildMeshBvh output: DFS from the root, every
// child index in range, no node revisited (a cycle would make traversal
// hang), every leaf's triangle chunk inside the slice, and all nodes
// reachable.  Catches malformed trees directly, independent of traversal.
bool testBuildStructure(const std::vector<Triangle>& tris, int maxDepth, int leafSize)
{
    std::vector<BvhNode> nodes;
    std::vector<Triangle> dummy;   // reordered output is not this test's concern
    const int root = bvh::buildMeshBvh(nodes, dummy, tris, 0, (int)tris.size(), maxDepth, leafSize);
    if (root < 0)
    {
        printf("FAIL structure: buildMeshBvh returned no root\n");
        return false;
    }

    std::vector<int> visited(nodes.size(), 0);
    std::vector<int> stack{ root };
    int reached = 0;
    while (!stack.empty())
    {
        const int idx = stack.back();
        stack.pop_back();
        if (idx < 0 || idx >= (int)nodes.size())
        {
            printf("FAIL structure: child index %d out of range (nodes=%zu)\n", idx, nodes.size());
            return false;
        }
        if (visited[idx])
        {
            printf("FAIL structure: node %d revisited (cycle)\n", idx);
            return false;
        }
        visited[idx] = 1;
        reached++;

        const BvhNode& n = nodes[idx];
        if (n.isLeaf)
        {
            if (n.right < 0 || n.left < 0 || n.left + n.right > (int)tris.size())
            {
                printf("FAIL structure: leaf chunk [%d,+%d) out of triangle range (%zu)\n",
                       n.left, n.right, tris.size());
                return false;
            }
        }
        else
        {
            stack.push_back(n.left);
            stack.push_back(n.right);
        }
    }
    if (reached != (int)nodes.size())
    {
        printf("FAIL structure: reached %d of %zu nodes\n", reached, nodes.size());
        return false;
    }
    return true;
}

// Build a two-mesh scene through buildSceneBvh and verify the flatten
// pass: per-mesh reordered slices contain the same triangles, an empty
// geom gets rootNodeIndex = -1, and traversal on the reordered buffer
// matches brute force.
bool testSceneFlatten()
{
    const std::vector<Triangle> cube  = makeCube(1.0f);
    const std::vector<Triangle> cloud = makeRandomCloud(120, 0xABCDEFu);

    // Two meshes concatenated — the loader's layout (per-mesh slices).
    std::vector<Triangle> hostTris = cube;
    hostTris.insert(hostTris.end(), cloud.begin(), cloud.end());

    std::vector<Geom> geoms(3);
    geoms[0].meshTriangleOffset = 0;
    geoms[0].meshTriangleCount  = (int)cube.size();
    geoms[1].meshTriangleOffset = (int)cube.size();
    geoms[1].meshTriangleCount  = (int)cloud.size();
    // geoms[2] stays default (offset -1, count 0) → empty mesh.

    BvhBuffers out;
    bvh::buildSceneBvh(out, hostTris, geoms, 24, 4);

    if (out.hostTriangles.size() != hostTris.size())
    {
        printf("FAIL flatten: reordered size %zu != original %zu\n",
               out.hostTriangles.size(), hostTris.size());
        return false;
    }
    if (out.hostBvhMeta.size() != geoms.size())
    {
        printf("FAIL flatten: meta size %zu != geoms %zu\n",
               out.hostBvhMeta.size(), geoms.size());
        return false;
    }
    if (out.hostBvhMeta[2].rootNodeIndex != -1)
    {
        printf("FAIL flatten: empty geom expected root -1, got %d\n",
               out.hostBvhMeta[2].rootNodeIndex);
        return false;
    }

    for (int m = 0; m < 2; m++)
    {
        const int off = geoms[m].meshTriangleOffset;
        const int cnt = geoms[m].meshTriangleCount;
        const int root = out.hostBvhMeta[m].rootNodeIndex;
        if (root < 0)
        {
            printf("FAIL flatten: mesh %d expected a root, got %d\n", m, root);
            return false;
        }

        // Every reordered triangle's vertices must belong to the mesh's
        // original slice — catches cross-mesh contamination / drops.
        std::vector<glm::vec3> orig;
        for (int j = 0; j < cnt; j++)
        {
            orig.push_back(hostTris[off + j].v0);
            orig.push_back(hostTris[off + j].v1);
            orig.push_back(hostTris[off + j].v2);
        }
        for (int j = 0; j < cnt; j++)
        {
            const Triangle& rt = out.hostTriangles[off + j];
            const glm::vec3 p[3] = { rt.v0, rt.v1, rt.v2 };
            for (int k = 0; k < 3; k++)
            {
                bool found = false;
                for (const glm::vec3& q : orig)
                    if (glm::length(p[k] - q) < 1e-4f) { found = true; break; }
                if (!found)
                {
                    printf("FAIL flatten: mesh %d tri %d vert %d not in original slice\n",
                           m, j, k);
                    return false;
                }
            }
        }

        // Traversal on the reordered buffer must match brute force on
        // BOTH the reordered and the original slices.
        const std::vector<Ray> rays = makeRayBatch(hostTris, 0x1234u + (unsigned)m);
        for (const Ray& ray : rays)
        {
            float bt; glm::vec3 bn;
            const bool bhit = bruteForceClosest(ray, out.hostTriangles, off, cnt, bt, bn);
            float bo; glm::vec3 no;
            const bool borig = bruteForceClosest(ray, hostTris, off, cnt, bo, no);
            if (bhit != borig)
            {
                printf("FAIL flatten: brute-force differs after reorder (mesh %d)\n", m);
                return false;
            }

            float vt = LARGE_T; glm::vec3 vn;
            const bool vhit = traverseBvhClosest(ray, out.hostNodes.data(), root,
                                                 out.hostTriangles.data(), vt, vn);
            if (vhit != bhit)
            {
                printf("FAIL flatten: BVH hit %d != brute %d (mesh %d)\n", vhit, bhit, m);
                return false;
            }
            if (vhit && (fabsf(vt - bt) > 1e-3f || glm::dot(vn, bn) < 0.9999f))
            {
                printf("FAIL flatten: BVH t=%.6f brute t=%.6f dot=%.6f (mesh %d)\n",
                       vt, bt, glm::dot(vn, bn), m);
                return false;
            }
        }
    }
    return true;
}

} // namespace

int main()
{
    const std::vector<Triangle> cube  = makeCube(1.0f);
    const std::vector<Triangle> cloud = makeRandomCloud(120, 0xDEADBEEFu);
    const std::vector<Triangle> degen = makeDegenerate(40);

    struct Case { const char* name; const std::vector<Triangle>& tris; };
    const Case cases[] = {
        { "cube",         cube  },
        { "random_cloud", cloud },
        { "degenerate",   degen },
    };
    const int depths[] = { 8, 24 };
    const int leaves[] = { 1, 4 };

    int failures = 0;

    // Empty mesh → no root.
    {
        std::vector<BvhNode> nodes;
        std::vector<Triangle> dummy;
        const int r = bvh::buildMeshBvh(nodes, dummy, cube, 0, 0, 8, 4);
        if (r != -1) { printf("FAIL: empty mesh expected root -1, got %d\n", r); failures++; }
    }

    for (const Case& c : cases)
        for (int d : depths)
            for (int l : leaves)
            {
                if (!testBuildStructure(c.tris, d, l)) failures++;
                if (!testMeshTraversal(c.name, c.tris, d, l)) failures++;
            }

    if (!testSceneFlatten()) failures++;

    if (failures == 0)
    {
        printf("ALL PASS\n");
        return 0;
    }
    printf("%d FAILURE(S)\n", failures);
    return 1;
}
