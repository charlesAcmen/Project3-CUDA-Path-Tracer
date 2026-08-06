// ====================================================================
// bvh_test — host-side validation of the single world-space BVH.
//
// Compiles the PRODUCTION code (src/bvh/bvh.h + src/bvh/bvh.cu) into the
// test TU, so the tests exercise exactly what the renderer runs.  There
// are no traced duplicates anymore — the traced/ copies that mirrored the
// OLD per-mesh, object-space build were deleted when the build switched to
// a single world-space tree.
//
// Validates:
//   1. World-space bake: buildSceneBvh transforms each geom's triangles to
//      world space (vertices via transform, normals via invTranspose),
//      tags materialId, and flattens leaves into contiguous runs.
//   2. Single-tree build: root is node 0, every node reachable, leaf
//      chunks within the triangle array.
//   3. traverseBvhClosest (near-child-first) matches a brute-force O(N)
//      scan on the baked array — hit flag, t, normal, materialId.
//   4. intersectRayAABBEntry entry-distance correctness.
//   5. Empty scene: no nodes, traversal misses.
//
// Host-only: no kernels launched, no GPU required.
// ====================================================================

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>

#include "bvh/bvh.h"             // production: AABB, BvhNode, BvhBuffers, traverseBvhClosest
#include "bvh/aabb.h"            // production: intersectRayAABBEntry
#include "constants.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

// Production build code (world-space bake + SAH + flatten).  Compiled once
// here; do NOT add src/bvh/bvh.cu to the test's CMake sources.
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
    return Triangle{ a, b, c, n, n, n };   // materialId stays -1 (source geometry)
}

// World-space transform builder (translation → rotation XYZ → scale).
glm::mat4 makeTransform(const glm::vec3& trans, const glm::vec3& rotRad, const glm::vec3& scale)
{
    glm::mat4 m(1.0f);
    m = glm::translate(m, trans);
    m = glm::rotate(m, rotRad.x, glm::vec3(1, 0, 0));
    m = glm::rotate(m, rotRad.y, glm::vec3(0, 1, 0));
    m = glm::rotate(m, rotRad.z, glm::vec3(0, 0, 1));
    m = glm::scale(m, scale);
    return m;
}

// Geom referencing a contiguous triangle slice [triOffset, triOffset+triCount).
Geom makeGeom(int materialId, int triOffset, int triCount, const glm::mat4& T)
{
    Geom g;
    g.materialid          = materialId;
    g.transform           = T;
    g.inverseTransform    = glm::inverse(T);
    g.invTranspose        = glm::inverseTranspose(T);
    g.meshTriangleOffset  = triOffset;
    g.meshTriangleCount   = triCount;
    return g;
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

// Closest-hit linear scan over a triangle slice — the reference the BVH
// traversal is validated against.  Also returns the hit triangle index so
// the test can compare materialId with the BVH's triIndex.
bool bruteForceClosest(const Ray& ray, const std::vector<Triangle>& tris,
                       int offset, int count, float& t, glm::vec3& nrm, int& idx)
{
    t = LARGE_T;
    idx = -1;
    bool hit = false;
    for (int j = 0; j < count; j++)
    {
        float tt;
        glm::vec3 nn;
        if (triangleIntersectionTest(ray, tris[offset + j], tt, nn))
        {
            if (tt < t) { t = tt; nrm = nn; idx = offset + j; hit = true; }
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
    const glm::vec3 center = 0.5f * (bounds.min + bounds.max);
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
// World-space bake reference + comparison
// ---------------------------------------------------------------------

// Expected bake of one source triangle under a geom's transform:
// vertices via transform (w=1), normals via invTranspose (w=0, deliberately
// NOT re-normalized — triangleIntersectionTest normalizes the interpolated
// result), materialId tagged.  Mirrors buildSceneBvh's bake.
Triangle bakeExpected(const Geom& g, const Triangle& src)
{
    Triangle d;
    d.v0         = glm::vec3(g.transform * glm::vec4(src.v0, 1.0f));
    d.v1         = glm::vec3(g.transform * glm::vec4(src.v1, 1.0f));
    d.v2         = glm::vec3(g.transform * glm::vec4(src.v2, 1.0f));
    d.n0         = glm::vec3(g.invTranspose * glm::vec4(src.n0, 0.0f));
    d.n1         = glm::vec3(g.invTranspose * glm::vec4(src.n1, 0.0f));
    d.n2         = glm::vec3(g.invTranspose * glm::vec4(src.n2, 0.0f));
    d.materialId = g.materialid;
    return d;
}

bool nearVec(const glm::vec3& a, const glm::vec3& b, float eps)
{
    return glm::length(a - b) < eps;
}

// Two triangles are equal iff materialId matches, vertices match, and the
// baked normals are parallel (they may be scaled by invTranspose).
bool triEqual(const Triangle& a, const Triangle& b, float eps)
{
    if (a.materialId != b.materialId) return false;
    if (!nearVec(a.v0, b.v0, eps) || !nearVec(a.v1, b.v1, eps) || !nearVec(a.v2, b.v2, eps))
        return false;
    auto normalEq = [](const glm::vec3& x, const glm::vec3& y) {
        const float lx = glm::length(x), ly = glm::length(y);
        if (lx < 1e-6f && ly < 1e-6f) return true;
        if (lx < 1e-6f || ly < 1e-6f) return false;
        return glm::dot(x / lx, y / ly) > 1.0f - 1e-5f;
    };
    return normalEq(a.n0, b.n0) && normalEq(a.n1, b.n1) && normalEq(a.n2, b.n2);
}

// Whether `got` is a permutation of `expected` (flatten reorders but must
// never drop, duplicate, or alter a baked triangle).
bool sameMultiset(const std::vector<Triangle>& expected, const std::vector<Triangle>& got, float eps)
{
    if (expected.size() != got.size()) return false;
    std::vector<char> used(got.size(), 0);
    for (const Triangle& e : expected)
    {
        bool found = false;
        for (size_t j = 0; j < got.size() && !found; j++)
        {
            if (!used[j] && triEqual(e, got[j], eps)) { used[j] = 1; found = true; }
        }
        if (!found) return false;
    }
    return true;
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

// Test 1: world-space bake.  A cube under a NON-UNIFORM scale (exercises
// the invTranspose normal path) must come out exactly as transform(source),
// with materialId tagged.
bool testWorldBake()
{
    const std::vector<Triangle> cube = makeCube(1.0f);
    const glm::mat4 T = makeTransform({ 1, 2, 3 }, { 0.4f, 0.2f, 0.3f }, { 2, 1, 3 });
    const Geom g = makeGeom(7, 0, (int)cube.size(), T);

    std::vector<Geom> geoms{ g };
    BvhBuffers out;
    bvh::buildSceneBvh(out, cube, geoms);

    if (out.hostTriangles.size() != cube.size())
    {
        printf("FAIL bake: baked size %zu != source %zu\n", out.hostTriangles.size(), cube.size());
        return false;
    }

    std::vector<Triangle> expected;
    for (const Triangle& t : cube) expected.push_back(bakeExpected(g, t));

    if (!sameMultiset(expected, out.hostTriangles, 1e-3f))
    {
        printf("FAIL bake: baked triangles do not match transform(source)\n");
        return false;
    }
    return true;
}

// Test 2: multi-geom bake.  Two meshes with DIFFERENT transforms and
// materials are concatenated (the loader's layout).  The bake must keep
// both sets intact, tagged with their own materialId; an empty geom is
// ignored.
bool testMultiGeomBake()
{
    const std::vector<Triangle> cube  = makeCube(1.0f);
    const std::vector<Triangle> cloud = makeRandomCloud(120, 0xABCDEFu);

    std::vector<Triangle> hostTris = cube;
    hostTris.insert(hostTris.end(), cloud.begin(), cloud.end());

    const glm::mat4 T0 = makeTransform({ -4, 0, 0 }, { 0, 0.5f, 0 }, { 1, 2, 1 });
    const glm::mat4 T1 = makeTransform({  3, 1, 2 }, { 0.3f, 0, 0.7f }, { 0.5f, 0.5f, 2 });

    std::vector<Geom> geoms(3);
    geoms[0] = makeGeom(3, 0,            (int)cube.size(),  T0);
    geoms[1] = makeGeom(9, (int)cube.size(), (int)cloud.size(), T1);
    // geoms[2] stays default (offset -1, count 0) → empty mesh, ignored.

    BvhBuffers out;
    bvh::buildSceneBvh(out, hostTris, geoms);

    if (out.hostTriangles.size() != hostTris.size())
    {
        printf("FAIL multi-geom: baked size %zu != source %zu\n", out.hostTriangles.size(), hostTris.size());
        return false;
    }

    std::vector<Triangle> expected;
    for (const Triangle& t : cube)  expected.push_back(bakeExpected(geoms[0], t));
    for (const Triangle& t : cloud) expected.push_back(bakeExpected(geoms[1], t));

    if (!sameMultiset(expected, out.hostTriangles, 1e-3f))
    {
        printf("FAIL multi-geom: baked triangles do not match per-geom transforms\n");
        return false;
    }
    return true;
}

// Test 3: traverseBvhClosest (near-child-first) must equal a brute-force
// O(N) scan on the baked array — hit flag, t, normal, and the materialId
// of the hit triangle (triIndex).  Runs over several mesh kinds ×
// transforms, including a non-uniform scale (invTranspose normals).
bool testTraversalVsBrute(const char* name, const std::vector<Triangle>& tris,
                          const glm::mat4& T, int materialId)
{
    std::vector<Geom> geoms(1);
    geoms[0] = makeGeom(materialId, 0, (int)tris.size(), T);

    BvhBuffers out;
    bvh::buildSceneBvh(out, tris, geoms);
    if (out.hostNodes.empty())
    {
        printf("FAIL %s: no tree built\n", name);
        return false;
    }
    const int root = 0;   // single tree → root is node 0

    // Rays aim at the BAKED geometry (world space), like the renderer.
    const std::vector<Ray> rays = makeRayBatch(out.hostTriangles, 0x1234567u);
    for (const Ray& ray : rays)
    {
        float bt = LARGE_T; glm::vec3 bn; int bidx = -1;
        const bool bhit = bruteForceClosest(ray, out.hostTriangles, 0, (int)out.hostTriangles.size(), bt, bn, bidx);

        const BvhHit vhit = traverseBvhClosest(ray, out.hostNodes.data(), root,
                                               out.hostTriangles.data(), LARGE_T);

        if (bhit != vhit.hit)
        {
            printf("FAIL %s: hit flag brute=%d bvh=%d\n", name, bhit, vhit.hit);
            return false;
        }
        if (!bhit) continue;

        if (fabsf(bt - vhit.t) > 1e-3f || glm::dot(bn, vhit.normal) < 0.9999f)
        {
            printf("FAIL %s: t brute=%.6f bvh=%.6f dot=%.6f\n", name, bt, vhit.t, glm::dot(bn, vhit.normal));
            return false;
        }
        if (vhit.triIndex < 0 || vhit.triIndex >= (int)out.hostTriangles.size())
        {
            printf("FAIL %s: triIndex %d out of range\n", name, vhit.triIndex);
            return false;
        }
        if (out.hostTriangles[vhit.triIndex].materialId != out.hostTriangles[bidx].materialId)
        {
            printf("FAIL %s: BVH materialId %d != brute %d\n", name,
                   out.hostTriangles[vhit.triIndex].materialId, out.hostTriangles[bidx].materialId);
            return false;
        }
    }
    return true;
}

// Test 4: tree structure.  Root is node 0; DFS reaches every node exactly
// once (no cycles, no orphans), and every leaf chunk lies inside the baked
// triangle array.
bool testBuildStructure(const std::vector<Triangle>& tris, const glm::mat4& T, int materialId)
{
    std::vector<Geom> geoms(1);
    geoms[0] = makeGeom(materialId, 0, (int)tris.size(), T);

    BvhBuffers out;
    bvh::buildSceneBvh(out, tris, geoms);
    if (out.hostNodes.empty())
    {
        printf("FAIL structure: no nodes built\n");
        return false;
    }

    std::vector<int> visited(out.hostNodes.size(), 0);
    std::vector<int> stack{ 0 };
    int reached = 0;
    while (!stack.empty())
    {
        const int idx = stack.back();
        stack.pop_back();
        if (idx < 0 || idx >= (int)out.hostNodes.size())
        {
            printf("FAIL structure: child index %d out of range (nodes=%zu)\n", idx, out.hostNodes.size());
            return false;
        }
        if (visited[idx])
        {
            printf("FAIL structure: node %d revisited (cycle)\n", idx);
            return false;
        }
        visited[idx] = 1;
        reached++;

        const BvhNode& n = out.hostNodes[idx];
        if (n.isLeaf)
        {
            const int off = n.leafTriOffset();
            const int cnt = n.leafTriCount();
            if (off < 0 || cnt < 0 || off + cnt > (int)out.hostTriangles.size())
            {
                printf("FAIL structure: leaf chunk [%d,+%d) out of triangle range (%zu)\n",
                       off, cnt, out.hostTriangles.size());
                return false;
            }
        }
        else
        {
            stack.push_back(n.childL());
            stack.push_back(n.childR());
        }
    }
    if (reached != (int)out.hostNodes.size())
    {
        printf("FAIL structure: reached %d of %zu nodes\n", reached, out.hostNodes.size());
        return false;
    }
    return true;
}

// Test 5: intersectRayAABBEntry entry-distance correctness.  The near-first
// traversal orders children by this value, so it must report the ray-box
// entry distance exactly.
bool testAabbEntry()
{
    AABB box;
    box.min = glm::vec3(0, 0, 0);
    box.max = glm::vec3(2, 1, 1);

    // invDir must be the true 1/dir (as traverseBvhClosest computes it):
    // a zero direction component yields ±inf slabs, which the slab test
    // leaves unrestricted.  Passing a literal 0 here would zero the slab and
    // wrongly reject the ray.
    const glm::vec3 dirPX(1.0f, 0.0f, 0.0f);
    const glm::vec3 invDirPX = glm::vec3(1.0f) / dirPX;   // (1, inf, inf)
    const glm::vec3 dirNX(-1.0f, 0.0f, 0.0f);
    const glm::vec3 invDirNX = glm::vec3(1.0f) / dirNX;   // (-1, inf, inf)

    // Ray from (-5, 0.5, 0.5) along +x enters at x=0 → t = 5.
    {
        const glm::vec3 o(-5.0f, 0.5f, 0.5f);
        float entry = -1.0f;
        if (!intersectRayAABBEntry(o, invDirPX, box, RAY_EPSILON, LARGE_T, entry))
        {
            printf("FAIL aabb-entry: expected hit, got miss\n");
            return false;
        }
        if (fabsf(entry - 5.0f) > 1e-4f)
        {
            printf("FAIL aabb-entry: entry=%.6f, expected 5.0\n", entry);
            return false;
        }
    }

    // Ray from (5, 0.5, 0.5) along -x enters at x=2 → t = 3.
    {
        const glm::vec3 o(5.0f, 0.5f, 0.5f);
        float entry = -1.0f;
        if (!intersectRayAABBEntry(o, invDirNX, box, RAY_EPSILON, LARGE_T, entry))
        {
            printf("FAIL aabb-entry: expected hit, got miss (neg dir)\n");
            return false;
        }
        if (fabsf(entry - 3.0f) > 1e-4f)
        {
            printf("FAIL aabb-entry: entry=%.6f, expected 3.0\n", entry);
            return false;
        }
    }

    // Ray from (-5, 5, 0.5) along +x: y = 5 outside [0,1] → miss.
    {
        const glm::vec3 o(-5.0f, 5.0f, 0.5f);
        float entry = -1.0f;
        if (intersectRayAABBEntry(o, invDirPX, box, RAY_EPSILON, LARGE_T, entry))
        {
            printf("FAIL aabb-entry: expected miss, got hit entry=%.6f\n", entry);
            return false;
        }
    }

    // Far-plane pruning: box starts at t=5 but the far plane is 2 → miss.
    {
        const glm::vec3 o(-5.0f, 0.5f, 0.5f);
        float entry = -1.0f;
        if (intersectRayAABBEntry(o, invDirPX, box, RAY_EPSILON, 2.0f, entry))
        {
            printf("FAIL aabb-entry: expected miss (far plane 2 < entry 5)\n");
            return false;
        }
    }

    return true;
}

// Test 6: empty scene.  No triangles → no tree; traversal with root -1 must
// miss without touching any buffers.
bool testEmptyScene()
{
    std::vector<Triangle> none;
    std::vector<Geom> geoms(1);   // default: offset -1, count 0
    BvhBuffers out;
    bvh::buildSceneBvh(out, none, geoms);

    if (!out.hostNodes.empty() || !out.hostTriangles.empty())
    {
        printf("FAIL empty: expected empty tree\n");
        return false;
    }

    const Ray ray{ glm::vec3(0, 0, 0), glm::vec3(1, 0, 0) };
    const BvhHit h = traverseBvhClosest(ray, nullptr, -1, nullptr, LARGE_T);
    if (h.hit)
    {
        printf("FAIL empty: traversal with root -1 reported a hit\n");
        return false;
    }
    return true;
}

} // namespace

int main()
{
    int failures = 0;

    if (!testWorldBake()) failures++;
    if (!testMultiGeomBake()) failures++;
    if (!testAabbEntry()) failures++;
    if (!testEmptyScene()) failures++;

    // Traversal + structure across mesh kinds and transforms.
    const std::vector<Triangle> cube  = makeCube(1.0f);
    const std::vector<Triangle> cloud = makeRandomCloud(120, 0xDEADBEEFu);
    const std::vector<Triangle> degen = makeDegenerate(40);

    struct Case { const char* name; const std::vector<Triangle>& tris; glm::mat4 T; int mat; };
    const Case cases[] = {
        { "cube_identity",   cube,  makeTransform({ 0, 0, 0 }, { 0, 0, 0 }, { 1, 1, 1 }),      5 },
        { "cube_transform",  cube,  makeTransform({ 1, 2, 3 }, { 0.4f, 0.2f, 0.3f }, { 2, 1, 3 }), 6 },
        { "cloud_identity",  cloud, makeTransform({ 0, 0, 0 }, { 0, 0, 0 }, { 1, 1, 1 }),      7 },
        { "cloud_transform", cloud, makeTransform({ -3, 1, 0 }, { 0.1f, 0.9f, 0.2f }, { 0.5f, 2, 1 }), 8 },
        { "degenerate",      degen, makeTransform({ 0, 0, 0 }, { 0, 0, 0 }, { 1, 1, 1 }),      9 },
    };
    for (const Case& c : cases)
    {
        if (!testBuildStructure(c.tris, c.T, c.mat)) failures++;
        if (!testTraversalVsBrute(c.name, c.tris, c.T, c.mat)) failures++;
    }

    if (failures == 0)
    {
        printf("ALL PASS\n");
        return 0;
    }
    printf("%d FAILURE(S)\n", failures);
    return 1;
}
