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
//      chunks tile the triangle array exactly (contiguous, non-overlapping,
//      gap-free — the per-leaf sequential-read property), AND each chunk's
//      content is provably that leaf's own: every triangle in a leaf's
//      contiguous run lies fully inside the leaf's AABB, and the flattened
//      output is a bijection of the baked source (no drop/dup/alter).
//   3. traverseBvhClosest (near-child-first) matches a brute-force O(N)
//      scan on the baked array — hit flag, t, normal, materialId.
//   4. intersectRayAABBEntry entry-distance correctness + the near-first
//      ordering metric (two boxes, both ray directions, inside-box).
//   5. Scale-aware secondary-ray origin offsets remain representable at large
//      world coordinates while preserving the near-origin offset.
//   6. Empty scene: no nodes, traversal misses.
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
#include "intersection/intersections.h"

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

// Mirrors the renderer's source-geometry representation: positions and
// attributes have matching indices, without a temporary Triangle AoS.
struct TestMesh
{
    std::vector<TrianglePos> positions;
    std::vector<TriangleAttr> attrs;
};

void appendTriangle(TestMesh& mesh, const glm::vec3& a, const glm::vec3& b, const glm::vec3& c,
                    const glm::vec2& uva = glm::vec2(0.0f),
                    const glm::vec2& uvb = glm::vec2(0.0f),
                    const glm::vec2& uvc = glm::vec2(0.0f))
{
    glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));
    mesh.positions.push_back(TrianglePos{ a, b, c });
    TriangleAttr attr{};
    attr.n0 = n; attr.n1 = n; attr.n2 = n;
    attr.uv0 = uva; attr.uv1 = uvb; attr.uv2 = uvc;
    mesh.attrs.push_back(attr);
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
TestMesh makeCube(float half = 1.0f)
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
    // Per-face corner UVs: each face spans the full [0,1]² unit square, so
    // the interpolated UV at any hit is non-trivial (not a constant (0,0)).
    const glm::vec2 uv[8] = {
        { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 },   // -z face
        { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 }    // +z face
    };
    TestMesh mesh;
    for (const auto& f : faces)
    {
        appendTriangle(mesh, c[f[0]], c[f[1]], c[f[2]], uv[f[0]], uv[f[1]], uv[f[2]]);
        appendTriangle(mesh, c[f[0]], c[f[2]], c[f[3]], uv[f[0]], uv[f[2]], uv[f[3]]);
    }
    return mesh;
}

// Random triangle soup in a box — spatially incoherent, exercises the
// SAH split on a noisier distribution than a clean cube.
TestMesh makeRandomCloud(int n, unsigned int seed)
{
    Lcg rng(seed);
    TestMesh mesh;
    for (int i = 0; i < n; i++)
    {
        glm::vec3 a(rng.range(-3, 3), rng.range(-3, 3), rng.range(-3, 3));
        glm::vec3 b = a + glm::vec3(rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1));
        glm::vec3 c = a + glm::vec3(rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1));
        appendTriangle(mesh, a, b, c);
    }
    return mesh;
}

// Degenerate set: collinear (zero-area) triangles.  They never register
// a hit (Möller-Trumbore near-parallel rejection) but their zero-volume
// AABBs exercise the surfaceArea() guard and the degenerate-split path.
TestMesh makeDegenerate(int n)
{
    const glm::vec3 normal(0.0f, 1.0f, 0.0f);
    TestMesh mesh;
    for (int i = 0; i < n; i++)
    {
        const glm::vec3 p((float)i * 0.5f, 0.0f, 0.0f);
        mesh.positions.push_back(TrianglePos{ p, p + glm::vec3(1, 0, 0), p + glm::vec3(2, 0, 0) });
        TriangleAttr attr{};
        attr.n0 = normal; attr.n1 = normal; attr.n2 = normal;
        mesh.attrs.push_back(attr);
    }
    return mesh;
}

// ---------------------------------------------------------------------
// Brute force reference
// ---------------------------------------------------------------------

// Closest-hit linear scan over a triangle slice — the reference the BVH
// traversal is validated against.  Also returns the hit triangle index so
// the test can compare materialId with the BVH's triIndex.
bool bruteForceClosest(const Ray& ray, const TestMesh& mesh,
                       int offset, int count, float& t, glm::vec3& nrm,
                       glm::vec2& uv, int& idx)
{
    t = LARGE_T;
    idx = -1;
    uv = glm::vec2(0.0f);
    bool hit = false;
    for (int j = 0; j < count; j++)
    {
        float tt;
        float u, v;
        if (intersectTrianglePositions(ray, mesh.positions[offset + j], tt, u, v))
        {
            if (tt < t)
            {
                glm::vec4 tangent;
                glm::vec3 vertexColor;
                t = tt;
                interpolateTriangleAttributes(mesh.positions[offset + j], mesh.attrs[offset + j],
                                              u, v, true, nrm, uv, tangent, vertexColor);
                idx = offset + j;
                hit = true;
            }
        }
    }
    return hit;
}

// Deterministic ray batch: origins on a sphere around the mesh aimed at
// the centroid, random rays, and axis-aligned rays (zero direction
// components) to exercise the slab test's zero-direction handling.
std::vector<Ray> makeRayBatch(const std::vector<TrianglePos>& positions, unsigned int seed)
{
    AABB bounds;
    for (const TrianglePos& pos : positions)
        bounds.expand(pos);
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
// NOT re-normalized — interpolation normalizes the final result), materialId
// tagged separately. Mirrors buildSceneBvh's bake.
void appendBakedExpected(TestMesh& dst, std::vector<int>& materialIds,
                         const Geom& g, const TrianglePos& srcPos,
                         const TriangleAttr& srcAttr)
{
    TrianglePos pos;
    TriangleAttr attr = srcAttr;
    pos.v0      = glm::vec3(g.transform * glm::vec4(srcPos.v0, 1.0f));
    pos.v1      = glm::vec3(g.transform * glm::vec4(srcPos.v1, 1.0f));
    pos.v2      = glm::vec3(g.transform * glm::vec4(srcPos.v2, 1.0f));
    attr.n0     = glm::vec3(g.invTranspose * glm::vec4(srcAttr.n0, 0.0f));
    attr.n1     = glm::vec3(g.invTranspose * glm::vec4(srcAttr.n1, 0.0f));
    attr.n2     = glm::vec3(g.invTranspose * glm::vec4(srcAttr.n2, 0.0f));
    // UVs are texture-space coordinates — the geometry transform does NOT
    // apply to them; copy through unchanged (mirrors the bake in bvh.cu).
    dst.positions.push_back(pos);
    dst.attrs.push_back(attr);
    materialIds.push_back(g.materialid);
}

bool nearVec(const glm::vec3& a, const glm::vec3& b, float eps)
{
    return glm::length(a - b) < eps;
}

// Two source entries are equal iff materialId matches, vertices match, the
// baked normals are parallel (they may be scaled by invTranspose), and the
// surface-binding ids + UVs match (both are copied through the bake unchanged).
bool triEqual(const TrianglePos& aPos, const TriangleAttr& aAttr, int aMaterialId,
              const TrianglePos& bPos, const TriangleAttr& bAttr, int bMaterialId, float eps)
{
    if (aMaterialId != bMaterialId) return false;
    if (!nearVec(aPos.v0, bPos.v0, eps) || !nearVec(aPos.v1, bPos.v1, eps) || !nearVec(aPos.v2, bPos.v2, eps))
        return false;
    auto normalEq = [](const glm::vec3& x, const glm::vec3& y) {
        const float lx = glm::length(x), ly = glm::length(y);
        if (lx < 1e-6f && ly < 1e-6f) return true;
        if (lx < 1e-6f || ly < 1e-6f) return false;
        return glm::dot(x / lx, y / ly) > 1.0f - 1e-5f;
    };
    if (!normalEq(aAttr.n0, bAttr.n0) || !normalEq(aAttr.n1, bAttr.n1) || !normalEq(aAttr.n2, bAttr.n2))
        return false;
    const auto uvEq = [](const glm::vec2& x, const glm::vec2& y) {
        return glm::length(x - y) < 1e-4f;
    };
    if (!uvEq(aAttr.uv0, bAttr.uv0) || !uvEq(aAttr.uv1, bAttr.uv1) || !uvEq(aAttr.uv2, bAttr.uv2))
        return false;
    if (!nearVec(aAttr.c0, bAttr.c0, eps) || !nearVec(aAttr.c1, bAttr.c1, eps) || !nearVec(aAttr.c2, bAttr.c2, eps))
        return false;
    if (aAttr.surfaceId != bAttr.surfaceId)
        return false;
    return true;
}

// Whether `got` is a permutation of `expected` (flatten reorders but must
// never drop, duplicate, or alter a baked triangle).
bool sameMultiset(const TestMesh& expected, const std::vector<int>& expectedMaterialIds,
                  const TestMesh& got, const std::vector<int>& gotMaterialIds, float eps)
{
    if (expected.positions.size() != got.positions.size()) return false;
    std::vector<char> used(got.positions.size(), 0);
    for (size_t i = 0; i < expected.positions.size(); ++i)
    {
        bool found = false;
        for (size_t j = 0; j < got.positions.size() && !found; j++)
        {
            if (!used[j] && triEqual(expected.positions[i], expected.attrs[i], expectedMaterialIds[i],
                                     got.positions[j], got.attrs[j], gotMaterialIds[j], eps))
            { used[j] = 1; found = true; }
        }
        if (!found) return false;
    }
    return true;
}

// Convert flattened runtime Surface ids back to the source-binding ids used
// by the bake reference, while exposing the material ids that now live only
// in BvhBuffers::hostSurfaces.
void unpackRuntimeSurfaces(const BvhBuffers& out, TestMesh& mesh,
                           std::vector<int>& materialIds)
{
    materialIds.resize(mesh.attrs.size());
    for (size_t i = 0; i < mesh.attrs.size(); ++i)
    {
        const Surface& surface = out.hostSurfaces[mesh.attrs[i].surfaceId];
        materialIds[i] = surface.materialId;
        mesh.attrs[i].surfaceId = surface.surfaceBindingId;
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

// Test 1: world-space bake.  A cube under a NON-UNIFORM scale (exercises
// the invTranspose normal path) must come out exactly as transform(source),
// with materialId tagged.
bool testWorldBake()
{
    TestMesh cube = makeCube(1.0f);
    // Stamp non-default surface-binding ids so the bake's copy-through is
    // exercised (triEqual compares them); must survive bake + flatten.
    cube.attrs[0].surfaceId = 3;
    cube.attrs[1].surfaceId = 5;
    // Non-white colors prove that the world-space bake preserves COLOR_0
    // instead of resetting it to TriangleAttr's default white values.
    cube.attrs[0].c0 = glm::vec3(1.0f, 0.0f, 0.0f);
    cube.attrs[0].c1 = glm::vec3(0.0f, 1.0f, 0.0f);
    cube.attrs[0].c2 = glm::vec3(0.0f, 0.0f, 1.0f);
    const glm::mat4 T = makeTransform({ 1, 2, 3 }, { 0.4f, 0.2f, 0.3f }, { 2, 1, 3 });
    const Geom g = makeGeom(7, 0, (int)cube.positions.size(), T);

    std::vector<Geom> geoms{ g };
    BvhBuffers out;
    bvh::buildSceneBvh(out, cube.positions, cube.attrs, geoms);

    if (out.hostTrianglePositions.size() != cube.positions.size())
    {
        printf("FAIL bake: baked size %zu != source %zu\n", out.hostTrianglePositions.size(), cube.positions.size());
        return false;
    }

    TestMesh expected;
    std::vector<int> expectedMaterialIds;
    for (size_t i = 0; i < cube.positions.size(); ++i)
        appendBakedExpected(expected, expectedMaterialIds, g, cube.positions[i], cube.attrs[i]);

    TestMesh actual{ out.hostTrianglePositions, out.hostTriangleAttrs };
    std::vector<int> actualMaterialIds;
    unpackRuntimeSurfaces(out, actual, actualMaterialIds);
    if (!sameMultiset(expected, expectedMaterialIds, actual, actualMaterialIds, 1e-3f))
    {
        printf("FAIL bake: baked triangles do not match transform(source)\n");
        return false;
    }
    if (out.hostSurfaces.size() != 3)
    {
        printf("FAIL bake: expected 3 unique (material, binding) surfaces, got %zu\n",
               out.hostSurfaces.size());
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
    const TestMesh cube  = makeCube(1.0f);
    const TestMesh cloud = makeRandomCloud(120, 0xABCDEFu);

    TestMesh hostMesh = cube;
    hostMesh.positions.insert(hostMesh.positions.end(), cloud.positions.begin(), cloud.positions.end());
    hostMesh.attrs.insert(hostMesh.attrs.end(), cloud.attrs.begin(), cloud.attrs.end());

    const glm::mat4 T0 = makeTransform({ -4, 0, 0 }, { 0, 0.5f, 0 }, { 1, 2, 1 });
    const glm::mat4 T1 = makeTransform({  3, 1, 2 }, { 0.3f, 0, 0.7f }, { 0.5f, 0.5f, 2 });

    std::vector<Geom> geoms(3);
    geoms[0] = makeGeom(3, 0, (int)cube.positions.size(), T0);
    geoms[1] = makeGeom(9, (int)cube.positions.size(), (int)cloud.positions.size(), T1);
    // geoms[2] stays default (offset -1, count 0) → empty mesh, ignored.

    BvhBuffers out;
    bvh::buildSceneBvh(out, hostMesh.positions, hostMesh.attrs, geoms);

    if (out.hostTrianglePositions.size() != hostMesh.positions.size())
    {
        printf("FAIL multi-geom: baked size %zu != source %zu\n", out.hostTrianglePositions.size(), hostMesh.positions.size());
        return false;
    }

    TestMesh expected;
    std::vector<int> expectedMaterialIds;
    for (size_t i = 0; i < cube.positions.size(); ++i)
        appendBakedExpected(expected, expectedMaterialIds, geoms[0], cube.positions[i], cube.attrs[i]);
    for (size_t i = 0; i < cloud.positions.size(); ++i)
        appendBakedExpected(expected, expectedMaterialIds, geoms[1], cloud.positions[i], cloud.attrs[i]);

    TestMesh actual{ out.hostTrianglePositions, out.hostTriangleAttrs };
    std::vector<int> actualMaterialIds;
    unpackRuntimeSurfaces(out, actual, actualMaterialIds);
    if (!sameMultiset(expected, expectedMaterialIds, actual, actualMaterialIds, 1e-3f))
    {
        printf("FAIL multi-geom: baked triangles do not match per-geom transforms\n");
        return false;
    }
    if (out.hostSurfaces.size() != 2)
    {
        printf("FAIL multi-geom: expected one shared surface per material, got %zu\n",
               out.hostSurfaces.size());
        return false;
    }
    return true;
}

// Test 3: traverseBvhClosest (near-child-first) must equal a brute-force
// O(N) scan on the baked array.  The BVH returns only t/u/v + triIndex;
// this test expands the winning triangle afterward and verifies that its
// normal/UV still match the full brute-force result.  Runs over several mesh kinds ×
// transforms, including a non-uniform scale (invTranspose normals).
bool testTraversalVsBrute(const char* name, const TestMesh& mesh,
                          const glm::mat4& T, int materialId)
{
    std::vector<Geom> geoms(1);
    geoms[0] = makeGeom(materialId, 0, (int)mesh.positions.size(), T);

    BvhBuffers out;
    bvh::buildSceneBvh(out, mesh.positions, mesh.attrs, geoms);
    if (out.hostNodes.empty())
    {
        printf("FAIL %s: no tree built\n", name);
        return false;
    }

    // Rays aim at the BAKED geometry (world space), like the renderer.
    const std::vector<Ray> rays = makeRayBatch(out.hostTrianglePositions, 0x1234567u);
    const TestMesh baked{ out.hostTrianglePositions, out.hostTriangleAttrs };
    for (const Ray& ray : rays)
    {
        float bt = LARGE_T; glm::vec3 bn; glm::vec2 buv; int bidx = -1;
        const bool bhit = bruteForceClosest(ray, baked, 0, (int)baked.positions.size(), bt, bn, buv, bidx);

        const BvhHit vhit = traverseBvhClosest(ray, out.hostNodes.data(),
                                               out.hostTrianglePositions.data(), LARGE_T);

        if (bhit != vhit.hit)
        {
            printf("FAIL %s: hit flag brute=%d bvh=%d\n", name, bhit, vhit.hit);
            return false;
        }
        if (!bhit) continue;

        if (vhit.triIndex < 0 || vhit.triIndex >= (int)out.hostTrianglePositions.size())
        {
            printf("FAIL %s: triIndex %d out of range\n", name, vhit.triIndex);
            return false;
        }

        glm::vec3 vn;
        glm::vec2 vuv;
        glm::vec4 vtangent;
        glm::vec3 vcolor;
        interpolateTriangleAttributes(out.hostTrianglePositions[vhit.triIndex],
                                      out.hostTriangleAttrs[vhit.triIndex],
                                      vhit.u, vhit.v, true,
                                      vn, vuv, vtangent, vcolor);

        if (fabsf(bt - vhit.t) > 1e-3f || glm::dot(bn, vn) < 0.9999f)
        {
            printf("FAIL %s: t brute=%.6f bvh=%.6f dot=%.6f\n", name, bt, vhit.t, glm::dot(bn, vn));
            return false;
        }
        if (glm::length(buv - vuv) > 1e-3f)
        {
            printf("FAIL %s: uv brute=(%.4f,%.4f) bvh=(%.4f,%.4f)\n", name,
                   buv.x, buv.y, vuv.x, vuv.y);
            return false;
        }
        const int bvhMaterialId = out.hostSurfaces[out.hostTriangleAttrs[vhit.triIndex].surfaceId].materialId;
        const int bruteMaterialId = out.hostSurfaces[out.hostTriangleAttrs[bidx].surfaceId].materialId;
        if (bvhMaterialId != bruteMaterialId)
        {
            printf("FAIL %s: BVH materialId %d != brute %d\n", name,
                   bvhMaterialId, bruteMaterialId);
            return false;
        }
    }
    return true;
}

// Test 4: tree structure.  Root is node 0; DFS reaches every node exactly
// once (no cycles, no orphans), and the flattened triangle array is a
// PERFECT TILING of the leaves: every leaf chunk is a contiguous run, and
// the runs partition [0, N) with no overlap and no gap.  That tiling is
// exactly the property that makes the traversal's per-leaf sequential read
// `tris[triBase + j]` cache-friendly.
//
// Two content assertions close the loop that tiling alone leaves open:
//   (a) every triangle in a leaf's contiguous chunk is fully inside that
//       leaf's AABB (the build expanded `bounds` over exactly these
//       triangles), so the chunk really holds that leaf's triangles and
//   (b) the flattened output is a bijection of the baked source, so the
//       flatten never dropped, duplicated, or altered a triangle.
bool testBuildStructure(const TestMesh& mesh, const glm::mat4& T, int materialId)
{
    std::vector<Geom> geoms(1);
    geoms[0] = makeGeom(materialId, 0, (int)mesh.positions.size(), T);

    BvhBuffers out;
    bvh::buildSceneBvh(out, mesh.positions, mesh.attrs, geoms);
    if (out.hostNodes.empty())
    {
        printf("FAIL structure: no nodes built\n");
        return false;
    }

    // (b) Per-case content-completeness: the flattened output must be a
    // bijection of the baked source — no triangle dropped, duplicated, or
    // altered by the flatten.  testTraversalVsBrute scans the SAME baked
    // array on both sides, so it cannot see flatten bugs; the standalone
    // bake tests only cover their own inputs.  This closes that gap for
    // every structure case.
    TestMesh expected;
    std::vector<int> expectedMaterialIds;
    for (size_t i = 0; i < mesh.positions.size(); ++i)
        appendBakedExpected(expected, expectedMaterialIds, geoms[0], mesh.positions[i], mesh.attrs[i]);
    TestMesh actual{ out.hostTrianglePositions, out.hostTriangleAttrs };
    std::vector<int> actualMaterialIds;
    unpackRuntimeSurfaces(out, actual, actualMaterialIds);
    if (!sameMultiset(expected, expectedMaterialIds, actual, actualMaterialIds, 1e-3f))
    {
        printf("FAIL structure: flattened output is not a bijection of the baked source\n");
        return false;
    }

    const int total = (int)out.hostTrianglePositions.size();
    std::vector<int> coverage(total, 0);   // cells covered by exactly one leaf ⇒ tiling
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
            if (off < 0 || cnt < 1 || off + cnt > total)
            {
                printf("FAIL structure: leaf chunk [%d,+%d) out of triangle range (%d)\n",
                       off, cnt, total);
                return false;
            }
            // Mark this leaf's contiguous run; a cell already covered ⇒ overlap.
            for (int j = 0; j < cnt; j++)
                if (coverage[off + j]++)
                {
                    printf("FAIL structure: leaf chunks overlap at triangle %d\n", off + j);
                    return false;
                }

            // (a) Memory-contiguity content check: every triangle in this
            // leaf's CONTIGUOUS chunk must lie fully inside the leaf's AABB.
            // The build expanded `bounds` over exactly these triangles
            // (their union — min/max is commutative, so this is bit-exact),
            // hence containment holds by construction.  A flatten bug that
            // wrote a DIFFERENT leaf's triangles into this chunk breaks it
            // immediately.  The coverage/tiling loop above only proves the
            // array is partitioned into contiguous runs, not that each run
            // holds its own leaf's triangles.
            for (int j = 0; j < cnt; j++)
            {
                AABB tb;
                tb.expand(out.hostTrianglePositions[off + j]);
                if (tb.min.x < n.bounds.min.x - 1e-4f || tb.max.x > n.bounds.max.x + 1e-4f ||
                    tb.min.y < n.bounds.min.y - 1e-4f || tb.max.y > n.bounds.max.y + 1e-4f ||
                    tb.min.z < n.bounds.min.z - 1e-4f || tb.max.z > n.bounds.max.z + 1e-4f)
                {
                    printf("FAIL structure: leaf %d chunk [%d,+%d) triangle %d outside leaf AABB\n",
                           idx, off, cnt, off + j);
                    return false;
                }
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

    // Every triangle covered by exactly one leaf ⇒ leaves tile [0, N):
    // no gaps, no duplication.  Combined with testWorldBake's multiset
    // check, each leaf's run is exactly the build's assigned triangles.
    for (int i = 0; i < total; i++)
        if (coverage[i] != 1)
        {
            printf("FAIL structure: triangle %d covered by %d leaves (want 1)\n", i, coverage[i]);
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

    // ---- Near-first ordering metric ----
    // traverseBvhClosest orders children by tEntry: the smaller entry is the
    // nearer child and is descended first.  These cases lock that tEntry is
    // the true ray-distance entry, so the ordering picks the physically
    // first-hit box (including under negative direction).
    const AABB boxNear = box;                       // [0,2] × [0,1] × [0,1]
    AABB boxFar;
    boxFar.min = glm::vec3(3, 0, 0);
    boxFar.max = glm::vec3(5, 1, 1);

    // +x ray from (-5, .5, .5): boxNear entered at 5, boxFar at 8 → boxNear
    // has the smaller entry (near child).
    {
        const glm::vec3 o(-5.0f, 0.5f, 0.5f);
        float en = -1.0f, ef = -1.0f;
        if (!intersectRayAABBEntry(o, invDirPX, boxNear, RAY_EPSILON, LARGE_T, en) ||
            !intersectRayAABBEntry(o, invDirPX, boxFar,  RAY_EPSILON, LARGE_T, ef))
        {
            printf("FAIL aabb-entry: both boxes should hit (+x)\n");
            return false;
        }
        if (fabsf(en - 5.0f) > 1e-4f || fabsf(ef - 8.0f) > 1e-4f || !(en < ef))
        {
            printf("FAIL aabb-entry: +x near=%f far=%f (want near 5 < far 8)\n", en, ef);
            return false;
        }
    }

    // -x ray from (7, .5, .5): the far box [3,5] is hit FIRST (entry 2), the
    // near box second (entry 5).  The ordering must still pick the smaller
    // entry — the box the ray physically reaches first.
    {
        const glm::vec3 o(7.0f, 0.5f, 0.5f);
        float en = -1.0f, ef = -1.0f;
        if (!intersectRayAABBEntry(o, invDirNX, boxNear, RAY_EPSILON, LARGE_T, en) ||
            !intersectRayAABBEntry(o, invDirNX, boxFar,  RAY_EPSILON, LARGE_T, ef))
        {
            printf("FAIL aabb-entry: both boxes should hit (-x)\n");
            return false;
        }
        if (fabsf(ef - 2.0f) > 1e-4f || fabsf(en - 5.0f) > 1e-4f || !(ef < en))
        {
            printf("FAIL aabb-entry: -x first-hit=%f second=%f (want first-hit entry 2 < second 5)\n", ef, en);
            return false;
        }
    }

    // Ray origin INSIDE a box: entry clips to RAY_EPSILON (the box is "here").
    {
        const glm::vec3 o(1.0f, 0.5f, 0.5f);   // inside [0,2]×[0,1]×[0,1]
        float entry = -1.0f;
        if (!intersectRayAABBEntry(o, invDirPX, box, RAY_EPSILON, LARGE_T, entry))
        {
            printf("FAIL aabb-entry: expected hit from inside box\n");
            return false;
        }
        if (fabsf(entry - RAY_EPSILON) > 1e-6f)
        {
            printf("FAIL aabb-entry: inside-box entry=%f, want RAY_EPSILON\n", entry);
            return false;
        }
    }

    return true;
}

// Test 6: a near child is already AABB-tested by its parent, while a popped
// far child must be retested after the near subtree may have tightened t.
bool testKnownNearChildTraversal()
{
    auto makeLeaf = [](const TrianglePos& tri, int offset) {
        BvhNode leaf{};
        leaf.bounds.expand(tri);
        leaf.left = offset;
        leaf.right = 1;
        leaf.isLeaf = true;
        return leaf;
    };

    // Both child AABBs hit; the near triangle at z=2 must win and prune the
    // far child at z=5 when it is popped with the reduced far plane.
    {
        const TrianglePos nearTri{ {-1, -1, 2}, {1, -1, 2}, {0, 1, 2} };
        const TrianglePos farTri { {-1, -1, 5}, {1, -1, 5}, {0, 1, 5} };
        BvhNode nodes[3]{};
        nodes[1] = makeLeaf(nearTri, 0);
        nodes[2] = makeLeaf(farTri, 1);
        nodes[0].bounds.expand(nodes[1].bounds);
        nodes[0].bounds.expand(nodes[2].bounds);
        nodes[0].left = 1;
        nodes[0].right = 2;

        const TrianglePos tris[] = { nearTri, farTri };
        const BvhHit hit = traverseBvhClosest(
            Ray{ glm::vec3(0, 0, 0), glm::vec3(0, 0, 1) }, nodes, tris, LARGE_T);
        if (!hit.hit || hit.triIndex != 0 || fabsf(hit.t - 2.0f) > 1e-4f)
        {
            printf("FAIL known-near: expected near triangle at t=2\\n");
            return false;
        }
    }

    // The near leaf's box hits but its triangle does not. The far leaf must
    // still be popped and traversed, producing its triangle at z=5.
    {
        const TrianglePos nearMiss{ {0, 0, 2}, {2, 0, 2}, {0, 2, 2} };
        const TrianglePos farTri  { {0.5f, 0.5f, 5}, {2.5f, 0.5f, 5}, {1.5f, 2.5f, 5} };
        BvhNode nodes[3]{};
        nodes[1] = makeLeaf(nearMiss, 0);
        nodes[2] = makeLeaf(farTri, 1);
        nodes[0].bounds.expand(nodes[1].bounds);
        nodes[0].bounds.expand(nodes[2].bounds);
        nodes[0].left = 1;
        nodes[0].right = 2;

        const TrianglePos tris[] = { nearMiss, farTri };
        const BvhHit hit = traverseBvhClosest(
            Ray{ glm::vec3(1.5f, 1.5f, 0), glm::vec3(0, 0, 1) }, nodes, tris, LARGE_T);
        if (!hit.hit || hit.triIndex != 1 || fabsf(hit.t - 5.0f) > 1e-4f)
        {
            printf("FAIL known-near: expected far triangle after near-leaf miss\\n");
            return false;
        }
    }

    return true;
}

// Test 7: a fixed EPSILON does not change 1000.0f, but the production
// scale-aware offset must move both sides of a large-coordinate surface.
bool testScaleAwareRayOffset()
{
    const glm::vec3 point(1000.0f, -600.0f, 0.25f);
    const glm::vec3 normal(1.0f, 0.0f, 0.0f);
    const glm::vec3 positive = offsetRayOrigin(point, normal, 1.0f);
    const glm::vec3 negative = offsetRayOrigin(point, normal, -1.0f);
    const glm::vec3 nearOrigin = offsetRayOrigin(glm::vec3(0.0f), normal, 1.0f);

    if (!(positive.x > point.x) || !(negative.x < point.x) ||
        !(nearOrigin.x >= EPSILON))
    {
        printf("[FAIL] scale-aware ray origin offset\n");
        return false;
    }
    printf("[PASS] scale-aware ray origin offset\n");
    return true;
}

// Test 8: empty scene.  No triangles → no tree; traversal with null buffers
// must miss without touching any memory.
bool testEmptyScene()
{
    std::vector<TrianglePos> noPositions;
    std::vector<TriangleAttr> noAttrs;
    std::vector<Geom> geoms(1);   // default: offset -1, count 0
    BvhBuffers out;
    bvh::buildSceneBvh(out, noPositions, noAttrs, geoms);

    if (!out.hostNodes.empty() || !out.hostTrianglePositions.empty() ||
        !out.hostTriangleAttrs.empty() || !out.hostSurfaces.empty())
    {
        printf("FAIL empty: expected empty tree\n");
        return false;
    }

    const Ray ray{ glm::vec3(0, 0, 0), glm::vec3(1, 0, 0) };
    const BvhHit h = traverseBvhClosest(ray, nullptr, nullptr, LARGE_T);
    if (h.hit)
    {
        printf("FAIL empty: traversal with null buffers reported a hit\n");
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
    if (!testKnownNearChildTraversal()) failures++;
    if (!testScaleAwareRayOffset()) failures++;
    if (!testEmptyScene()) failures++;

    // Traversal + structure across mesh kinds and transforms.
    const TestMesh cube  = makeCube(1.0f);
    const TestMesh cloud = makeRandomCloud(120, 0xDEADBEEFu);
    const TestMesh degen = makeDegenerate(40);

    struct Case { const char* name; const TestMesh& mesh; glm::mat4 T; int mat; };
    const Case cases[] = {
        { "cube_identity",   cube,  makeTransform({ 0, 0, 0 }, { 0, 0, 0 }, { 1, 1, 1 }),      5 },
        { "cube_transform",  cube,  makeTransform({ 1, 2, 3 }, { 0.4f, 0.2f, 0.3f }, { 2, 1, 3 }), 6 },
        { "cloud_identity",  cloud, makeTransform({ 0, 0, 0 }, { 0, 0, 0 }, { 1, 1, 1 }),      7 },
        { "cloud_transform", cloud, makeTransform({ -3, 1, 0 }, { 0.1f, 0.9f, 0.2f }, { 0.5f, 2, 1 }), 8 },
        { "degenerate",      degen, makeTransform({ 0, 0, 0 }, { 0, 0, 0 }, { 1, 1, 1 }),      9 },
    };
    for (const Case& c : cases)
    {
        if (!testBuildStructure(c.mesh, c.T, c.mat)) failures++;
        if (!testTraversalVsBrute(c.name, c.mesh, c.T, c.mat)) failures++;
    }

    if (failures == 0)
    {
        printf("ALL PASS\n");
        return 0;
    }
    printf("%d FAILURE(S)\n", failures);
    return 1;
}
