/**
 * @file loader_test.cu
 * @brief Edge-case tests for the whole scene loader (OBJ + glTF + JSON dispatch).
 *
 * Includes the REAL scene-loader sources (obj / gltf / texture / scene .cpp,
 * which carry the tinyobjloader, cgltf, and stb implementations) and links
 * utils/utilities.cu for buildTransformationMatrix, so this exercises the
 * production load path.  Because the sources are in this TU, the
 * SceneLoader::appendTriangle / loadOBJ / loadGLTF helpers are callable directly,
 * and SceneLoader::loadFromJSON covers the JSON dispatch.
 *
 * Covers extreme cases: face-normal fallback (no NORMAL / zero NORMAL / NaN),
 * degenerate triangles, non-triangle primitives, missing POSITION, indexed vs
 * non-indexed, multiple meshes, out-of-range indices, index counts not a
 * multiple of 3, external .bin buffers, node-transform application,
 * normalized int8 POSITION, and JSON dispatch (missing mesh / unsupported ext).
 *
 * Build (from tests/loader_test/):
 *   cmake -G "Visual Studio 17 2022" -A x64 -B build .
 *   cmake --build build --config Release
 * Then run build/Release/loader_test.exe
 */

// The scene loader is split into four TUs (obj / gltf / texture / scene).
// Each is #included here so this test links the production load path without
// touching the root CMake build, and so the SceneLoader helpers are callable
// directly in this TU.
//
// texture_loader.cpp calls stbi_load for glTF/MTL images; this TU must supply
// the implementation (the main app gets it from src/stb.cpp).
//
// ORDER MATTERS: stb_image v2.06 keeps the implementation block OUTSIDE the
// include guard, so STB_IMAGE_IMPLEMENTATION must NOT be defined when
// texture_loader.cpp's own `#include <stb_image.h>` runs (that would compile
// the implementation a second time).  Include the loaders first (declarations
// only), then define the implementation in a second include.
#include "scene/loader_internal.h"
#include "scene/texture_loader.cpp"
#include "scene/obj_loader.cpp"
#include "scene/gltf_loader.cpp"
#include "scene/scene_loader.cpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <io.h>   // _dup / _dup2 / _close
#endif

static int failures = 0;

static void check(bool cond, const char* what)
{
    if (!cond) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       { std::printf("  ok:   %s\n", what); }
}

// ---- Geometry helpers -------------------------------------------------

static bool isUnit(const glm::vec3& n, float eps = 1e-4f)
{
    return std::fabs(glm::dot(n, n) - 1.0f) < eps;
}

struct LoadedMesh
{
    std::vector<TrianglePos> positions;
    std::vector<TriangleAttr> attrs;
    size_t size() const { return positions.size(); }
    bool empty() const { return positions.empty(); }
};

static glm::vec3 faceNormal(const TrianglePos& pos)
{
    return glm::normalize(glm::cross(pos.v1 - pos.v0, pos.v2 - pos.v0));
}

// True when every vertex normal is unit and parallel to the geometric face
// normal (i.e. appendTriangle fell back to fn).  Not valid on degenerate triangles.
static bool normalsFollowFace(const TrianglePos& pos, const TriangleAttr& attr, float eps = 1e-4f)
{
    glm::vec3 fn = faceNormal(pos);
    for (const glm::vec3* n : { &attr.n0, &attr.n1, &attr.n2 })
    {
        if (!isUnit(*n)) return false;
        if (std::fabs(glm::dot(*n, fn) - 1.0f) > eps) return false;
    }
    return true;
}

// ---- Loader helpers ---------------------------------------------------

static LoadedMesh loadOBJTris(const std::string& exeDir, const std::string& asset)
{
    LoadedMesh mesh;
    SceneLoader::loadOBJ(exeDir + "/assets/" + asset, mesh.positions, mesh.attrs);
    return mesh;
}

static LoadedMesh loadGLTFTris(const std::string& exeDir, const std::string& asset)
{
    Scene scene;
    SceneLoader::loadGLTF(scene, exeDir + "/assets/" + asset);
    return LoadedMesh{ scene.hostTrianglePositions, scene.hostTriangleAttrs };
}

#if defined(_WIN32)
// RAII redirect of fd 2 (stderr, where Log::warn writes) to a file, so the
// test can assert that a specific warning fired.  Single-threaded test makes
// this safe.
struct StderrCapture
{
    int       savedFd = -1;
    FILE*     cap     = nullptr;
    std::string capPath;

    explicit StderrCapture(const std::string& file) : capPath(file)
    {
        std::fflush(stderr);
        savedFd = _dup(_fileno(stderr));
        cap = std::fopen(capPath.c_str(), "w");
        if (cap) _dup2(_fileno(cap), _fileno(stderr));
    }
    ~StderrCapture()
    {
        if (cap)
        {
            std::fflush(cap);
            _dup2(savedFd, _fileno(stderr));   // restore fd 2
            std::fclose(cap);
            _close(savedFd);
        }
    }
    std::string text() const
    {
        std::ifstream f(capPath.c_str());
        return std::string(std::istreambuf_iterator<char>(f),
                           std::istreambuf_iterator<char>());
    }
};
#endif

// =====================================================================
// Part A — appendTriangle unit checks
// =====================================================================
static void testAppendTriangle()
{
    std::printf("=== appendTriangle ===\n");

    LoadedMesh mesh;
    SceneLoader::appendTriangle(mesh.positions, mesh.attrs,
        glm::vec3(0, 0, 0), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0),
        glm::vec3(0, 0, 1), glm::vec3(0, 0, 1), glm::vec3(0, 0, 1));
    check(mesh.attrs[0].n0 == glm::vec3(0, 0, 1) && mesh.attrs[0].n1 == glm::vec3(0, 0, 1) &&
          mesh.attrs[0].n2 == glm::vec3(0, 0, 1),
          "valid unit normals pass through unchanged");

    SceneLoader::appendTriangle(mesh.positions, mesh.attrs,
        glm::vec3(0, 0, 0), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0),
        glm::vec3(0, 0, 0), glm::vec3(0, 0, 0), glm::vec3(0, 0, 0));
    check(mesh.attrs[1].n0 == glm::vec3(0, 0, 1) && mesh.attrs[1].n1 == glm::vec3(0, 0, 1) &&
          mesh.attrs[1].n2 == glm::vec3(0, 0, 1) && isUnit(mesh.attrs[1].n0),
          "zero-length normals fall back to the face normal");

    glm::vec3 nanv = glm::vec3(std::numeric_limits<float>::quiet_NaN());
    SceneLoader::appendTriangle(mesh.positions, mesh.attrs,
        glm::vec3(0, 0, 0), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0),
        nanv, nanv, nanv);
    check(mesh.attrs[2].n0 == glm::vec3(0, 0, 1) && isUnit(mesh.attrs[2].n0),
          "NaN normals fall back to the face normal");

    SceneLoader::appendTriangle(mesh.positions, mesh.attrs,
        glm::vec3(0, 0, 0), glm::vec3(0, 0, 0), glm::vec3(0, 0, 0),
        glm::vec3(0, 0, 0), glm::vec3(0, 0, 0), glm::vec3(0, 0, 0));
    check(mesh.attrs[3].n0 == glm::vec3(0, 1, 0) && mesh.attrs[3].n1 == glm::vec3(0, 1, 0) &&
          mesh.attrs[3].n2 == glm::vec3(0, 1, 0),
          "degenerate triangle falls back to (0,1,0)");

    std::printf("\n");
}

// =====================================================================
// Part B — loadOBJ edge cases
// =====================================================================
static void testLoadOBJ(const std::string& exeDir)
{
    std::printf("=== loadOBJ ===\n");

    {
        auto tris = loadOBJTris(exeDir, "tri_vn.obj");
        check(tris.size() == 1, "tri_vn.obj -> 1 triangle");
        if (tris.size() == 1)
            check(tris.attrs[0].n0 == glm::vec3(0, 0, 1),
                  "tri_vn.obj -> vertex normals loaded from vn entries");
    }
    {
        auto tris = loadOBJTris(exeDir, "tri_uv.obj");
        check(tris.size() == 1, "tri_uv.obj -> 1 triangle");
        if (tris.size() == 1)
            check(tris.attrs[0].uv0 == glm::vec2(0, 0) && tris.attrs[0].uv1 == glm::vec2(1, 0) &&
                  tris.attrs[0].uv2 == glm::vec2(0, 1),
                  "tri_uv.obj -> per-corner UVs loaded from vt entries");
    }
    {
        auto tris = loadOBJTris(exeDir, "tri_color.obj");
        check(tris.size() == 1, "tri_color.obj -> 1 triangle");
        if (tris.size() == 1)
            check(tris.attrs[0].c0 == glm::vec3(1, 0, 0) &&
                  tris.attrs[0].c1 == glm::vec3(0, 1, 0) &&
                  tris.attrs[0].c2 == glm::vec3(0, 0, 1),
                  "tri_color.obj -> vertex colors loaded from v entries");
    }
    {
        // vt exists for some corners but not the face → per-corner guard.
        auto tris = loadOBJTris(exeDir, "tri_no_vn.obj");
        check(tris.size() == 1, "tri_no_vn.obj -> 1 triangle (no vt)");
        if (tris.size() == 1)
            check(tris.attrs[0].uv0 == glm::vec2(0, 0) && tris.attrs[0].uv1 == glm::vec2(0, 0) &&
                  tris.attrs[0].uv2 == glm::vec2(0, 0),
                  "tri_no_vn.obj -> missing vt falls back to (0,0)");
    }
    {
        auto tris = loadOBJTris(exeDir, "tri_no_vn.obj");
        check(tris.size() == 1, "tri_no_vn.obj -> 1 triangle");
        if (tris.size() == 1)
            check(normalsFollowFace(tris.positions[0], tris.attrs[0]),
                  "tri_no_vn.obj -> face-normal fallback");
    }
    {
        auto tris = loadOBJTris(exeDir, "quad.obj");
        check(tris.size() == 2, "quad.obj -> 2 triangles (tinyobj triangulates)");
    }
    {
        auto tris = loadOBJTris(exeDir, "degenerate.obj");
        check(tris.size() == 1, "degenerate.obj -> 1 triangle (no crash)");
        if (tris.size() == 1)
            check(tris.attrs[0].n0 == glm::vec3(0, 1, 0),
                  "degenerate.obj -> fallback normal (0,1,0)");
    }
    {
        auto tris = loadOBJTris(exeDir, "oob.obj");
        check(tris.size() == 1, "oob.obj -> 1 triangle (tinyobj skips OOB quad)");
    }
    {
        LoadedMesh tris;
        auto slice = SceneLoader::loadOBJ(exeDir + "/assets/missing.obj", tris.positions, tris.attrs);
        check(slice.first == -1 && slice.second == 0,
              "missing.obj -> {-1, 0}");
    }

    std::printf("\n");
}

// =====================================================================
// Part C — loadGLTF edge cases
// =====================================================================
static void testLoadGLTF(const std::string& exeDir)
{
    std::printf("=== loadGLTF ===\n");

    {
        auto tris = loadGLTFTris(exeDir, "cube_embed.gltf");
        check(tris.size() == 12, "cube_embed.gltf -> 12 triangles");
        if (tris.size() == 12)
        {
            bool ok = true;
            for (size_t i = 0; i < tris.size(); ++i)
            {
                const TrianglePos& pos = tris.positions[i];
                const TriangleAttr& attr = tris.attrs[i];
                for (const glm::vec3* n : { &attr.n0, &attr.n1, &attr.n2 })
                    if (!isUnit(*n)) ok = false;
                for (const glm::vec3* v : { &pos.v0, &pos.v1, &pos.v2 })
                    if (v->x < -1.01f || v->x > 1.01f || v->y < -1.01f ||
                        v->y > 1.01f || v->z < -1.01f || v->z > 1.01f)
                        ok = false;
            }
            check(ok, "cube_embed.gltf -> unit normals, verts in unit cube");
        }
    }
    {
        auto tris = loadGLTFTris(exeDir, "cube_glb.glb");
        check(tris.size() == 12, "cube_glb.glb -> 12 triangles");
    }
    {
        auto tris = loadGLTFTris(exeDir, "cube_nonindexed.gltf");
        check(tris.size() == 12, "cube_nonindexed.gltf -> 12 triangles");
        if (tris.size() == 12)
        {
            bool ok = true;
            for (const TriangleAttr& attr : tris.attrs)
                for (const glm::vec3* n : { &attr.n0, &attr.n1, &attr.n2 })
                    if (!isUnit(*n)) ok = false;
            check(ok, "cube_nonindexed.gltf -> unit normals");
        }
    }
    {
        auto tris = loadGLTFTris(exeDir, "cube_nonormal.gltf");
        check(tris.size() == 12, "cube_nonormal.gltf -> 12 triangles (no NORMAL)");
        if (tris.size() == 12)
        {
            bool ok = true;
            for (size_t i = 0; i < tris.size(); ++i)
                if (!normalsFollowFace(tris.positions[i], tris.attrs[i])) ok = false;
            check(ok, "cube_nonormal.gltf -> face-normal fallback");
        }
    }
    {
        auto tris = loadGLTFTris(exeDir, "mode_line.gltf");
        check(tris.size() == 12, "mode_line.gltf -> 12 triangles (points primitive skipped)");
    }
    {
        auto tris = loadGLTFTris(exeDir, "no_position.gltf");
        check(tris.empty(), "no_position.gltf -> 0 triangles (no POSITION)");
    }
    {
        auto tris = loadGLTFTris(exeDir, "no_mode.gltf");
        check(tris.size() == 1, "no_mode.gltf -> 1 triangle (mode defaults to triangles)");
    }
    {
        auto tris = loadGLTFTris(exeDir, "multi_mesh.gltf");
        check(tris.size() == 3, "multi_mesh.gltf -> 3 triangles (2 + 1 across 2 meshes)");
    }
    {
        // Out-of-range vertex index (999 >= 24).  This cgltf version's
        // cgltf_validate cross-checks index bounds against the vertex count
        // (cgltf.h ~1717-1722), so it rejects the whole file before the
        // loader iterates — loadGLTF returns {-1,0}.  The loader's own
        // out-of-range guard in the indexed path is defense-in-depth.
#if defined(_WIN32)
        Scene scene;
        StderrCapture cap(exeDir + "/stderr_capture.log");
        auto slice = SceneLoader::loadGLTF(scene, exeDir + "/assets/oob_index.gltf");
        check(slice.first == -1 && slice.second == 0,
              "oob_index.gltf -> {-1, 0} (cgltf_validate rejects OOB indices)");
        check(cap.text().find("glTF validation failed") != std::string::npos,
              "oob_index.gltf -> 'glTF validation failed' error emitted");
#else
        Scene scene;
        auto slice = SceneLoader::loadGLTF(scene, exeDir + "/assets/oob_index.gltf");
        check(slice.first == -1 && slice.second == 0,
              "oob_index.gltf -> {-1, 0} (cgltf_validate rejects OOB indices)");
#endif
    }
    {
        // NORMAL count (2) != POSITION count (4): cgltf_validate requires all
        // attributes to share a count (cgltf.h ~1696), so the file is rejected.
        Scene scene;
        auto slice = SceneLoader::loadGLTF(scene, exeDir + "/assets/normal_mismatch.gltf");
        check(slice.first == -1 && slice.second == 0,
              "normal_mismatch.gltf -> {-1, 0} (attribute count mismatch rejected)");
    }
    {
        // Index count not a multiple of 3: the whole primitive is skipped.
#if defined(_WIN32)
        Scene scene;
        StderrCapture cap(exeDir + "/stderr_capture2.log");
        auto slice = SceneLoader::loadGLTF(scene, exeDir + "/assets/index_not_mult3.gltf");
        check(slice.second == 0, "index_not_mult3.gltf -> 0 triangles (skipped)");
        check(cap.text().find("not a multiple of 3") != std::string::npos,
              "index_not_mult3.gltf -> 'not a multiple of 3' warning emitted");
#else
        auto tris = loadGLTFTris(exeDir, "index_not_mult3.gltf");
        check(tris.empty(), "index_not_mult3.gltf -> 0 triangles (skipped)");
#endif
    }
    {
        auto tris = loadGLTFTris(exeDir, "degenerate.gltf");
        check(tris.size() == 1, "degenerate.gltf -> 1 triangle (no crash)");
        if (tris.size() == 1)
            check(tris.attrs[0].n0 == glm::vec3(0, 1, 0) && tris.attrs[0].n1 == glm::vec3(0, 1, 0) &&
                  tris.attrs[0].n2 == glm::vec3(0, 1, 0),
                  "degenerate.gltf -> fallback normal (0,1,0)");
    }
    {
        auto tris = loadGLTFTris(exeDir, "zero_normal.gltf");
        check(tris.size() == 1, "zero_normal.gltf -> 1 triangle");
        if (tris.size() == 1)
            check(normalsFollowFace(tris.positions[0], tris.attrs[0]),
                  "zero_normal.gltf -> zero NORMALs fall back to face normal");
    }
    {
        auto tris = loadGLTFTris(exeDir, "external_bin.gltf");
        check(tris.size() == 1, "external_bin.gltf -> 1 triangle (external .bin)");
    }
    {
        auto tris = loadGLTFTris(exeDir, "node_transform.gltf");
        check(tris.size() == 1, "node_transform.gltf -> 1 triangle");
        if (tris.size() == 1)
            // Node carries a translation matrix of (5,5,5), which the loader
            // applies: raw verts (0,0,0)/(1,0,0)/(0,1,0) become (5,5,5)/(6,5,5)/(5,6,5).
            check(tris.positions[0].v0 == glm::vec3(5, 5, 5) && tris.positions[0].v1 == glm::vec3(6, 5, 5) &&
                  tris.positions[0].v2 == glm::vec3(5, 6, 5),
                  "node_transform.gltf -> node translation (5,5,5) applied to verts");
    }
    {
        auto tris = loadGLTFTris(exeDir, "norm_i8.gltf");
        check(tris.size() == 1, "norm_i8.gltf -> 1 triangle");
        if (tris.size() == 1)
        {
            const glm::vec3& v1 = tris.positions[0].v1;
            const glm::vec3& v2 = tris.positions[0].v2;
            bool ok = std::fabs(tris.positions[0].v0.x) < 1e-3f && std::fabs(tris.positions[0].v0.y) < 1e-3f &&
                      std::fabs(tris.positions[0].v0.z) < 1e-3f &&
                      std::fabs(v1.x - 1.0f) < 1e-3f && std::fabs(v1.y) < 1e-3f && std::fabs(v1.z) < 1e-3f &&
                      std::fabs(v2.x) < 1e-3f && std::fabs(v2.y - 1.0f) < 1e-3f && std::fabs(v2.z) < 1e-3f;
            check(ok, "norm_i8.gltf -> normalized int8 POSITION decoded");
        }
    }
    {
        // TEXCOORD_0 present: loader must surface the per-vertex UVs.
        auto tris = loadGLTFTris(exeDir, "tri_uv.gltf");
        check(tris.size() == 1, "tri_uv.gltf -> 1 triangle");
        if (tris.size() == 1)
        {
            const TriangleAttr& attr = tris.attrs[0];
            bool ok = attr.uv0 == glm::vec2(0, 0) && attr.uv1 == glm::vec2(1, 0) &&
                      attr.uv2 == glm::vec2(0, 1);
            check(ok, "tri_uv.gltf -> per-vertex UVs read from TEXCOORD_0");
        }
    }
    {
        // glTF texture auto-load: external PNG baseColor + normal + occlusion
        // slots all referencing the SAME image.  The loader must load it ONCE
        // into Scene::textures (dedup by cgltf image index), stamp the global
        // id on the triangle's TextureBinding, and linearize the sRGB
        // baseColor (gray 128/255 ≈ 0.502 → linear ≈ 0.216).
        Scene scene;
        SceneLoader::loadGLTF(scene, exeDir + "/assets/tex_cube.gltf");
        check(scene.hostTrianglePositions.size() == 1, "tex_cube.gltf -> 1 triangle");
        check(scene.textures.size() == 1,
              "tex_cube.gltf -> 3 slots sharing one image dedup to 1 texture");
        if (scene.textures.size() == 1)
        {
            const TextureData& td = scene.textures[0];
            check(td.width == 2 && td.height == 2, "tex_cube.gltf -> texture is 2x2");
            check(std::fabs(td.pixels[0].r - 0.2158f) < 5e-3f &&
                  std::fabs(td.pixels[0].g - 0.2158f) < 5e-3f &&
                  std::fabs(td.pixels[0].b - 0.2158f) < 5e-3f,
                  "tex_cube.gltf -> baseColor linearized (gray 128 → ~0.216)");
        }
        if (scene.hostTriangleAttrs.size() == 1)
        {
            const SurfaceBinding& b = scene.surfaceBindings[scene.hostTriangleAttrs[0].surfaceId];
            check(b.baseColor == 0 && b.normal == 0 && b.occlusion == 0,
                  "tex_cube.gltf -> bound slots stamped with texture id 0");
            check(b.metallicRoughness == -1 && b.emissive == -1,
                  "tex_cube.gltf -> unbound slots stay -1");
        }
    }
    {
        // No TEXCOORD_0: UVs must fall back to (0,0) — not garbage.
        auto tris = loadGLTFTris(exeDir, "no_mode.gltf");
        check(tris.size() == 1, "no_mode.gltf -> 1 triangle (no UVs)");
        if (tris.size() == 1)
            check(tris.attrs[0].uv0 == glm::vec2(0, 0) && tris.attrs[0].uv1 == glm::vec2(0, 0) &&
                  tris.attrs[0].uv2 == glm::vec2(0, 0),
                  "no_mode.gltf -> missing TEXCOORD_0 falls back to (0,0)");
    }

    std::printf("\n");
}

// =====================================================================
// Part D — loadFromJSON (JSON dispatch + geom slicing)
// =====================================================================
static void testLoadFromJSON(const std::string& exeDir)
{
    std::printf("=== loadFromJSON ===\n");

    {
        Scene scene = SceneLoader::loadFromJSON(exeDir + "/scene_baseline.json");
        check(scene.geoms.size() == 1, "scene_baseline: 1 geom");
        if (scene.geoms.size() == 1)
            check(scene.geoms[0].meshTriangleOffset == 0 &&
                  scene.geoms[0].meshTriangleCount == 12,
                  "scene_baseline: geom slice (0, 12)");
        check(scene.hostTrianglePositions.size() == 12, "scene_baseline: 12 host triangles");
    }
    {
        Scene scene = SceneLoader::loadFromJSON(exeDir + "/scene_glb.json");
        check(scene.hostTrianglePositions.size() == 12, "scene_glb: 12 host triangles");
    }
    {
        Scene scene = SceneLoader::loadFromJSON(exeDir + "/scene_quad_obj.json");
        check(scene.hostTrianglePositions.size() == 2, "scene_quad_obj: 2 host triangles");
    }
    {
        Scene scene = SceneLoader::loadFromJSON(exeDir + "/scene_missing_mesh.json");
        check(scene.geoms.size() == 1, "scene_missing_mesh: geom still pushed");
        if (scene.geoms.size() == 1)
            check(scene.geoms[0].meshTriangleOffset == -1 &&
                  scene.geoms[0].meshTriangleCount == 0,
                  "scene_missing_mesh: {-1, 0} slice");
        check(scene.hostTrianglePositions.empty(), "scene_missing_mesh: no host triangles");
    }
    {
        Scene scene = SceneLoader::loadFromJSON(exeDir + "/scene_unsupported_ext.json");
        check(scene.geoms.empty(),
              "scene_unsupported_ext: no geom (unsupported extension skipped)");
    }
    {
        Scene scene = SceneLoader::loadFromJSON(exeDir + "/scene_multi.json");
        check(scene.geoms.size() == 2, "scene_multi: 2 geoms");
        if (scene.geoms.size() == 2)
            check(scene.geoms[0].meshTriangleOffset == 0 && scene.geoms[0].meshTriangleCount == 3 &&
                  scene.geoms[1].meshTriangleOffset == 3 && scene.geoms[1].meshTriangleCount == 3,
                  "scene_multi: slices (0,3) and (3,3)");
        check(scene.hostTrianglePositions.size() == 6, "scene_multi: 6 host triangles");
    }
    std::printf("\n");
}

// =====================================================================
// Part E — sweep mode: load every glTF under a models root
//
//   loader_test --sweep <modelsRoot>
//
// Walks <modelsRoot>/**/*.gltf|glb (glTF-Sample-Assets layout: each model
// has glTF/ glTF-Binary/ glTF-Embedded/ variant dirs) and loads each with
// the REAL production loadGLTF.  The point is scale: M1–M6 unit tests prove
// each edge case in isolation; this proves no real-world asset crashes the
// loader, that every material-slot image that EXISTS on disk actually gets
// loaded (the M6 auto-load contract), and that each triangle's five texture
// slots point at the RIGHT image (the slot-mapping contract, checked by
// pixel content — see verifyGltfSlotMapping).
//
// Per-model assertions:
//   - loadGLTF returns a valid slice (load-failures are listed, not fatal —
//     Draco-compressed models and a few spec-violators legitimately fail).
//   - When the file parses and the material slots reference external image
//     URIs whose files exist next to the .gltf, Scene::textures must end up
//     with EXACTLY that many entries (dedup by cgltf image index).  A shortfall
//     is a real bug — an existing PNG silently skipped.
//   - Every distinct TextureBinding stamped on the loaded triangles must be
//     the binding some material's slots imply: each bound slot's texture
//     content must equal its image file (decoded exactly as loadTextureFile
//     stores it).  A mismatch is a real bug — the wrong PNG bound to a slot.
// Exit code: 0 iff no texture shortfall and no slot mismatch.  Crashes abort
// the whole sweep (the report then shows how far it got).
// =====================================================================
namespace fs = std::filesystem;

// Resolve a material slot's image to its external file path (relative to the
// .gltf's directory), or "" when the slot should NOT load an image — absent,
// embedded (bufferView), data: URI, or the percent-decoded file is missing.
// Mirrors the loader's "unsupported → -1" contract (resolveGltfTextureSlot).
static std::string loadableImageFile(const fs::path& dir,
                                     const cgltf_texture_view& view)
{
    if (view.texture == nullptr || view.texture->image == nullptr)
        return "";
    const cgltf_image* img = view.texture->image;
    if (img->buffer_view || img->uri == nullptr) return "";      // embedded
    const std::string uri = img->uri;
    if (uri.rfind("data:", 0) == 0) return "";                   // data URI
    std::string    decoded(uri.begin(), uri.end());
    std::vector<char> buf(decoded.begin(), decoded.end());
    buf.push_back('\0');
    cgltf_decode_uri(buf.data());
    decoded.assign(buf.data());
    const fs::path p = dir / decoded;
    return fs::exists(p) ? p.string() : "";
}

// Count distinct glTF image indices that the material slots reference and that
// SHOULD load: external file URIs (not bufferView-embedded, not data:) whose
// percent-decoded file exists next to the .gltf.  -1 if the file doesn't
// parse/validate (no texture assertion then).
static int countExpectedExternalTextures(const std::string& path)
{
    cgltf_options opts{};
    cgltf_data*   data = nullptr;
    if (cgltf_parse_file(&opts, path.c_str(), &data) != cgltf_result_success)
        return -1;
    if (cgltf_validate(data) != cgltf_result_success)
    {
        cgltf_free(data);
        return -1;
    }

    const fs::path dir = fs::path(path).parent_path();
    std::set<int>  distinct;
    for (size_t m = 0; m < data->materials_count; ++m)
    {
        const cgltf_material* mat = &data->materials[m];
        const cgltf_texture_view views[] = {
            mat->pbr_metallic_roughness.base_color_texture,
            mat->normal_texture,
            mat->pbr_metallic_roughness.metallic_roughness_texture,
            mat->occlusion_texture,
            mat->emissive_texture,
        };
        for (const cgltf_texture_view& v : views)
            if (!loadableImageFile(dir, v).empty())
                distinct.insert((int)(v.texture->image - data->images));
    }
    cgltf_free(data);
    return (int)distinct.size();
}

// =====================================================================
// Slot-mapping verification
//
// Every distinct TextureBinding stamped on the loaded triangles must be
// exactly the binding that SOME material's five slots imply — matched by
// PIXEL CONTENT, not by predicted texture id (ids are first-come during the
// scene walk, so predicting them would re-implement the loader).  This is
// what catches "the wrong PNG bound to a slot" — a swap, a duplicate, an
// out-of-range id — that the texture-COUNT check alone cannot see.
// =====================================================================

static uint64_t fnv1a64(const void* data, size_t n)
{
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; ++i)
        h = (h ^ p[i]) * 0x100000001b3ULL;
    return h;
}

// Hash of a loaded texture's pixels as the loader stored them (linear floats).
static uint64_t hashTexels(const std::vector<glm::vec3>& pixels)
{
    return pixels.empty()
        ? 0
        : fnv1a64(pixels.data(), pixels.size() * sizeof(glm::vec3));
}

// Decode an image file EXACTLY as loadTextureFile stores it (stbi_load with
// req_comp=3, optional sRGB→linear for color maps) and hash the resulting
// float texels.  0 on decode failure.  Same TU → same stb implementation, so a
// texture the loader loaded bit-matches its source file's hash.
static uint64_t hashImageFile(const std::string& path, bool srgb)
{
    int w = 0, h = 0, comp = 0;
    stbi_uc* data = stbi_load(path.c_str(), &w, &h, &comp, 3);
    if (data == nullptr)
        return 0;

    // Must replicate loadTextureFile's linearization byte-for-byte.
    const auto lin = [](float c) {
        return (c <= 0.04045f) ? c / 12.92f
                               : std::pow((c + 0.055f) / 1.055f, 2.4f);
    };

    std::vector<float> texels((size_t)w * (size_t)h * 3);
    for (int i = 0; i < w * h; i++)
    {
        const float r = data[3 * i + 0] / 255.0f;
        const float g = data[3 * i + 1] / 255.0f;
        const float b = data[3 * i + 2] / 255.0f;
        texels[3 * i + 0] = srgb ? (float)lin(r) : r;
        texels[3 * i + 1] = srgb ? (float)lin(g) : g;
        texels[3 * i + 2] = srgb ? (float)lin(b) : b;
    }
    stbi_image_free(data);
    return fnv1a64(texels.data(), texels.size() * sizeof(float));
}

// Content hash of an image file decoded under a treatment, cached per
// (path, srgb) so each file decodes at most once per sweep.
static uint64_t imageContentHash(
    std::map<std::pair<std::string, bool>, uint64_t>& cache,
    const std::string& file, bool srgb)
{
    const auto key = std::make_pair(file, srgb);
    const auto it  = cache.find(key);
    if (it != cache.end())
        return it->second;
    const uint64_t h = hashImageFile(file, srgb);
    cache.emplace(key, h);
    return h;
}

// Does a texture's content equal this slot's image under either the slot's
// canonical treatment or the opposite one?  The loader picks the sRGB treatment
// of the FIRST slot to reference an image (dedup by image index), so a data
// slot can legally hold a linearized texture (and vice versa) when one file is
// shared across roles.  Tolerate both — but only both.
static bool contentMatches(uint64_t texelHash, const std::string& file,
                           bool slotColor,
                           std::map<std::pair<std::string, bool>, uint64_t>& cache)
{
    const uint64_t canonical = imageContentHash(cache, file, slotColor);
    if (canonical != 0 && texelHash == canonical) return true;
    const uint64_t other = imageContentHash(cache, file, !slotColor);
    return other != 0 && texelHash == other;
}

// A material's five texture slots, in the loader's resolution order
// (bindGltfMaterial).  `view` points into the cgltf_material; `color` marks the
// sRGB→linear color slots (baseColor/emissive).
struct SlotSpec
{
    const char* name;
    const cgltf_texture_view* view;
    bool color;
};

static void materialSlots(const cgltf_material* mat, SlotSpec out[5])
{
    out[0] = { "baseColor",         &mat->pbr_metallic_roughness.base_color_texture, true };
    out[1] = { "normal",            &mat->normal_texture, false };
    out[2] = { "metallicRoughness", &mat->pbr_metallic_roughness.metallic_roughness_texture, false };
    out[3] = { "occlusion",         &mat->occlusion_texture, false };
    out[4] = { "emissive",          &mat->emissive_texture, true };
}

// Verify the slot mapping of a loaded glTF scene by independent re-parse.
// Returns the number of distinct bindings on the loaded triangles that no
// material's slot pattern can explain (0 = mapping is correct).
static int verifyGltfSlotMapping(const std::string& path, const Scene& scene)
{
    cgltf_options opts{};
    cgltf_data*   data = nullptr;
    if (cgltf_parse_file(&opts, path.c_str(), &data) != cgltf_result_success)
        return 0;   // can't independently parse → nothing to check
    if (cgltf_validate(data) != cgltf_result_success)
    {
        cgltf_free(data);
        return 0;
    }

    const fs::path dir = fs::path(path).parent_path();

    // Hash every loaded texture once.
    std::vector<uint64_t> texHash(scene.textures.size());
    for (size_t k = 0; k < scene.textures.size(); ++k)
        texHash[k] = hashTexels(scene.textures[k].pixels);

    // Distinct bindings referenced by the triangle attributes → how many triangles carry each.
    std::map<std::array<int, 5>, int> bindings;
    for (const TriangleAttr& attr : scene.hostTriangleAttrs)
    {
        const SurfaceBinding& binding = scene.surfaceBindings[attr.surfaceId];
        std::array<int, 5> k = { binding.baseColor, binding.normal,
                                 binding.metallicRoughness, binding.occlusion,
                                 binding.emissive };
        ++bindings[k];
    }

    std::map<std::pair<std::string, bool>, uint64_t> imgHash;
    int failures = 0;

    for (const auto& kv : bindings)
    {
        const std::array<int, 5>& k   = kv.first;
        const int                 nTri = kv.second;

        bool ok = false;
        for (size_t m = 0; m < data->materials_count && !ok; ++m)
        {
            SlotSpec slots[5];
            materialSlots(&data->materials[m], slots);

            bool match = true;
            for (int s = 0; s < 5 && match; ++s)
            {
                const std::string file = loadableImageFile(dir, *slots[s].view);
                if (file.empty())
                {
                    if (k[s] != -1) match = false;
                }
                else
                {
                    if (k[s] < 0 || k[s] >= (int)scene.textures.size())
                    { match = false; break; }
                    if (!contentMatches(texHash[k[s]], file, slots[s].color, imgHash))
                        match = false;
                }
            }
            if (match) ok = true;
        }

        if (!ok)
        {
            ++failures;
            std::printf("    [SLOT MISMATCH] %s  baseColor=%d normal=%d "
                        "metallicRoughness=%d occlusion=%d emissive=%d "
                        "(on %d triangles)\n",
                        path.c_str(), k[0], k[1], k[2], k[3], k[4], nTri);
        }
    }

    cgltf_free(data);
    return failures;
}

// Walk a tree depth-first calling `visit` for every regular file, without
// throwing on un-enumerable directories.  Some glTF-Sample-Assets names use
// unusual unicode (e.g. "Unicode❤♻Test") that makes std::filesystem throw on
// Windows; one bad path must not abort the whole sweep.
static void walkFiles(const fs::path& root,
                      const std::function<void(const fs::path&)>& visit)
{
    std::vector<fs::path> stack = { root };
    while (!stack.empty())
    {
        const fs::path dir = stack.back();
        stack.pop_back();

        std::error_code ec;
        fs::directory_iterator it(dir, ec), end;
        if (ec) { ec.clear(); continue; }            // unenumerable dir → skip
        while (it != end)
        {
            const auto& e = *it;
            std::error_code fec;
            if (e.is_directory(fec))
                stack.push_back(e.path());
            else if (e.is_regular_file(fec))
                visit(e.path());
            it.increment(ec);
            if (ec) { ec.clear(); break; }           // error mid-iteration → stop this dir
        }
    }
}

static int sweepGltfRoot(const std::string& root)
{
    int files = 0, loaded = 0, texExpected = 0, texLoaded = 0, shortfalls = 0;
    int slotFailures = 0;
    std::vector<std::string> failList, shortfallList, slotFailList;

    const auto process = [&](const fs::path& p) {
        if (p.extension() != ".gltf" && p.extension() != ".glb") return;

        Scene scene;
        auto slice = SceneLoader::loadGLTF(scene, p.string());
        ++files;

        const bool ok = slice.first >= 0;
        if (ok) { ++loaded; }
        else    { failList.push_back(p.string()); }

        const int expected = countExpectedExternalTextures(p.string());
        if (ok && expected >= 0)
        {
            texExpected += expected;
            texLoaded   += (int)scene.textures.size();
            if ((int)scene.textures.size() < expected)
            {
                ++shortfalls;
                shortfallList.push_back(
                    p.string() + "  expected " + std::to_string(expected) +
                    " textures, got " + std::to_string(scene.textures.size()));
            }
        }

        const int slotFails = ok ? verifyGltfSlotMapping(p.string(), scene) : 0;
        if (slotFails > 0)
        {
            slotFailures += slotFails;
            slotFailList.push_back(p.string());
        }

        std::printf("%-70s %s  tris=%-8d tex=%-3d slot=%s\n",
                    p.filename().string().c_str(),
                    ok ? "ok " : "FAIL",
                    (int)scene.hostTrianglePositions.size(),
                    (int)scene.textures.size(),
                    slotFails > 0 ? "FAIL" : "ok");
    };

    walkFiles(root, process);

    std::printf("\n===== SWEEP SUMMARY =====\n");
    std::printf("files scanned      : %d\n", files);
    std::printf("loaded ok          : %d\n", loaded);
    std::printf("load failures      : %d\n", (int)failList.size());
    std::printf("textures exp/ok    : %d / %d\n", texExpected, texLoaded);
    std::printf("texture shortfalls : %d\n", shortfalls);
    std::printf("slot mismatches    : %d\n", slotFailures);

    for (const auto& s : failList)
        std::printf("  [load fail] %s\n", s.c_str());
    for (const auto& s : shortfallList)
        std::printf("  [SHORTFALL] %s\n", s.c_str());
    for (const auto& s : slotFailList)
        std::printf("  [SLOT FAIL] %s\n", s.c_str());

    return (shortfalls == 0 && slotFailures == 0) ? 0 : 1;
}

int main(int argc, char** argv)
{
    // Sweep mode: loader_test --sweep <modelsRoot>
    if (argc >= 3 && std::string(argv[1]) == "--sweep")
        return sweepGltfRoot(argv[2]);

    // Assets + scene JSONs are copied next to the exe by POST_BUILD.
    // Allow an explicit base dir override:  loader_test <baseDir>
    const std::string baseDir = (argc > 1) ? argv[1]
        : std::filesystem::path(argv[0]).parent_path().string();
    const std::string exeDir = baseDir;

    if (!std::filesystem::exists(exeDir + "/assets/cube_embed.gltf"))
    {
        std::printf("Cannot find %s — did the POST_BUILD copy run?\n",
                    (exeDir + "/assets").c_str());
        return 2;
    }

    testAppendTriangle();
    testLoadOBJ(exeDir);
    testLoadGLTF(exeDir);
    testLoadFromJSON(exeDir);

    std::printf(failures == 0 ? "ALL PASS\n" : "FAILURES: %d\n", failures);
    return failures == 0 ? 0 : 1;
}
