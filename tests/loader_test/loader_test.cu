/**
 * @file loader_test.cu
 * @brief Edge-case tests for the whole scene loader (OBJ + glTF + JSON dispatch).
 *
 * Includes the REAL scene_loader.cpp (which carries the tinyobjloader and cgltf
 * implementations) and links utils/utilities.cu for buildTransformationMatrix,
 * so this exercises the production load path.  Because the .cpp is in this TU,
 * the internal `static` SceneLoader::makeTri / loadOBJ / loadGLTF are callable
 * directly, and SceneLoader::loadFromJSON covers the JSON dispatch.
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

// scene_loader.cpp calls stbi_load for TEXTURE images; this TU must supply
// the implementation (the main app gets it from src/stb.cpp).
//
// ORDER MATTERS: stb_image v2.06 keeps the implementation block OUTSIDE the
// include guard, so STB_IMAGE_IMPLEMENTATION must NOT be defined when
// scene_loader.cpp's own `#include <stb_image.h>` runs (that would compile
// the implementation a second time).  Include the loader first (declarations
// only), then define the implementation in a second include.
#include "scene/scene_loader.cpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
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

static glm::vec3 faceNormal(const Triangle& t)
{
    return glm::normalize(glm::cross(t.v1 - t.v0, t.v2 - t.v0));
}

// True when every vertex normal is unit and parallel to the geometric face
// normal (i.e. makeTri fell back to fn).  Not valid on degenerate triangles.
static bool normalsFollowFace(const Triangle& t, float eps = 1e-4f)
{
    glm::vec3 fn = faceNormal(t);
    for (const glm::vec3* n : { &t.n0, &t.n1, &t.n2 })
    {
        if (!isUnit(*n)) return false;
        if (std::fabs(glm::dot(*n, fn) - 1.0f) > eps) return false;
    }
    return true;
}

// ---- Loader helpers ---------------------------------------------------

static std::vector<Triangle> loadOBJTris(const std::string& exeDir,
                                         const std::string& asset)
{
    std::vector<Triangle> tris;
    SceneLoader::loadOBJ(exeDir + "/assets/" + asset, tris);
    return tris;
}

static std::vector<Triangle> loadGLTFTris(const std::string& exeDir,
                                          const std::string& asset)
{
    Scene scene;
    SceneLoader::loadGLTF(scene, exeDir + "/assets/" + asset);
    return scene.hostTriangles;
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
// Part A — makeTri unit checks
// =====================================================================
static void testMakeTri()
{
    std::printf("=== makeTri ===\n");

    Triangle t = SceneLoader::makeTri(
        glm::vec3(0, 0, 0), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0),
        glm::vec3(0, 0, 1), glm::vec3(0, 0, 1), glm::vec3(0, 0, 1));
    check(t.n0 == glm::vec3(0, 0, 1) && t.n1 == glm::vec3(0, 0, 1) &&
          t.n2 == glm::vec3(0, 0, 1),
          "valid unit normals pass through unchanged");

    Triangle t0 = SceneLoader::makeTri(
        glm::vec3(0, 0, 0), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0),
        glm::vec3(0, 0, 0), glm::vec3(0, 0, 0), glm::vec3(0, 0, 0));
    check(t0.n0 == glm::vec3(0, 0, 1) && t0.n1 == glm::vec3(0, 0, 1) &&
          t0.n2 == glm::vec3(0, 0, 1) && isUnit(t0.n0),
          "zero-length normals fall back to the face normal");

    glm::vec3 nanv = glm::vec3(std::numeric_limits<float>::quiet_NaN());
    Triangle tN = SceneLoader::makeTri(
        glm::vec3(0, 0, 0), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0),
        nanv, nanv, nanv);
    check(tN.n0 == glm::vec3(0, 0, 1) && isUnit(tN.n0),
          "NaN normals fall back to the face normal");

    Triangle tD = SceneLoader::makeTri(
        glm::vec3(0, 0, 0), glm::vec3(0, 0, 0), glm::vec3(0, 0, 0),
        glm::vec3(0, 0, 0), glm::vec3(0, 0, 0), glm::vec3(0, 0, 0));
    check(tD.n0 == glm::vec3(0, 1, 0) && tD.n1 == glm::vec3(0, 1, 0) &&
          tD.n2 == glm::vec3(0, 1, 0),
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
            check(tris[0].n0 == glm::vec3(0, 0, 1),
                  "tri_vn.obj -> vertex normals loaded from vn entries");
    }
    {
        auto tris = loadOBJTris(exeDir, "tri_uv.obj");
        check(tris.size() == 1, "tri_uv.obj -> 1 triangle");
        if (tris.size() == 1)
            check(tris[0].uv0 == glm::vec2(0, 0) && tris[0].uv1 == glm::vec2(1, 0) &&
                  tris[0].uv2 == glm::vec2(0, 1),
                  "tri_uv.obj -> per-corner UVs loaded from vt entries");
    }
    {
        // vt exists for some corners but not the face → per-corner guard.
        auto tris = loadOBJTris(exeDir, "tri_no_vn.obj");
        check(tris.size() == 1, "tri_no_vn.obj -> 1 triangle (no vt)");
        if (tris.size() == 1)
            check(tris[0].uv0 == glm::vec2(0, 0) && tris[0].uv1 == glm::vec2(0, 0) &&
                  tris[0].uv2 == glm::vec2(0, 0),
                  "tri_no_vn.obj -> missing vt falls back to (0,0)");
    }
    {
        auto tris = loadOBJTris(exeDir, "tri_no_vn.obj");
        check(tris.size() == 1, "tri_no_vn.obj -> 1 triangle");
        if (tris.size() == 1)
            check(normalsFollowFace(tris[0]),
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
            check(tris[0].n0 == glm::vec3(0, 1, 0),
                  "degenerate.obj -> fallback normal (0,1,0)");
    }
    {
        auto tris = loadOBJTris(exeDir, "oob.obj");
        check(tris.size() == 1, "oob.obj -> 1 triangle (tinyobj skips OOB quad)");
    }
    {
        std::vector<Triangle> tris;
        auto slice = SceneLoader::loadOBJ(exeDir + "/assets/missing.obj", tris);
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
            for (const auto& t : tris)
            {
                for (const glm::vec3* n : { &t.n0, &t.n1, &t.n2 })
                    if (!isUnit(*n)) ok = false;
                for (const glm::vec3* v : { &t.v0, &t.v1, &t.v2 })
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
            for (const auto& t : tris)
                for (const glm::vec3* n : { &t.n0, &t.n1, &t.n2 })
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
            for (const auto& t : tris)
                if (!normalsFollowFace(t)) ok = false;
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
            check(tris[0].n0 == glm::vec3(0, 1, 0) && tris[0].n1 == glm::vec3(0, 1, 0) &&
                  tris[0].n2 == glm::vec3(0, 1, 0),
                  "degenerate.gltf -> fallback normal (0,1,0)");
    }
    {
        auto tris = loadGLTFTris(exeDir, "zero_normal.gltf");
        check(tris.size() == 1, "zero_normal.gltf -> 1 triangle");
        if (tris.size() == 1)
            check(normalsFollowFace(tris[0]),
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
            check(tris[0].v0 == glm::vec3(5, 5, 5) && tris[0].v1 == glm::vec3(6, 5, 5) &&
                  tris[0].v2 == glm::vec3(5, 6, 5),
                  "node_transform.gltf -> node translation (5,5,5) applied to verts");
    }
    {
        auto tris = loadGLTFTris(exeDir, "norm_i8.gltf");
        check(tris.size() == 1, "norm_i8.gltf -> 1 triangle");
        if (tris.size() == 1)
        {
            const glm::vec3& v1 = tris[0].v1;
            const glm::vec3& v2 = tris[0].v2;
            bool ok = std::fabs(tris[0].v0.x) < 1e-3f && std::fabs(tris[0].v0.y) < 1e-3f &&
                      std::fabs(tris[0].v0.z) < 1e-3f &&
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
            const Triangle& t = tris[0];
            bool ok = t.uv0 == glm::vec2(0, 0) && t.uv1 == glm::vec2(1, 0) &&
                      t.uv2 == glm::vec2(0, 1);
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
        check(scene.hostTriangles.size() == 1, "tex_cube.gltf -> 1 triangle");
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
        if (scene.hostTriangles.size() == 1)
        {
            const TextureBinding& b = scene.hostTriangles[0].tex;
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
            check(tris[0].uv0 == glm::vec2(0, 0) && tris[0].uv1 == glm::vec2(0, 0) &&
                  tris[0].uv2 == glm::vec2(0, 0),
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
        check(scene.hostTriangles.size() == 12, "scene_baseline: 12 host triangles");
    }
    {
        Scene scene = SceneLoader::loadFromJSON(exeDir + "/scene_glb.json");
        check(scene.hostTriangles.size() == 12, "scene_glb: 12 host triangles");
    }
    {
        Scene scene = SceneLoader::loadFromJSON(exeDir + "/scene_quad_obj.json");
        check(scene.hostTriangles.size() == 2, "scene_quad_obj: 2 host triangles");
    }
    {
        Scene scene = SceneLoader::loadFromJSON(exeDir + "/scene_missing_mesh.json");
        check(scene.geoms.size() == 1, "scene_missing_mesh: geom still pushed");
        if (scene.geoms.size() == 1)
            check(scene.geoms[0].meshTriangleOffset == -1 &&
                  scene.geoms[0].meshTriangleCount == 0,
                  "scene_missing_mesh: {-1, 0} slice");
        check(scene.hostTriangles.empty(), "scene_missing_mesh: no host triangles");
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
        check(scene.hostTriangles.size() == 6, "scene_multi: 6 host triangles");
    }
    {
        // Procedural checkerboard material: TEXTURE = "checkerboard" must map
        // to the checkerboard sentinel, and UV_SCALE to uvScale (textureId
        // stays -2 → no image loaded → Scene::textures empty).
        Scene scene = SceneLoader::loadFromJSON(exeDir + "/scene_texture.json");
        check(scene.materials.size() == 1, "scene_texture: 1 material");
        if (scene.materials.size() == 1)
        {
            check(scene.materials[0].textureId == kCheckerboardTextureId,
                  "scene_texture: TEXTURE \"checkerboard\" → kCheckerboardTextureId");
            check(scene.materials[0].uvScale == 2.0f,
                  "scene_texture: UV_SCALE 2.0 → uvScale");
        }
        check(scene.textures.empty(),
              "scene_texture: no image texture loaded for checkerboard");
        check(scene.hostTriangles.size() == 12, "scene_texture: 12 host triangles");
    }
    {
        // JSON TEXTURE wins over the glTF baseColor map: the object's material
        // declares TEXTURE "checkerboard" while the mesh (tex_cube.gltf) carries
        // its own glTF baseColor slot.  parseObjects must zero the slice's
        // tex.baseColor so the shading fallback (tex.baseColor → m.textureId)
        // resolves to the JSON checkerboard, not the model's PNG.
        Scene scene = SceneLoader::loadFromJSON(exeDir + "/scene_texture_override.json");
        check(scene.hostTriangles.size() == 1,
              "scene_texture_override: 1 host triangle");
        if (scene.hostTriangles.size() == 1)
            check(scene.hostTriangles[0].tex.baseColor == -1,
                  "scene_texture_override: JSON TEXTURE zeroes glTF tex.baseColor");
        check(scene.materials.size() == 1 &&
              scene.materials[0].textureId == kCheckerboardTextureId,
              "scene_texture_override: material keeps checkerboard textureId");
    }

    std::printf("\n");
}

int main(int argc, char** argv)
{
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

    testMakeTri();
    testLoadOBJ(exeDir);
    testLoadGLTF(exeDir);
    testLoadFromJSON(exeDir);

    std::printf(failures == 0 ? "ALL PASS\n" : "FAILURES: %d\n", failures);
    return failures == 0 ? 0 : 1;
}
