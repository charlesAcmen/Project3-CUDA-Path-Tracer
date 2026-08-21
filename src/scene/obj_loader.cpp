// ====================================================================
// OBJ mesh loading (tinyobjloader) + the shared triangle constructor.
//
// This is the ONLY translation unit that defines TINYOBJLOADER_IMPLEMENTATION,
// so the tinyobjloader implementation lives here and never leaks into the
// other scene-loader TUs.
// ====================================================================

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "scene/loader_internal.h"

#include "constants.h"        // RAY_EPSILON (face-normal fallback in appendTriangle)
#include "utils/logger.h"

#include <cmath>              // isnan / sqrt (appendTriangle)
#include <filesystem>
#include <unordered_map>
#include <vector>

using namespace std;

namespace SceneLoader {

// -----------------------------------------------------------------------
// Shared triangle construction (OBJ + glTF)
// -----------------------------------------------------------------------

/**
 * Append one split triangle from three vertices and three vertex normals.
 *
 * Shared by loadOBJ and loadGLTF.  Any vertex normal that is NaN or has
 * (near-)zero length is replaced by the geometric face normal — the same
 * safe fallback the old OBJ path applied inline.
 *
 * @param v0,v1,v2  Vertex positions (object space)
 * @param n0,n1,n2  Vertex normals; NaN / zero-length entries fall back
 *                  to the face normal
 * @param u0,u1,u2  Corner UVs (texture space); default (0,0) when the
 *                  source file provides none
 */
void appendTriangle(
    vector<TrianglePos>& positions, vector<TriangleAttr>& attrs,
    const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
    const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2,
    const glm::vec2& u0, const glm::vec2& u1, const glm::vec2& u2,
    const glm::vec3& c0, const glm::vec3& c1, const glm::vec3& c2)
{
    // ---- Face normal calculation (safe fallback) ----
    glm::vec3 e1 = v1 - v0;
    glm::vec3 e2 = v2 - v0;
    glm::vec3 crossE = glm::cross(e1, e2);
    float cLen2 = glm::dot(crossE, crossE);
    glm::vec3 fn = (std::isnan(cLen2) || cLen2 < RAY_EPSILON)
        ? glm::vec3(0.0f, 1.0f, 0.0f)
        : crossE * (1.0f / std::sqrt(cLen2));

    auto validOr = [&](const glm::vec3& n) {
        float len2 = glm::dot(n, n);
        return (std::isnan(len2) || len2 < RAY_EPSILON) ? fn : n;
    };

    positions.push_back({ v0, v1, v2 });
    TriangleAttr attr;
    attr.n0 = validOr(n0); attr.n1 = validOr(n1); attr.n2 = validOr(n2);
    attr.uv0 = u0; attr.uv1 = u1; attr.uv2 = u2;
    attr.c0 = c0; attr.c1 = c1; attr.c2 = c2;
    attrs.push_back(attr);
}

// -----------------------------------------------------------------------
// OBJ Mesh Loading
// -----------------------------------------------------------------------

// Emit every triangular face of an OBJ into `triangles`.  Walks
// tinyobjloader's index array. Normals and UVs are indexed per-corner;
// colors are indexed per vertex.
static void appendObjGeometry(const tinyobj::attrib_t& attrib,
                              const vector<tinyobj::shape_t>& shapes,
                              bool hasNormals, bool hasUvs, bool hasColors,
                              vector<Triangle>& triangles, int& count)
{
    for (const auto& shape : shapes)
    {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
        {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3)
            {
                // Skip non-triangular faces — pre-triangulate your OBJ.
                index_offset += fv;
                continue;
            }

            tinyobj::index_t idx0 = shape.mesh.indices[index_offset + 0];
            tinyobj::index_t idx1 = shape.mesh.indices[index_offset + 1];
            tinyobj::index_t idx2 = shape.mesh.indices[index_offset + 2];

            glm::vec3 v0(
                attrib.vertices[3 * (size_t)idx0.vertex_index + 0],
                attrib.vertices[3 * (size_t)idx0.vertex_index + 1],
                attrib.vertices[3 * (size_t)idx0.vertex_index + 2]);
            glm::vec3 v1(
                attrib.vertices[3 * (size_t)idx1.vertex_index + 0],
                attrib.vertices[3 * (size_t)idx1.vertex_index + 1],
                attrib.vertices[3 * (size_t)idx1.vertex_index + 2]);
            glm::vec3 v2(
                attrib.vertices[3 * (size_t)idx2.vertex_index + 0],
                attrib.vertices[3 * (size_t)idx2.vertex_index + 1],
                attrib.vertices[3 * (size_t)idx2.vertex_index + 2]);

            // ---- Vertex normals ----
            glm::vec3 n0, n1, n2;
            if (hasNormals &&
                idx0.normal_index >= 0 && (size_t)(3 * idx0.normal_index + 2) < attrib.normals.size() &&
                idx1.normal_index >= 0 && (size_t)(3 * idx1.normal_index + 2) < attrib.normals.size() &&
                idx2.normal_index >= 0 && (size_t)(3 * idx2.normal_index + 2) < attrib.normals.size())
            {
                // Load vertex normals from OBJ vn entries.
                n0 = glm::vec3(
                    attrib.normals[3 * (size_t)idx0.normal_index + 0],
                    attrib.normals[3 * (size_t)idx0.normal_index + 1],
                    attrib.normals[3 * (size_t)idx0.normal_index + 2]);
                n1 = glm::vec3(
                    attrib.normals[3 * (size_t)idx1.normal_index + 0],
                    attrib.normals[3 * (size_t)idx1.normal_index + 1],
                    attrib.normals[3 * (size_t)idx1.normal_index + 2]);
                n2 = glm::vec3(
                    attrib.normals[3 * (size_t)idx2.normal_index + 0],
                    attrib.normals[3 * (size_t)idx2.normal_index + 1],
                    attrib.normals[3 * (size_t)idx2.normal_index + 2]);
            }
            else
            {
                // No vertex normals in OBJ → appendTriangle falls back to the face normal.
                n0 = n1 = n2 = glm::vec3(0.0f);
            }

            // ---- Texture coordinates (vt entries) ----
            // OBJ UVs are PER-CORNER (indexed by the same face indices), so
            // each corner is guarded individually — a face may mix corners
            // with and without vt.  Missing UVs fall back to (0,0).
            glm::vec2 uv0(0.0f), uv1(0.0f), uv2(0.0f);
            if (hasUvs &&
                idx0.texcoord_index >= 0 && (size_t)(2 * idx0.texcoord_index + 1) < attrib.texcoords.size() &&
                idx1.texcoord_index >= 0 && (size_t)(2 * idx1.texcoord_index + 1) < attrib.texcoords.size() &&
                idx2.texcoord_index >= 0 && (size_t)(2 * idx2.texcoord_index + 1) < attrib.texcoords.size())
            {
                uv0 = glm::vec2(attrib.texcoords[2 * (size_t)idx0.texcoord_index + 0],
                                attrib.texcoords[2 * (size_t)idx0.texcoord_index + 1]);
                uv1 = glm::vec2(attrib.texcoords[2 * (size_t)idx1.texcoord_index + 0],
                                attrib.texcoords[2 * (size_t)idx1.texcoord_index + 1]);
                uv2 = glm::vec2(attrib.texcoords[2 * (size_t)idx2.texcoord_index + 0],
                                attrib.texcoords[2 * (size_t)idx2.texcoord_index + 1]);
            }

            // ---- Vertex colors (extension: `v x y z r g b`) ----
            // tinyobjloader stores RGB alongside positions, so colors use the
            // vertex index rather than OBJ's per-corner normal/UV indices.
            // Missing or malformed color data stays white, which preserves
            // the renderer's "no vertex-color modulation" default.
            glm::vec3 c0(1.0f), c1(1.0f), c2(1.0f);
            auto loadColor = [&](int vertexIndex, glm::vec3& color) {
                if (hasColors && vertexIndex >= 0 &&
                    (size_t)(3 * vertexIndex + 2) < attrib.colors.size())
                {
                    color = glm::vec3(
                        attrib.colors[3 * (size_t)vertexIndex + 0],
                        attrib.colors[3 * (size_t)vertexIndex + 1],
                        attrib.colors[3 * (size_t)vertexIndex + 2]);
                }
            };
            loadColor(idx0.vertex_index, c0);
            loadColor(idx1.vertex_index, c1);
            loadColor(idx2.vertex_index, c2);

            appendTriangle(positions, attrs, v0, v1, v2, n0, n1, n2,
                           uv0, uv1, uv2, c0, c1, c2);
            count++;
            index_offset += fv;
        }
    }
}

// Stamp the companion .mtl's per-material texture slots (map_Kd / map_Bump /
// map_Ke) onto the triangles this mesh pushed, resolved by material_id.  The
// .mtl's flat colors (Kd/Ks/Ns) are intentionally NOT applied — the scene
// JSON's MATERIAL governs the shading model, same as the glTF path.
static void stampMtlTextures(const string& objPath,
                             const vector<tinyobj::material_t>& materials,
                             const vector<tinyobj::shape_t>& shapes,
                             size_t offset, Scene& scene,
                             vector<TriangleAttr>& attrs)
{
    unordered_map<string, int> texCache;   // dedup by resolved texture path
    const filesystem::path objDir = filesystem::path(objPath).parent_path();
    auto resolveMtlTex = [&](const string& texname, bool srgb) -> int
    {
        if (texname.empty()) return -1;
        filesystem::path texPath(texname);
        if (!texPath.is_absolute()) texPath = objDir / texPath;
        const string key = texPath.lexically_normal().generic_string();
        const auto it = texCache.find(key);
        if (it != texCache.end()) return it->second;
        const int id = loadTextureFile(scene, key, srgb);
        texCache.emplace(key, id);
        if (id >= 0)
            Log::info("Scene", "Auto-loaded MTL texture '%s'", key.c_str());
        return id;
    };

    vector<SurfaceBinding> matBindings(materials.size());
    for (size_t mi = 0; mi < materials.size(); ++mi)
    {
        const auto& m = materials[mi];
        matBindings[mi].baseColor = resolveMtlTex(m.diffuse_texname, true);
        matBindings[mi].normal    = resolveMtlTex(m.bump_texname, false);
        matBindings[mi].emissive  = resolveMtlTex(m.emissive_texname, true);
        // MTL Ke scales the emissive texture — the OBJ mirror of glTF
        // emissiveFactor.  An all-zero Ke (tinyobj default) with a bound
        // map_Ke means "texture as-is", the same viewer convention as glTF.
        if (matBindings[mi].emissive >= 0)
        {
            glm::vec3 ke(m.emission[0], m.emission[1], m.emission[2]);
            matBindings[mi].emissiveFactor = (ke != glm::vec3(0.0f)) ? ke
                                                                      : glm::vec3(1.0f);
        }
    }

    vector<int> surfaceBindingIds(materials.size(), -1);
    for (size_t mi = 0; mi < matBindings.size(); ++mi)
        surfaceBindingIds[mi] = internSurfaceBinding(scene, matBindings[mi]);

    // Link each face to its shared surface binding (same face order as pushed,
    // so triIndex tracks the mesh's slice starting at `offset`).
    size_t triIndex = offset;
    for (const auto& shape : shapes)
    {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f)
        {
            const int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3) { index_offset += fv; continue; }
            const int matId = shape.mesh.material_ids[f];
            if (matId >= 0 && (size_t)matId < matBindings.size() &&
                triIndex < triangles.size())
                triangles[triIndex].tex = matBindings[matId];
            ++triIndex;
            index_offset += fv;
        }
    }
}

/**
 * Load triangles from a Wavefront OBJ file and append them to the
 * hostTriangles vector.
 *
 * @param objPath   Path to the .obj file on disk
 * @param triangles [out] Flat array of object-space triangles to append to
 * @param scene     Optional texture sink.  When non-null, the companion .mtl's
 *                  image maps (map_Kd / map_Bump / map_Ke) are resolved into
 *                  per-material SurfaceBindings linked to each face by
 *                  material_id.  Null (the loader_test's 2-arg
 *                  calls) skips MTL texture loading — geometry only.
 * @return (offset, count) — the slice of `triangles` this mesh occupies
 */
pair<int, int> loadOBJ(const string& objPath,
                       vector<Triangle>& triangles,
                       Scene* scene)
{
    tinyobj::attrib_t attrib;
    vector<tinyobj::shape_t> shapes;
    vector<tinyobj::material_t> materials;
    string warn, err;

    // Pass the OBJ's directory as the MTL base dir — tinyobjloader's default
    // MaterialFileReader otherwise resolves `mtllib box.mtl` against the
    // process CWD and fails to find the .mtl next to the mesh.
    const string objDir = filesystem::path(objPath).parent_path().string();
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                          objPath.c_str(), objDir.c_str()))
    {
        Log::error("Scene", "Failed to load: %s", objPath.c_str());
        if (!warn.empty()) Log::warn("Scene", "%s", warn.c_str());
        if (!err.empty())  Log::error("Scene", "%s", err.c_str());
        return {-1, 0};
    }
    if (!warn.empty()) Log::warn("Scene", "%s", warn.c_str());
    if (!err.empty())  Log::warn("Scene", "%s", err.c_str());

    const int offset = (int)triangles.size();
    int       count  = 0;

    // Triangles from the face loop, then — when a Scene sink is present —
    // the companion .mtl's texture slots stamped by material_id.  A 2-arg call
    // (loader_test's geometry assertions) keeps scene == nullptr and skips MTL
    // texture loading entirely.
    appendObjGeometry(attrib, shapes, !attrib.normals.empty(),
                      !attrib.texcoords.empty(), !attrib.colors.empty(),
                      triangles, count);
    if (scene != nullptr && !materials.empty())
        stampMtlTextures(objPath, materials, shapes, (size_t)offset,
                         *scene, triangles);

    Log::info("Scene", "Loaded mesh: %s  (%d triangles, total %zu)",
              objPath.c_str(), count, triangles.size());
    return {offset, count};
}

} // namespace SceneLoader
