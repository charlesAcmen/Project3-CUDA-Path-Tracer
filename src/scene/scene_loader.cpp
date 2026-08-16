// Define the implementation of tinyobjloader in this translation unit
// so it doesn't leak into other compilation units.
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

// Define the implementation of cgltf (glTF parser) in this TU.
// Header vendored at external/include/cgltf.h (jkuhlmann/cgltf, MIT).
#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#include "scene/scene_loader.h"

#include "constants.h"
#include "utils/logger.h"
#include "utils/utilities.h"

#include <stb_image.h>   // stbi_load for PNG/JPG texture files

#include <glm/gtc/matrix_inverse.hpp>   // glm::inverse / glm::inverseTranspose

#include <json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

namespace SceneLoader {

// -----------------------------------------------------------------------
// Shared triangle construction (OBJ + glTF)
// -----------------------------------------------------------------------

/**
 * Build a Triangle from three vertices and three vertex normals.
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
static Triangle makeTri(
    const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
    const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2,
    const glm::vec2& u0 = glm::vec2(0.0f),
    const glm::vec2& u1 = glm::vec2(0.0f),
    const glm::vec2& u2 = glm::vec2(0.0f))
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

    return Triangle{ v0, v1, v2, validOr(n0), validOr(n1), validOr(n2),
                     u0, u1, u2 };
}

// -----------------------------------------------------------------------
// OBJ Mesh Loading
// -----------------------------------------------------------------------

/**
 * Load triangles from a Wavefront OBJ file and append them to the
 * hostTriangles vector.
 *
 * @param objPath   Path to the .obj file on disk
 * @param triangles [out] Flat array of object-space triangles to append to
 * @return (offset, count) — the slice of `triangles` this mesh occupies
 */
static pair<int, int> loadOBJ(const string& objPath,
                              vector<Triangle>& triangles)
{
    tinyobj::attrib_t attrib;
    vector<tinyobj::shape_t> shapes;
    vector<tinyobj::material_t> materials;
    string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                          objPath.c_str()))
    {
        Log::error("Scene", "Failed to load: %s", objPath.c_str());
        if (!warn.empty()) Log::warn("Scene", "%s", warn.c_str());
        if (!err.empty())  Log::error("Scene", "%s", err.c_str());
        return {-1, 0};
    }
    if (!warn.empty()) Log::warn("Scene", "%s", warn.c_str());
    if (!err.empty())  Log::warn("Scene", "%s", err.c_str());

    int offset = (int)triangles.size();
    int count  = 0;

    // Determine if the OBJ provides vertex normals (vn entries) and UVs
    // (vt entries).
    const bool hasNormals = (!attrib.normals.empty());
    const bool hasUvs     = (!attrib.texcoords.empty());

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
                // No vertex normals in OBJ → makeTri falls back to the face normal.
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

            triangles.push_back(makeTri(v0, v1, v2, n0, n1, n2, uv0, uv1, uv2));
            count++;
            index_offset += fv;
        }
    }

    Log::info("Scene", "Loaded mesh: %s  (%d triangles, total %zu)",
              objPath.c_str(), count, triangles.size());
    return {offset, count};
}

// ---- Texture auto-load (defined here so the glTF walk can stamp
// per-triangle slots; loadTextureFile is forward-declared and defined in the
// Texture Loading section) -------------------------------------------------

static int loadTextureFile(Scene& scene, const string& path, bool srgb = true);

// Loader context threaded through the glTF walk: the scene (texture sink),
// the cgltf document (image-index dedup), and the .gltf's directory (URI
// resolution — glTF image URIs are relative to the .gltf file, exactly like
// mesh paths are relative to the scene JSON).
struct GltfLoadCtx
{
    Scene&           scene;
    cgltf_data*      data;
    filesystem::path dir;            // parent dir of the .gltf / .glb
    std::vector<int> imageToTexId;   // data->images index → Scene::textures index
};

// Resolve one glTF material slot (a cgltf_texture_view) to a global
// Scene::textures index, loading the image on first use.
//
// - Dedup by cgltf image INDEX: a texture shared across primitives or across
//   material slots loads once.  (Consequence: the FIRST slot to reference an
//   image sets its sRGB treatment; no real file shares one image between a
//   color and a data role.)
// - External PNG/JPG file URIs only.  Images embedded in the .glb buffer
//   (buffer_view) or as data: URIs are skipped with a warning — shading falls
//   back to the material's own value.  Deferred, not supported yet.
// - URIs are percent-decoded (glTF allows %20, unicode, …) before use.
//
// @return Scene::textures index (>= 0), or -1 (empty slot / unsupported
//         image source / load failure).
static int resolveGltfTextureSlot(GltfLoadCtx& ctx,
                                  const cgltf_texture_view& view,
                                  bool srgb)
{
    if (view.texture == nullptr || view.texture->image == nullptr)
        return -1;

    cgltf_image* image = view.texture->image;
    const int imageIdx = (int)(image - ctx.data->images);

    if (ctx.imageToTexId[imageIdx] >= 0)
        return ctx.imageToTexId[imageIdx];   // already loaded

    const bool embedded = (image->uri == nullptr || image->buffer_view != nullptr);
    const bool dataUri  = (!embedded && string(image->uri).rfind("data:", 0) == 0);
    if (embedded || dataUri)
    {
        Log::warn("Scene",
                  "glTF image #%d ('%s'): embedded (bufferView) and data: "
                  "textures are not supported yet; using material color",
                  imageIdx, image->name ? image->name : "unnamed");
        ctx.imageToTexId[imageIdx] = -1;   // remember — warn once per image
        return -1;
    }

    // Percent-decode into a copy (cgltf_decode_uri mutates in place, and the
    // uri string is owned by the cgltf document).
    string uri = image->uri;
    vector<char> decoded(uri.begin(), uri.end());
    decoded.push_back('\0');
    cgltf_decode_uri(decoded.data());
    uri.assign(decoded.data());

    const int id = loadTextureFile(ctx.scene,
                                   (ctx.dir / uri).generic_string(), srgb);
    if (id >= 0)
        Log::info("Scene", "Auto-loaded glTF texture '%s' (image #%d)",
                  uri.c_str(), imageIdx);
    ctx.imageToTexId[imageIdx] = id;
    return id;
}

// Resolve a glTF material's five texture slots into a TextureBinding.
// baseColor / emissive are color maps (sRGB → linearized on load); normal /
// metallicRoughness / occlusion are data maps (raw bytes kept).
static TextureBinding bindGltfMaterial(GltfLoadCtx& ctx,
                                       const cgltf_material* mat)
{
    TextureBinding b;
    if (mat == nullptr)
        return b;
    b.baseColor         = resolveGltfTextureSlot(ctx,
        mat->pbr_metallic_roughness.base_color_texture, true);
    b.normal            = resolveGltfTextureSlot(ctx, mat->normal_texture, false);
    b.metallicRoughness = resolveGltfTextureSlot(ctx,
        mat->pbr_metallic_roughness.metallic_roughness_texture, false);
    b.occlusion         = resolveGltfTextureSlot(ctx, mat->occlusion_texture, false);
    b.emissive          = resolveGltfTextureSlot(ctx, mat->emissive_texture, true);
    // glTF's own roughness default (cgltf fills the spec default 1.0 when the
    // file omits it) — a fallback for Specular materials whose JSON did not
    // declare ROUGHNESS.  Non-glTF meshes leave the -1 sentinel.
    b.roughnessFactor   = mat->pbr_metallic_roughness.roughness_factor;
    return b;
}

// -----------------------------------------------------------------------
// glTF 2.0 Mesh Loading
// -----------------------------------------------------------------------

// Local transform of a glTF node.  glTF 2.0 defines it either as an
// explicit column-major 4x4 `matrix`, or as TRS: the composition T·R·S
// (translation, then rotation, then scale) applied to the node's mesh and
// children.  GLM reads column-major arrays directly, so `glm::mat4(m)` is
// the exact glTF matrix.
static glm::mat4 nodeLocalMatrix(const cgltf_node* n)
{
    if (n->has_matrix)
        return glm::make_mat4(n->matrix);

    // glTF 2.0 TRS: local = T·R·S.  The standard GLM idiom composes them in
    // the order translate → rotate → scale, each helper RIGHT-multiplying
    // (translate(m,v) = m·T, rotate/scale likewise), which yields exactly
    // T·R·S — translation applied LAST to the vertex (scale, then rotate,
    // then move into place).
    glm::mat4 m(1.0f);
    if (n->has_translation)
        m = glm::translate(m, glm::vec3(n->translation[0], n->translation[1],
                                        n->translation[2]));
    if (n->has_rotation)
        // cgltf stores the quaternion as (x, y, z, w); GLM's constructor is (w, x, y, z).
        m = m * glm::mat4_cast(glm::quat(n->rotation[3], n->rotation[0],
                                         n->rotation[1], n->rotation[2]));
    if (n->has_scale)
        m = glm::scale(m, glm::vec3(n->scale[0], n->scale[1], n->scale[2]));
    return m;   // m = T·R·S
}

// Emit one primitive's triangles, transformed into the node's accumulated
// frame.  Vertices via `world`; normals via the inverse-transpose — left
// UNnormalized, so the world-space bake + hit-time normalize compose with
// it by linearity, exactly as with the per-geom transform.
static void appendPrimitiveTriangles(const cgltf_primitive* prim,
                                     const glm::mat4& world,
                                     const glm::mat4& worldIT,
                                     vector<Triangle>& triangles, int& count,
                                     GltfLoadCtx& ctx)
{
    if (prim->type != cgltf_primitive_type_triangles)
    {
        Log::warn("Scene", "glTF primitive is not triangles (mode %d); skipping",
                  (int)prim->type);
        return;
    }

    // ---- Locate POSITION / NORMAL / TEXCOORD_0 accessors ----
    const cgltf_accessor* posAcc = nullptr;
    const cgltf_accessor* nrmAcc = nullptr;
    const cgltf_accessor* uvAcc  = nullptr;
    for (cgltf_size ai = 0; ai < prim->attributes_count; ++ai)
    {
        const cgltf_attribute* attr = &prim->attributes[ai];
        if (attr->type == cgltf_attribute_type_position)
            posAcc = attr->data;
        else if (attr->type == cgltf_attribute_type_normal)
            nrmAcc = attr->data;
        else if (attr->type == cgltf_attribute_type_texcoord && attr->index == 0)
            uvAcc = attr->data;   // TEXCOORD_0 (we ignore higher sets)
    }
    if (posAcc == nullptr)
    {
        Log::warn("Scene", "glTF primitive has no POSITION accessor; skipping");
        return;
    }

    // Per-triangle texture slots from the primitive's glTF material.  All five
    // roles (baseColor/normal/ORM/occlusion/emissive) resolve into the global
    // texture table and are stamped on every triangle; only baseColor is
    // sampled by the current shading — the rest are data for future features.
    const TextureBinding binding = bindGltfMaterial(ctx, prim->material);

    const cgltf_size vertCount = posAcc->count;

    // Normals are indexed by the same vertex indices as positions, so a
    // NORMAL accessor must cover every vertex — otherwise norm(i) below
    // reads nrm out of bounds.  cgltf_validate() also requires all
    // attributes to share a count, so this is defense-in-depth.
    if (nrmAcc != nullptr && nrmAcc->count != vertCount)
    {
        Log::warn("Scene",
                  "glTF primitive NORMAL count (%zu) != POSITION count "
                  "(%zu); skipping",
                  (size_t)nrmAcc->count, (size_t)vertCount);
        return;
    }

    // Unpack all vertex positions.  read_float handles integer and
    // normalized component types, so a VEC3 always yields 3 floats.
    vector<float> pos(3 * (size_t)vertCount);
    for (cgltf_size i = 0; i < vertCount; ++i)
        cgltf_accessor_read_float(posAcc, i, &pos[3 * (size_t)i], 3);

    // Normals (optional; makeTri falls back to the face normal).
    vector<float> nrm;
    if (nrmAcc != nullptr)
    {
        nrm.resize(3 * (size_t)vertCount);
        for (cgltf_size i = 0; i < vertCount; ++i)
            cgltf_accessor_read_float(nrmAcc, i, &nrm[3 * (size_t)i], 3);
    }

    // UVs (optional; default (0,0)).  glTF UVs are PER-VERTEX — shared by
    // every face that references the vertex — so unlike OBJ there is no
    // per-corner guard; a present TEXCOORD_0 accessor covers all vertices.
    vector<float> uv;
    if (uvAcc != nullptr)
    {
        if (uvAcc->count != vertCount)
        {
            Log::warn("Scene",
                      "glTF primitive TEXCOORD_0 count (%zu) != POSITION "
                      "count (%zu); ignoring UVs",
                      (size_t)uvAcc->count, (size_t)vertCount);
            uvAcc = nullptr;
        }
        else
        {
            uv.resize(2 * (size_t)vertCount);
            for (cgltf_size i = 0; i < vertCount; ++i)
                cgltf_accessor_read_float(uvAcc, i, &uv[2 * (size_t)i], 2);
        }
    }

    auto vert = [&](cgltf_size i) -> glm::vec3 {
        return glm::vec3(pos[3 * (size_t)i + 0],
                         pos[3 * (size_t)i + 1],
                         pos[3 * (size_t)i + 2]);
    };
    auto norm = [&](cgltf_size i) -> glm::vec3 {
        return glm::vec3(nrm[3 * (size_t)i + 0],
                         nrm[3 * (size_t)i + 1],
                         nrm[3 * (size_t)i + 2]);
    };
    // UVs are texture-space — NOT transformed by the node matrix.
    auto uvAt = [&](cgltf_size i) -> glm::vec2 {
        if (uvAcc == nullptr) return glm::vec2(0.0f);
        return glm::vec2(uv[2 * (size_t)i + 0],
                         uv[2 * (size_t)i + 1]);
    };

    // Node-transform the local vertex / normal into the scene frame.
    auto vTrans = [&](cgltf_size i) -> glm::vec3 {
        return glm::vec3(world * glm::vec4(vert(i), 1.0f));
    };
    auto nTrans = [&](cgltf_size i) -> glm::vec3 {
        return (nrmAcc != nullptr)
            ? glm::vec3(worldIT * glm::vec4(norm(i), 0.0f))
            : glm::vec3(0.0f);   // no vertex normal → makeTri's face-normal fallback
    };

    // ---- Emit triangles ----
    if (prim->indices != nullptr)
    {
        const cgltf_accessor* idxAcc = prim->indices;

        // Index count must be a whole number of triangles.
        if (idxAcc->count % 3 != 0)
        {
            Log::warn("Scene",
                      "glTF indexed primitive has %zu indices "
                      "(not a multiple of 3); skipping",
                      (size_t)idxAcc->count);
            return;
        }

        const cgltf_size nTri = idxAcc->count / 3;
        for (cgltf_size t = 0; t < nTri; ++t)
        {
            cgltf_size i0 = cgltf_accessor_read_index(idxAcc, 3 * t + 0);
            cgltf_size i1 = cgltf_accessor_read_index(idxAcc, 3 * t + 1);
            cgltf_size i2 = cgltf_accessor_read_index(idxAcc, 3 * t + 2);

            // cgltf_validate() also cross-checks index bounds when buffer
            // data is loaded, so this guard is defense-in-depth.
            if (i0 >= vertCount || i1 >= vertCount || i2 >= vertCount)
            {
                Log::warn("Scene",
                          "glTF indexed primitive has out-of-range "
                          "vertex index (%zu/%zu/%zu of %zu); "
                          "skipping triangle",
                          (size_t)i0, (size_t)i1, (size_t)i2,
                          (size_t)vertCount);
                continue;
            }

            triangles.push_back(makeTri(vTrans(i0), vTrans(i1), vTrans(i2),
                                        nTrans(i0), nTrans(i1), nTrans(i2),
                                        uvAt(i0), uvAt(i1), uvAt(i2)));
            triangles.back().tex = binding;   // glTF texture slots
            ++count;
        }
    }
    else
    {
        // Non-indexed primitive: vertices are already in triangle order.
        if (vertCount % 3 != 0)
        {
            Log::warn("Scene", "glTF non-indexed primitive has %zu vertices (not a multiple of 3); skipping", (size_t)vertCount);
            return;
        }
        for (cgltf_size t = 0; t < vertCount; t += 3)
        {
            triangles.push_back(makeTri(vTrans(t + 0), vTrans(t + 1), vTrans(t + 2),
                                        nTrans(t + 0), nTrans(t + 1), nTrans(t + 2),
                                        uvAt(t + 0), uvAt(t + 1), uvAt(t + 2)));
            triangles.back().tex = binding;   // glTF texture slots
            ++count;
        }
    }
}

// Emit all of a mesh's primitives under one accumulated transform.
static void appendMeshTriangles(const cgltf_mesh* mesh, const glm::mat4& world,
                                vector<Triangle>& triangles, int& count,
                                GltfLoadCtx& ctx)
{
    const glm::mat4 worldIT = glm::inverseTranspose(world);
    for (cgltf_size pi = 0; pi < mesh->primitives_count; ++pi)
        appendPrimitiveTriangles(&mesh->primitives[pi], world, worldIT,
                                 triangles, count, ctx);
}

// Depth-first walk of the scene graph.  The accumulated matrix `parentWorld`
// is the composition of every ancestor's local transform; multiplying it by
// this node's local transform gives the world matrix that places its mesh.
static void walkNode(const cgltf_node* node, const glm::mat4& parentWorld,
                     vector<Triangle>& triangles, int& count, GltfLoadCtx& ctx)
{
    const glm::mat4 world = parentWorld * nodeLocalMatrix(node);
    if (node->mesh != nullptr)
        appendMeshTriangles(node->mesh, world, triangles, count, ctx);
    for (cgltf_size c = 0; c < node->children_count; ++c)
        walkNode(node->children[c], world, triangles, count, ctx);
}

/**
 * Load triangles from a glTF 2.0 file (.gltf JSON or .glb binary) and
 * append them to the scene's hostTriangles vector.
 *
 * - The scene graph is walked and every node's accumulated transform is
 *   applied, so multi-part models scattered across nodes are assembled
 *   exactly as the file specifies (each (node, mesh) instance is emitted
 *   once, with its own transform).
 * - Only triangle primitives (mode 4) are loaded; point/line primitives
 *   are skipped with a warning.
 * - Each primitive's material texture slots (baseColor / normal /
 *   metallicRoughness / occlusion / emissive) are auto-loaded into
 *   Scene::textures and stamped on the triangles as a TextureBinding.
 *   Only external PNG/JPG file URIs load; .glb-embedded (bufferView) and
 *   data: images warn + fall back to the material color.  The scene JSON's
 *   MATERIAL still governs the shading model (an explicit JSON TEXTURE
 *   overrides the glTF baseColor — see parseObjects).
 * - Draco-compressed primitives are not supported (cgltf does not decode).
 *
 * @param scene     Scene to append to (hostTriangles AND textures)
 * @param gltfPath  Path to the .gltf or .glb file
 * @return (offset, count) — the slice of scene.hostTriangles this file occupies
 */
static pair<int, int> loadGLTF(Scene& scene, const string& gltfPath)
{
    cgltf_options options{};
    cgltf_data*   data = nullptr;

    cgltf_result result = cgltf_parse_file(&options, gltfPath.c_str(), &data);
    if (result != cgltf_result_success)
    {
        Log::error("Scene", "Failed to parse glTF: %s", gltfPath.c_str());
        return {-1, 0};
    }

    result = cgltf_load_buffers(&options, data, gltfPath.c_str());
    if (result != cgltf_result_success)
    {
        Log::error("Scene", "Failed to load glTF buffers: %s", gltfPath.c_str());
        cgltf_free(data);
        return {-1, 0};
    }

    result = cgltf_validate(data);
    if (result != cgltf_result_success)
    {
        Log::error("Scene", "glTF validation failed: %s", gltfPath.c_str());
        cgltf_free(data);
        return {-1, 0};
    }

    vector<Triangle>& triangles = scene.hostTriangles;
    const int offset = (int)triangles.size();
    int       count  = 0;

    // Texture auto-load context: glTF image URIs resolve relative to the
    // .gltf's directory; images dedup by cgltf image index (a file shared by
    // several materials / primitives loads once).
    GltfLoadCtx ctx = {
        scene, data,
        filesystem::path(gltfPath).parent_path(),
        vector<int>(data->images_count, -1)
    };

    // ---- Walk the scene graph, emitting one transformed instance per
    // (node, mesh).  Parts scattered across nodes are assembled by the
    // accumulated node transforms — the glTF-spec behavior.
    const cgltf_scene* gltfScene = data->scene;   // default scene
    if (gltfScene != nullptr && gltfScene->nodes_count > 0)
    {
        for (cgltf_size i = 0; i < gltfScene->nodes_count; ++i)
            walkNode(gltfScene->nodes[i], glm::mat4(1.0f), triangles, count, ctx);
    }
    else
    {
        // No scene graph (some minimal files): emit every mesh untransformed.
        for (cgltf_size mi = 0; mi < data->meshes_count; ++mi)
            appendMeshTriangles(&data->meshes[mi], glm::mat4(1.0f), triangles, count, ctx);
    }

    cgltf_free(data);

    Log::info("Scene", "Loaded mesh: %s  (%d triangles, total %zu)",
              gltfPath.c_str(), count, triangles.size());
    return {offset, count};
}

// -----------------------------------------------------------------------
// Texture Loading
// -----------------------------------------------------------------------

/**
 * Load an image file (PNG/JPG via stb_image) into Scene::textures.
 *
 * The shading pipeline works in linear space, and color PNG/JPG texels are
 * sRGB, so linearization happens here once, at load time — the GPU sampler
 * then returns linear colors that feed the accumulation buffer directly.
 * glTF also has DATA maps (normal / metallic-roughness / occlusion) whose
 * bytes are already linear; those pass through untouched (srgb=false).
 *
 * @param scene  Scene to append the texture to
 * @param path   Absolute path to the image file
 * @param srgb   True (default): linearize sRGB texels on load (color maps).
 *               False: keep raw byte values (normal/ORM/occlusion maps).
 * @return       Index into Scene::textures (>= 0), or -1 on failure
 */
static int loadTextureFile(Scene& scene, const string& path, bool srgb)
{
    int w = 0, h = 0, comp = 0;
    // req_comp = 3 forces RGB (3 channels), so texels are always vec3.
    stbi_uc* data = stbi_load(path.c_str(), &w, &h, &comp, 3);
    if (data == nullptr)
    {
        Log::error("Scene", "Failed to load texture image: %s", path.c_str());
        return -1;
    }

    TextureData td;
    td.width  = w;
    td.height = h;
    td.pixels.resize((size_t)w * (size_t)h);

    // sRGB → linear: c <= 0.04045 ? c/12.92 : ((c+0.055)/1.055)^2.4
    const auto lin = [](float c) {
        return (c <= 0.04045f) ? c / 12.92f
                               : std::pow((c + 0.055f) / 1.055f, 2.4f);
    };
    for (int i = 0; i < w * h; i++)
    {
        const float r = data[3 * i + 0] / 255.0f;
        const float g = data[3 * i + 1] / 255.0f;
        const float b = data[3 * i + 2] / 255.0f;
        td.pixels[i] = srgb
            ? glm::vec3(lin(r), lin(g), lin(b))
            : glm::vec3(r, g, b);
    }
    stbi_image_free(data);

    const int id = (int)scene.textures.size();
    scene.textures.push_back(std::move(td));
    return id;
}

// -----------------------------------------------------------------------
// JSON Section Parsing
// -----------------------------------------------------------------------

/**
 * Parse the "Materials" section into Scene::materials.
 *
 * @param data         Parsed scene JSON
 * @param scene        Scene to append materials to (also the texture sink)
 * @param jsonDir      Directory of the scene JSON — relative TEXTURE paths
 *                     resolve against it (same as mesh FILE paths)
 * @param MatNameToID  [out] Map from material name -> index into
 *                     Scene::materials; parseObjects resolves each object's
 *                     MATERIAL field through it.
 */
static void parseMaterials(
    const json& data, Scene& scene,
    const filesystem::path& jsonDir,
    unordered_map<string, uint32_t>& MatNameToID)
{
    // ---- Materials ----------------------------------------------------
    // Dedup texture files by resolved path: two materials referencing the
    // same image share one TextureData (and one device slice).
    unordered_map<string, int> textureCache;

    const auto& materialsData = data["Materials"];
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        newMaterial.indexOfRefraction = 1.0f;
        newMaterial.invIndexOfRefraction = 1.0f;
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = MaterialType::Diffuse;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = MaterialType::Emissive;
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            // Specular tint: read SPECULAR_COLOR if present, fall back to RGB.
            // RGB alone serves both diffuse albedo and specular tint, but the
            // two can diverge for physically accurate metals (RGB ≈ black,
            // SPECULAR_COLOR = reflectivity tint per wavelength).
            if (p.contains("SPECULAR_COLOR"))
            {
                const auto& sc = p["SPECULAR_COLOR"];
                newMaterial.specular.color = glm::vec3(sc[0], sc[1], sc[2]);
            }
            else
            {
                newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
            }
            newMaterial.type = MaterialType::Reflective;
            // Store only the raw roughness; the ROUGHNESS_THRESHOLD / exponent
            // conversion (2/r² − 2) is done per-hit by resolveGlossyExponent,
            // because the source chain also feeds per-texel ORM and glTF factor
            // values — a precomputed exponent here would be redundant.
            if (p.contains("ROUGHNESS"))
                newMaterial.specular.roughness =
                    glm::clamp((float)p["ROUGHNESS"], 0.0f, 1.0f);
            // else: -1 (unspecified) → the shader falls through to the mesh's
            // glTF roughnessFactor, then a fixed default.
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = MaterialType::Refractive;
            newMaterial.indexOfRefraction = p.value("IOR", 1.5f);
            newMaterial.invIndexOfRefraction =
                1.0f / newMaterial.indexOfRefraction;
        }
        else
        {
            Log::warn("Scene",
                "Unknown material TYPE '%s' for '%s' defaulting to Diffuse",
                      p["TYPE"].get<std::string>().c_str(), name.c_str());
        }

        // ---- Texture mapping (optional) ----
        // TEXTURE: image path relative to the scene JSON, OR the literal
        // "checkerboard" for the procedural 8x8 pattern.  Resolves to
        // Material::textureId: -1 = flat color (default), -2 = checkerboard,
        // >= 0 = index into Scene::textures.  Only the diffuse albedo is
        // sampled this milestone; other material types keep their color.
        if (p.contains("TEXTURE"))
        {
            const string tex = p["TEXTURE"].get<string>();
            if (tex == "checkerboard")
            {
                newMaterial.textureId = kCheckerboardTextureId;
            }
            else
            {
                const string texPath = (jsonDir / tex).generic_string();
                const auto it = textureCache.find(texPath);
                if (it != textureCache.end())
                {
                    newMaterial.textureId = it->second;
                }
                else
                {
                    newMaterial.textureId = loadTextureFile(scene, texPath);
                    if (newMaterial.textureId >= 0)
                        textureCache[texPath] = newMaterial.textureId;
                }
            }
            newMaterial.uvScale = p.value("UV_SCALE", 1.0f);
        }

        MatNameToID[name] = scene.materials.size();
        scene.materials.emplace_back(newMaterial);
    }
}

/**
 * Parse the "Objects" section into Scene::geoms, dispatching each mesh's
 * FILE entry to the OBJ / glTF loader by file extension.
 *
 * @param data         Parsed scene JSON
 * @param scene        Scene to append geoms to (also the triangle sink for
 *                     mesh loading)
 * @param jsonDir      Directory of the scene JSON — relative mesh paths
 *                     resolve against it
 * @param MatNameToID  Material-name -> index map built by parseMaterials
 */
static void parseObjects(
    const json& data, Scene& scene,
    const filesystem::path& jsonDir,
    unordered_map<string, uint32_t>& MatNameToID)
{
    // ---- Objects (geometries) -----------------------------------------
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        Geom newGeom{};

        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation    = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale       = glm::vec3(scale[0], scale[1], scale[2]);

        newGeom.meshTriangleOffset = -1;
        newGeom.meshTriangleCount  = 0;

        filesystem::path objRel = p.value("FILE", string(""));
        if (objRel.empty())
        {
            Log::warn("Scene", "Mesh object with no FILE field; skipping");
            continue;
        }

        // Dispatch on file extension: .obj (tinyobjloader) or
        // .gltf / .glb (cgltf).
        string meshPath = (jsonDir / objRel).generic_string();
        string ext = filesystem::path(meshPath).extension().string();
        transform(ext.begin(), ext.end(), ext.begin(),
                  [](unsigned char c) { return (char)tolower(c); });

        pair<int, int> slice = {-1, 0};
        if (ext == ".obj")
        {
            slice = loadOBJ(meshPath, scene.hostTriangles);
        }
        else if (ext == ".gltf" || ext == ".glb")
        {
            slice = loadGLTF(scene, meshPath);
        }
        else
        {
            Log::warn("Scene", "Unsupported mesh format '%s' for '%s'; skipping",
                      ext.c_str(), objRel.string().c_str());
            continue;
        }

        newGeom.meshTriangleOffset = slice.first;
        newGeom.meshTriangleCount  = slice.second;

        // An explicit JSON TEXTURE on the object's material wins over the
        // model's glTF baseColor map (the scene author overrides the
        // asset's own albedo).  Zero the slice's tex.baseColor so the
        // shading fallback chain (tex.baseColor → m.textureId) resolves
        // to the JSON-declared image.
        if (slice.first >= 0 && slice.second > 0)
        {
            const string matName = p["MATERIAL"].get<string>();
            if (data["Materials"].contains(matName) &&
                data["Materials"][matName].contains("TEXTURE"))
            {
                for (int i = slice.first;
                     i < slice.first + slice.second; ++i)
                    scene.hostTriangles[i].tex.baseColor = -1;
            }
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose     =
            glm::inverseTranspose(newGeom.transform);

        scene.geoms.push_back(newGeom);
    }
}

/**
 * Parse the "Camera" section and derive the camera frame — fov, the
 * view/right basis, pixel length — and the host image buffer.
 */
static void parseCamera(const json& data, Scene& scene)
{
    // ---- Camera -------------------------------------------------------
    const auto& cameraData  = data["Camera"];
    Camera&      camera     = scene.state.camera;
    RenderState& state      = scene.state;
    camera.resolution.x     = cameraData["RES"][0];
    camera.resolution.y     = cameraData["RES"][1];
    float fovy              = cameraData["FOVY"];
    state.iterations        = cameraData["ITERATIONS"];
    state.traceDepth        = cameraData["DEPTH"];
    state.rrMinBounces      = cameraData.value("RR_DEPTH", 3);
    state.imageName         = cameraData["FILE"];

    const auto& pos    = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up     = cameraData["UP"];
    camera.position    = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt      = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up          = glm::vec3(up[0], up[1], up[2]);
    camera.lensRadius      = cameraData.value("LENS_RADIUS", 0.0f);
    camera.focalDistance   = cameraData.value("FOCAL_DISTANCE", 0.0f);

    float yscaled = tan(fovy * DEG_TO_RAD);
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx    = atan(xscaled) * RAD_TO_DEG;
    camera.fov    = glm::vec2(fovx, fovy);

    camera.view        = glm::normalize(camera.lookAt - camera.position);
    camera.right       = glm::normalize(
        glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(
        2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    fill(state.image.begin(), state.image.end(), glm::vec3());
}

// -----------------------------------------------------------------------
// JSON Scene Loading
// -----------------------------------------------------------------------

/**
 * Load a complete scene from a JSON file.
 *
 * Parses materials, objects (mesh objects dispatch to the OBJ / glTF
 * loaders), and camera settings from the JSON.  Mesh file paths are
 * resolved relative to the JSON file's directory.
 */
Scene loadFromJSON(const std::string& jsonName)
{
    Scene scene;

    Log::info("Scene", "Reading: %s", jsonName.c_str());

    auto ext = jsonName.substr(jsonName.find_last_of('.'));
    if (ext != ".json")
    {
        Log::error("Scene", "Unsupported scene format: %s", ext.c_str());
        exit(-1);
    }

    // Resolve the JSON file's directory so relative mesh paths work.
    filesystem::path jsonDir = filesystem::path(jsonName).parent_path();

    ifstream f(jsonName);
    json data = json::parse(f);

    // Parse the sections in dependency order: materials first (objects
    // reference them by name), then objects (geometries), then the camera.
    unordered_map<string, uint32_t> MatNameToID;
    parseMaterials(data, scene, jsonDir, MatNameToID);
    parseObjects(data, scene, jsonDir, MatNameToID);
    parseCamera(data, scene);

    return scene;
}

} // namespace SceneLoader
