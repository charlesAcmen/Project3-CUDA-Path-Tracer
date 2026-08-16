// ====================================================================
// glTF 2.0 mesh loading (cgltf): scene-graph walk, node-transform bake, and
// per-material texture auto-load.
//
// This is the ONLY translation unit that defines CGLTF_IMPLEMENTATION, so the
// cgltf implementation lives here and never leaks into the other scene-loader
// TUs.
// ====================================================================

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include "scene/loader_internal.h"

#include "utils/logger.h"

#include <glm/gtc/matrix_inverse.hpp>   // glm::inverseTranspose
#include <glm/gtc/matrix_transform.hpp> // glm::translate / glm::scale (node TRS)
#include <glm/gtc/quaternion.hpp>       // glm::mat4_cast (node rotation)
#include <glm/gtc/type_ptr.hpp>         // glm::make_mat4 (node matrix)

#include <filesystem>
#include <string>
#include <vector>

using namespace std;

namespace SceneLoader {

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
// - External PNG/JPG file URIs (relative to the .gltf/.glb) AND images
//   embedded in the .glb buffer (buffer_view, decoded from memory) load;
//   data: URIs are skipped with a warning — shading falls back to the
//   material's own value.
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

    // data: URIs stay unsupported (neither a file nor a bufferView) — warn
    // once per image and fall back to the material's own value.
    if (image->uri != nullptr && string(image->uri).rfind("data:", 0) == 0)
    {
        Log::warn("Scene",
                  "glTF image #%d ('%s'): data: URIs are not supported; "
                  "using material color",
                  imageIdx, image->name ? image->name : "unnamed");
        ctx.imageToTexId[imageIdx] = -1;   // remember — warn once per image
        return -1;
    }

    // Embedded (bufferView) images — .glb's default layout packs the PNG/JPG
    // bytes into the binary buffer.  cgltf_load_buffers already loaded those
    // bytes, so decode straight from memory (stb auto-detects the format).
    if (image->buffer_view != nullptr)
    {
        const cgltf_buffer* buf = image->buffer_view->buffer;
        if (buf == nullptr || buf->data == nullptr || image->buffer_view->size == 0)
        {
            Log::warn("Scene", "glTF image #%d ('%s'): embedded image has no "
                      "buffer data; using material color",
                      imageIdx, image->name ? image->name : "unnamed");
            ctx.imageToTexId[imageIdx] = -1;
            return -1;
        }
        const unsigned char* bytes =
            (const unsigned char*)buf->data + image->buffer_view->offset;
        const int id = loadTextureMemory(ctx.scene, bytes,
                                         (int)image->buffer_view->size, srgb);
        if (id >= 0)
            Log::info("Scene", "Auto-loaded embedded glTF texture (image #%d)",
                      imageIdx);
        ctx.imageToTexId[imageIdx] = id;
        return id;
    }

    // No URI and no bufferView — nothing to load.
    if (image->uri == nullptr)
    {
        ctx.imageToTexId[imageIdx] = -1;
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
    // glTF's own pbrMetallicRoughness scalar defaults (cgltf fills the spec
    // defaults — roughness 1.0, metallic 1.0, baseColorFactor (1,1,1,1) —
    // when the file omits them).  These are fallbacks for Pbr materials whose
    // JSON did not declare ROUGHNESS / METALLIC / an explicit color; non-glTF
    // meshes keep the -1 sentinels.
    b.roughnessFactor = mat->pbr_metallic_roughness.roughness_factor;
    b.metallicFactor  = mat->pbr_metallic_roughness.metallic_factor;
    b.baseColorFactor = glm::vec3(mat->pbr_metallic_roughness.base_color_factor[0],
                                  mat->pbr_metallic_roughness.base_color_factor[1],
                                  mat->pbr_metallic_roughness.base_color_factor[2]);
    return b;
}

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

// Locate the POSITION / NORMAL / TEXCOORD_0 accessors of a primitive.
// TEXCOORD_0 is preferred (higher sets are ignored).
static void findAttributeAccessors(const cgltf_primitive* prim,
                                   const cgltf_accessor*& posAcc,
                                   const cgltf_accessor*& nrmAcc,
                                   const cgltf_accessor*& uvAcc)
{
    posAcc = nullptr;
    nrmAcc = nullptr;
    uvAcc  = nullptr;
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
}

// Unpack an accessor's component values into a flat float array.
// cgltf_accessor_read_float handles integer and normalized component types,
// so a VEC3 always yields 3 floats.
static vector<float> unpackAccessor(const cgltf_accessor* acc, int comps)
{
    vector<float> out(comps * (size_t)acc->count);
    for (cgltf_size i = 0; i < acc->count; ++i)
        cgltf_accessor_read_float(acc, i, &out[comps * (size_t)i], comps);
    return out;
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
    findAttributeAccessors(prim, posAcc, nrmAcc, uvAcc);
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

    // Unpack all vertex positions / normals / UVs (cgltf_accessor_read_float
    // handles integer and normalized component types).
    vector<float> pos = unpackAccessor(posAcc, 3);

    // Normals (optional; makeTri falls back to the face normal).
    vector<float> nrm;
    if (nrmAcc != nullptr)
        nrm = unpackAccessor(nrmAcc, 3);

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
            uv = unpackAccessor(uvAcc, 2);
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

    // Emit one triangle with the primitive's material texture slots.
    auto emit = [&](cgltf_size i0, cgltf_size i1, cgltf_size i2) {
        triangles.push_back(makeTri(vTrans(i0), vTrans(i1), vTrans(i2),
                                    nTrans(i0), nTrans(i1), nTrans(i2),
                                    uvAt(i0), uvAt(i1), uvAt(i2)));
        triangles.back().tex = binding;
        ++count;
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

            emit(i0, i1, i2);
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
            emit(t + 0, t + 1, t + 2);
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
 *   Images load from external PNG/JPG file URIs or from the .glb binary
 *   buffer (bufferView); data: URIs warn + fall back to the material color.
 *   The scene JSON's MATERIAL still governs the shading model (an explicit
 *   JSON TEXTURE overrides the glTF baseColor — see parseObjects).
 * - Draco-compressed primitives are not supported (cgltf does not decode).
 *
 * @param scene     Scene to append to (hostTriangles AND textures)
 * @param gltfPath  Path to the .gltf or .glb file
 * @return (offset, count) — the slice of scene.hostTriangles this file occupies
 */
pair<int, int> loadGLTF(Scene& scene, const string& gltfPath)
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

} // namespace SceneLoader
