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

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <string>
#include <thread>
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

// One glTF image queued for the parallel pre-decode (index = cgltf image
// index).  `decode` is true when the image is referenced by a material slot
// AND has a loadable source; `srgb` is the treatment the FIRST slot to
// reference it asked for (mirrors the sequential loader's first-reference-
// wins rule; no real file shares one image between a color and a data role).
struct PendingTexture
{
    bool        decode = false;
    bool        srgb   = false;
    string      uri;                 // percent-decoded URI (file source, for logs)
    string      path;                // dir-joined absolute path (file source)
    const unsigned char* bytes = nullptr;   // .glb bufferView payload
    int         len    = 0;
};

// Decode result for one queued image.
struct TextureDecodeOut
{
    TextureData td;
    bool        ok = false;
};

// Queue every image the material walk below will bind, in first-reference
// order.  Replicates the old per-slot resolveGltfTextureSlot decision logic:
// the first slot to reference an image sets its sRGB treatment, and a skipped
// image (data: URI / no buffer / no URI) is never retried — the same cached
// -1 semantics, just decided up front instead of mid-walk.
static void queueMaterialSlots(GltfLoadCtx& ctx, const cgltf_material* mat,
                               vector<PendingTexture>& pending,
                               vector<bool>& decided)
{
    if (mat == nullptr)
        return;

    const auto queueSlot = [&](const cgltf_texture_view& view, bool srgb) {
        if (view.texture == nullptr || view.texture->image == nullptr)
            return;
        const int idx = (int)(view.texture->image - ctx.data->images);
        if (decided[(size_t)idx])
            return;                                // already decided
        decided[(size_t)idx] = true;
        PendingTexture& p = pending[(size_t)idx];
        p.srgb = srgb;

        const cgltf_image* image = view.texture->image;

        // data: URIs stay unsupported (neither a file nor a bufferView).
        if (image->uri != nullptr && string(image->uri).rfind("data:", 0) == 0)
        {
            Log::warn("Scene",
                      "glTF image #%d ('%s'): data: URIs are not supported; "
                      "using material color",
                      idx, image->name ? image->name : "unnamed");
            return;                                // p.decode stays false
        }

        // Embedded (bufferView) images — .glb's default layout packs the
        // PNG/JPG bytes into the binary buffer; cgltf_load_buffers already
        // loaded them, so decode straight from memory (stb detects format).
        if (image->buffer_view != nullptr)
        {
            const cgltf_buffer* buf = image->buffer_view->buffer;
            if (buf == nullptr || buf->data == nullptr || image->buffer_view->size == 0)
            {
                Log::warn("Scene", "glTF image #%d ('%s'): embedded image has no "
                          "buffer data; using material color",
                          idx, image->name ? image->name : "unnamed");
                return;
            }
            p.bytes = (const unsigned char*)buf->data + image->buffer_view->offset;
            p.len   = (int)image->buffer_view->size;
            p.decode = true;
            return;
        }

        // No URI and no bufferView — nothing to load.
        if (image->uri == nullptr)
            return;

        // Percent-decode into a copy (cgltf_decode_uri mutates in place, and
        // the uri string is owned by the cgltf document).
        string uri = image->uri;
        vector<char> decoded(uri.begin(), uri.end());
        decoded.push_back('\0');
        cgltf_decode_uri(decoded.data());
        uri.assign(decoded.data());
        p.uri  = uri;
        p.path = (ctx.dir / uri).generic_string();
        p.decode = true;
    };

    // Visit the same 5 slots in the same order as bindGltfMaterial.
    queueSlot(mat->pbr_metallic_roughness.base_color_texture, true);
    queueSlot(mat->normal_texture, false);
    queueSlot(mat->pbr_metallic_roughness.metallic_roughness_texture, false);
    queueSlot(mat->occlusion_texture, false);
    queueSlot(mat->emissive_texture, true);
}

// Depth-first pre-walk mirroring walkNode / appendMeshTriangles, collecting
// each (node, mesh) instance's primitive materials in traversal order.  No
// triangle work — just enough to know WHICH images the graph walk will bind,
// and in what order, so the first-reference sRGB decision matches it.
static void collectPrimitiveMaterials(const cgltf_node* node,
                                      vector<const cgltf_material*>& mats)
{
    if (node->mesh != nullptr)
        for (cgltf_size pi = 0; pi < node->mesh->primitives_count; ++pi)
            if (node->mesh->primitives[pi].material != nullptr)
                mats.push_back(node->mesh->primitives[pi].material);
    for (cgltf_size c = 0; c < node->children_count; ++c)
        collectPrimitiveMaterials(node->children[c], mats);
}

// Decode every queued glTF image in parallel, then assemble the survivors
// into Scene::textures in image-index order and fill ctx.imageToTexId.
//
// stbi_load / stbi_load_from_memory are thread-safe per call, and decode +
// sRGB-linearize touch no shared state, so the ~30 × 2048² JPEG dump that
// took seconds of serial decode becomes one wave of worker threads.  The
// ASSEMBLY stays serial: texture ids are Scene::textures indices, and
// push_back must be ordered deterministically.
static void preloadGltfTextures(GltfLoadCtx& ctx,
                                const vector<const cgltf_material*>& mats)
{
    const int nImages = (int)ctx.data->images_count;
    if (nImages == 0)
        return;

    vector<PendingTexture> pending((size_t)nImages);
    vector<bool>           decided((size_t)nImages, false);
    for (const cgltf_material* mat : mats)
        queueMaterialSlots(ctx, mat, pending, decided);

    // ---- Parallel decode (atomic index pull keeps workers balanced) ----
    int decodeCount = 0;
    for (const PendingTexture& p : pending)
        if (p.decode) ++decodeCount;

    vector<TextureDecodeOut> results((size_t)nImages);
    if (decodeCount > 0)
    {
        const unsigned nThreads = (unsigned)std::max<size_t>(
            1, std::min<size_t>(std::thread::hardware_concurrency(),
                                (size_t)decodeCount));
        std::atomic<int> next{0};
        vector<std::thread> workers;
        workers.reserve(nThreads);
        for (unsigned t = 0; t < nThreads; ++t)
            workers.emplace_back([&] {
                for (int i = next.fetch_add(1, std::memory_order_relaxed);
                     i < nImages;
                     i = next.fetch_add(1, std::memory_order_relaxed))
                {
                    const PendingTexture& p = pending[(size_t)i];
                    if (!p.decode) continue;
                    results[(size_t)i].ok = decodeTexture(
                        p.path, p.bytes, p.len, p.srgb, results[(size_t)i].td);
                }
            });
        for (std::thread& t : workers)
            t.join();                       // join → decodes visible
    }

    // ---- Serial assembly: deterministic ids, logs in image order ----
    for (int i = 0; i < nImages; ++i)
    {
        const PendingTexture& p = pending[(size_t)i];
        if (!p.decode)
        {
            ctx.imageToTexId[(size_t)i] = -1;   // unreferenced or skipped
            continue;
        }
        const TextureDecodeOut& r = results[(size_t)i];
        if (!r.ok)
        {
            if (p.bytes)
                Log::error("Scene", "Failed to decode embedded texture image "
                          "(%d bytes)", p.len);
            else
                Log::error("Scene", "Failed to load texture image: %s",
                          p.path.c_str());
            ctx.imageToTexId[(size_t)i] = -1;
            continue;
        }
        const int id = (int)ctx.scene.textures.size();
        ctx.scene.textures.push_back(std::move(r.td));
        ctx.imageToTexId[(size_t)i] = id;
        if (p.bytes)
            Log::info("Scene", "Auto-loaded embedded glTF texture (image #%d)", i);
        else
            Log::info("Scene", "Auto-loaded glTF texture '%s' (image #%d)",
                      p.uri.c_str(), i);
    }
}

// Resolve one glTF material slot (a cgltf_texture_view) to its global
// Scene::textures index.  All referenced images were pre-decoded in parallel
// by preloadGltfTextures, so this is a pure lookup — the decode and the
// first-reference sRGB decision both happened up front.
static int resolveGltfTextureSlot(GltfLoadCtx& ctx,
                                  const cgltf_texture_view& view)
{
    if (view.texture == nullptr || view.texture->image == nullptr)
        return -1;
    const int imageIdx = (int)(view.texture->image - ctx.data->images);
    return ctx.imageToTexId[imageIdx];
}

// Resolve a glTF material's five texture slots into a shared SurfaceBinding.
// baseColor / emissive are color maps (sRGB → linearized on load); normal /
// metallicRoughness / occlusion are data maps (raw bytes kept).
static SurfaceBinding bindGltfMaterial(GltfLoadCtx& ctx,
                                       const cgltf_material* mat)
{
    SurfaceBinding b;
    if (mat == nullptr)
        return b;
    b.baseColor         = resolveGltfTextureSlot(ctx,
        mat->pbr_metallic_roughness.base_color_texture);
    b.normal            = resolveGltfTextureSlot(ctx, mat->normal_texture);
    b.metallicRoughness = resolveGltfTextureSlot(ctx,
        mat->pbr_metallic_roughness.metallic_roughness_texture);
    b.occlusion         = resolveGltfTextureSlot(ctx, mat->occlusion_texture);
    b.emissive          = resolveGltfTextureSlot(ctx, mat->emissive_texture);
    // glTF's pbrMetallicRoughness scalar factors:
    // When an ORM texture is bound, factors act as multipliers (glTF spec default 1.0).
    // When no ORM texture is bound, cgltf fills 1.0 by default — if factors were left at
    // default 1.0 without an ORM texture, keep -1 sentinels so the Pbr dielectric fallback
    // (metallic 0.0, roughness 0.5) is reachable rather than forcing untextured meshes into rough metal.
    if (mat->has_pbr_metallic_roughness)
    {
        if (b.metallicRoughness >= 0)
        {
            b.roughnessFactor = mat->pbr_metallic_roughness.roughness_factor;
            b.metallicFactor  = mat->pbr_metallic_roughness.metallic_factor;
        }
        else
        {
            if (mat->pbr_metallic_roughness.metallic_factor != 1.0f)
                b.metallicFactor = mat->pbr_metallic_roughness.metallic_factor;
            if (mat->pbr_metallic_roughness.roughness_factor != 1.0f)
                b.roughnessFactor = mat->pbr_metallic_roughness.roughness_factor;
        }
        b.baseColorFactor = glm::vec3(mat->pbr_metallic_roughness.base_color_factor[0],
                                      mat->pbr_metallic_roughness.base_color_factor[1],
                                      mat->pbr_metallic_roughness.base_color_factor[2]);
    }
    // glTF emissive intensity (see SurfaceBinding::emissiveFactor / Strength):
    //   Le = (emissiveTexture.rgb or white) · emissiveFactor
    //        · KHR_materials_emissive_strength.
    // cgltf fills emissive_factor with the spec default [0,0,0] when absent,
    // which disables emission even when an emissive texture is bound.
    // emissiveStrength defaults to 1.0 (no-op).
    b.emissiveFactor = glm::vec3(mat->emissive_factor[0],
                                 mat->emissive_factor[1],
                                 mat->emissive_factor[2]);
    b.emissiveStrength = mat->has_emissive_strength
                             ? mat->emissive_strength.emissive_strength : 1.0f;
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

// Locate the POSITION / NORMAL / TEXCOORD_0 / COLOR_0 accessors of a primitive.
// TEXCOORD_0 and COLOR_0 are preferred (higher sets are ignored).
static void findAttributeAccessors(const cgltf_primitive* prim,
                                   const cgltf_accessor*& posAcc,
                                   const cgltf_accessor*& nrmAcc,
                                   const cgltf_accessor*& uvAcc,
                                   const cgltf_accessor*& colAcc)
{
    posAcc = nullptr;
    nrmAcc = nullptr;
    uvAcc  = nullptr;
    colAcc = nullptr;
    for (cgltf_size ai = 0; ai < prim->attributes_count; ++ai)
    {
        const cgltf_attribute* attr = &prim->attributes[ai];
        if (attr->type == cgltf_attribute_type_position)
            posAcc = attr->data;
        else if (attr->type == cgltf_attribute_type_normal)
            nrmAcc = attr->data;
        else if (attr->type == cgltf_attribute_type_texcoord && attr->index == 0)
            uvAcc = attr->data;   // TEXCOORD_0 (we ignore higher sets)
        else if (attr->type == cgltf_attribute_type_color && attr->index == 0)
            colAcc = attr->data;   // COLOR_0 (we ignore higher sets)
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
                                     vector<TrianglePos>& positions,
                                     vector<TriangleAttr>& attrs, int& count,
                                     GltfLoadCtx& ctx)
{
    if (prim->type != cgltf_primitive_type_triangles)
    {
        Log::warn("Scene", "glTF primitive is not triangles (mode %d); skipping",
                  (int)prim->type);
        return;
    }

    // ---- Locate POSITION / NORMAL / TEXCOORD_0 / COLOR_0 accessors ----
    const cgltf_accessor* posAcc = nullptr;
    const cgltf_accessor* nrmAcc = nullptr;
    const cgltf_accessor* uvAcc  = nullptr;
    const cgltf_accessor* colAcc  = nullptr;
    findAttributeAccessors(prim, posAcc, nrmAcc, uvAcc, colAcc);
    if (posAcc == nullptr)
    {
        Log::warn("Scene", "glTF primitive has no POSITION accessor; skipping");
        return;
    }

    // Per-primitive surface binding from the glTF material.  All five roles
    // (baseColor/normal/ORM/occlusion/emissive) resolve into the global
    // texture table, then the resulting binding is interned once and linked
    // from every emitted triangle by a compact id.  baseColor, the ORM
    // (metallicRoughness) channels, and the normal slot are sampled by the
    // current shading; occlusion/emissive are data for future features.
    const int surfaceBindingId = internSurfaceBinding(
        ctx.scene, bindGltfMaterial(ctx, prim->material));

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

    // Normals (optional; appendTriangle falls back to the face normal).
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

    // Vertex colors (optional; default (1,1,1) = no effect).  glTF COLOR_0
    // can be VEC3 or VEC4; we only use RGB, ignoring alpha.
    vector<float> col;
    if (colAcc != nullptr)
    {
        if (colAcc->count != vertCount)
        {
            Log::warn("Scene",
                      "glTF primitive COLOR_0 count (%zu) != POSITION "
                      "count (%zu); ignoring vertex colors",
                      (size_t)colAcc->count, (size_t)vertCount);
            colAcc = nullptr;
        }
        else
        {
            // Unpack colors: VEC3 -> 3 floats, VEC4 -> 4 floats (we ignore alpha)
            int colComps = (colAcc->type == cgltf_type_vec3) ? 3 : 4;
            col = unpackAccessor(colAcc, colComps);
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
    // Vertex colors are per-vertex and NOT transformed.
    auto colAt = [&](cgltf_size i) -> glm::vec3 {
        if (colAcc == nullptr) return glm::vec3(1.0f);  // default white = no effect
        // Handle both VEC3 and VEC4 (ignore alpha)
        int stride = (colAcc->type == cgltf_type_vec3) ? 3 : 4;
        return glm::vec3(col[stride * (size_t)i + 0],
                         col[stride * (size_t)i + 1],
                         col[stride * (size_t)i + 2]);
    };

    // Node-transform the local vertex / normal into the scene frame.
    auto vTrans = [&](cgltf_size i) -> glm::vec3 {
        return glm::vec3(world * glm::vec4(vert(i), 1.0f));
    };
    auto nTrans = [&](cgltf_size i) -> glm::vec3 {
        return (nrmAcc != nullptr)
            ? glm::vec3(worldIT * glm::vec4(norm(i), 0.0f))
            : glm::vec3(0.0f);   // no vertex normal → appendTriangle's face-normal fallback
    };

    // Emit one triangle linked to the primitive's shared surface binding.
    auto emit = [&](cgltf_size i0, cgltf_size i1, cgltf_size i2) {
        appendTriangle(positions, attrs,
                       vTrans(i0), vTrans(i1), vTrans(i2),
                       nTrans(i0), nTrans(i1), nTrans(i2),
                       uvAt(i0), uvAt(i1), uvAt(i2),
                       colAt(i0), colAt(i1), colAt(i2));
        attrs.back().surfaceId = surfaceBindingId;
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
                                vector<TrianglePos>& positions,
                                vector<TriangleAttr>& attrs, int& count,
                                GltfLoadCtx& ctx)
{
    const glm::mat4 worldIT = glm::inverseTranspose(world);
    for (cgltf_size pi = 0; pi < mesh->primitives_count; ++pi)
        appendPrimitiveTriangles(&mesh->primitives[pi], world, worldIT,
                                 positions, attrs, count, ctx);
}

// Depth-first walk of the scene graph.  The accumulated matrix `parentWorld`
// is the composition of every ancestor's local transform; multiplying it by
// this node's local transform gives the world matrix that places its mesh.
static void walkNode(const cgltf_node* node, const glm::mat4& parentWorld,
                     vector<TrianglePos>& positions,
                     vector<TriangleAttr>& attrs, int& count, GltfLoadCtx& ctx)
{
    const glm::mat4 world = parentWorld * nodeLocalMatrix(node);
    if (node->mesh != nullptr)
        appendMeshTriangles(node->mesh, world, positions, attrs, count, ctx);
    for (cgltf_size c = 0; c < node->children_count; ++c)
        walkNode(node->children[c], world, positions, attrs, count, ctx);
}

/**
 * Load triangles from a glTF 2.0 file (.gltf JSON or .glb binary) and
 * append them to the scene's matched host position / attribute arrays.
 *
 * - The scene graph is walked and every node's accumulated transform is
 *   applied, so multi-part models scattered across nodes are assembled
 *   exactly as the file specifies (each (node, mesh) instance is emitted
 *   once, with its own transform).
 * - Only triangle primitives (mode 4) are loaded; point/line primitives
 *   are skipped with a warning.
 * - Each primitive's material texture slots (baseColor / normal /
 *   metallicRoughness / occlusion / emissive) are auto-loaded into
 *   Scene::textures and referenced by emitted triangles through a shared
 *   SurfaceBinding id.
 *   Images load from external PNG/JPG file URIs or from the .glb binary
 *   buffer (bufferView); data: URIs warn + fall back to the material color.
 *   The scene JSON's MATERIAL still selects the renderer's BSDF; texture
 *   roles stay with the glTF material.
 * - Draco-compressed primitives are not supported (cgltf does not decode).
 *
 * @param scene     Scene to append to (parallel geometry arrays AND textures)
 * @param gltfPath  Path to the .gltf or .glb file
 * @return (offset, count) — the source-triangle slice this file occupies
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

    vector<TrianglePos>& positions = scene.hostTrianglePositions;
    vector<TriangleAttr>& attrs = scene.hostTriangleAttrs;
    const int offset = (int)positions.size();
    int       count  = 0;

    // Texture auto-load context: glTF image URIs resolve relative to the
    // .gltf's directory; images dedup by cgltf image index (a file shared by
    // several materials / primitives loads once).
    GltfLoadCtx ctx = {
        scene, data,
        filesystem::path(gltfPath).parent_path(),
        vector<int>(data->images_count, -1)
    };

    const cgltf_scene* gltfScene = data->scene;   // default scene

    // ---- Parallel texture pre-decode ----
    // Collect the materials the walk below will bind (in walk order), then
    // decode every referenced image in parallel.  The graph walk itself is
    // fast; the ~30 × 2048² JPEG decode was the serial startup bottleneck.
    vector<const cgltf_material*> mats;
    if (gltfScene != nullptr && gltfScene->nodes_count > 0)
    {
        for (cgltf_size i = 0; i < gltfScene->nodes_count; ++i)
            collectPrimitiveMaterials(gltfScene->nodes[i], mats);
    }
    else
    {
        // No scene graph (some minimal files): every mesh's material binds.
        for (cgltf_size mi = 0; mi < data->meshes_count; ++mi)
            for (cgltf_size pi = 0; pi < data->meshes[mi].primitives_count; ++pi)
                if (data->meshes[mi].primitives[pi].material != nullptr)
                    mats.push_back(data->meshes[mi].primitives[pi].material);
    }
    preloadGltfTextures(ctx, mats);

    // ---- Walk the scene graph, emitting one transformed instance per
    // (node, mesh).  Parts scattered across nodes are assembled by the
    // accumulated node transforms — the glTF-spec behavior.
    if (gltfScene != nullptr && gltfScene->nodes_count > 0)
    {
        for (cgltf_size i = 0; i < gltfScene->nodes_count; ++i)
            walkNode(gltfScene->nodes[i], glm::mat4(1.0f), positions, attrs, count, ctx);
    }
    else
    {
        // No scene graph (some minimal files): emit every mesh untransformed.
        for (cgltf_size mi = 0; mi < data->meshes_count; ++mi)
            appendMeshTriangles(&data->meshes[mi], glm::mat4(1.0f), positions, attrs, count, ctx);
    }

    cgltf_free(data);

    Log::info("Scene", "Loaded mesh: %s  (%d triangles, total %zu)",
              gltfPath.c_str(), count, positions.size());
    return {offset, count};
}

} // namespace SceneLoader
