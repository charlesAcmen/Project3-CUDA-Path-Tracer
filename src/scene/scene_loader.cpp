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

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>   // glm::translate / glm::scale (node TRS)
#include <glm/gtc/quaternion.hpp>         // glm::mat4_cast (node rotation)
#include <glm/gtc/type_ptr.hpp>           // glm::make_mat4 (node matrix)
#include "json.hpp"

#include <algorithm>
#include <cctype>
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
 */
static Triangle makeTri(
    const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
    const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2)
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

    return Triangle{ v0, v1, v2, validOr(n0), validOr(n1), validOr(n2) };
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

    // Determine if the OBJ provides vertex normals (vn entries).
    const bool hasNormals = (!attrib.normals.empty());

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

            triangles.push_back(makeTri(v0, v1, v2, n0, n1, n2));
            count++;
            index_offset += fv;
        }
    }

    Log::info("Scene", "Loaded mesh: %s  (%d triangles, total %zu)",
              objPath.c_str(), count, triangles.size());
    return {offset, count};
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

    glm::mat4 m(1.0f);
    if (n->has_scale)
        m = glm::scale(m, glm::vec3(n->scale[0], n->scale[1], n->scale[2]));
    if (n->has_rotation)
        // cgltf stores the quaternion as (x, y, z, w); GLM's constructor is (w, x, y, z).
        m = glm::mat4_cast(glm::quat(n->rotation[3], n->rotation[0],
                                     n->rotation[1], n->rotation[2])) * m;
    if (n->has_translation)
        m = glm::translate(m, glm::vec3(n->translation[0], n->translation[1],
                                        n->translation[2]));
    return m;   // m = T·R·S
}

// Emit one primitive's triangles, transformed into the node's accumulated
// frame.  Vertices via `world`; normals via the inverse-transpose — left
// UNnormalized, so the world-space bake + hit-time normalize compose with
// it by linearity, exactly as with the per-geom transform.
static void appendPrimitiveTriangles(const cgltf_primitive* prim,
                                     const glm::mat4& world,
                                     const glm::mat4& worldIT,
                                     vector<Triangle>& triangles, int& count)
{
    if (prim->type != cgltf_primitive_type_triangles)
    {
        Log::warn("Scene", "glTF primitive is not triangles (mode %d); skipping",
                  (int)prim->type);
        return;
    }

    // ---- Locate POSITION / NORMAL accessors ----
    const cgltf_accessor* posAcc = nullptr;
    const cgltf_accessor* nrmAcc = nullptr;
    for (cgltf_size ai = 0; ai < prim->attributes_count; ++ai)
    {
        const cgltf_attribute* attr = &prim->attributes[ai];
        if (attr->type == cgltf_attribute_type_position)
            posAcc = attr->data;
        else if (attr->type == cgltf_attribute_type_normal)
            nrmAcc = attr->data;
    }
    if (posAcc == nullptr)
    {
        Log::warn("Scene", "glTF primitive has no POSITION accessor; skipping");
        return;
    }

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
                                        nTrans(i0), nTrans(i1), nTrans(i2)));
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
                                        nTrans(t + 0), nTrans(t + 1), nTrans(t + 2)));
            ++count;
        }
    }
}

// Emit all of a mesh's primitives under one accumulated transform.
static void appendMeshTriangles(const cgltf_mesh* mesh, const glm::mat4& world,
                                vector<Triangle>& triangles, int& count)
{
    const glm::mat4 worldIT = glm::inverseTranspose(world);
    for (cgltf_size pi = 0; pi < mesh->primitives_count; ++pi)
        appendPrimitiveTriangles(&mesh->primitives[pi], world, worldIT,
                                 triangles, count);
}

// Depth-first walk of the scene graph.  The accumulated matrix `parentWorld`
// is the composition of every ancestor's local transform; multiplying it by
// this node's local transform gives the world matrix that places its mesh.
static void walkNode(const cgltf_node* node, const glm::mat4& parentWorld,
                     vector<Triangle>& triangles, int& count)
{
    const glm::mat4 world = parentWorld * nodeLocalMatrix(node);
    if (node->mesh != nullptr)
        appendMeshTriangles(node->mesh, world, triangles, count);
    for (cgltf_size c = 0; c < node->children_count; ++c)
        walkNode(node->children[c], world, triangles, count);
}

/**
 * Load triangles from a glTF 2.0 file (.gltf JSON or .glb binary) and
 * append them to the hostTriangles vector.
 *
 * - The scene graph is walked and every node's accumulated transform is
 *   applied, so multi-part models scattered across nodes are assembled
 *   exactly as the file specifies (each (node, mesh) instance is emitted
 *   once, with its own transform).
 * - Only triangle primitives (mode 4) are loaded; point/line primitives
 *   are skipped with a warning.
 * - glTF materials/textures are ignored — the scene JSON's MATERIAL field
 *   governs shading (texture mapping is a separate feature).
 * - Draco-compressed primitives are not supported (cgltf does not decode).
 *
 * @param gltfPath  Path to the .gltf or .glb file
 * @param triangles [out] Flat array of object-space triangles to append to
 * @return (offset, count) — the slice of `triangles` this file occupies
 */
static pair<int, int> loadGLTF(const string& gltfPath,
                               vector<Triangle>& triangles)
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

    int offset = (int)triangles.size();
    int count  = 0;

    // ---- Walk the scene graph, emitting one transformed instance per
    // (node, mesh).  Parts scattered across nodes are assembled by the
    // accumulated node transforms — the glTF-spec behavior.
    const cgltf_scene* scene = data->scene;   // default scene
    if (scene != nullptr && scene->nodes_count > 0)
    {
        for (cgltf_size i = 0; i < scene->nodes_count; ++i)
            walkNode(scene->nodes[i], glm::mat4(1.0f), triangles, count);
    }
    else
    {
        // No scene graph (some minimal files): emit every mesh untransformed.
        for (cgltf_size mi = 0; mi < data->meshes_count; ++mi)
            appendMeshTriangles(&data->meshes[mi], glm::mat4(1.0f), triangles, count);
    }

    cgltf_free(data);

    Log::info("Scene", "Loaded mesh: %s  (%d triangles, total %zu)",
              gltfPath.c_str(), count, triangles.size());
    return {offset, count};
}

// -----------------------------------------------------------------------
// JSON Scene Loading
// -----------------------------------------------------------------------

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

    // Resolve the JSON file's directory so relative OBJ paths work.
    filesystem::path jsonDir = filesystem::path(jsonName).parent_path();

    ifstream f(jsonName);
    json data = json::parse(f);

    // ---- Materials ----------------------------------------------------
    const auto& materialsData = data["Materials"];
    unordered_map<string, uint32_t> MatNameToID;
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
            if (p.contains("ROUGHNESS"))
            {
                float r = glm::clamp((float)p["ROUGHNESS"], 0.0f, 1.0f);
                if (r < ROUGHNESS_THRESHOLD)
                {
                    newMaterial.specular.exponent = -1.0f;
                }
                else
                {
                    newMaterial.specular.exponent =
                        (2.0f / (r * r)) - 2.0f;
                }
            }
            else
            {
                newMaterial.specular.exponent = -1.0f;
            }
            // Precompute 1/(exponent+1) for the Phong lobe here (host-side) so
            // samplePhongSpecularDir never divides on the GPU.  Mirror
            // (exponent = -1) never reaches the Phong branch — value is 0.
            newMaterial.specular.invExponentPlusOne =
                (newMaterial.specular.exponent >= 0.0f)
                    ? 1.0f / (newMaterial.specular.exponent + 1.0f)
                    : 0.0f;
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
        MatNameToID[name] = scene.materials.size();
        scene.materials.emplace_back(newMaterial);
    }

    // ---- Objects (geometries) -----------------------------------------
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
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

        if (type == "mesh")
        {
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
                slice = loadGLTF(meshPath, scene.hostTriangles);
            }
            else
            {
                Log::warn("Scene", "Unsupported mesh format '%s' for '%s'; skipping",
                          ext.c_str(), objRel.string().c_str());
                continue;
            }

            newGeom.meshTriangleOffset = slice.first;
            newGeom.meshTriangleCount  = slice.second;
        }
        else
        {
            Log::warn("Scene", "Unknown object TYPE '%s'; skipping",
                      p["TYPE"].get<std::string>().c_str());
            continue;
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose     =
            glm::inverseTranspose(newGeom.transform);

        scene.geoms.push_back(newGeom);
    }

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

    return scene;
}

} // namespace SceneLoader
