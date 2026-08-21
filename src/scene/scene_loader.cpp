// ====================================================================
// JSON scene loading — the public SceneLoader interface (loadFromJSON).
//
// Parses the scene JSON's Materials / Objects / Camera sections.  Mesh FILE
// entries dispatch to the OBJ / glTF loaders in obj_loader.cpp /
// gltf_loader.cpp; texture loading lives in texture_loader.cpp.  This TU no
// longer includes tinyobjloader / cgltf / stb — the loader implementation
// macros are defined in the per-format loader TUs.
// ====================================================================

#include "scene/scene_loader.h"
#include "scene/loader_internal.h"

#include "constants.h"        // DEG_TO_RAD / RAD_TO_DEG
#include "utils/logger.h"
#include "utils/utilities.h"  // buildTransformationMatrix

#include <glm/gtc/matrix_inverse.hpp>   // glm::inverse / glm::inverseTranspose

#include <json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

namespace SceneLoader {

// -----------------------------------------------------------------------
// JSON Section Parsing
// -----------------------------------------------------------------------

// Parse one material's TYPE dispatch into `m` (the material's other fields
// are untouched).  TYPE is matched case-insensitively ("pbr" == "PBR").
static void applyMaterialType(const json& p, const string& name, Material& m)
{
    const std::string rawType = p["TYPE"].get<std::string>();
    std::string typeStr = rawType;
    std::transform(typeStr.begin(), typeStr.end(), typeStr.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

    if (typeStr == "DIFFUSE")
    {
        const auto& col = p["RGB"];
        m.color = glm::vec3(col[0], col[1], col[2]);
        m.type = MaterialType::Diffuse;
    }
    else if (typeStr == "EMITTING")
    {
        const auto& col = p["RGB"];
        m.color = glm::vec3(col[0], col[1], col[2]);
        m.type = MaterialType::Emissive;
        m.emittance = p["EMITTANCE"];
    }
    else if (typeStr == "SPECULAR")
    {
        const auto& col = p["RGB"];
        m.color = glm::vec3(col[0], col[1], col[2]);
        // Specular tint: read SPECULAR_COLOR if present, fall back to RGB.
        // RGB alone serves both diffuse albedo and specular tint, but the
        // two can diverge for physically accurate metals (RGB ≈ black,
        // SPECULAR_COLOR = reflectivity tint per wavelength).
        if (p.contains("SPECULAR_COLOR"))
        {
            const auto& sc = p["SPECULAR_COLOR"];
            m.specular.color = glm::vec3(sc[0], sc[1], sc[2]);
        }
        else
        {
            m.specular.color = glm::vec3(col[0], col[1], col[2]);
        }
        m.type = MaterialType::Reflective;
    }
    else if (typeStr == "PBR")
    {
        // Unified metallic-roughness surface (GGX).  RGB is the base color
        // — the diffuse albedo, which also tints the conductor F0 by metallic.
        // Roughness and metallic parameters are resolved per-hit from mesh
        // textures (ORM) or glTF factors (or standard defaults).
        //
        // Special case: if RGB is missing but the material references a glTF mesh,
        // set a sentinel value so the glTF material's own colors are used.
        if (p.contains("RGB"))
        {
            const auto& col = p["RGB"];
            m.color = glm::vec3(col[0], col[1], col[2]);
        }
        else
        {
            // Missing RGB means "use glTF material colors" for glTF meshes
            // Default to black (0,0,0) — resolveBaseColor will skip this and use glTF textures/factors
            m.color = glm::vec3(-1.0f);  // Sentinel for "use glTF material"
        }
        m.type = MaterialType::Pbr;
    }
    else if (typeStr == "REFRACTIVE")
    {
        const auto& col = p["RGB"];
        m.color = glm::vec3(col[0], col[1], col[2]);
        m.type = MaterialType::Refractive;
        m.indexOfRefraction = p.value("IOR", 1.5f);
        m.invIndexOfRefraction =
            1.0f / m.indexOfRefraction;
    }
    else
    {
        Log::warn("Scene",
            "Unknown material TYPE '%s' for '%s' defaulting to Diffuse",
                  rawType.c_str(), name.c_str());
        // Leave a VISIBLE fallback: the value-initialized color is black,
        // which renders as a pitch-black object — indistinguishable from a
        // missing mesh or an unlit scene.  White keeps a typo debuggable.
        m.color = glm::vec3(1.0f);
    }
}

// Parse one material's optional TEXTURE slot (image path relative to the
// scene JSON) into `m.textureId`, deduping by resolved path.
static void applyMaterialTexture(const json& p,
                                 const filesystem::path& jsonDir,
                                 Scene& scene,
                                 unordered_map<string, int>& textureCache,
                                 Material& m)
{
    // TEXTURE: image path relative to the scene JSON.  Resolves to
    // Material::textureId: -1 = flat color (default; an unreadable image
    // falls back here with an error logged), >= 0 = index into
    // Scene::textures.  This is the diffuse albedo source for Diffuse and
    // Pbr surfaces (resolveBaseColor: glTF baseColor binding wins, else the
    // JSON TEXTURE, else the flat color); other material types keep their
    // color.
    if (p.contains("TEXTURE"))
    {
        const string tex = p["TEXTURE"].get<string>();
        const string texPath = (jsonDir / tex).generic_string();
        const auto it = textureCache.find(texPath);
        if (it != textureCache.end())
        {
            m.textureId = it->second;
        }
        else
        {
            m.textureId = loadTextureFile(scene, texPath);
            if (m.textureId >= 0)
                textureCache[texPath] = m.textureId;
        }
        m.uvScale = p.value("UV_SCALE", 1.0f);
    }
}

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

        // Material TYPE + optional JSON TEXTURE.
        applyMaterialType(p, name, newMaterial);
        applyMaterialTexture(p, jsonDir, scene, textureCache, newMaterial);

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
            slice = loadOBJ(meshPath, scene.hostTrianglePositions, scene.hostTriangleAttrs, &scene);
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
    camera.up          = glm::normalize(
        glm::cross(camera.right, camera.view));
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
