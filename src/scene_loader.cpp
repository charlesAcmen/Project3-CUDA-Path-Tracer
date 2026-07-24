// Define the implementation of tinyobjloader in this translation unit
// so it doesn't leak into other compilation units.
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "scene_loader.h"

#include "constants.h"
#include "logger.h"
#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include "json.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

namespace SceneLoader {

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

            // ---- Face normal calculation (safe fallback) ----
            glm::vec3 e1 = v1 - v0;
            glm::vec3 e2 = v2 - v0;
            glm::vec3 crossE = glm::cross(e1, e2);
            float cLen2 = glm::dot(crossE, crossE);
            glm::vec3 fn = (std::isnan(cLen2) || cLen2 < RAY_EPSILON) ? glm::vec3(0.0f, 1.0f, 0.0f) : crossE * (1.0f / std::sqrt(cLen2));

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

                if (std::isnan(glm::dot(n0, n0)) || glm::dot(n0, n0) < RAY_EPSILON) n0 = fn;
                if (std::isnan(glm::dot(n1, n1)) || glm::dot(n1, n1) < RAY_EPSILON) n1 = fn;
                if (std::isnan(glm::dot(n2, n2)) || glm::dot(n2, n2) < RAY_EPSILON) n2 = fn;
            }
            else
            {
                // No vertex normals in OBJ → use face normal for all three vertices.
                n0 = n1 = n2 = fn;
            }

            triangles.push_back({v0, v1, v2, n0, n1, n2});
            count++;
            index_offset += fv;
        }
    }

    Log::info("Scene", "Loaded mesh: %s  (%d triangles, total %zu)",
              objPath.c_str(), count, triangles.size());
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
        // TODO: handle materials loading differently
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

            auto [offset, count] =
                loadOBJ((jsonDir / objRel).generic_string(),
                        scene.hostTriangles);
            newGeom.meshTriangleOffset = offset;
            newGeom.meshTriangleCount  = count;
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
    state.fresnelMode = static_cast<FresnelMode>(
        cameraData.value("FRESNEL_MODE", 0));
    state.imageName         = cameraData["FILE"];

    const auto& pos    = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up     = cameraData["UP"];
    camera.position    = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt      = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up          = glm::vec3(up[0], up[1], up[2]);
    camera.lensRadius      = cameraData.value("LENS_RADIUS", 0.0f);
    camera.focalDistance   = cameraData.value("FOCAL_DISTANCE", 0.0f);

    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx    = (atan(xscaled) * 180) / PI;
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
