#pragma once

// ====================================================================
// Internal loader API (not part of the public SceneLoader interface)
//
// These helpers are shared across the split scene-loader translation
// units (obj_loader.cpp / gltf_loader.cpp / texture_loader.cpp /
// scene_loader.cpp) and are kept non-static so each TU can call the
// others' helpers.  They are declared here — NOT in scene_loader.h —
// because only SceneLoader::loadFromJSON is the public interface.
//
// Default arguments live ONLY in these declarations; the definitions
// in the .cpp files omit them.
// ====================================================================

#include "scene/scene.h"

#include <glm/glm.hpp>

#include <string>
#include <utility>
#include <vector>

namespace SceneLoader {

// Shared triangle construction (OBJ + glTF).  Any vertex normal that is
// NaN or (near-)zero-length is replaced by the geometric face normal.
// UVs default to (0,0) when the source file provides none.
Triangle makeTri(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
                 const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2,
                 const glm::vec2& u0 = glm::vec2(0.0f),
                 const glm::vec2& u1 = glm::vec2(0.0f),
                 const glm::vec2& u2 = glm::vec2(0.0f));

// Load an image file (PNG/JPG via stb_image) into Scene::textures.
// Returns the Scene::textures index (>= 0), or -1 on failure.
int loadTextureFile(Scene& scene, const std::string& path, bool srgb = true);

// Load an image decoded from in-memory bytes (a .glb bufferView / embedded
// glTF buffer) into Scene::textures.  Returns the index (>= 0), or -1.
int loadTextureMemory(Scene& scene, const unsigned char* bytes, int len,
                      bool srgb);

// Decode one image into a host TextureData WITHOUT touching Scene::textures
// (thread-safe — used by the parallel glTF texture pre-pass).  `bytes`
// non-null → decode from memory; else decode from `path`.  On success fills
// `out` and returns true.
bool decodeTexture(const std::string& path, const unsigned char* bytes, int len,
                   bool srgb, TextureData& out);

// Load triangles from a Wavefront OBJ file, appending to `triangles`.
// When `scene` is non-null, the companion .mtl's image maps
// (map_Kd / map_Bump / map_Ke) are stamped onto each face's triangles by
// material_id.  Returns (offset, count) — the slice this mesh occupies
// (offset -1 on failure).
std::pair<int, int> loadOBJ(const std::string& objPath,
                            std::vector<Triangle>& triangles,
                            Scene* scene = nullptr);

// Load triangles from a glTF 2.0 file (.gltf JSON or .glb binary), walking
// the scene graph and applying accumulated node transforms.  Returns
// (offset, count) — the slice of scene.hostTriangles this file occupies.
std::pair<int, int> loadGLTF(Scene& scene, const std::string& gltfPath);

} // namespace SceneLoader
