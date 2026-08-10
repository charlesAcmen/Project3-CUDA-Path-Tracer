#pragma once

// ====================================================================
// Scene Data Container
//
// Pure data — no parsing, no file I/O.  Populated by SceneLoader or
// constructed programmatically for testing.
//
// Mesh triangles live in a flat hostTriangles vector (OBJECT space — the
// source geometry); each Geom references a slice via meshTriangleOffset /
// ::count.  buildSceneBvh (src/bvh/bvh.cu) bakes them to world space and
// reorders them into its own BvhBuffers::hostTriangles.
// ====================================================================

#include "sceneStructs.h"
#include <vector>

// A texture image loaded from disk (host side).  `pixels` holds width*height
// LINEAR-RGB texels, row-major.  Referenced by Material::textureId (>= 0).
struct TextureData
{
    std::vector<glm::vec3> pixels;   // w*h linear-RGB texels
    int width  = 0;
    int height = 0;
};

struct Scene {
    std::vector<Geom>     geoms;
    std::vector<Material> materials;
    RenderState state;

    // Texture images referenced by Material::textureId (>= 0).  Empty for
    // scenes with no textures.
    std::vector<TextureData> textures;

    // Flat array of all mesh triangles (object-space).  Each mesh
    // geometry references a contiguous slice via
    // Geom::meshTriangleOffset / ::meshTriangleCount.
    std::vector<Triangle> hostTriangles;
};

// ---- Scene Statistics -------------------------------------------------
// Lightweight descriptor of scene complexity.  Useful for startup
// summary, ImGui overlay, and Profiler CSV metadata.

struct SceneStats {
    int numObjects    = 0;
    int numMeshes     = 0;
    int numMaterials  = 0;
    int numTriangles  = 0;
};

SceneStats computeSceneStats(const Scene& scene);
