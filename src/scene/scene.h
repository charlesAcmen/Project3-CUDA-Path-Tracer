#pragma once

// ====================================================================
// Scene Data Container
//
// Pure data — no parsing, no file I/O.  Populated by SceneLoader or
// constructed programmatically for testing.
//
// Mesh geometry lives in parallel object-space position / attribute arrays;
// each Geom references the same slice in both via meshTriangleOffset / ::count.
// buildSceneBvh (src/bvh/bvh.cu) bakes and reorders those arrays directly.
// ====================================================================

#include "sceneStructs.h"
#include <vector>

// A texture image loaded from disk (host side).  `pixels` holds width*height
// LINEAR-RGB texels, row-major.  Referenced by SurfaceBinding texture slots.
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

    // Texture images referenced by SurfaceBinding slots.  Empty for scenes
    // with no textures.
    std::vector<TextureData> textures;

    // Source primitive / OBJ-face surface inputs.  TriangleAttr stores only
    // an index into this table, so the same glTF material binding is not
    // copied once per triangle.  The loader interns equal bindings in source order.
    //(intern:驻留一份在内存，避免重复存储；in source order:按照源文件顺序)
    std::vector<SurfaceBinding> surfaceBindings;

    // Flat object-space triangle storage.  Corresponding entries in the two
    // arrays form one source triangle; each mesh references a contiguous
    // slice via Geom::meshTriangleOffset / ::meshTriangleCount.
    std::vector<TrianglePos>  hostTrianglePositions;
    std::vector<TriangleAttr> hostTriangleAttrs;
};

// Return the stable scene-local id for a source surface binding.  Exact
// equality is intentional: all fields originate from parsed asset values and
// are copied unchanged; approximate matching could merge distinct materials.
int internSurfaceBinding(Scene& scene, const SurfaceBinding& binding);

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
