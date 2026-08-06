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

struct Scene {
    std::vector<Geom>     geoms;
    std::vector<Material> materials;
    RenderState state;

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
