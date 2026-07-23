#pragma once

// ====================================================================
// Scene Data Container
//
// Pure data — no parsing, no file I/O.  Populated by SceneLoader or
// constructed programmatically for testing.
//
// Mesh triangles live in a flat hostTriangles vector; each Geom with
// type == MESH references a slice via meshTriangleOffset / ::count.
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
    int numSpheres    = 0;
    int numCubes      = 0;
    int numMaterials  = 0;
    int numTriangles  = 0;
};

SceneStats computeSceneStats(const Scene& scene);
