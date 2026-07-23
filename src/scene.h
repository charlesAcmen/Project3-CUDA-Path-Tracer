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

    // Flat array of all mesh triangles (object-space).  Each mesh geometry
    // references a contiguous slice via Geom::meshTriangleOffset / ::count.
    std::vector<Triangle> hostTriangles;
};
