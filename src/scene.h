#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);

    // Load a single OBJ file and append its triangles into hostTriangles.
    // Returns (offset, count) into hostTriangles for the loaded mesh.
    // objPath is resolved relative to the JSON scene file's directory.
    std::pair<int, int> loadOBJ(const std::string& objPath);

public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    // Flat array of all mesh triangles (object-space).  Each mesh geometry
    // references a contiguous slice via Geom::meshTriangleOffset / ::count.
    std::vector<Triangle> hostTriangles;
};
