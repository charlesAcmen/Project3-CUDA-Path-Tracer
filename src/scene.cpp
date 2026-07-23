#include "scene.h"

// ====================================================================
// Scene Statistics
// ====================================================================

SceneStats computeSceneStats(const Scene& scene)
{
    SceneStats s;
    s.numObjects   = (int)scene.geoms.size();
    s.numMaterials = (int)scene.materials.size();
    s.numTriangles = (int)scene.hostTriangles.size();

    for (const auto& g : scene.geoms)
    {
        if      (g.type == MESH)   s.numMeshes++;
        else if (g.type == SPHERE) s.numSpheres++;
        else if (g.type == CUBE)   s.numCubes++;
    }
    return s;
}
