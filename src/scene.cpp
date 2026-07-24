#include "scene.h"

// ====================================================================
// Scene Statistics
// ====================================================================

SceneStats computeSceneStats(const Scene& scene)
{
    SceneStats s;
    s.numObjects   = (int)scene.geoms.size();
    s.numMeshes    = (int)scene.geoms.size();
    s.numMaterials = (int)scene.materials.size();
    s.numTriangles = (int)scene.hostTriangles.size();
    return s;
}
