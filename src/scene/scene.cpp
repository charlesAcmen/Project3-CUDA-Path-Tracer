#include "scene/scene.h"

// ====================================================================
// Scene Statistics
// ====================================================================

SceneStats computeSceneStats(const Scene& scene)
{
    SceneStats s;
    s.numObjects   = (int)scene.geoms.size();
    
    // Count actual meshes (geometries with triangle data).
    s.numMeshes = 0;
    for (const auto& geom : scene.geoms) {
        if (geom.meshTriangleCount > 0) {
            s.numMeshes++;
        }
    }
    
    s.numMaterials = (int)scene.materials.size();
    s.numTriangles = (int)scene.hostTriangles.size();
    return s;
}
