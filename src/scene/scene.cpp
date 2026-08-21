#include "scene/scene.h"

// ====================================================================
// Scene Statistics
// ====================================================================

int internSurfaceBinding(Scene& scene, const SurfaceBinding& binding)
{
    for (size_t i = 0; i < scene.surfaceBindings.size(); ++i)
    {
        const SurfaceBinding& candidate = scene.surfaceBindings[i];
        if (candidate.baseColor == binding.baseColor &&
            candidate.normal == binding.normal &&
            candidate.metallicRoughness == binding.metallicRoughness &&
            candidate.occlusion == binding.occlusion &&
            candidate.emissive == binding.emissive &&
            candidate.roughnessFactor == binding.roughnessFactor &&
            candidate.metallicFactor == binding.metallicFactor &&
            candidate.baseColorFactor == binding.baseColorFactor &&
            candidate.emissiveFactor == binding.emissiveFactor &&
            candidate.emissiveStrength == binding.emissiveStrength)
        {
            return (int)i;
        }
    }

    scene.surfaceBindings.push_back(binding);
    return (int)scene.surfaceBindings.size() - 1;
}

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
    s.numTriangles = (int)scene.hostTrianglePositions.size();
    return s;
}
