#pragma once

// ====================================================================
// Emissive Triangle Sampling
//
// Host construction scans the BVH-flattened world-space triangle arrays and
// builds a Walker/Vose alias table.  Device shading reads the resulting
// LightSamplingView without touching the BVH traversal layout.
// ====================================================================

#include "scene/scene.h"

#include <vector>

struct HostLightSampling
{
    std::vector<LightTriangle>  triangles;
    std::vector<LightAliasEntry> aliasEntries;
    std::vector<int>            lightIndexByTriangle;
};

// Build a world-space emissive-triangle sampler after BVH flattening.  The
// source texture data is used only for an importance estimate; device shading
// still evaluates emitted radiance at the sampled UV.
HostLightSampling buildLightSampling(
    const std::vector<TrianglePos>& trianglePositions,
    const std::vector<TriangleAttr>& triangleAttrs,
    const std::vector<Surface>& surfaces,
    const std::vector<Material>& materials,
    const std::vector<SurfaceBinding>& surfaceBindings,
    const std::vector<TextureData>& textures);

// One uniform sample chooses an alias-table column and its in-column branch.
// u is expected in [0, 1); an empty view returns -1.
__host__ __device__ inline int sampleLightTriangle(
    const LightSamplingView& lights, float u)
{
    if (lights.count <= 0 || lights.aliasEntries == nullptr) return -1;

    const float scaled = u * static_cast<float>(lights.count);
    int column = static_cast<int>(scaled);
    if (column >= lights.count) column = lights.count - 1;
    const float fractional = scaled - static_cast<float>(column);
    const LightAliasEntry entry = lights.aliasEntries[column];
    return (fractional < entry.q) ? column : entry.alias;
}
