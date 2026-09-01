#include "lighting/light_sampling.h"

#include <cmath>
#include <limits>

namespace {

// Below this mean linear luminance, an emissive surface remains visible to
// BSDF paths but is not worth selecting as an explicit NEE light. This filters
// compression residue in nominally black emissive textures without bias: an
// omitted triangle has pLight=0 and its BSDF-hit emission keeps MIS weight 1.
constexpr float kMinimumDirectLightPower = 1e-3f;

float luminance(const glm::vec3& c)
{
    return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
}

glm::vec3 meanTextureColor(const std::vector<TextureData>& textures, int textureIndex)
{
    if (textureIndex < 0 || textureIndex >= static_cast<int>(textures.size()))
        return glm::vec3(1.0f);

    const std::vector<glm::vec3>& pixels = textures[textureIndex].pixels;
    if (pixels.empty()) return glm::vec3(1.0f);

    glm::vec3 sum(0.0f);
    for (const glm::vec3& pixel : pixels) sum += pixel;
    return sum / static_cast<float>(pixels.size());
}

glm::vec3 estimateEmittedRadiance(
    const Material& material,
    const SurfaceBinding& binding,
    const glm::vec3& meanEmissiveTextureColor)
{
    glm::vec3 emitted(0.0f);
    if (binding.emissiveFactor != glm::vec3(0.0f))
    {
        emitted = meanEmissiveTextureColor *
                  binding.emissiveFactor * binding.emissiveStrength;
    }
    else if (material.emittance > 0.0f)
    {
        emitted = material.color;
    }

    return (material.emittance > 0.0f) ? emitted * material.emittance : emitted;
}

void buildAliasTable(const std::vector<float>& pmf, std::vector<LightAliasEntry>& out)
{
    const int count = static_cast<int>(pmf.size());
    out.assign(count, LightAliasEntry{});
    if (count == 0) return;

    std::vector<float> scaled(count);
    std::vector<int> small;
    std::vector<int> large;
    small.reserve(count);
    large.reserve(count);

    for (int i = 0; i < count; ++i)
    {
        scaled[i] = pmf[i] * static_cast<float>(count);
        if (scaled[i] < 1.0f) small.push_back(i);
        else large.push_back(i);
    }

    while (!small.empty() && !large.empty())
    {
        const int low = small.back(); small.pop_back();
        const int high = large.back(); large.pop_back();
        out[low].q = scaled[low];
        out[low].alias = high;
        scaled[high] = (scaled[high] + scaled[low]) - 1.0f;
        if (scaled[high] < 1.0f) small.push_back(high);
        else large.push_back(high);
    }

    for (int i : large) out[i] = LightAliasEntry{ 1.0f, i };
    for (int i : small) out[i] = LightAliasEntry{ 1.0f, i };
}

} // namespace

HostLightSampling buildLightSampling(
    const std::vector<TrianglePos>& trianglePositions,
    const std::vector<TriangleAttr>& triangleAttrs,
    const std::vector<Surface>& surfaces,
    const std::vector<Material>& materials,
    const std::vector<SurfaceBinding>& surfaceBindings,
    const std::vector<TextureData>& textures)
{
    HostLightSampling result;
    const int triangleCount = static_cast<int>(trianglePositions.size());
    result.lightIndexByTriangle.assign(triangleCount, -1);
    if (triangleAttrs.size() != trianglePositions.size()) return result;

    std::vector<float> weights;
    weights.reserve(triangleCount);

    // A mesh commonly has thousands of triangles sharing one emissive map.
    // Cache each map's mean by texture id: scanning it per triangle turns a
    // 2048² texture on a 15k-triangle mesh into tens of billions of host reads.
    std::vector<glm::vec3> meanTextureCache(textures.size());
    std::vector<unsigned char> meanTextureCached(textures.size(), 0);
    const auto cachedMeanTextureColor = [&](int textureIndex) {
        if (textureIndex < 0 || textureIndex >= static_cast<int>(textures.size()))
            return glm::vec3(1.0f);
        if (!meanTextureCached[textureIndex])
        {
            meanTextureCache[textureIndex] = meanTextureColor(textures, textureIndex);
            meanTextureCached[textureIndex] = 1;
        }
        return meanTextureCache[textureIndex];
    };

    const SurfaceBinding emptyBinding{};
    for (int triangleIndex = 0; triangleIndex < triangleCount; ++triangleIndex)
    {
        const TriangleAttr& attr = triangleAttrs[triangleIndex];
        if (attr.surfaceId < 0 || attr.surfaceId >= static_cast<int>(surfaces.size()))
            continue;

        const Surface& surface = surfaces[attr.surfaceId];
        if (surface.materialId < 0 || surface.materialId >= static_cast<int>(materials.size()))
            continue;

        const SurfaceBinding& binding =
            (surface.surfaceBindingId >= 0 &&
             surface.surfaceBindingId < static_cast<int>(surfaceBindings.size()))
            ? surfaceBindings[surface.surfaceBindingId] : emptyBinding;
        const glm::vec3 meanEmissiveTextureColor =
            binding.emissiveFactor != glm::vec3(0.0f)
            ? cachedMeanTextureColor(binding.emissive) : glm::vec3(1.0f);
        const glm::vec3 emitted = estimateEmittedRadiance(
            materials[surface.materialId], binding, meanEmissiveTextureColor);
        const float power = luminance(emitted);
        if (!(power > kMinimumDirectLightPower) || !std::isfinite(power))
            continue;

        const TrianglePos& triangle = trianglePositions[triangleIndex];
        const float area = 0.5f * glm::length(glm::cross(
            triangle.v1 - triangle.v0, triangle.v2 - triangle.v0));
        if (!(area > 0.0f) || !std::isfinite(area)) continue;

        result.lightIndexByTriangle[triangleIndex] =
            static_cast<int>(result.triangles.size());
        result.triangles.push_back(LightTriangle{ triangleIndex, area, 0.0f });
        weights.push_back(area * power);
    }

    float totalWeight = 0.0f;
    for (float weight : weights) totalWeight += weight;
    if (!(totalWeight > 0.0f) || !std::isfinite(totalWeight))
    {
        result.triangles.clear();
        result.lightIndexByTriangle.assign(triangleCount, -1);
        return result;
    }

    std::vector<float> pmf(weights.size());
    for (size_t i = 0; i < weights.size(); ++i)
    {
        pmf[i] = weights[i] / totalWeight;
        result.triangles[i].selectPmf = pmf[i];
    }
    buildAliasTable(pmf, result.aliasEntries);
    return result;
}
