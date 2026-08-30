// ====================================================================
// light_sampling_test — host-side validation of emissive triangle sampling.
//
// Compiles the production host light-table builder directly.  No kernel is
// launched and no GPU is required.
// ====================================================================

#include <cmath>
#include <cstdio>
#include <vector>

#include "lighting/light_sampling.h"
#include "lighting/light_sampling.cpp"

namespace {

TrianglePos makeTriangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c)
{
    return TrianglePos{ a, b, c };
}

bool testLightTable()
{
    std::vector<TrianglePos> positions = {
        makeTriangle(glm::vec3(0), glm::vec3(2, 0, 0), glm::vec3(0, 1, 0)),
        makeTriangle(glm::vec3(0), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0)),
        makeTriangle(glm::vec3(0), glm::vec3(0), glm::vec3(0))
    };
    std::vector<TriangleAttr> attrs(positions.size());
    attrs[0].surfaceId = 0;
    attrs[1].surfaceId = 1;
    attrs[2].surfaceId = 2;

    Material bright{};
    bright.color = glm::vec3(4.0f);
    bright.emittance = 2.0f;
    Material dim{};
    dim.color = glm::vec3(1.0f);
    dim.emittance = 1.0f;
    // Near-black JPEG emissive residue must not turn an entire mesh into
    // explicit light candidates. It remains valid emission-hit transport.
    Material nearBlack{};
    nearBlack.color = glm::vec3(1.0f);
    nearBlack.emittance = 1e-4f;
    std::vector<Material> materials = { bright, dim, nearBlack };
    std::vector<Surface> surfaces = {
        Surface{ 0, -1 }, Surface{ 1, -1 }, Surface{ 2, -1 }
    };

    const HostLightSampling lights = buildLightSampling(
        positions, attrs, surfaces, materials, {}, {});
    if (lights.triangles.size() != 2 || lights.aliasEntries.size() != 2 ||
        lights.lightIndexByTriangle.size() != positions.size())
    {
        printf("[FAIL] light table shape\n");
        return false;
    }
    if (lights.lightIndexByTriangle[0] != 0 || lights.lightIndexByTriangle[1] != 1 ||
        lights.lightIndexByTriangle[2] != -1)
    {
        printf("[FAIL] light triangle reverse lookup\n");
        return false;
    }

    float pmfSum = 0.0f;
    for (const LightTriangle& light : lights.triangles) pmfSum += light.selectPmf;
    if (fabsf(pmfSum - 1.0f) > 1e-6f ||
        fabsf(lights.triangles[0].area - 1.0f) > 1e-6f ||
        fabsf(lights.triangles[1].area - 0.5f) > 1e-6f)
    {
        printf("[FAIL] light table area or PMF\n");
        return false;
    }
    printf("[PASS] light table\n");
    return true;
}

bool testAliasDistribution()
{
    LightTriangle triangles[2] = {
        LightTriangle{ 7, 1.0f, 0.75f },
        LightTriangle{ 9, 1.0f, 0.25f }
    };
    LightAliasEntry aliases[2] = {
        LightAliasEntry{ 1.0f, 0 }, LightAliasEntry{ 0.5f, 0 }
    };
    const LightSamplingView view{ triangles, aliases, nullptr, 2 };
    int counts[2] = { 0, 0 };
    unsigned int state = 0x12345678u;
    constexpr int samples = 100000;
    for (int i = 0; i < samples; ++i)
    {
        state = state * 1664525u + 1013904223u;
        const float u = static_cast<float>(state & 0x00ffffffu) / 16777216.0f;
        const int index = sampleLightTriangle(view, u);
        if (index < 0 || index >= 2)
        {
            printf("[FAIL] alias index range\n");
            return false;
        }
        counts[index]++;
    }
    const float p0 = static_cast<float>(counts[0]) / samples;
    if (fabsf(p0 - 0.75f) > 0.01f)
    {
        printf("[FAIL] alias distribution %.6f\n", p0);
        return false;
    }
    printf("[PASS] alias distribution\n");
    return true;
}

} // namespace

int main()
{
    int failures = 0;
    if (!testLightTable()) failures++;
    if (!testAliasDistribution()) failures++;

    if (failures == 0)
    {
        printf("ALL PASS\n");
        return 0;
    }
    printf("%d FAILURE(S)\n", failures);
    return 1;
}
