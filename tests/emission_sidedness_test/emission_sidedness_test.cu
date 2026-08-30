// ====================================================================
// emission_sidedness_test — host-side contract test for directed emission.
//
// No kernel launches: this verifies the shared helper used by both NEE and
// BSDF-hit emission, so front/back semantics cannot drift between them.
// ====================================================================

#include "interactions/interactions.h"

#include <cmath>
#include <cstdio>

namespace {

bool closeTo(const glm::vec3& a, const glm::vec3& b, float epsilon = 1e-6f)
{
    return std::fabs(a.x - b.x) < epsilon &&
           std::fabs(a.y - b.y) < epsilon &&
           std::fabs(a.z - b.z) < epsilon;
}

bool testEmissionSidedness()
{
    Material emitter{};
    emitter.color = glm::vec3(0.25f, 0.5f, 0.75f);
    emitter.emittance = 4.0f;
    const SurfaceBinding binding{};
    const TextureTable textures{};
    const glm::vec3 normal(0.0f, 0.0f, 1.0f);
    const glm::vec3 front(0.0f, 0.0f, 1.0f);
    const glm::vec3 back(0.0f, 0.0f, -1.0f);
    const glm::vec3 expected = emitter.color * emitter.emittance;

    if (emissionCosine(emitter, normal, front) != 1.0f ||
        emissionCosine(emitter, normal, back) != 1.0f ||
        !closeTo(evaluateEmittedRadiance(binding, textures, glm::vec2(0.0f),
                                          emitter, normal, front), expected) ||
        !closeTo(evaluateEmittedRadiance(binding, textures, glm::vec2(0.0f),
                                          emitter, normal, back), expected))
    {
        std::printf("[FAIL] legacy two-sided emission\n");
        return false;
    }

    emitter.emissionSidedness = EmissionSidedness::OneSided;
    if (emissionCosine(emitter, normal, front) != 1.0f ||
        emissionCosine(emitter, normal, back) != 0.0f ||
        !closeTo(evaluateEmittedRadiance(binding, textures, glm::vec2(0.0f),
                                          emitter, normal, front), expected) ||
        !closeTo(evaluateEmittedRadiance(binding, textures, glm::vec2(0.0f),
                                          emitter, normal, back), glm::vec3(0.0f)))
    {
        std::printf("[FAIL] one-sided emission\n");
        return false;
    }

    std::printf("[PASS] emission sidedness\n");
    return true;
}

} // namespace

int main()
{
    return testEmissionSidedness() ? 0 : 1;
}
