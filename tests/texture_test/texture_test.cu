/**
 * @file texture_test.cu
 * @brief Standalone host-side test for the texture sampling functions.
 *
 * Compiles with nvcc as a pure host program (no kernel launches), like
 * rng_test.  Includes the real interactions/interactions.h and exercises
 * the actual `__host__ __device__` sampleTexture / sampleCheckerboard
 * implementations from CPU code — a kernel launch would test the same
 * arithmetic, just slower to debug.
 *
 * Usage:
 *   texture_test
 *   → prints one "ok: ..." / "FAIL: ..." line per check; exits 0 iff all pass.
 */

#include "interactions/interactions.h"

#include <cstdio>
#include <cmath>
#include <vector>

// -----------------------------------------------------------------------
// Check harness — same shape as loader_test / bvh_test
// -----------------------------------------------------------------------
static int g_failures = 0;

static void check(bool ok, const char* what)
{
    std::printf("  %s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) g_failures++;
}

static bool closeTo(const glm::vec3& a, const glm::vec3& b, float eps = 1e-4f)
{
    return std::fabs(a.x - b.x) < eps && std::fabs(a.y - b.y) < eps &&
           std::fabs(a.z - b.z) < eps;
}

// -----------------------------------------------------------------------
// sampleTexture — 2×2 image with four distinct texels
// -----------------------------------------------------------------------
static void testSampleTexture()
{
    std::printf("=== sampleTexture ===\n");

    // 8 texels: the same 2×2 image stored at BOTH pixelOffset 0 and 4.
    // A second TextureInfo points at offset 4 to prove the offset math —
    // both tables must sample identically.
    //    (0,0) red    (1,0) green
    //    (0,1) blue   (1,1) yellow
    std::vector<glm::vec3> buf(8);
    for (int copy = 0; copy < 2; copy++)
    {
        const int o = copy * 4;
        buf[o + 0] = glm::vec3(1.0f, 0.0f, 0.0f);
        buf[o + 1] = glm::vec3(0.0f, 1.0f, 0.0f);
        buf[o + 2] = glm::vec3(0.0f, 0.0f, 1.0f);
        buf[o + 3] = glm::vec3(1.0f, 1.0f, 0.0f);
    }

    const TextureInfo ti0{ 0, 2, 2 };
    const TextureInfo ti4{ 4, 2, 2 };

    const glm::vec3 RED(1, 0, 0), GREEN(0, 1, 0), BLUE(0, 0, 1), YELLOW(1, 1, 0);

    // Exact corner → that texel (fx = fy = 0 collapses the blend).
    check(closeTo(sampleTexture(buf.data(), ti0, glm::vec2(0.0f, 0.0f)), RED),
          "(0,0) corner → red texel");
    check(closeTo(sampleTexture(buf.data(), ti0, glm::vec2(0.5f, 0.0f)), GREEN),
          "(0.5,0) corner → green texel (x1 = width handled)");
    check(closeTo(sampleTexture(buf.data(), ti0, glm::vec2(0.0f, 0.5f)), BLUE),
          "(0,0.5) corner → blue texel");
    check(closeTo(sampleTexture(buf.data(), ti0, glm::vec2(0.5f, 0.5f)), YELLOW),
          "(0.5,0.5) corner → yellow texel");

    // Center of the 2×2 grid: uv = (0.25, 0.25) is halfway between all four
    // texel corners → exact average (1+0+0+1, 0+1+0+1, 0+0+1+0)/4.
    check(closeTo(sampleTexture(buf.data(), ti0, glm::vec2(0.25f, 0.25f)),
                  glm::vec3(0.5f, 0.5f, 0.25f)),
          "(0.25,0.25) center → average of all four texels");

    // Repeat-wrap: (1.5, 0.25) → u folds to 0.5 → same as (0.5, 0.25).
    const glm::vec3 wrapped = sampleTexture(buf.data(), ti0, glm::vec2(0.5f, 0.25f));
    check(closeTo(sampleTexture(buf.data(), ti0, glm::vec2(1.5f, 0.25f)), wrapped),
          "(1.5,0.25) wraps to (0.5,0.25)");
    // Negative wraps the same way: -0.5 → 0.5.
    check(closeTo(sampleTexture(buf.data(), ti0, glm::vec2(-0.5f, 0.25f)), wrapped),
          "(-0.5,0.25) wraps to (0.5,0.25)");

    // pixelOffset: identical image at a non-zero offset must sample identically.
    check(closeTo(sampleTexture(buf.data(), ti4, glm::vec2(0.25f, 0.25f)),
                  glm::vec3(0.5f, 0.5f, 0.25f)),
          "pixelOffset=4 → same samples as offset 0");

    std::printf("\n");
}

// -----------------------------------------------------------------------
// sampleCheckerboard — 8×8 grid derived from the base color
// -----------------------------------------------------------------------
static void testSampleCheckerboard()
{
    std::printf("=== sampleCheckerboard ===\n");

    const glm::vec3 base(0.8f, 0.6f, 0.4f);
    const glm::vec3 dark = base * 0.25f;

    // (0,0) is an even cell (u+v = 0) → base.
    check(closeTo(sampleCheckerboard(glm::vec2(0.0f, 0.0f), base), base),
          "(0,0) even cell → base");
    // Adjacent cells flip parity in each axis.
    check(closeTo(sampleCheckerboard(glm::vec2(0.125f, 0.0f), base), dark),
          "(0.125,0) u+1 → dark");
    check(closeTo(sampleCheckerboard(glm::vec2(0.0f, 0.125f), base), dark),
          "(0,0.125) v+1 → dark");
    // Diagonal is same parity as the origin.
    check(closeTo(sampleCheckerboard(glm::vec2(0.125f, 0.125f), base), base),
          "(0.125,0.125) diagonal → base");
    // Last cell (u=7) is dark, so the 8-cell grid ends dark.
    check(closeTo(sampleCheckerboard(glm::vec2(0.9375f, 0.0f), base), dark),
          "(0.9375,0) u=7 last cell → dark");

    // Period 8 in uv: uv=1.125 is cell u=9, same parity as u=1.
    check(closeTo(sampleCheckerboard(glm::vec2(1.125f, 0.0f), base), dark),
          "(1.125,0) wraps to u=9 → dark (period 8)");

    // Negative uv keeps consistent parity (C++ % preserves parity under
    // truncate-toward-zero): (-0.0625,0) → u=-1 → dark; add v=1 → even → base.
    check(closeTo(sampleCheckerboard(glm::vec2(-0.0625f, 0.0f), base), dark),
          "(-0.0625,0) u=-1 → dark");
    check(closeTo(sampleCheckerboard(glm::vec2(-0.0625f, 0.125f), base), base),
          "(-0.0625,0.125) u+v=0 → base");

    std::printf("\n");
}

int main()
{
    testSampleTexture();
    testSampleCheckerboard();

    if (g_failures == 0)
        std::printf("ALL PASS\n");
    else
        std::printf("%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
