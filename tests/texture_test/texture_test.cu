/**
 * @file texture_test.cu
 * @brief Standalone host-side test for the texture sampling functions.
 *
 * Compiles with nvcc as a pure host program (no kernel launches), like
 * rng_test.  Includes the real interactions/interactions.h and exercises
 * the actual `__host__ __device__` sampleTexture implementation from CPU
 * code — a kernel launch would test the same
 * arithmetic, just slower to debug.
 *
 * Usage:
 *   texture_test
 *   → prints one "ok: ..." / "FAIL: ..." line per check; exits 0 iff all pass.
 */

#include "interactions/interactions.h"

#include <cstdio>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

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
// resolvePbrSurfaceParams — unified metallic-roughness GGX surface params
// -----------------------------------------------------------------------
static void testResolvePbrSurfaceParams()
{
    std::printf("=== resolvePbrSurfaceParams ===\n");

    // 2×2 ORM texture (data map: G = roughness, B = metallic) + a 2×2
    // baseColor texture with a distinct albedo at each corner.
    //   ORM (slice 0):  (0,0) G=0.5 B=0.0     (1,0) G=0.5 B=1.0
    //                   (0,1) G=1.0 B=0.0     (1,1) G=0.0 B=0.5
    //   base (slice 1): (0,0) (0.8,0.2,0.1)   (1,0) (0.1,0.9,0.1)
    //                   (0,1) (0.2,0.1,0.8)   (1,1) (0.9,0.9,0.9)
    std::vector<glm::vec3> ormBuf(4);
    ormBuf[0] = glm::vec3(0.0f, 0.50f, 0.00f);
    ormBuf[1] = glm::vec3(0.0f, 0.50f, 1.00f);
    ormBuf[2] = glm::vec3(0.0f, 1.00f, 0.00f);
    ormBuf[3] = glm::vec3(0.0f, 0.0001f, 0.50f);
    std::vector<glm::vec3> baseBuf(4);
    baseBuf[0] = glm::vec3(0.8f, 0.2f, 0.1f);
    baseBuf[1] = glm::vec3(0.1f, 0.9f, 0.1f);
    baseBuf[2] = glm::vec3(0.2f, 0.1f, 0.8f);
    baseBuf[3] = glm::vec3(0.9f, 0.9f, 0.9f);

    std::vector<glm::vec3> tbl(8);      // ORM at offset 0, base at offset 4
    std::copy(ormBuf.begin(), ormBuf.end(), tbl.begin());
    std::copy(baseBuf.begin(), baseBuf.end(), tbl.begin() + 4);
    std::vector<TextureInfo> infos = { TextureInfo{ 0, 2, 2 }, TextureInfo{ 4, 2, 2 } };
    TextureTable table;
    table.pixels = tbl.data();
    table.infos  = infos.data();
    table.count  = 2;

    const glm::vec3 ALBEDO = baseBuf[0];   // the Pbr baseColor at (0,0)
    const glm::vec3 D3(0.04f);             // dielectric F0

    float r, metallic, alpha;
    glm::vec3 F0, diff;

    // ---- ORM bound: dielectric (B=0) — F0 = 0.04, full diffuse ----
    {
        Material m; m.type = MaterialType::Pbr;
        SurfaceBinding tex; tex.metallicRoughness = 0; tex.baseColor = 1;
        resolvePbrSurfaceParams(r, metallic, alpha, F0, diff, tex, table,
                                glm::vec2(0.0f, 0.0f), m, glm::vec3(1.0f));
        check(std::fabs(r - 0.5f) < 1e-6f && std::fabs(alpha - 0.25f) < 1e-6f,
              "ORM dielectric: G=0.5 → r=0.5, alpha=r²=0.25");
        check(closeTo(F0, D3) && closeTo(diff, ALBEDO),
              "ORM dielectric: B=0 → F0=0.04, diffuse=baseColor");
    }

    // ---- ORM bound: metal (B=1) — F0 = baseColor, no diffuse ----
    {
        Material m; m.type = MaterialType::Pbr;
        SurfaceBinding tex; tex.metallicRoughness = 0; tex.baseColor = 1;
        resolvePbrSurfaceParams(r, metallic, alpha, F0, diff, tex, table,
                                glm::vec2(0.5f, 0.0f), m, glm::vec3(1.0f));
        check(std::fabs(r - 0.5f) < 1e-6f, "ORM metal: G=0.5 → r=0.5");
        check(closeTo(F0, baseBuf[1]) && closeTo(diff, glm::vec3(0.0f)),
              "ORM metal: B=1 → F0=baseColor, diffuse=0");
    }

    // ---- ORM bound: r below ROUGHNESS_THRESHOLD + metallic 0.5 ----
    {
        Material m; m.type = MaterialType::Pbr;
        SurfaceBinding tex; tex.metallicRoughness = 0; tex.baseColor = 1;
        resolvePbrSurfaceParams(r, metallic, alpha, F0, diff, tex, table,
                                glm::vec2(0.5f, 0.5f), m, glm::vec3(1.0f));
        check(r < ROUGHNESS_THRESHOLD,
              "ORM G=0.0001 < ROUGHNESS_THRESHOLD → mirror path");
        check(closeTo(F0, glm::mix(D3, baseBuf[3], 0.5f)) &&
              closeTo(diff, baseBuf[3] * 0.5f),
              "ORM B=0.5 → F0=mix(0.04,base,0.5), diffuse=base·0.5");
    }

    // ---- Unbound: glTF factors govern roughness and metallic ----
    {
        Material m; m.type = MaterialType::Pbr;
        m.color = ALBEDO;
        SurfaceBinding tex;
        tex.roughnessFactor = 0.9f; tex.metallicFactor = 0.7f;
        resolvePbrSurfaceParams(r, metallic, alpha, F0, diff, tex, table,
                                glm::vec2(0.25f, 0.25f), m, glm::vec3(1.0f));
        check(std::fabs(alpha - 0.81f) < 1e-6f,
              "unbound: glTF roughnessFactor 0.9 → alpha=0.81");
        check(closeTo(F0, glm::mix(D3, ALBEDO, 0.7f)) &&
              closeTo(diff, ALBEDO * 0.3f),
              "unbound: glTF metallicFactor 0.7 → F0=mix, diffuse=base·0.3");
    }

    // ---- Type default: Reflective (legacy chrome) — metallic 1.0 ----
    {
        Material m; m.type = MaterialType::Reflective;
        m.specular.color = glm::vec3(0.9f, 0.8f, 0.7f);   // chrome tint
        SurfaceBinding unbound;                          // all slots -1
        resolvePbrSurfaceParams(r, metallic, alpha, F0, diff, unbound, table,
                                glm::vec2(0.25f, 0.25f), m, glm::vec3(1.0f));
        check(closeTo(F0, m.specular.color) && closeTo(diff, glm::vec3(0.0f)),
              "Reflective default: metallic=1 → F0=specular.color, diffuse=0 (chrome)");
    }

    // ---- Type default: Pbr — metallic 0.0, roughness 0.5, baseColor m.color ----
    {
        Material m; m.type = MaterialType::Pbr;
        m.color = ALBEDO;
        SurfaceBinding unbound;
        resolvePbrSurfaceParams(r, metallic, alpha, F0, diff, unbound, table,
                                glm::vec2(0.25f, 0.25f), m, glm::vec3(1.0f));
        check(std::fabs(r - 0.5f) < 1e-6f && std::fabs(alpha - 0.25f) < 1e-6f,
              "Pbr default: roughness 0.5, alpha 0.25 (incomplete model not a mirror)");
        check(closeTo(F0, D3) && closeTo(diff, ALBEDO),
              "Pbr default: metallic=0 → F0=0.04, diffuse=baseColor");
    }

    // ---- baseColorFactor applies when the glTF baseColor slot wins ----
    {
        Material m; m.type = MaterialType::Pbr;
        m.color = glm::vec3(1.0f, 0.0f, 0.0f);   // would win if no texture
        SurfaceBinding tex; tex.baseColor = 1;
        tex.baseColorFactor = glm::vec3(0.5f, 0.5f, 0.5f);
        float rr, mm, aa; glm::vec3 f0, dd;
        resolvePbrSurfaceParams(rr, mm, aa, f0, dd, tex, table,
                                glm::vec2(0.0f, 0.0f), m, glm::vec3(1.0f));
        check(closeTo(dd, baseBuf[0] * 0.5f),
              "glTF baseColorFactor × texture → diffuse = texel·factor");
    }

    // ---- factor-only glTF colors do not require a roughness factor ----
    {
        Material m; m.type = MaterialType::Pbr;
        m.color = glm::vec3(-1.0f);  // glTF-material sentinel
        SurfaceBinding tex;
        tex.metallicFactor = 0.0f;   // Rockstar logo's Black material
        tex.baseColorFactor = glm::vec3(0.0f);
        float rr, mm, aa; glm::vec3 f0, dd;
        resolvePbrSurfaceParams(rr, mm, aa, f0, dd, tex, table,
                                glm::vec2(0.0f), m, glm::vec3(1.0f));
        check(closeTo(dd, glm::vec3(0.0f)),
              "factor-only glTF black stays black without roughnessFactor");
    }

    std::printf("\n");
}

// -----------------------------------------------------------------------
// testPbrBrdf — Fresnel endpoints, GGX half-vector sampling, energy
// -----------------------------------------------------------------------
static void testPbrBrdf()
{
    std::printf("=== testPbrBrdf ===\n");

    const glm::vec3 D3(0.04f);

    // ---- Conductor Fresnel (Schlick) endpoints ----
    check(closeTo(fresnelSchlickF0(1.0f, D3), D3),
          "Fresnel cos=1 → F0 (0.04)");
    check(closeTo(fresnelSchlickF0(0.0f, D3), glm::vec3(1.0f)),
          "Fresnel cos=0 → 1.0 (grazing white-out)");
    // cos=0.5 → 0.04 + 0.96·(0.5)⁵ = 0.04 + 0.96·0.03125 = 0.07.
    const glm::vec3 mid = fresnelSchlickF0(0.5f, D3);
    check(std::fabs(mid.x - 0.07f) < 1e-6f,
          "Fresnel cos=0.5 → 0.04 + 0.96·(0.5)⁵ = 0.07");
    // Out-of-range cos clamps, never NaNs.
    check(closeTo(fresnelSchlickF0(5.0f, D3), D3) &&
          closeTo(fresnelSchlickF0(-5.0f, D3), glm::vec3(1.0f)),
          "Fresnel clamps cos outside [0,1]");

    // ---- Smith G1 masking-shadowing ----
    // α=0 → G1 = 2v/(v + sqrt(v²)) = 1 for any v>0 (perfectly smooth).
    check(std::fabs(smithG1Ggx(0.0f, 1.0f) - 1.0f) < 1e-5f,
          "G1(α=0, NdotV=1) → 1");
    check(std::fabs(smithG1Ggx(0.0f, 1e-3f) - 1.0f) < 1e-5f,
          "G1(α=0, grazing) → 1 (limit)");
    // Roughness increases masking: G1 drops as α grows.
    check(smithG1Ggx(0.25f, 0.5f) < smithG1Ggx(0.05f, 0.5f),
          "G1 rougher → more masking (lower G1)");
    // Grazing NdotV → G1 → 0 (a nearly-tangent ray is almost surely shadowed).
    check(smithG1Ggx(0.25f, 1e-3f) < 1e-2f,
          "G1 grazing NdotV → ≈ 0");

    // ---- GGX half-vector sampling ----
    // α=0: cosθh² = (1−ξ)/(1+(0−1)ξ) = 1 ⇒ H == N exactly (perfect mirror H).
    {
        const glm::vec3 N(0.0f, 0.0f, 1.0f);
        RngState rng = makeRngState(0, 0, 0, RngMode::LCG);
        const glm::vec3 H = sampleGgxHalfVector(N, 0.0f, rng);
        check(glm::dot(H, N) > 0.99999f,
              "α=0 half-vector → H ≈ N (mirror collapse)");
    }
    // α=1: H stays in the N hemisphere and unit length (cosine-ish spread).
    {
        const glm::vec3 N(0.0f, 0.0f, 1.0f);
        RngState rng = makeRngState(1, 0, 0, RngMode::LCG);
        const glm::vec3 H = sampleGgxHalfVector(N, 1.0f, rng);
        check(glm::dot(H, N) > 0.0f && std::fabs(glm::length(H) - 1.0f) < 1e-6f,
              "α=1 half-vector → in hemisphere, unit length");
    }

    // ---- Directional-hemispherical energy (compensated) ----
    // Per-bounce reflected energy = ∫ f_spec·(N·L) dω + diffuseCompensation,
    // where diffuseCompensation = albedo·(1−metallic)·(1−lum(F_view)).  A naive
    // additive model (compensation 1.0) sums to ≈2 for dielectrics; the
    // compensated model stays ≤ 1 + tolerance.  Numerically integrate the
    // specular lobe f·(N·L) over a spherical grid around the view direction.
    {
        const glm::vec3 N(0.0f, 0.0f, 1.0f);
        const glm::vec3 wo(0.0f, 0.0f, -1.0f);      // view straight down
        const float NdotV = glm::clamp(glm::dot(N, -wo), 1e-4f, 1.0f);
        const float alpha = 0.1f;                    // fairly smooth dielectric

        // Integrate over the HALF-VECTOR, not wi: the change of variables
        // dω_l = 4·(V·H)·dω_h is what GGX half-vector sampling is built on, and
        // it sidesteps the catastrophic cancellation of normalize(wo+wi) near
        // the mirror direction (wo+wi → 0 at the specular peak).
        //   f_spec·(N·L)·dω_l = F·G·D·(V·H)/(N·V) · dω_h
        // with V=wo=(0,0,-1):  V·H = N·H = cosθh, N·V = 1, and
        //   wi = reflect(wo,H) ⇒ N·L = 2·cosθh² − 1  (>0 only for θh < 45°).
        float specSum = 0.0f;
        const int NTH = 200, NPH = 400;              // fine enough for <0.1%
        for (int i = 0; i < NTH; i++)
        {
            const float th    = (i + 0.5f) / NTH * (PI * 0.5f);   // θh
            const float sinTh = sinf(th), cosTh = cosf(th);
            for (int j = 0; j < NPH; j++)
            {
                const float ph    = (j + 0.5f) / NPH * TWO_PI;
                const float NdotL = 2.0f * cosTh * cosTh - 1.0f;   // reflect(wo,H)·N
                if (NdotL <= 0.0f) continue;                       // outside the lobe
                const float G = smithG1Ggx(alpha, NdotV) * smithG1Ggx(alpha, NdotL);
                const float D = ggxD(alpha, cosTh);                // N·H = cosθh
                const float F = fresnelSchlickF0(cosTh, D3).r;   // V·H = cosθh
                specSum += (F * G * D * cosTh / NdotV)
                           * sinTh * (PI * 0.5f / NTH) * (TWO_PI / NPH);
            }
        }

        // F_view at NdotV=1 → Fresnel(1) = 0.04; compensated diffuse energy
        // for albedo=1, metallic=0 is (1 − 0.04) = 0.96.
        const float diffEnergy = 1.0f * (1.0f - luminance(fresnelSchlickF0(1.0f, D3)));
        const float total = specSum + diffEnergy;
        check(total <= 1.0f + 2e-2f,
              "compensated GGX + diffuse energy ≤ 1.02");
        check(specSum < 0.1f,
              "dielectric specular directional albedo < 0.1 at normal incidence");

        // Discriminator: the naive additive model (uncompensated diffuse = 1.0)
        // would exceed the bound — prove the compensation is what keeps ≤ 1.02.
        check(specSum + 1.0f > 1.0f + 2e-2f,
              "naive (uncompensated) model would exceed the bound");
    }

    std::printf("\n");
}

// -----------------------------------------------------------------------
// evaluateBsdf — finite continuous-result contract for NEE/MIS callers
// -----------------------------------------------------------------------
static void testEvaluateBsdfFiniteContract()
{
    std::printf("=== evaluateBsdf finite contract ===\n");

    Material material{};
    material.type = MaterialType::Pbr;
    material.color = glm::vec3(0.8f);

    SurfaceBinding binding{};
    binding.roughnessFactor = 0.4f;
    binding.metallicFactor = 0.0f;

    ShadeableIntersection hit{};
    hit.surfaceNormal = glm::vec3(0.0f, 0.0f, 1.0f);
    hit.vertexColor = glm::vec3(1.0f);
    hit.surface = &binding;

    const BsdfEvaluation valid = evaluateBsdf(
        hit, material, TextureTable{}, glm::vec3(0.0f, 0.0f, -1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f));
    check(!valid.isDelta && std::isfinite(valid.pdfOmega) && valid.pdfOmega > 0.0f &&
          std::isfinite(valid.value.x) && std::isfinite(valid.value.y) &&
          std::isfinite(valid.value.z),
          "valid rough PBR returns a finite continuous BSDF");

    // The shading kernel resolves texture-backed state once, then evaluates it
    // for both the sampled light direction and the continuation direction.
    // Keep that path exactly equivalent to the compatibility evaluator.
    const ResolvedBsdf resolved = resolveBsdf(
        hit, material, TextureTable{}, glm::vec3(0.0f, 0.0f, -1.0f));
    const BsdfEvaluation reused = evaluateBsdf(
        resolved, material, glm::vec3(0.0f, 0.0f, -1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f));
    check(!reused.isDelta && fabsf(reused.pdfOmega - valid.pdfOmega) < 1e-6f &&
          glm::length(reused.value - valid.value) < 1e-6f &&
          glm::length(reused.shadingNormal - valid.shadingNormal) < 1e-6f,
          "resolved BSDF evaluation matches compatibility path");

    // Exercise the exact reuse path with both baseColor and ORM slots bound:
    // this is the expensive Rockstar-style case the compact state is meant to
    // avoid re-sampling for direct light, GGX scatter, and continuation PDF.
    glm::vec3 texturePixels[] = {
        glm::vec3(0.7f, 0.2f, 0.1f),
        glm::vec3(0.0f, 0.35f, 0.6f)
    };
    TextureInfo textureInfos[] = { TextureInfo{ 0, 1, 1 }, TextureInfo{ 1, 1, 1 } };
    const TextureTable texturedTable{ texturePixels, textureInfos, 2 };
    binding.baseColor = 0;
    binding.metallicRoughness = 1;
    binding.roughnessFactor = 0.8f;
    binding.metallicFactor = 0.5f;
    const BsdfEvaluation texturedCompatibility = evaluateBsdf(
        hit, material, texturedTable, glm::vec3(0.0f, 0.0f, -1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f));
    const ResolvedBsdf texturedResolved = resolveBsdf(
        hit, material, texturedTable, glm::vec3(0.0f, 0.0f, -1.0f));
    const BsdfEvaluation texturedReused = evaluateBsdf(
        texturedResolved, material, glm::vec3(0.0f, 0.0f, -1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f));
    check(fabsf(texturedReused.pdfOmega - texturedCompatibility.pdfOmega) < 1e-6f &&
          glm::length(texturedReused.value - texturedCompatibility.value) < 1e-6f,
          "textured resolved BSDF matches compatibility path");

    // This models a corrupt PBR texture sample.  The public evaluator must
    // preserve its no-continuous-contribution contract for its NEE/MIS users
    // instead of exposing the non-finite intermediate result to them.
    binding = SurfaceBinding{};
    binding.roughnessFactor = 0.4f;
    binding.metallicFactor = 0.0f;
    material.color = glm::vec3(std::numeric_limits<float>::quiet_NaN());
    const BsdfEvaluation invalid = evaluateBsdf(
        hit, material, TextureTable{}, glm::vec3(0.0f, 0.0f, -1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f));
    check(!invalid.isDelta && invalid.pdfOmega == 0.0f &&
          invalid.value.x == 0.0f && invalid.value.y == 0.0f &&
          invalid.value.z == 0.0f,
          "non-finite PBR evaluation returns zero continuous contribution");

    std::printf("\n");
}

int main()
{
    testSampleTexture();
    testResolvePbrSurfaceParams();
    testPbrBrdf();
    testEvaluateBsdfFiniteContract();

    if (g_failures == 0)
        std::printf("ALL PASS\n");
    else
        std::printf("%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
