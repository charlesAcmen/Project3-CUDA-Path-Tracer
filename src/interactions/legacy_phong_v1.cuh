#pragma once

// ============================================================================
// VERSION 1.0 ARCHIVE — Legacy Phong specular shading (pre-GGX)
// ============================================================================
//
// This header preserves the ENTIRE v1.0 Phong specular implementation that the
// v2.0 unified metallic-roughness GGX surface (interactions.cu) REPLACED.
//
//   v1.0 (Phong, this file)   — cosine-power lobe × flat specular color,
//                               non-energy-conserving; Reflective only, Pbr
//                               materials fell through to Lambert diffuse.
//   v2.0 (GGX, interactions.cu) — microfacet BRDF (GGX NDF + separable Smith
//                               G + Schlick conductor Fresnel), metallic-
//                               roughness workflow, diffuse/specular handled
//                               as one unbiased probabilistic mixture.
//
// STATUS: NOT compiled, NOT called by v2.0.  This file is not #included
// anywhere.  It exists purely as a reference archive for:
//   * code review / diffing the v1 → v2 change,
//   * a fallback if the GGX surface needs to be reverted or A/B-compared.
//
// If you DO want to compile it (e.g. a standalone test), all functions are
// `inline __host__ __device__` so a single #include is safe; nothing else in
// the renderer references these symbols.
//
// Preserved verbatim from `git show HEAD:src/interactions/interactions.cu`
// (the pre-GGX commit 9a61242), except:
//   * `inline` added to every definition (header-safe),
//   * `SPECULAR_EXPONENT_ZERO_EPSILON` defined locally — it was deleted from
//     constants.h by the v2.0 change,
//   * the old `case MaterialType::Reflective` branch body lifted into
//     `scatterPhongReflectiveV1()` so the full v1 behavior is reconstructible.
// ============================================================================

#include "interactions/interactions.h"   // RngState, HaltonDim, Material, sampleTexture, ...
#include "constants.h"                   // EPSILON, TWO_PI, ROUGHNESS_THRESHOLD

// Deleted from constants.h by the v2.0 GGX change; kept here so this archive
// stays self-contained.  A specular exponent ≈ 0 means maximum roughness
// (ROUGHNESS = 1 → exponent = 2/1² − 2 = 0), where the Phong lobe is uniform
// and the powf(x, 1/(n+1)) in samplePhongSpecularDir reduces to x.
#define SPECULAR_EXPONENT_ZERO_EPSILON 0.00001f

// Defined in interactions.cu (v2.0 keeps it); forward-declared here so the
// archive compiles standalone.
__host__ __device__ void buildOrthonormalBasis(
    glm::vec3 normal,
    glm::vec3 &tangent,
    glm::vec3 &bitangent);

/**
 * V1.0 — Importance-sample the Phong specular lobe around the perfect
 * reflection direction `reflectDir`.
 *
 * Lobe distribution: f ∝ cos^n(θ) where θ is measured from reflectDir.
 * Inverse-CDF sampling: cosθ = ξ1^(1/(n+1)).  n = 0 → uniform hemisphere
 * (maximum roughness); the explicit zero check skips the powf.
 */
inline __host__ __device__ glm::vec3 samplePhongSpecularDir(
    glm::vec3 reflectDir,
    float exponent,
    float invExponentPlusOne,   // 1/(exponent+1), precomputed per-hit by the caller
    RngState& rng)
{
    float xi1 = rng.next(HaltonDim::SpecularTheta);  // dim 6 (prime 17): specular lobe theta
    float xi2 = rng.next(HaltonDim::SpecularPhi);  // dim 7 (prime 19): specular lobe phi

    // Eq.7: cos(theta_s) = xi1^(1/(n+1))
    // Optimization: if exponent is 0.0f (i.e. maximum roughness, r=1.0f), we skip powf
    float cosTheta = (exponent < SPECULAR_EXPONENT_ZERO_EPSILON) ? xi1 : powf(xi1, invExponentPlusOne);
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

    // Eq.8: phi_s = 2*pi*xi2
    float phi = TWO_PI * xi2;

    // Eq.9: Local Cartesian coordinates
    float xs = sinTheta * cosf(phi);
    float ys = sinTheta * sinf(phi);
    float zs = cosTheta;

    // Construct local ONB with reflectDir as the up direction (Z-axis)
    glm::vec3 tangent, bitangent;
    buildOrthonormalBasis(reflectDir, tangent, bitangent);

    return glm::normalize(xs * tangent + ys * bitangent + zs * reflectDir);
}

/**
 * V1.0 — Resolve the Phong-lobe exponent + precomputed 1/(exponent+1) for a hit.
 *
 * A bound metallicRoughness texture (glTF ORM: G = roughness, B = metallic; a
 * data map loaded raw, so both are already linear [0,1]) overrides the
 * material's JSON roughness scalar; r below ROUGHNESS_THRESHOLD yields a
 * perfect mirror (exponent = -1).  Mirrors the loader's ROUGHNESS conversion
 * (2/r² − 2) exactly, so a texture-driven hit behaves like the JSON scalar did.
 *
 * `metallic` is the ORM B channel, read and RESERVED for a future PBR BRDF —
 * the Lambert/Phong model has no metallic concept, so the caller currently
 * ignores it.  Returns whether the ORM slot drove the result (false = the
 * material scalar is authoritative).
 */
inline __host__ __device__ bool resolveGlossyExponent(
    float& exponent, float& invExpPlusOne, float& metallic,
    const SurfaceBinding& tex, const TextureTable& textures, glm::vec2 uv,
    const Material& m)
{
    // Roughness source — first hit wins, priority is the read order:
    //   ORM texture G (per-texel) → JSON ROUGHNESS → glTF roughnessFactor
    //   → fixed default 0.5 (incomplete models must not silently become mirrors)
    float r;
    metallic = 0.0f;
    const bool fromTexture = tex.metallicRoughness >= 0;
    if (fromTexture)
    {
        const glm::vec3 orm = sampleTexture(textures.pixels,
                                            textures.infos[tex.metallicRoughness],
                                            uv * m.uvScale);
        metallic = orm.z;                                  // B = metallic (reserved)
        r        = glm::clamp(orm.y, 0.0f, 1.0f);          // G = roughness
    }
    else if (m.specular.roughness >= 0.0f)
    {
        r = m.specular.roughness;                          // explicit JSON ROUGHNESS
    }
    else if (tex.roughnessFactor >= 0.0f)
    {
        r = tex.roughnessFactor;                           // glTF roughnessFactor
    }
    else
    {
        r = 0.5f;                                          // fixed default (medium gloss)
    }
    // Same conversion the loader applies to JSON ROUGHNESS:
    //   r < ROUGHNESS_THRESHOLD → perfect mirror (exponent = -1)
    //   otherwise Phong exponent 2/r² − 2, invExponentPlusOne = 1/(exponent+1).
    if (r < ROUGHNESS_THRESHOLD) { exponent = -1.0f; invExpPlusOne = 0.0f; }
    else
    {
        const float r2 = r * r;
        exponent      = (2.0f / r2) - 2.0f;
        invExpPlusOne = r2 / (2.0f - r2);              // one division, no 1/(x+1)
    }
    return fromTexture;
}

/**
 * V1.0 — Faithful lift of the old `case MaterialType::Reflective` branch of
 * scatterRay (HEAD interactions.cu:441-472) into a standalone function.
 *
 * This is exactly what v1.0 did for a Reflective hit:
 *   glossy (exponent >= 0) → samplePhongSpecularDir, fall back to the perfect
 *   reflection if the candidate went below the surface;
 *   perfect mirror (exponent < 0) → pure reflection;
 *   throughput ×= the FLAT specular color (no Fresnel).
 *
 * Kept as a callable whole so the v1 behavior is A/B-comparable against the
 * v2.0 GGX surface without re-editing interactions.cu.
 */
inline __host__ __device__ void scatterPhongReflectiveV1(
    PathSegment& pathSegment,
    const glm::vec3& intersect,
    const glm::vec3& shadingNormal,
    const SurfaceBinding& tex,
    const TextureTable& textures,
    glm::vec2 uv,
    const Material& m,
    RngState& rng)
{
    float exponent, invExpPlusOne, metallic;
    resolveGlossyExponent(exponent, invExpPlusOne, metallic, tex, textures, uv, m);

    glm::vec3 reflectedDir = glm::reflect(pathSegment.ray.direction, shadingNormal);
    glm::vec3 scatterDir;

    if (exponent >= 0.0f)
    {
        // Glossy specular (imperfect specular)
        glm::vec3 candidate = samplePhongSpecularDir(reflectedDir, exponent, invExpPlusOne, rng);
        // Ensure the ray goes outward from the surface, otherwise fallback to perfect reflection
        scatterDir = (glm::dot(candidate, shadingNormal) > 0.0f) ? candidate : reflectedDir;
    }
    else
    {
        // exponent < 0 (i.e. -1.0f): Perfect specular mirror
        scatterDir = reflectedDir;
    }

    float offsetSign = glm::dot(scatterDir, shadingNormal) > 0.0f ? 1.0f : -1.0f;
    pathSegment.ray.origin = intersect + shadingNormal * (EPSILON * offsetSign);
    pathSegment.ray.direction = scatterDir;
    pathSegment.color *= m.specular.color;
}