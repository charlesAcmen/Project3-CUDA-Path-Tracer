#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

#include "rng/rng.h"

/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    RngState& rng);

// eta = n1/n2, precomputed by the caller (invIOR on entry, IOR on exit).
__host__ __device__ float fresnelSchlick(float cosThetaI, float eta);
__host__ __device__ float fresnelAccurate(float cosThetaI, float eta);
__host__ __device__ HitSide classifyRefraction(
    glm::vec3 rayDir,
    glm::vec3 surfaceNormal,
    float& outCosThetaI);

/**
 * Sample a file-loaded texture at a UV coordinate.
 *
 * Repeat-wrap: uv folds into [0,1) per axis (negative coordinates wrap too).
 * Bilinear: the four texels surrounding the wrapped position are blended by
 * the fractional texel offset.  At an exact texel corner (uv mapping to an
 * integer texel position) the sample collapses to that texel's color.
 *
 * The returned color is expected to be linear (the loader converts sRGB
 * texels to linear on upload), so it feeds the existing linear pipeline
 * directly.
 *
 * @param pixels   Concatenated linear-RGB texel buffer
 * @param ti       Which texture inside `pixels` (pixelOffset/width/height)
 * @param uv       Texture coordinate; any real values are valid (wrapped)
 * @return         Sampled linear color
 */
__host__ __device__ glm::vec3 sampleTexture(
    const glm::vec3* pixels,
    const TextureInfo& ti,
    glm::vec2 uv);

// ---- Unified metallic-roughness GGX surface (the PBR path) ---------------

// Schlick Fresnel with per-channel F0: F0 + (1−F0)(1−cosθ)⁵.  Handles both
// conductors (F0 = metal tint from baseColor) and dielectrics (F0 ≈ 0.04);
// the (1−cos)⁵ is computed with explicit multiplies (never powf) to stay
// fast-math friendly, matching the scalar fresnelSchlick.
__host__ __device__ glm::vec3 fresnelSchlickF0(float cosTheta, const glm::vec3& F0);

// Rec.709 relative luminance — used for the diffuse/specular split probability.
__host__ __device__ float luminance(const glm::vec3& c);

// Separable Smith masking-shadowing for GGX (glTF form): alpha = roughness².
// G1(v) = 2(N·v) / ((N·v) + sqrt(alpha² + (1−alpha²)(N·v)²)).
__host__ __device__ float smithG1Ggx(float alpha, float NdotV);

// GGX normal-distribution function (for the energy integral test; the sampling
// path needs it only implicitly — it cancels in the weight).
__host__ __device__ float ggxD(float alpha, float NdotH);

// Importance-sample the GGX NDF's half-vector: cosθh = sqrt((1−ξ₁)/(1+(α²−1)ξ₁)),
// φ = 2π·ξ₂, in an ONB around `normal`.  α→0 collapses to H→N (a perfect mirror).
__host__ __device__ glm::vec3 sampleGgxHalfVector(const glm::vec3& normal, float alpha, RngState& rng);

// Resolve the diffuse albedo.  Source chain — first hit wins:
//   glTF baseColor texture (tex.baseColor, × baseColorFactor) >
//   flat material color m.color.
// Vertex colors (COLOR_0) multiply the final albedo: albedo *= vertexColor.
__host__ __device__ glm::vec3 resolveBaseColor(
    const SurfaceBinding& tex, const TextureTable& textures, glm::vec2 uv,
    const Material& m, const glm::vec3& vertexColor);

// Resolve the per-hit emissive radiance.  glTF/OBJ emission is
//   (emissive texture RGB or white) × emissiveFactor × emissiveStrength.
// A zero factor denotes no model-defined emission; JSON Emitting then falls
// back to its flat material color.
// The caller scales by material.emittance for JSON Emitting light sources;
// auto-glow (a non-zero emissive factor with no emittance) adds it as-is.
__host__ __device__ glm::vec3 resolveEmissive(
    const SurfaceBinding& tex, const TextureTable& textures, glm::vec2 uv,
    const Material& m);

// Resolve the per-hit GGX surface parameters for a Reflective / Pbr material.
//
// Roughness source chain — first hit wins:
//   1. ORM texture G channel (per-texel, when `tex.metallicRoughness` is bound) × factor
//   2. glTF `pbrMetallicRoughness.roughnessFactor`  (`tex.roughnessFactor >= 0`)
//   3. type default — Reflective = 0.0 (mirror), Pbr = 0.5 (medium gloss)
// Metallic source chain:
//   1. ORM texture B channel (per-texel, when `tex.metallicRoughness` is bound) × factor
//   2. glTF `pbrMetallicRoughness.metallicFactor`  (`tex.metallicFactor >= 0`)
//   3. type default — Reflective (legacy JSON Specular) = 1.0 (chrome),
//      Pbr = 0.0 (dielectric)
//
// baseColor role per type: Reflective uses specular.color as the metal tint;
// Pbr resolves the albedo (baseColor texture / material color).  From those:
//   alpha = r² ;  F0 = mix(0.04, baseColor, metallic) ;
//   diffuseColor = baseColor·(1 − metallic)
// The ORM texture is a data map (srgb=false), so G/B are already linear [0,1].
__host__ __device__ void resolvePbrSurfaceParams(
    float& roughness, float& metallic, float& alpha, glm::vec3& F0, glm::vec3& diffuseColor,
    const SurfaceBinding& tex, const TextureTable& textures, glm::vec2 uv,
    const Material& m, const glm::vec3& vertexColor);

// Resolve the per-hit SHADING normal from a glTF normal texture (tangent
// space) when one is bound.  Samples the normal slot (a data map, raw linear
// bytes), remaps [0,1]→[-1,1] (glTF: n = 2·texel − 1), and rotates it into
// world space via the per-triangle TBN frame:
//     T = per-triangle tangent (orthogonalized against the geometric normal)
//     B = cross(N, T) · tangent.w        (glTF/OpenGL +V → bitangent;
//         tangent.w = UV handedness, +1 regular / -1 mirrored island)
//     worldN = T·n.x + B·n.y + N·n.z
// Returns `geometricNormal` unchanged when no normal slot is bound or the
// tangent is the (0,0,0,0) sentinel (degenerate UVs).
__host__ __device__ glm::vec3 resolveShadingNormal(
    const glm::vec3& geometricNormal,
    const glm::vec4& tangent,
    const SurfaceBinding& tex,
    const TextureTable& textures,
    glm::vec2 uv);

// Continuous BSDF evaluation used by direct-light sampling and MIS.  `value`
// is f(wo, wi); pdfOmega is in solid-angle measure.  Delta events return
// isDelta=true and pdfOmega=0 because a finite-area light sample cannot
// generate their single direction.  Invalid numerical evaluations return the
// zero-initialized result, preserving this same no-continuous-contribution
// contract without clamping valid samples.
struct BsdfEvaluation
{
    glm::vec3 value{ 0.0f };
    glm::vec3 shadingNormal{ 0.0f };
    float pdfOmega = 0.0f;
    bool isDelta = false;
};

// Texture-resolved, incident-direction-dependent BSDF state for one hit.
// Direct lighting and continuation scattering consume the same values, so a
// normal/base-color/ORM lookup happens once per shaded non-emissive hit.
// This deliberately contains only material parameters; light samples and
// shadow-traversal temporaries remain local to the direct-light path.
struct ResolvedBsdf
{
    glm::vec3 shadingNormal{ 0.0f };
    glm::vec3 baseColor{ 0.0f }; // Lambert albedo, PBR baseColor, or Reflective tint
    float roughness = 0.0f;
    float metallic = 0.0f;
};

__host__ __device__ ResolvedBsdf resolveBsdf(
    const ShadeableIntersection& hit,
    const Material& material,
    const TextureTable& textures,
    const glm::vec3& incidentRayDirection);

__host__ __device__ BsdfEvaluation evaluateBsdf(
    const ResolvedBsdf& resolved,
    const Material& material,
    const glm::vec3& incidentRayDirection,
    const glm::vec3& outgoingDirection);

// Compatibility entry point for focused BSDF tests.  Shading code should
// resolve once, then use the overload above for every direction at that hit.
__host__ __device__ BsdfEvaluation evaluateBsdf(
    const ShadeableIntersection& hit,
    const Material& material,
    const TextureTable& textures,
    const glm::vec3& incidentRayDirection,
    const glm::vec3& outgoingDirection);

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * Texture input is resolved from the hit's shared SurfaceBinding and UV before
 * this overload is called: diffuse receives baseColor albedo, while GGX
 * receives baseColor/metallicRoughness-derived parameters. Refractive
 * materials keep their material color and ignore textures. The compatibility
 * overload below performs that one-time resolution for callers without NEE.
 *
 * The hit geometry (point, smooth/geometric normals, UV, texture slots) is passed as the
 * ShadeableIntersection record rather than as loose scalars: it is exactly
 * the surface state the traversal produced, so the caller hands over the
 * whole hit.  The exact hit point is derived inside from
 * `pathSegment.ray.origin + hit.t * direction` (the ray is unit length).
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    const ShadeableIntersection& hit,
    const Material& m,
    const ResolvedBsdf& resolved,
    RngState& rng);

// Compatibility entry point for focused tests and callers that do not also
// perform direct-light evaluation.  The main shading kernel resolves once and
// calls the overload above, avoiding repeated texture work.
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    const ShadeableIntersection& hit,
    const Material& m,
    RngState& rng,
    const TextureTable& textures);

