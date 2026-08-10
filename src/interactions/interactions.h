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

/**
 * Sample a procedural checkerboard texture at a UV coordinate.
 *
 * An 8×8 grid of alternating squares: bright squares use `base`, dark
 * squares use `base * 0.25f` — the pattern reads clearly while keeping the
 * material's hue.  The grid repeats every 1.0 in uv (period 8 cells).
 *
 * @param uv    Texture coordinate
 * @param base  Material albedo the checker is derived from
 * @return      base (bright cell) or base * 0.25f (dark cell)
 */
__host__ __device__ glm::vec3 sampleCheckerboard(
    glm::vec2 uv,
    const glm::vec3& base);

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
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    RngState& rng,
    const TextureTable& textures);

// Glossy specular: samples a direction around the reflected direction
// using a Phong lobe with the given exponent.
__host__ __device__ glm::vec3 samplePhongSpecularDir(
    glm::vec3 reflectDir,
    float exponent,
    float invExponentPlusOne,   // precomputed 1/(exponent+1) — no GPU division
    RngState& rng);
