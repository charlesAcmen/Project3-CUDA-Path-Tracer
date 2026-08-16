#include "interactions/interactions.h"

#include "utils/utilities.h"

#include "rng/rng.h"

/**
 * Generates a random direction vector in a hemisphere oriented around a given surface normal,
 * with cosine-weighted distribution (importance sampling for diffuse surfaces).
 * 
 * This function implements Monte Carlo importance sampling for physically-based rendering.
 * Cosine weighting means rays closer to the normal are more likely to be sampled than
 * rays near the horizon, which matches the cosine term in Lambert's law and reduces variance.
 * 
 * The algorithm consists of three main steps:
 * 1. Sample spherical coordinates using inverse transform sampling
 * 2. Construct an orthonormal basis (ONB) aligned with the surface normal
 * 3. Transform the local sample to world space coordinates
 * 
 * @param normal The surface normal vector (assumed to be normalized)
 * @param rng Random number generator for Monte Carlo sampling
 * @return A unit direction vector in the hemisphere, cosine-weighted around the normal
 */
__host__ __device__ void buildOrthonormalBasis(
    glm::vec3 normal,
    glm::vec3 &tangent,
    glm::vec3 &bitangent)
{
    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    else
    {
        directionNotNormal = glm::vec3(0.0f, 0.0f, 1.0f);
    }

    // Generate two perpendicular direction vectors using cross products
    // These form the tangent (U) and bitangent (V) vectors of our local coordinate system
    // 
    // First perpendicular direction (U-axis): orthogonal to both normal and our helper vector
    tangent = glm::normalize(glm::cross(normal, directionNotNormal));

    // Second perpendicular direction (V-axis): orthogonal to both normal and first perpendicular
    // This completes the right-handed orthonormal basis {U, V, N}
    bitangent = glm::normalize(glm::cross(normal, tangent));
}

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    RngState &rng)
{
    // STEP 1: Polar Coordinate Sampling (Inverse Transform Sampling)
    // ----------------------------------------------------------------
    // To achieve cosine-weighted distribution, we use the probability density function:
    // p(theta) = cos(theta) * sin(theta), where theta is the polar angle from the normal
    // 
    // The cumulative distribution function (CDF) is: CDF(theta) = sin^2(theta)
    // Inverting the CDF gives: theta = arcsin(sqrt(xi_1)), where xi_1 is a uniform random variable
    // 
    // This simplifies to:
    //   cos(theta) = sqrt(xi_1)         (vertical component, weighted toward normal)
    //   sin(theta) = sqrt(1 - xi_1)     (horizontal component, from Pythagorean identity)
    
    float up = sqrt(rng.next(HaltonDim::DiffuseTheta)); // dim 4 (prime 11): diffuse hemisphere theta (cos(theta))
    float over = sqrt(1 - up * up); // sin(theta) - derived from trigonometric identity sin^2(theta) + cos^2(theta) = 1
    float around = rng.next(HaltonDim::DiffusePhi) * TWO_PI; // dim 5 (prime 13): diffuse hemisphere phi

    // At this point, we have spherical coordinates (theta, phi) that represent a direction
    // in a LOCAL coordinate system where the normal is aligned with the Z-axis:
    //   Local coordinates: (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))

    // STEP 2: Construct Orthonormal Basis (ONB)
    // ------------------------------------------
    // To transform our local sample into world space, we need to build a coordinate frame
    // where the normal becomes the "up" direction (local Z-axis), and we need two
    // perpendicular vectors to complete the basis (local X and Y axes).
    // 
    // Peter Kutz's Trick: Since the normal is a unit vector (x^2 + y^2 + z^2 = 1),
    // it's mathematically impossible for all three components to have absolute values
    // greater than sqrt(1/3) ~= 0.577. At least one component must be smaller.
    // 
    // By selecting the axis corresponding to the smallest component, we guarantee
    // that the cross product won't degenerate to zero (which would happen if the
    // normal and our chosen axis were parallel or nearly parallel).

    glm::vec3 perpendicularDirection1;
    glm::vec3 perpendicularDirection2;
    buildOrthonormalBasis(normal, perpendicularDirection1, perpendicularDirection2);

    // STEP 3: Transform from Local to World Space
    // --------------------------------------------
    // Now we have an orthonormal basis:
    //   - normal (N): the "up" direction (local Z-axis)
    //   - perpendicularDirection1 (U): tangent direction (local X-axis)
    //   - perpendicularDirection2 (V): bitangent direction (local Y-axis)
    // 
    // Our sampled direction in local coordinates is:
    //   local_dir = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
    //             = (over * cos(around), over * sin(around), up)
    // 
    // To transform to world space, we compute the linear combination:
    //   world_dir = cos(theta) * N + sin(theta)cos(phi) * U + sin(theta)sin(phi) * V
    //             = up * normal + cos(around) * over * U + sin(around) * over * V
    // 
    // This gives us our final cosine-weighted random direction in world space.

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ float fresnelSchlick(float cosThetaI, float eta)
{
    cosThetaI = fminf(fmaxf(cosThetaI, 0.0f), 1.0f);
    // Schlick's approximation for Fresnel reflectance.
    // R(θ) = R₀ + (1 - R₀) · (1 - cosθ)⁵
    // eta = n1/n2 is precomputed by the caller (invIOR on entry, IOR on exit),
    // so the n1/n2 ratio never divides on the GPU.
    float r0 = (eta - 1.0f) / (eta + 1.0f);
    r0 = r0 * r0;

    // When light travels from a denser medium into a rarer one (eta > 1),
    // the physically correct argument to Schlick is cos(θₜ) — the cosine
    // of the *transmitted* angle — rather than cos(θᵢ).  Otherwise
    // reflectance is severely underestimated near the critical angle.
    //   Ref: "Reflections and Refractions in Ray Tracing" — Bram de Greve
    //   Ref: PBRT 4th ed. §9.2.1 — FresnelDielectric
    float cosTheta = cosThetaI;
    if (eta > 1.0f)
    {
        float sinThetaI = sqrtf(fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI));
        float sinThetaT = eta * sinThetaI;
        if (sinThetaT >= 1.0f) return 1.0f;           // total internal reflection
        cosTheta = sqrtf(fmaxf(0.0f, 1.0f - sinThetaT * sinThetaT)); // cos(θₜ)
    }

    float oneMinusCos = 1.0f - cosTheta;
    float oneMinusCos2 = oneMinusCos * oneMinusCos;
    float oneMinusCos5 = oneMinusCos2 * oneMinusCos2 * oneMinusCos;
    return r0 + (1.0f - r0) * oneMinusCos5;
}

//returns the fraction of non-polarized light reflected at the interface between two materials with indices of refraction n1 and n2,
//given the cosine of the incident angle cosThetaI.  eta = n1/n2 (precomputed).
__host__ __device__ float fresnelAccurate(float cosThetaI, float eta)
{
    cosThetaI = fminf(fmaxf(cosThetaI, 0.0f), 1.0f);
    float sinThetaI = sqrtf(fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI));
    //SNELL'S LAW: n1 * sin(thetaI) = n2 * sin(thetaT)  →  sin(thetaT) = eta * sin(thetaI)
    float sinThetaT = eta * sinThetaI;
    if (sinThetaT >= 1.0f)
    {// Total internal reflection occurs when the angle of incidence exceeds the critical angle, 
        //resulting in no refraction.
        return 1.0f;
    }

    float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sinThetaT * sinThetaT));
    // Divide both numerator and denominator by n2 → eta = n1/n2 form.
    float rParallel = (cosThetaI - eta * cosThetaT) /
                      (cosThetaI + eta * cosThetaT);
    // Correct perpendicular (s-polarized) Fresnel term.
    float rPerpendicular = (eta * cosThetaI - cosThetaT) /
                           (eta * cosThetaI + cosThetaT);
    return (rParallel * rParallel + rPerpendicular * rPerpendicular) * 0.5f;
}

__host__ __device__ HitSide classifyRefraction(
    glm::vec3 rayDir,//assumed to be normalized
    glm::vec3 surfaceNormal,//assumed to be normalized
    float& outCosThetaI//a positive value of costheta incident angle
)
{
    //>=0:exit the object
    //<0:enter the object
    float cosTheta = glm::dot(rayDir, surfaceNormal);

    if (cosTheta < 0.0f)
    {
        outCosThetaI = -cosTheta;//invert the sign to make it positive
        return HitSide::Outside;
    }
    else
    {
        outCosThetaI = cosTheta;
        return HitSide::Inside;
    }
}

// ---- Texture sampling ---------------------------------------------------
__host__ __device__ glm::vec3 sampleTexture(
    const glm::vec3* pixels,
    const TextureInfo& ti,
    glm::vec2 uv)
{
    // Repeat-wrap: fold uv into [0,1) per axis.  floorf handles negative
    // coordinates: uv.x = -0.3 → floor(-0.3) = -1 → u = 0.7, which is the
    // correct wrap.  After this u,v are strictly < 1 (u = x - floor(x) can
    // never equal 1.0), so the texel indices below are always in range.
    const float u = uv.x - floorf(uv.x);
    const float v = uv.y - floorf(uv.y);

    // Texture space → texel space.  u = 0 lands on texel column 0's left
    // edge, u = 1 − ε on the last column's — a "texel-corner" convention,
    // which makes exact corners collapse to that texel (fx = 0).
    const float x = u * (float)ti.width;
    const float y = v * (float)ti.height;

    const int   x0 = (int)x;
    const int   y0 = (int)y;
    const float fx = x - (float)x0;
    const float fy = y - (float)y0;

    // x0 ∈ [0, width-1] (u < 1 guarantees it).  x0+1 is the next texel; it
    // only equals `width` when u → 1, which never happens post-wrap — the
    // clamp is defense-in-depth so the index is never out of range.
    const int x1 = (x0 + 1 < ti.width)  ? x0 + 1 : x0;
    const int y1 = (y0 + 1 < ti.height) ? y0 + 1 : y0;

    const int base = ti.pixelOffset;
    const glm::vec3& p00 = pixels[base + (y0 * ti.width + x0)];
    const glm::vec3& p10 = pixels[base + (y0 * ti.width + x1)];
    const glm::vec3& p01 = pixels[base + (y1 * ti.width + x0)];
    const glm::vec3& p11 = pixels[base + (y1 * ti.width + x1)];

    // Two horizontal interpolations, then one vertical — the standard
    // bilinear blend.
    const glm::vec3 top = p00 + fx * (p10 - p00);
    const glm::vec3 bot = p01 + fx * (p11 - p01);
    return top + fy * (bot - top);
}

// ---- Unified metallic-roughness GGX surface (the PBR path) ---------------
// v2.0: replaces the v1.0 Phong lobe (archived in legacy_phong_v1.cuh — NOT
// called here).  The old Phong lobe (cosine-power lobe × flat specular color)
// was not energy-conserving and split diffuse/specular into mutually-exclusive
// material types.  This replaces it with a microfacet BRDF — GGX NDF +
// separable Smith geometry + Schlick Fresnel (F0 parameterized) — sampled via
// the GGX half-vector, where diffuse and specular are drawn probabilistically
// and each weight divided by its branch probability (unbiased mixture).

__host__ __device__ glm::vec3 fresnelSchlickF0(float cosTheta, const glm::vec3& F0)
{
    const float c  = fminf(fmaxf(cosTheta, 0.0f), 1.0f);
    const float omc = 1.0f - c;
    const float omc2 = omc * omc;
    const float omc5 = omc2 * omc2 * omc;   // explicit multiplies — never powf
    return F0 + (1.0f - F0) * omc5;
}

__host__ __device__ float luminance(const glm::vec3& c)
{
    return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
}

__host__ __device__ float smithG1Ggx(float alpha, float NdotV)
{
    // Separable Smith masking-shadowing (glTF GGX form), alpha = roughness².
    const float a2 = alpha * alpha;
    const float ndv = fmaxf(NdotV, 0.0f);
    return (2.0f * ndv) / (ndv + sqrtf(a2 + (1.0f - a2) * ndv * ndv));
}

__host__ __device__ float ggxD(float alpha, float NdotH)
{
    // GGX NDF: α² / (π ((N·H)²(α²−1) + 1)²).  Used by the energy test; the
    // sampling weight cancels it (see the specular branch in scatterRay).
    const float a2 = alpha * alpha;
    const float t  = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / (PI * t * t);
}

__host__ __device__ glm::vec3 sampleGgxHalfVector(const glm::vec3& normal, float alpha, RngState& rng)
{
    // Importance-sample the GGX NDF half-vector.  Inverse-CDF of D(θ):
    //   cosθh² = (1−ξ₁) / (1 + (α²−1)ξ₁)   →  α→0 ⇒ cosθh→1 ⇒ H→N (mirror),
    //   α=1 ⇒ cosθh = sqrt(1−ξ₁) (cosine distribution).
    const float xi1 = rng.next(HaltonDim::SpecularTheta);
    const float xi2 = rng.next(HaltonDim::SpecularPhi);
    const float a2  = alpha * alpha;
    const float cosThetaSq = (1.0f - xi1) / (1.0f + (a2 - 1.0f) * xi1);
    const float cosTheta  = sqrtf(fmaxf(cosThetaSq, 0.0f));
    const float sinTheta  = sqrtf(fmaxf(1.0f - cosTheta * cosTheta, 0.0f));
    const float phi       = TWO_PI * xi2;

    glm::vec3 tangent, bitangent;
    buildOrthonormalBasis(normal, tangent, bitangent);
    return glm::normalize(sinTheta * cosf(phi) * tangent +
                          sinTheta * sinf(phi) * bitangent +
                          cosTheta  * normal);
}

// Resolve the diffuse albedo.  Source chain — first hit wins:
//   glTF baseColor texture (tex.baseColor, × baseColorFactor) >
//   JSON-declared Material::textureId >
//   flat material color m.color.
__host__ __device__ glm::vec3 resolveBaseColor(
    const TextureBinding& tex, const TextureTable& textures, glm::vec2 uv,
    const Material& m)
{
    int bid = tex.baseColor;
    if (bid < 0) bid = m.textureId;
    glm::vec3 albedo = m.color;
    if (bid >= 0)
    {
        albedo = sampleTexture(textures.pixels, textures.infos[bid], uv * m.uvScale);
        // glTF semantics: baseColor = texture.rgb · baseColorFactor.  The
        // factor applies only when the winning slot is the glTF baseColor
        // binding (tex.baseColor), not a JSON-declared TEXTURE.
        if (tex.baseColor >= 0)
            albedo *= glm::vec3(tex.baseColorFactor);
    }
    else if (tex.roughnessFactor >= 0.0f)
    {
        // glTF material whose color is factor-only (no baseColorTexture):
        // baseColorFactor IS the glTF material's own tint (e.g. the flat
        // red/yellow spheres in transmission_test).  Only glTF triangles
        // carry roughnessFactor >= 0 (OBJ keeps the -1 sentinel), so the
        // scene JSON color still governs plain meshes.
        albedo = glm::vec3(tex.baseColorFactor);
    }
    return albedo;
}

// Resolve the per-hit GGX surface parameters for a Reflective / Pbr material
// (chains documented in interactions.h).  `roughness` (r) drives the mirror
// threshold and α; `alpha`, `F0`, `diffuseColor` feed the BRDF directly.
__host__ __device__ void resolvePbrSurfaceParams(
    float& roughness, float& metallic, float& alpha, glm::vec3& F0, glm::vec3& diffuseColor,
    const TextureBinding& tex, const TextureTable& textures, glm::vec2 uv,
    const Material& m)
{
    // Roughness source — first hit wins, priority is the read order:
    //   ORM texture G (per-texel) → JSON ROUGHNESS → glTF roughnessFactor
    //   → fixed default 0.5 (incomplete models must not silently become mirrors)
    float r;
    if (tex.metallicRoughness >= 0)
    {
        // ORM texture drives BOTH channels per-texel (a data map, already linear).
        // glTF spec: final value = texture × factor.  The factor defaults to
        // 1.0 (via cgltf) when the file omits it, so the multiply is a no-op
        // for files that only have a texture.  A -1 sentinel means this is not
        // a glTF material (plain OBJ) — treat as factor 1.
        const glm::vec3 orm = sampleTexture(textures.pixels,
                                            textures.infos[tex.metallicRoughness],
                                            uv * m.uvScale);
        const float rFactor = (tex.roughnessFactor >= 0.0f) ? tex.roughnessFactor : 1.0f;
        const float mFactor = (tex.metallicFactor  >= 0.0f) ? tex.metallicFactor  : 1.0f;
        metallic = glm::clamp(orm.z * mFactor, 0.0f, 1.0f);  // B = metallic × factor
        r        = glm::clamp(orm.y * rFactor, 0.0f, 1.0f);  // G = roughness × factor
    }
    else
    {
        // Roughness source — first hit wins: JSON ROUGHNESS > glTF factor > 0.5.
        r = (m.specular.roughness >= 0.0f)     ? m.specular.roughness
          : (tex.roughnessFactor >= 0.0f)      ? tex.roughnessFactor
          : 0.5f;
        // Metallic source — first hit wins: JSON METALLIC > glTF factor >
        // type default (Reflective = chrome 1.0, Pbr = dielectric 0.0).
        metallic = (m.metallic >= 0.0f)             ? m.metallic
                 : (tex.metallicFactor >= 0.0f)     ? tex.metallicFactor
                 : (m.type == MaterialType::Reflective) ? 1.0f : 0.0f;
    }

    // baseColor role per type: legacy chrome uses specular.color as its metal
    // tint (F0); the Pbr surface resolves the albedo like the diffuse branch.
    glm::vec3 baseColor;
    if (m.type == MaterialType::Reflective)
        baseColor = m.specular.color;
    else
        baseColor = resolveBaseColor(tex, textures, uv, m);

    roughness     = r;
    alpha         = r * r;
    F0            = glm::mix(glm::vec3(0.04f), baseColor, metallic);
    diffuseColor  = baseColor * (1.0f - metallic);
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    const ShadeableIntersection &hit,
    const Material &m,
    RngState &rng,
    const TextureTable &textures)
{
    // Scatter a ray according to the material's BSDF.
    // Diffuse: cosine-weighted hemisphere sampling.
    // Reflective / Pbr: 
    //   Unified metallic-roughness PBR workflow.
    //   GGX microfacet NDF surface (Trowbridge-Reitz variant, Walter et al. 2007):
    //   - Smooth surfaces (r < ROUGHNESS_THRESHOLD): collapse to a mirror lobe
    //     to avoid numerical instability where NDF approaches a Dirac delta.
    //   - Rough surfaces: Fresnel-weighted probabilistic split between GGX
    //     half-vector specular reflection and diffuse Lambert reflection,
    //     with unbiased throughput compensation (divided by branch probability).
    // Refractive: Fresnel-weighted Russian roulette between reflection and
    //             refraction (glm::refract), with normal flipped for exit rays.

    // The hit record carries everything the scatter needs; unpack it here so
    // the branch bodies below stay compact.  The exact hit point is derived
    // from the path ray (unit length) and hit.t — identical to what the
    // caller's getExactPointOnRay would compute, so passing it separately
    // would be redundant.
    const glm::vec3        intersect = pathSegment.ray.origin + hit.t * pathSegment.ray.direction;
    const glm::vec3        normal    = hit.surfaceNormal;
    const glm::vec2&       uv        = hit.uv;
    const TextureBinding&  tex       = hit.tex;

    // Generate new random direction for diffuse reflection (cosine-weighted hemisphere sampling)
    // Common mistake: offsetting along newDirection instead of normal
    // - When newDirection is nearly parallel to the surface (grazing angle),
    //   offset along newDirection has almost zero normal component
    // - This causes the ray to start below the surface -> self-intersection -> shadow acne
    // 
    // Opaque (double-sided) materials shade on the hit side regardless of the
    // model's winding: orient the shading normal toward the incoming ray so
    // the diffuse hemisphere / reflection lobe is on the correct side.
    // Refraction keeps the TRUE normal — its sign (dot with the ray) is what
    // classifyRefraction uses to distinguish entry from exit.
    const glm::vec3 rayDir        = pathSegment.ray.direction;
    const glm::vec3 shadingNormal = (glm::dot(normal, rayDir) > 0.0f) ? -normal : normal;

    // Offset the new ray origin off the surface by EPSILON so it cannot
    // immediately re-hit the same triangle.  The offset direction is chosen
    // per branch below: refractive keys off the entering/exiting state
    // (numerically stable near grazing angles), reflective keys off the
    // shading-normal orientation, diffuse always pushes outward.
    switch (m.type)
    {
        case MaterialType::Refractive:
        {
            float cosThetaI;
            const HitSide hitSide = classifyRefraction(pathSegment.ray.direction, normal, cosThetaI);
            const bool entering = (hitSide == HitSide::Outside);
            // Use invIndexOfRefraction to avoid division on entry.
            // The offset sign is keyed off the entering/exiting state rather than
            // the new direction's dot product, which is numerically unstable near grazing angles.
            const float etaRatio = entering ? m.invIndexOfRefraction : m.indexOfRefraction;
            const glm::vec3 refractNormal = entering ? normal : -normal;

            // etaRatio = n1/n2, already precomputed (invIOR on entry, IOR on
            // exit) — the Fresnel functions take it directly instead of
            // dividing by n1/n2 on the GPU.
            // Both Fresnel functions return exactly 1.0 on total internal
            // reflection (see fresnelSchlick / fresnelAccurate), so the roulette
            // below normally takes the reflection branch whenever refraction is
            // impossible.  The explicit `tir ||` below makes that unconditional.
            //
            // Fresnel evaluator is hardcoded to Accurate (the default renderer
            // choice).  To switch to Schlick, change this one call — there is
            // no runtime mode dispatch by design.
            const float reflectance = fresnelAccurate(cosThetaI, etaRatio);

            // Russian roulette: reflect with prob R, refract with prob 1-R.
            // Throughput multiplier = (energy fraction) / (probability):
            //   reflection:  R * color / R     = color
            //   refraction: (1-R) * color / (1-R) = color
            // → Fresnel factor cancels out in both branches.
            //
            // TIR detection from refract()'s OUTPUT — not a recomputation of k.
            // GLM computes k = 1 - eta²(1-dot²) internally and returns a NaN
            // vector when k < 0 (sqrt of a negative); inspecting the result
            // catches that.  A valid refracted direction is always unit length
            // (squared length 1.0) and NaN compares false against ANY
            // threshold, so one check catches every degenerate shape:
            //   valid refraction:  dot = 1.0  → !(1.0 > 0.5) = false
            //   TIR → NaN:         dot = NaN  → !(NaN > 0.5) = true
            //   zero vector:       dot = 0.0  → !(0.0 > 0.5) = true
            // (glm::isnan is deliberately avoided: its CUDA branch recurses
            // under the MSVC host pass — compiler warning C4717.)
            glm::vec3 refractedDir = glm::refract(pathSegment.ray.direction, refractNormal, etaRatio);
            const bool tir = !(glm::dot(refractedDir, refractedDir) > REFRACT_VALID_SQ_LEN_MIN);

            if (tir || rng.next(HaltonDim::FresnelRR) < reflectance)  // dim 8 (prime 23): Fresnel roulette
            {
                glm::vec3 reflectedDir = glm::reflect(pathSegment.ray.direction, normal);
                const float offsetSign = entering ? 1.0f : -1.0f;
                pathSegment.ray.origin = intersect + normal * (EPSILON * offsetSign);
                pathSegment.ray.direction = reflectedDir;
            }
            else
            {
                // !tir here ⇒ refractedDir is a finite unit vector; the offset
                // pushes to the far side of the surface.
                const float offsetSign = entering ? -1.0f : 1.0f;
                pathSegment.ray.origin = intersect + normal * (EPSILON * offsetSign);
                pathSegment.ray.direction = refractedDir;
            }
            pathSegment.color *= m.color;
            break;
        }
        case MaterialType::Reflective:
        case MaterialType::Pbr:
        {
            // Unified metallic-roughness GGX surface (legacy JSON Specular →
            // Reflective is just the metallic=1 chrome case).  Resolve the
            // per-hit surface params: roughness (mirror threshold below
            // ROUGHNESS_THRESHOLD), alpha = r², conductor F0, diffuse albedo.
            float r, metallic, alpha;
            glm::vec3 F0, diffuseColor;
            resolvePbrSurfaceParams(r, metallic, alpha, F0, diffuseColor, tex, textures, uv, m);

            const float NdotV      = glm::clamp(glm::dot(shadingNormal, -rayDir), 1e-4f, 1.0f);
            const glm::vec3 F_view = fresnelSchlickF0(NdotV, F0);
            // Diffuse/specular split probability from the graze Fresnel, so a
            // surface that is mostly specular (metal, or dielectric at grazing
            // angles) spends most samples on the specular lobe.  Clamped so
            // both branch weights stay bounded (unbiased mixture — the divide
            // cancels the probability regardless of the clamp).
            const float specProb  = glm::clamp(luminance(F_view), 0.05f, 0.95f);

            glm::vec3 scatterDir;
            glm::vec3 throughput;

            if (r < ROUGHNESS_THRESHOLD && metallic > 0.5f)
            {
                // Smooth metal: a single mirror lobe — F0 ≈ baseColor, so
                // virtually all energy is in the specular reflection.  The
                // diffuse term is negligible (diffuseColor ≈ 0 for metallic ≈ 1).
                scatterDir = glm::reflect(rayDir, shadingNormal);
                throughput = fresnelSchlickF0(NdotV, F0);
            }
            else if (r < ROUGHNESS_THRESHOLD)
            {
                // Smooth dielectric: F0 ≈ 0.04, so most energy is diffuse.
                // Use the same probabilistic split as rough surfaces — specular
                // = mirror reflect with probability specProb, diffuse with
                // 1−specProb.  This avoids the old shortcut that dropped ~96%
                // of the energy for smooth dielectrics.
                if (rng.next(HaltonDim::PbrSplit) < specProb)
                {
                    scatterDir = glm::reflect(rayDir, shadingNormal);
                    throughput = fresnelSchlickF0(NdotV, F0) / specProb;
                }
                else
                {
                    scatterDir = calculateRandomDirectionInHemisphere(shadingNormal, rng);
                    throughput = diffuseColor * (glm::vec3(1.0f) - F_view) / (1.0f - specProb);
                }
            }
            else if (rng.next(HaltonDim::PbrSplit) < specProb)   // dim 10 (prime 31): GGX split
            {
                // Specular: importance-sample the GGX half-vector H, reflect
                // the view about it.  The NDF and the cos(N·L) cancel out of
                // the weight (D·(N·H)/(4·(V·H)) appears in both f and pdf):
                //   f = F·G·D/(4·(N·V)(N·L)) ,  pdf = D·(N·H)/(4·(V·H))
                //   weight = f·(N·L)/pdf = F·G·(V·H)/((N·V)·(N·H))
                const glm::vec3 H   = sampleGgxHalfVector(shadingNormal, alpha, rng);
                scatterDir = glm::reflect(rayDir, H);
                if (glm::dot(shadingNormal, scatterDir) > 0.0f)
                {
                    const float NdotL = glm::clamp(glm::dot(shadingNormal, scatterDir), 0.0f, 1.0f);
                    const float NdotH = glm::max(glm::dot(shadingNormal, H), 1e-4f);
                    const float VdotH = glm::max(glm::dot(-rayDir, H), 0.0f);
                    const float G = smithG1Ggx(alpha, NdotV) * smithG1Ggx(alpha, NdotL);
                    const glm::vec3 F = fresnelSchlickF0(VdotH, F0);
                    throughput = F * (G * VdotH / (NdotV * NdotH)) / specProb;
                }
                else
                {
                    // Below-surface sample (lobe backfacing the surface):
                    // zero weight, but scatter the mirror reflection so the
                    // path survives for Russian roulette to terminate it.
                    scatterDir = glm::reflect(rayDir, shadingNormal);
                    throughput = glm::vec3(0.0f);
                }
            }
            else
            {
                // Diffuse (probability 1 − specProb).  albedo is already
                // scaled by (1 − metallic); the per-channel (1 − F_view)
                // complement ensures specular + diffuse conserve energy.
                scatterDir = calculateRandomDirectionInHemisphere(shadingNormal, rng);
                throughput = diffuseColor * (glm::vec3(1.0f) - F_view) / (1.0f - specProb);
            }

            pathSegment.color *= throughput;
            float offsetSign = glm::dot(scatterDir, shadingNormal) > 0.0f ? 1.0f : -1.0f;
            pathSegment.ray.origin = intersect + shadingNormal * (EPSILON * offsetSign);
            pathSegment.ray.direction = scatterDir;
            break;
        }
        case MaterialType::Diffuse:
        default:
        {
            glm::vec3 newDirection = calculateRandomDirectionInHemisphere(shadingNormal, rng);
            pathSegment.ray.origin = intersect + shadingNormal * EPSILON;
            // Apply diffuse material color (energy attenuation)
            // multiplier = fr * cos theta/pdf(omega)
            // where pdf(omega) = cos theta / PI
            // BSDF of diffuse reflection: fr = R / PI
            pathSegment.ray.direction = newDirection;

            // Resolve the diffuse albedo: a per-triangle glTF baseColor binding
            // wins over the JSON-declared Material::textureId, then over the
            // flat material color.
            pathSegment.color *= resolveBaseColor(tex, textures, uv, m);
            break;
        }
    }

    // Decrement remaining bounces
    pathSegment.remainingBounces--;
}

