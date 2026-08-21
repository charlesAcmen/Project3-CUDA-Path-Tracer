#pragma once

// ====================================================================
// Mathematical constants used throughout the renderer.
//
// This header has zero dependencies -- it can be included from any
// compilation unit (host or device) without pulling in I/O libraries.
// ====================================================================

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define DEG_TO_RAD        (PI / 180.0f)   // degrees → radians (scene rotations, FOV)
#define RAD_TO_DEG        (180.0f / PI)   // radians → degrees (FOV back-calc)
#define EPSILON           0.00001f    // ray origin offset for self-intersection prevention
#define RAY_EPSILON       1e-10f      // minimum ray–scene distance; rejects near-parallel / self-hit
// Minimum squared length a refract() result must exceed to be a valid
// direction.  Unit |d|² = 1.0; a NaN (TIR: sqrt of negative) compares
// false against ANY threshold; a zero vector is 0.0.  0.5 sits safely
// between all three.
#define REFRACT_VALID_SQ_LEN_MIN 0.5f
#define ROUGHNESS_THRESHOLD 0.001f
// A smooth (r < ROUGHNESS_THRESHOLD) GGX surface with metallic above this
// value collapses to a single mirror lobe.  Below it the diffuse albedo
// (diffuseColor = baseColor·(1−metallic)) is non-negligible — at metallic 0.5
// that's half the albedo — so those surfaces must go through the specular/
// diffuse split in scatterRay or the diffuse lobe is silently dropped.
// 0.95 bounds the mirror-shortcut energy loss to ~(1−metallic)² ≈ 0.25%.
#define PBR_MIRROR_METALLIC_THRESHOLD 0.95f
// Type-default roughness used when a material provides no roughness source
// (see resolvePbrSurfaceParams): Reflective is a perfect mirror, Pbr is a
// generic dielectric rough surface.
#define REFLECTIVE_ROUGHNESS_DEFAULT 0.0f
#define PBR_ROUGHNESS_DEFAULT        0.5f
// Tangent-space normal mapping: degeneracy thresholds for the per-triangle
// tangent.  Two INDEPENDENT failure modes with different units, tuned
// separately (they happen to share the value 1e-8):
//   TANGENT_DET_EPSILON — the UV determinant det = ΔU1·ΔV2 − ΔU2·ΔV1 (twice
//     the signed area of the 2D UV triangle, uv² units).  |det| ≈ 0 → the UV
//     triangle is degenerate and the 2×2 solve for the tangent blows up, so
//     no tangent is emitted (triangle.h).
//   TANGENT_EPSILON     — squared-length degeneracy for a direction vector
//     (dimensionless): the orthogonalized tangent (triangle.h + the
//     re-orthogonalization in resolveShadingNormal), the (0,0,0,0) sentinel,
//     and the world-space TBN·n result.  A vector below this is (near) zero —
//     no usable direction → normal mapping skipped.
// NORMAL_MAP_TEXEL_EPSILON — squared-length degeneracy for the sampled
//   normal-map texel VECTOR (after the [0,1]→[-1,1] remap, before
//   re-normalization) in resolveShadingNormal.  A bilinear blend of two
//   nearly-opposite texels can shorten the vector toward zero; below this
//   the perturbation is degenerate noise, so shading falls back to the
//   geometric normal.  A DIFFERENT magnitude from TANGENT_EPSILON (1e-4 vs
//   1e-8): the texel vector lives in normal-map data space, where even a
//   pathological 4-texel blend rarely shortens it below ~0.1 — 1e-4 rejects
//   only truly degenerate samples.
#define TANGENT_DET_EPSILON 1e-8f
#define TANGENT_EPSILON     1e-8f
#define NORMAL_MAP_TEXEL_EPSILON 1e-4f
#define RR_P_MIN          0.2f
#define RR_P_MAX          1.0f
#define LARGE_T           1e30f       // sentinel > any valid ray–scene intersection

// Bloom post-processing limits — single source of truth shared by the device
// kernels (postprocess/bloom.cuh) and the host config (src/config.h), so the
// legal radius range can't drift between the two.
#define MAX_BLOOM_RADIUS 32           // max bloom Gaussian radius (device weights buffer size)
#define BLOOM_BLOCK_SIZE 256          // threads per 1D block for the bloom blur kernels

// ---- BVH build-time settings — single source of truth ----
// Tree depth and leaf size are compile-time constants (tune here, no runtime
// validation).  The traversal kernel's explicit stack holds at most
// BVH_MAX_STACK_DEPTH entries, so BVH_MAX_DEPTH must stay well below it:
// each tree level pushes up to two children.
constexpr int kBvhMaxDepth      = 24;   // max tree depth
constexpr int kBvhLeafSize      = 4;    // max triangles per leaf
constexpr int kMaxBvhStackDepth = 64;   // explicit-stack capacity
