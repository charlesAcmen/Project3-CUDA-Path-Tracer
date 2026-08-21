#pragma once

// ====================================================================
// Triangle Intersection — Möller-Trumbore (莫勒-特伦博尔算法)
//
// Double-sided ray–triangle intersection.  Standard Möller-Trumbore
// rejects back-face hits (a < 0); this version accepts them so that
// rays inside a closed mesh (e.g. inside a glass object) can hit the
// back face when exiting.
//
// The reported normal is the model's TRUE shading normal — it is NOT
// oriented toward the ray.  Winding / normal direction is meaningful:
// opaque shading (diffuse/reflective) orients it toward the ray in
// scatterRay, and refraction reads its sign (dot with the ray) to
// classify entry vs exit.  A flip here would erase the winding and
// make the refraction exit branch unreachable.
//
// Reference: Möller & Trumbore, "Fast, Minimum Storage Ray-Triangle
// Intersection", Journal of Graphics Tools, 1997.
// ====================================================================

#include "sceneStructs.h"
#include "constants.h"

/**
 * Double-sided ray–triangle intersection (Möller-Trumbore).
 *
 * @param ray       Ray in object space
 * @param tri       Triangle in object space
 * @param outT      [out] Distance along ray to hit
 * @param outNormal [out] Model's shading normal, TRUE orientation
 *                        (winding preserved, not oriented toward the ray)
 * @param outUv     [out] Interpolated texture coordinate at the hit
 * @param outTangent [out] Per-triangle tangent aligned with the texture's
 *                        +U axis (world space — edges are baked world space),
 *                        orthogonalized against the interpolated normal.
 *                        .xyz = the unit tangent; .w = UV handedness sign
 *                        (+1 regular layout, -1 mirrored island, glTF
 *                        TANGENT.w convention → B = cross(N, T)·w in shading).
 *                        (0,0,0,0) sentinel = degenerate UVs / no usable
 *                        tangent → the shading side skips normal mapping.
 * @param outVertexColor [out] Interpolated vertex color at the hit (default white = no effect)
 * @return          true on hit (either side)
 */
__host__ __device__ inline bool triangleIntersectionTest(
    const Ray& ray,
    const TrianglePos& tri,
    float& outT,
    glm::vec3& outNormal,
    glm::vec2& outUv,
    glm::vec4& outTangent,
    glm::vec3& outVertexColor)
{
    // ---- Step 1: edge vectors ----
    // Translate triangle so v0 is at origin, then compute the two
    // edges from v0.  This is the reference frame for barycentric
    // coordinates: any point on the triangle = v0 + u*e1 + v*e2.
    glm::vec3 e1 = tri.v1 - tri.v0;     // edge v0→v1
    glm::vec3 e2 = tri.v2 - tri.v0;     // edge v0→v2

    // ---- Step 2: determinant a = |e1  dir  e2| (scalar triple product) ----
    // p = dir × e2  — a vector perpendicular to both dir and e2.
    glm::vec3 p = glm::cross(ray.direction, e2);
    // a = dot(e1, p) = dot(e1, cross(dir, e2)).
    float a = glm::dot(e1, p);
    // Geometrically a is the signed volume of the parallelepiped
    // spanned by e1, dir, e2.  Its sign tells us which side of the
    // triangle the ray is hitting:
    //   a > 0  → front face (ray and geometric normal point opposite)
    //   a < 0  → back face  (ray and geometric normal point same way)

    // When |a| is near zero the ray is nearly parallel to the
    // triangle plane — no intersection.
    if (fabsf(a) < RAY_EPSILON)
        return false;

    float f = 1.0f / a;     // sign-preserving reciprocal
    // f > 0  → front-face hit,  f < 0 → back-face hit

    // ---- Step 3: barycentric u ----
    // s = ray.origin - v0  — vector from v0 to ray origin.
    glm::vec3 s = ray.origin - tri.v0;

    // u = f * dot(s, p)    — barycentric coordinate for edge e1.
    float u = f * glm::dot(s, p);
    // In the front-face case (f > 0) both f and dot(s, p) are
    // positive.  In the back-face case (f < 0) dot(s, p) flips sign
    // too, so u remains positive for a valid hit.
    if (u < 0.0f || u > 1.0f)
        return false;
    // u ∈ [0, 1] means the hit lies between v0 and v1 along e1.

    // ---- Step 4: barycentric v ----
    // q = s × e1  — perpendicular to both s and e1.
    glm::vec3 q = glm::cross(s, e1);
    // v = f * dot(dir, q)  — barycentric coordinate for edge e2.
    float v = f * glm::dot(ray.direction, q);
    // Same sign argument as u: f keeps the sign consistent.
    if (v < 0.0f || u + v > 1.0f)
        return false;

    // ---- Step 5: distance t ----
    // t = f * dot(e2, q)  — parametric distance along the ray.
    // A hit must be far enough from the origin to avoid the same
    // surface that was just scattered from (RAY_EPSILON guard).
    float t = f * glm::dot(e2, q);
    if (t < RAY_EPSILON)
        return false;

    outT = t;

    // ---- Step 6: interpolate vertex normal (smooth shading) ----
    // Executed ONLY on hit (rare path), zero overhead during traversal!
    glm::vec3 interp = (1.0f - u - v) * tri.n0 + u * tri.n1 + v * tri.n2;
    float len2 = glm::dot(interp, interp);
    if (len2 > RAY_EPSILON)
    {
        outNormal = interp * glm::inversesqrt(len2);
    }
    else
    {
        glm::vec3 geoNormal = glm::cross(e1, e2);
        float geoLen2 = glm::dot(geoNormal, geoNormal);
        outNormal = (geoLen2 > RAY_EPSILON) ? geoNormal * glm::inversesqrt(geoLen2) : tri.n0;
    }

    // Report the model's TRUE shading normal.  Opaque materials orient it
    // toward the ray in scatterRay; refraction reads its sign (dot with the
    // ray) to classify entry vs exit.  Flipping it here would erase the
    // model's winding and make classifyRefraction's exit branch unreachable.

    // ---- Step 7: interpolate the texture coordinate ----
    // Same barycentric weights as the normal above.  UVs live in texture
    // space, so unlike the normal there is no re-normalization step.
    outUv = (1.0f - u - v) * tri.uv0 + u * tri.uv1 + v * tri.uv2;

    // ---- Step 7.5: interpolate vertex color ----
    // Same barycentric weights.  Vertex colors are multiplied with
    // the base color in shading (glTF COLOR_0 × baseColorTexture).
    outVertexColor = (1.0f - u - v) * tri.c0 + u * tri.c1 + v * tri.c2;

    // ---- Step 8: per-triangle tangent (for tangent-space normal maps) ----
    // Derive the direction in which the texture's +U axis increases, from
    // the triangle's world-space edges and UV deltas.  Treat the triangle
    // as a 2×2 system in the tangent plane:
    //     e1 = ΔU1·T + ΔV1·B ,  e2 = ΔU2·T + ΔV2·B
    //     det = ΔU1·ΔV2 − ΔU2·ΔV1
    //     T   = (e1·ΔV2 − e2·ΔV1) / det     (world space, face plane)
    // Executed only on hit (rare path), like the normal interpolation.
    // UVs are texture-space, so this uses the model's own UV layout — the
    // bake never touches them.
    glm::vec2 du1 = tri.uv1 - tri.uv0;
    glm::vec2 du2 = tri.uv2 - tri.uv0;
    const float det = du1.x * du2.y - du2.x * du1.y;
    outTangent = glm::vec4(0.0f);   // sentinel: degenerate UVs / no tangent
    // TANGENT_DET_EPSILON guards the 2D UV triangle's area (det ≈ 0 → the solve below blows up);
    // TANGENT_EPSILON guards the derived 3D tangent's squared length (T ∥ N).
    if (fabsf(det) > TANGENT_DET_EPSILON)
    {
        glm::vec3 T = (e1 * du2.y - e2 * du1.y) * (1.0f / det);
        // Orthogonalize against the INTERPOLATED (smoothed) normal so the
        // TBN frame stays orthonormal under smooth shading.  A tangent that
        // collapses to ~parallel with N (flat geometry whose UV axis aligns
        // with the normal) keeps the sentinel.
        T = T - outNormal * glm::dot(outNormal, T);
        const float tlen2 = glm::dot(T, T);
        if (tlen2 > TANGENT_EPSILON)
        {
            // UV handedness (glTF TANGENT.w): the sign of the UV triangle's
            // 2D winding.  det > 0 = regular layout → w = +1; det < 0 =
            // mirrored island (author reused the map for both halves of a
            // symmetric mesh) → w = -1.  Shading computes
            // B = cross(N, T)·w so the green (+V) channel stays correct.
            const float handedness = (det < 0.0f) ? -1.0f : 1.0f;
            outTangent = glm::vec4(T * glm::inversesqrt(tlen2), handedness);
        }
    }

    return true;
}
