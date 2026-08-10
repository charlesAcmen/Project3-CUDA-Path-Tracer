/**
 * @file refraction_test.cu
 * @brief Standalone tests for the refractive branch of scatterRay in
 *        src/interactions/interactions.cu — specifically the TIR (total
 *        internal reflection) guard.
 *
 * Compiles with nvcc as a pure host program (no kernel launches).  It
 * #includes the REAL interactions.cu and rng.h from the path tracer so it
 * tests the actual implementation, not a re-implementation.
 *
 * Three layers:
 *   L1 premise    — glm::refract returns a NON-finite (NaN) vector on TIR,
 *                   NOT a zero vector.  Pins down the real GLM behavior the
 *                   guard was written for (an earlier "zero vector" claim
 *                   turned out to be wrong — see func_geometric.inl: the
 *                   result is multiplied by (k >= 0) AFTER sqrt(k) has been
 *                   computed, so k < 0 yields NaN, not zero).
 *   L2 invariant  — the guard's k < 0 test is exactly equivalent to
 *                   "glm::refract output is non-finite", across a sweep of
 *                   exit angles and IORs.  The guard and the call can never
 *                   disagree, because the guard evaluates the same expression
 *                   GLM uses internally (1 - eta²(1-dot²)).
 *   L3 behavior   — scatterRay never emits a NaN direction for a refractive
 *                   material at any exit angle; at TIR it deterministically
 *                   reflects (finite, unit-length, equal to glm::reflect).
 *
 * Exit code 0 = all passed; nonzero = a check failed.
 */

#include "interactions/interactions.cu"

#include "constants.h"

#include <cstdio>
#include <cmath>

// CHECK that always evaluates (assert() would be compiled out in Release).
#define CHECK(cond, ...) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "  FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n"); \
            return false; \
        } \
    } while (0)

static bool isFinite(const glm::vec3& v)
{
    return std::isfinite((double)v.x)
        && std::isfinite((double)v.y)
        && std::isfinite((double)v.z);
}

// ---------------------------------------------------------------------------
// L1: premise — GLM's refract returns NaN (not zero) on TIR
// ---------------------------------------------------------------------------
static bool testPremiseRefractNaNOnTIR()
{
    // Glass (ior 1.5) ray exiting toward air at θ ≈ 53.1°, beyond the
    // critical angle (41.8°).  Geometry matches scatterRay's exiting case:
    //   I             = incident ray inside the glass, heading out (+z)
    //   refractNormal = -surfaceNormal (points back into the glass)
    //   eta           = indexOfRefraction (n1/n2 for exit)
    const glm::vec3 I(0.8f, 0.0f, 0.6f);            // unit: sinθ=0.8, cosθ=0.6
    const glm::vec3 refractNormal(0.0f, 0.0f, -1.0f);
    const float eta = 1.5f;

    const float dotValue = glm::dot(refractNormal, I);
    const float k = 1.0f - eta * eta * (1.0f - dotValue * dotValue);
    CHECK(k < 0.0f, "test geometry broken: expected TIR (k=%g)", k);

    const glm::vec3 R = glm::refract(I, refractNormal, eta);
    const bool nonFinite = !isFinite(R);
    CHECK(nonFinite,
          "expected NaN on TIR, got R=(%g, %g, %g), len=%g.  If GLM now "
          "returns the zero vector (or a valid direction), the TIR guard in "
          "interactions.cu must be revisited.",
          R.x, R.y, R.z, glm::length(R));

    return true;
}

// ---------------------------------------------------------------------------
// L2: invariant — the scatterRay guard never misses / over-fires.
//     Ground truth: "refract() output is non-finite".  The guard's exact
//     expression (GLM-only: !(squared-length > REFRACT_VALID_SQ_LEN_MIN)) must match it, and both
//     must match the physics (TIR ⟺ k < 0).
// ---------------------------------------------------------------------------
static bool testGuardInvariant()
{
    // Sweep exit angles straight-out → near-grazing, for several IORs.
    // Exiting geometry: refractNormal = (0,0,-1), eta = ior.
    const float iors[] = { 1.2f, 1.5f, 2.0f };
    const glm::vec3 refractNormal(0.0f, 0.0f, -1.0f);

    for (float ior : iors)
    {
        for (float cosT = 0.999f; cosT > 0.001f; cosT -= 0.001f)
        {
            const float sinT = sqrtf(fmaxf(0.0f, 1.0f - cosT * cosT));
            const glm::vec3 I = glm::normalize(glm::vec3(sinT, 0.0f, cosT));

            const glm::vec3 R = glm::refract(I, refractNormal, ior);
            const bool nonFinite = !isFinite(R);

            // Physics: TIR (k < 0) ⟺ refract() produced a non-finite vector.
            const float dotValue = glm::dot(refractNormal, I);
            const float k = 1.0f - ior * ior * (1.0f - dotValue * dotValue);
            const bool tir = (k < 0.0f);
            CHECK(tir == nonFinite,
                  "ior=%g cosT=%g: TIR (k=%g<0=%d) but refract nonFinite=%d",
                  ior, cosT, k, (int)tir, (int)nonFinite);

            // The exact guard expression from scatterRay (GLM-only, no k
            // recomputation) must match that ground truth.  Note it relies on
            // NaN comparing false against ANY threshold: !(NaN > REFRACT_VALID_SQ_LEN_MIN) == true.
            const bool guardDegenerate = !(glm::dot(R, R) > REFRACT_VALID_SQ_LEN_MIN);
            CHECK(guardDegenerate == nonFinite,
                  "ior=%g cosT=%g: guard degenerate=%d but refract nonFinite=%d "
                  "(R=(%g, %g, %g))",
                  ior, cosT, (int)guardDegenerate, (int)nonFinite, R.x, R.y, R.z);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// L3: behavior — scatterRay never emits NaN; TIR deterministically reflects
// ---------------------------------------------------------------------------
static bool testScatterRayNeverNaN()
{
    // Glass slab.  Surface normal +z faces outward.  Exiting rays have
    // dot(I, normal) > 0 → HitSide::Inside → scatterRay refract branch.
    Material glass{};
    glass.type = MaterialType::Refractive;
    glass.color = glm::vec3(1.0f);
    glass.indexOfRefraction = 1.5f;
    glass.invIndexOfRefraction = 1.0f / 1.5f;

    const glm::vec3 normal(0.0f, 0.0f, 1.0f);   // geometric / shading normal

    int tirSeen = 0;
    int refractSeen = 0;
    for (float cosT = 0.999f; cosT > 0.05f; cosT -= 0.005f)
    {
        const float sinT = sqrtf(fmaxf(0.0f, 1.0f - cosT * cosT));
        const glm::vec3 I = glm::normalize(glm::vec3(sinT, 0.0f, cosT)); // exiting

        // Recompute the guard's k for this geometry (refractNormal = -normal).
        const glm::vec3 refractNormal = -normal;
        const float dotValue = glm::dot(refractNormal, I);
        const float k = 1.0f - glass.indexOfRefraction * glass.indexOfRefraction
                        * (1.0f - dotValue * dotValue);
        const bool tir = (k < 0.0f);
        if (tir) tirSeen++;
        else     refractSeen++;

        PathSegment p{};
        p.ray.origin = glm::vec3(0.0f);
        p.ray.direction = I;
        p.color = glm::vec3(1.0f);
        p.pixelIndex = 0;
        p.remainingBounces = 8;
        RngState rng = makeRngState(0, 0, 0, RngMode::LCG);   // fixed seed

        // Glass (refractive) never samples textures — empty hit record.
        // Ray origin is (0,0,0), so t = 0 reproduces intersect = (0,0,0).
        ShadeableIntersection hit{};
        hit.t = 0.0f;
        hit.surfaceNormal = normal;
        scatterRay(p, hit, glass, rng, TextureTable{});

        CHECK(isFinite(p.ray.direction),
              "cosT=%g: NaN direction out of scatterRay", cosT);

        const float len = glm::length(p.ray.direction);
        CHECK(fabsf(len - 1.0f) < 1e-3f,
              "cosT=%g: direction not unit (len=%g)", cosT, len);

        if (tir)
        {
            // TIR is not a choice — the result must be a deterministic
            // reflection (both Fresnel functions return 1.0 on TIR, and the
            // guard makes the reflection unconditional).
            const glm::vec3 expectedReflect = glm::reflect(I, normal);
            const float aligned = glm::dot(p.ray.direction, expectedReflect);
            CHECK(aligned > 0.999f,
                  "cosT=%g (TIR): expected reflection of (%g,%g,%g), "
                  "got (%g,%g,%g) dot=%g",
                  cosT, I.x, I.y, I.z,
                  p.ray.direction.x, p.ray.direction.y, p.ray.direction.z,
                  aligned);
        }
    }

    // Sanity: the sweep must have covered both regimes.
    CHECK(tirSeen > 0, "sweep never hit TIR — test geometry broken");
    CHECK(refractSeen > 0, "sweep never refracted — test geometry broken");

    return true;
}

// ---------------------------------------------------------------------------

int main()
{
    fprintf(stdout, "=== Refraction / TIR guard tests ===\n");

    bool allOk = true;

    bool r1 = testPremiseRefractNaNOnTIR();
    fprintf(stdout, "[%s] L1 premise: glm::refract returns NaN (not zero) on TIR\n",
            r1 ? "PASS" : "FAIL");
    allOk = allOk && r1;

    bool r2 = testGuardInvariant();
    fprintf(stdout, "[%s] L2 invariant: guard (degenerate output) == refract() non-finite == TIR\n",
            r2 ? "PASS" : "FAIL");
    allOk = allOk && r2;

    bool r3 = testScatterRayNeverNaN();
    fprintf(stdout, "[%s] L3 behavior: scatterRay never NaN (hardcoded Accurate Fresnel)\n",
            r3 ? "PASS" : "FAIL");
    allOk = allOk && r3;

    fprintf(stdout, "%s\n", allOk ? "ALL PASSED" : "FAILURES PRESENT");
    return allOk ? 0 : 1;
}
