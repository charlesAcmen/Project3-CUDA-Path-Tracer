#pragma once

#include "sceneStructs.h"   // RngMode, shared with ShadingConfig

/**
 * @file rng.h
 * @brief Unified random number generation for GPU Monte Carlo path tracing.
 *
 * Provides two RNG modes with a uniform interface:
 *   LCG    — thrust::default_random_engine
 *   HALTON — multi-dimensional Owen-scrambled Halton
 *
 * Key design for Halton mode:
 *   rng.next(dim) uses HALTON_PRIMES[dim] as the prime base.  All calls
 *   within one bounce share the same haltonIndex.
 *
 *   haltonIndex = iter
 *   The index walks CONSECUTIVELY across frames (0, 1, 2, …), which is
 *   exactly the ordering that gives Halton its O((log N)^d / N)
 *   low-discrepancy convergence guarantee.  iter is deliberately excluded
 *   from the scramble seed (see next()): if the seed changed every frame,
 *   each frame would be re-scrambled and the accumulated frames would no
 *   longer be the prefix of a single low-discrepancy sequence — the
 *   estimate would fall back to Monte-Carlo rate instead of QMC rate.
 *
 *   TRUE nested Owen scrambling (Owen 1997) replaces the old Cranley-
 *   Patterson rotation for per-pixel decorrelation.  The digit permutation
 *   at each base-b digit level depends on the digits that are more
 *   significant in the radical-inverse value, already read (the prefix):
 *       d'_k = pi_k(prefix_k)(d_k)
 *   This prefix-dependence is the definition of Owen scrambling — a weaker
 *   "linear" scramble (fixed permutation per level) was used first; it kept
 *   the net but left pixels nearly identical, see BUG 5.  Per-pixel
 *   decorrelation is completed by a fixed per-(pixel, bounce, dim) float
 *   toroidal shift (standard randomized-QMC / CP style, net-preserving in
 *   expectation over the shift — a fixed shift breaks the exact net, which
 *   is fine and standard).  The per-pixel per-bounce per-dim seed is
 *   derived from a chained hash (no linear-sum collisions).
 *
 *   KNOWN LIMITATION: for Halton's mixed prime bases, no digit-structure-
 *     preserving scramble can fully decorrelate pixels (small bases have few
 *     digit permutations; a strong base-2 hash (Burley-style) breaks the
 *     cross-dimensional net, see BUG 4).  True per-pixel independence would
 *     require an all-base-2 sequence (Sobol + Burley hash).
 *
 *   MEASURED LOW-SPP BEHAVIOR (2026-08, tests/rng_test, 16 px x 16k iter):
 *     At N ≲ a few hundred, Halton renders NOISIER than LCG.  This is
 *     EXPECTED, not a bug, and has two independent causes:
 *       (a) LARGE-PRIME-BASE CLUSTERING: dims 4-9 use bases 11..29; a base-b
 *           dimension pair is only well-distributed once N ≫ b².  Measured
 *           π-integral error (Owen vs LCG): the diffuse pair (11,13) is
 *           WORSE at N=16 (0.9x); the AA pair (2,3) crosses over near N≈64,
 *           the diffuse pair near N≈256.  The specular/RR dims (17..29) are
 *           the worst low-SPP offenders (variance spikes / fireflies).
 *       (b) STRUCTURED (NON-WHITE) ERROR: cross-pixel correlation (see KNOWN
 *           LIMITATION) makes the residual noise spatially correlated, which
 *           reads as "dirtier" than LCG's independent grain at the same RMS.
 *     The benefit is ASYMPTOTIC and COMPOUNDING: 1D star discrepancy 13.7x
 *     lower than LCG; 2D π-integral error equals ~55x more LCG samples at
 *     N=16384 (log-log slope -0.78 vs -0.59 for LCG, -1.0 QMC ideal).
 *     Practical crossover where Halton becomes visibly cleaner: simple scenes
 *     (low effective dimension) ~ a few hundred iters; complex scenes
 *     (specular/refractive/small lights) ~ a few thousand.
 *
 *   QMC ACROSS BOUNCES: bounceIndex is mixed into the seed, so each bounce
 *     gets an independent scramble.  This is standard per-bounce independence
 *     but means the JOINT of a path's bounces is a product of nets, not a
 *     single high-dimensional net (pure cross-bounce QMC would need one
 *     Halton dimension per (bounce, decision), which Halton cannot afford at
 *     depth — few primes, and high-dim Halton degrades).
 *
 * Usage:
 *   RngState rng = makeRngState(iter, pixelIdx, depth, rngMode);
 *   float u = rng.next(dim);  // dim indexes the Halton dimension table below
 */

#include "constants.h"

#include <thrust/random.h>

// ============================================================================
// utilhash — seed mixer for RNG
// ============================================================================

/**
 * Jenkins-style bit-mixing hash used to seed random number generators.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// ============================================================================
// Halton sequence constants
// ============================================================================

// --- Halton dimension assignment ---
// Each independent sampling decision in the pipeline gets a unique
// dimension index.
//
// Larger primes (higher dims) suffer large-base clustering at low sample
// counts — see "MEASURED LOW-SPP BEHAVIOR" in the file header.
//
//   Dim  Prime  Usage                          Location
//   ---  -----  -----------------------------  ----------------------------
//    0     2    AA jitter x                    generateRayFromCamera
//    1     3    AA jitter y                    generateRayFromCamera
//    2     5    Lens aperture u                generateRayFromCamera (DoF)
//    3     7    Lens aperture v                generateRayFromCamera (DoF)
//    4    11    Diffuse hemisphere θ           calculateRandomDirectionInHemisphere
//    5    13    Diffuse hemisphere φ           calculateRandomDirectionInHemisphere
//    6    17    Specular lobe θ (GGX half-vector)  sampleGgxHalfVector
//    7    19    Specular lobe φ (GGX half-vector)  sampleGgxHalfVector
//    8    23    Fresnel roulette               scatterRay (refractive branch)
//    9    29    Path Russian roulette          russianRouletteTerminate
//   10    31    Diffuse/specular split (PBR)   scatterRay (GGX surface)
//   ---  -----  -----------------------------  ----------------------------

/** Named constants for Halton dimension indices.
 *
 *  Use in place of raw integers at every rng.next() call site:
 *    rng.next(HaltonDim::AaJitterX)     instead of  rng.next(0)
 *    rng.next(HaltonDim::DiffuseTheta)  instead of  rng.next(4)
 *
 *  
 */
namespace HaltonDim {
    constexpr int AaJitterX      = 0;
    constexpr int AaJitterY      = 1;
    constexpr int LensApertureU  = 2;
    constexpr int LensApertureV  = 3;
    constexpr int DiffuseTheta   = 4;
    constexpr int DiffusePhi     = 5;
    constexpr int SpecularTheta  = 6;
    constexpr int SpecularPhi    = 7;
    constexpr int FresnelRR      = 8;
    constexpr int PathRR         = 9;
    constexpr int PbrSplit       = 10;   // diffuse vs specular roulette (GGX surface)
}

/**
 * Encodes bounce number for the `depth` argument of makeRngState.
 * Pass depth = bounceNum * MAX_DRAWS_PER_BOUNCE so each bounce gets a
 * distinct bounceIndex value inside the hash.  The actual multiplier (8)
 * is arbitrary — any value works since bounceIndex goes into a chained hash.
 */
constexpr int MAX_DRAWS_PER_BOUNCE = 16;

// ============================================================================
// RNG mode selection  (RngMode defined in sceneStructs.h, shared with
// ShadingConfig and PathTracerOptions)
// ============================================================================

/**
 * Cranley-Patterson rotation: shifts a Halton sample by a per-pixel,
 * per-dimension random offset, wrapped modulo 1.0.
 *
 *   result = (x + offset) mod 1.0
 */
__host__ __device__ inline float cpRotate(float x, float offset)
{
    float val = x + offset;
    if (val >= 1.0f) val -= 1.0f;
    return val;
}

// ============================================================================
// Owen Scrambling — nested digit permutations (Owen 1997)
// ============================================================================

/**
 * TRUE (nested) Owen-scrambled radical inverse for a prime base BASE.
 *
 * Owen scrambling (Owen 1997) permutes each digit of the radical inverse,
 * with the permutation at digit level k depending on the digits already
 * read that are more significant in the radical-inverse value (these are
 * the low-order digits of the integer n) — the "prefix":
 *     d'_k = pi_k(prefix_k)(d_k)
 * A prefix-dependent digit bijection is what distinguishes TRUE Owen
 * scrambling from linear/digit scrambling (fixed permutation per level).
 * Nested scrambling preserves the low-discrepancy net for every N; it is
 * the correct definition of Owen scrambling.
 *
 * The per-level permutation here is the affine map d -> (a·d + c) mod b
 * (branch-free; bijective for prime b since a ∈ [1,b-1]), with (a, c)
 * derived from a hash of (prefix, level) — so it depends on the more
 * significant digits already read, per the definition.  This is a
 * restricted instantiation of the full nested-Owen family; see the note in
 * next() about decorrelation.
 *
 * BASE is a template parameter so every `%` and `/` below is a compile-time
 * constant — nvcc folds them into multiply-shift sequences instead of
 * emitting slow integer division (~20+ instructions per op on the GPU).
 * Dispatch from a runtime dim via the owenRadicalInverse(int, ...) overload
 * below.
 */
template <int BASE>
__host__ __device__ inline float owenRadicalInverse(
    unsigned int n, unsigned int seed)
{
    constexpr float invBase = 1.0f / (float)BASE;
    float invBaseN = invBase;
    float result   = 0.0f;
    unsigned int prefix = seed;   // running hash of the more-significant fractional digits read so far
    unsigned int level  = 0;
    while (n > 0) {
        unsigned int digit = n % BASE;

        // Prefix-dependent digit permutation (TRUE nested Owen).
        unsigned int h = utilhash(prefix + level * 0x9e3779b9u + 0x1f123bb5u);
        unsigned int a = (h >> 8)  % BASE;
        unsigned int c = (h >> 16) % BASE;
        if (a == 0) a = BASE - 1;
        unsigned int permuted = (a * digit + c) % BASE;

        result += (float)permuted * invBaseN;
        invBaseN *= invBase;

        // Fold this digit into the prefix for the next (lower) level.
        prefix = utilhash(prefix ^ (digit * 0x85ebca6bu));
        n /= BASE;
        level++;
    }
    return result;
}

/**
 * Runtime-dim dispatch to the templated owenRadicalInverse<BASE>.
 *
 * Every case passes a literal prime (the same list as getHaltonPrime), so
 * the compiler can constant-fold all the integer division/modulo in the
 * digit loop.  At the renderer call sites `dim` is a compile-time HaltonDim
 * constant, so the switch collapses to the single matching case and the
 * fold is total.  A runtime `dim` (e.g. the rng_compare test) still works —
 * the switch just isn't folded.  Out-of-range dims fall back to base 53,
 * matching getHaltonPrime's clamp for dim >= 16.
 */
__host__ __device__ inline float owenRadicalInverse(
    int dim, unsigned int n, unsigned int seed)
{
    switch (dim) {
        case 0:  return owenRadicalInverse<2> (n, seed);
        case 1:  return owenRadicalInverse<3> (n, seed);
        case 2:  return owenRadicalInverse<5> (n, seed);
        case 3:  return owenRadicalInverse<7> (n, seed);
        case 4:  return owenRadicalInverse<11>(n, seed);
        case 5:  return owenRadicalInverse<13>(n, seed);
        case 6:  return owenRadicalInverse<17>(n, seed);
        case 7:  return owenRadicalInverse<19>(n, seed);
        case 8:  return owenRadicalInverse<23>(n, seed);
        case 9:  return owenRadicalInverse<29>(n, seed);
        case 10: return owenRadicalInverse<31>(n, seed);
        case 11: return owenRadicalInverse<37>(n, seed);
        case 12: return owenRadicalInverse<41>(n, seed);
        case 13: return owenRadicalInverse<43>(n, seed);
        case 14: return owenRadicalInverse<47>(n, seed);
        default: return owenRadicalInverse<53>(n, seed);  // dim 15 and out-of-range
    }
}

// ============================================================================
// RngState — unified RNG interface
// ============================================================================

/**
 * Unified RNG state.  Wraps both modes behind a uniform .next(dim) API.
 *
 * LCG mode (dim ignored):
 *   Delegates to thrust::default_random_engine, backward compatible.
 *
 * Halton mode (true nested-Owen Halton):
 *   - haltonIndex advances consecutively across iterations (= iter).
 *     This is the ordering that gives Halton its low-discrepancy guarantee.
 *   - All draws within one bounce share the same haltonIndex.  Different
 *     dimensions use different prime bases → proper multi-dimensional point.
 *   - next(dim) applies owenRadicalInverse(): TRUE nested Owen scrambling
 *     (per-level digit permutation depends on the prefix), with a per-pixel
 *     per-bounce per-dim seed (chained utilhash — no linear-sum collisions),
 *     plus a fixed per-(pixel, bounce, dim) float toroidal shift for
 *     decorrelation.  The result is a stratified, pixel-decorrelated sample
 *     in [0, 1).
 *   - The index does NOT advance within a bounce — every draw is a
 *     different dimension of the SAME scrambled Halton point.
 *   - Each bounce gets a fresh RngState with the same iter-based index
 *     but a different bounceIndex encoded in the scramble seed.
 *
 */
struct RngState {
    RngMode mode;

    // -- LCG branch --
    thrust::default_random_engine lcgEngine;

    // -- Halton branch --
    unsigned int haltonIndex;   // = iter (consecutive across frames)
    unsigned int pixelIndex;    // for Owen seed (per-pixel decorrelation)
    unsigned int bounceIndex;   // for Owen seed (per-bounce decorrelation)

    /** Returns a uniform random float in [0, 1) for the given dimension. */
    __host__ __device__ float next(int dim) {
        if (mode == RngMode::LCG) {
            thrust::uniform_real_distribution<float> u01(0, 1);
            return u01(lcgEngine);
        } else {
            // All dimensions within a bounce share the SAME haltonIndex.
            // This is proper multi-dimensional Halton: different prime bases
            // at the same index N form a well-distributed d-dimensional point.

            // TRUE nested Owen scramble (NOT a base-2-only Burley-style hash
            // scramble — for bases 3,5,7,11,… that destroys the
            // cross-dimensional net structure, see the file header).  Per-level
            // digit permutation depends on the prefix (the more-significant
            // fractional digits of the value), per the Owen definition.
            //
            // haltonIndex = iter stays a clean consecutive walk across frames
            // (NO per-pixel index jump).  The scramble seed below deliberately
            // EXCLUDES iter: the seed must be frame-independent so each frame
            // is the next point of the SAME scrambled sequence — a per-frame
            // seed would re-scramble every frame and the accumulated frames
            // would lose the low-discrepancy prefix property (QMC rate).
            //
            // Pixel/bounce/dim decorrelation comes from the chained-hash seed.
            // NOTE on decorrelation: nested digit scrambling preserves the net
            // but, for Halton's mixed prime bases, cannot fully decorrelate
            // pixels (small bases have few digit permutations).  The fixed
            // per-(pixel, bounce, dim) float toroidal shift below (standard
            // randomized-QMC / CP style, net-preserving in expectation over
            // the shift) closes most of that gap: mean
            // |cross-pixel corr| ~0.1-0.3 (adjacent pixels ~0.37; the worst
            // pairs of many still reach ~0.9) vs ~0.9+ without it.
            // See "MEASURED LOW-SPP BEHAVIOR" in the file header.
            unsigned int seed = utilhash(
                utilhash((unsigned int)pixelIndex  + 0x9e3779b9u)
                ^ ((unsigned int)bounceIndex       * 0x85ebca6bu)
                ^ ((unsigned int)dim               * 0xc2b2ae35u));

            // Dispatch on dim: at renderer call sites dim is a compile-time
            // HaltonDim constant, so this resolves to a single templated
            // owenRadicalInverse<BASE> with all div/mod constant-folded.
            float s = owenRadicalInverse(dim, haltonIndex, seed);
            float rot = (float)(seed & 0xFFFFFFu) * (1.0f / 16777216.0f);  // per-(pixel,bounce,dim) decorrelator
            return cpRotate(s, rot);
        }
    }
};

// ============================================================================
// makeRngState — factory function
// ============================================================================

/**
 * Creates an RngState in the requested mode.
 *
 * LCG mode:
 *   Seeds a thrust::default_random_engine using the same utilhash-based
 *   formula as the original makeSeededRandomEngine.  Note: the depth
 *   parameter now uses a bounce-indexed schedule (bounceNum * 8) instead
 *   of the old descending-remainingBounces schedule, so LCG sequences
 *   per-bounce are not identical to the original -- only the seed
 *   derivation formula is the same.
 *
 * Halton mode:
 *   haltonIndex = iter
 *     - All pixels share the same logical index counter.  Per-pixel
 *       decorrelation is handled entirely by the Owen seed in next(dim),
 *       which embeds pixelIndex and bounceIndex via chained hash.
 *     - Consecutive iter values → consecutive Halton indices → the
 *       sequence fills [0,1)^d at rate O((log N)^d / N).
 *
 * @param iter        Current iteration (frame) counter
 * @param pixelIndex  Linear pixel index
 * @param depth       bounceNum * MAX_DRAWS_PER_BOUNCE (bounce encoding)
 * @param mode        RNG mode (LCG or HALTON)
 * @return            Initialised RngState
 */
__host__ __device__ inline RngState makeRngState(
    int iter, int pixelIndex, int depth, RngMode mode)
{
    RngState state;
    state.mode = mode;

    if (mode == RngMode::LCG) {
        // Same seed formula as the original makeSeededRandomEngine (the
        // function itself was removed from pathtrace.cu):
        //   hash(depth, iter) ^ hash(pixelIndex)
        int h = utilhash((1u << 31) | ((unsigned int)depth << 22) | (unsigned int)iter)
                ^ utilhash((unsigned int)pixelIndex);
        state.lcgEngine = thrust::default_random_engine(h);
    } else {
        // The Halton index is derived from the iteration counter (iter).  While iter can
        // theoretically exceed 2^32 after ~4 billion samples, this is impractical for
        // interactive rendering (convergence occurs within 10k iterations).  If iter does
        // overflow, the sequence wraps to the start — low-discrepancy properties remain
        // intact because Halton sequences are periodic (period = infinity for irrational
        // bases, but the mantissa limits effective period to ~2^23 for float precision).
        // In practice, overflow is not a concern for any realistic use case.
        state.haltonIndex = (unsigned int)iter;
        state.pixelIndex  = (unsigned int)pixelIndex;
        state.bounceIndex = (unsigned int)depth;
    }
    return state;
}
