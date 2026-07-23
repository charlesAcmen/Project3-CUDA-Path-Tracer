#pragma once

#include "sceneStructs.h"   // RngMode, shared with ShadingConfig

/**
 * @file rng.h
 * @brief Unified random number generation for GPU Monte Carlo path tracing.
 *
 * Provides two RNG modes with a uniform interface:
 *   LCG    — thrust::default_random_engine (backward compatible, default)
 *   HALTON — multi-dimensional Cranley-Patterson scrambled Halton
 *
 * Key design for Halton mode:
 *   rng.next(dim) uses HALTON_PRIMES[dim] as the prime base.  All calls
 *   within one bounce share the same haltonIndex — this is proper multi-
 *   dimensional Halton: different prime bases at the same index N form a
 *   well-distributed d-dimensional point.
 *
 *   haltonIndex = baseOffset(pixelIndex, bounceIndex) + iter
 *   The baseOffset is a chained hash of (pixelIndex, bounceIndex), giving
 *   each (pixel, bounce) pair a unique start position and breaking the
 *   structured aliasing from pixel×stride formulas.  Adding iter makes the
 *   walk CONSECUTIVE across frames, preserving low-discrepancy convergence.
 *
 *   Cranley-Patterson rotation (per-pixel, per-iter, per-bounce, per-dim
 *   offset) decorrelates adjacent pixels while keeping each pixel stratified.
 *
 * Usage:
 *   RngState rng = makeRngState(iter, pixelIdx, depth, rngMode);
 *   float u = rng.next(dim);  // dim = 0..HALTON_NUM_DIMS-1
 */

#include "constants.h"

#include <thrust/random.h>

// ============================================================================
// utilhash — seed mixer for RNG
// ============================================================================

/**
 * Jenkins-style bit-mixing hash used to seed random number generators.
 * Provides spatial, temporal, and bounce-depth decorrelation when
 * combined with pixel index, iteration, and bounce depth.
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

constexpr int HALTON_NUM_DIMS = 16;

// --- Halton dimension assignment ---
// Each independent sampling decision in the pipeline gets a unique
// dimension index.  Currently 10 dimensions allocated (0–9); 6 remain
// available for future features (e.g., direct lighting).
//
//   Dim  Prime  Usage                          Location
//   ---  -----  -----------------------------  ----------------------------
//    0     2    AA jitter x                    generateRayFromCamera
//    1     3    AA jitter y                    generateRayFromCamera
//    2     5    Lens aperture u                generateRayFromCamera (DoF)
//    3     7    Lens aperture v                generateRayFromCamera (DoF)
//    4    11    Diffuse hemisphere θ           calculateRandomDirectionInHemisphere
//    5    13    Diffuse hemisphere φ           calculateRandomDirectionInHemisphere
//    6    17    Specular lobe θ                samplePhongSpecularDir
//    7    19    Specular lobe φ                samplePhongSpecularDir
//    8    23    Fresnel roulette               scatterRay (refractive branch)
//    9    29    Path Russian roulette          russianRouletteTerminate
//   ---  -----  -----------------------------  ----------------------------

/** Named constants for Halton dimension indices.
 *
 *  Use in place of raw integers at every rng.next() call site:
 *    rng.next(HaltonDim::AaJitterX)     instead of  rng.next(0)
 *    rng.next(HaltonDim::DiffuseTheta)  instead of  rng.next(4)
 *
 *  Dimensions 0-9 are allocated; 10-15 are reserved for future use.
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
}

/**
 * Returns the n-th prime number for use as a Halton sequence base.
 *
 * Accessible from both host and device code (avoids CUDA's host/device
 * symbol visibility issues with constexpr arrays).
 *
 * Halton dimension → prime base mapping:
 *   dim 0 → 2,   1 → 3,   2 → 5,   3 → 7,
 *   dim 4 → 11,  5 → 13,  6 → 17,  7 → 19,
 *   dim 8 → 23,  9 → 29,  10 → 31, 11 → 37,
 *   dim 12 → 41, 13 → 43, 14 → 47, 15 → 53
 */
__host__ __device__ inline int getHaltonPrime(int dim) {
    // constexpr array lets the compiler emit a single indexed load
    // from constant memory instead of a 16-branch if-else chain.
    constexpr int primes[] = {
        2, 3, 5, 7, 11, 13, 17, 19,
        23, 29, 31, 37, 41, 43, 47, 53
    };
    // Clamp to [0, 15] -- dim >= 16 is out of range for the current
    // dimension allocation (max used is 9).  Returning prime 2 for
    // out-of-range dims at least avoids a crash, but the Halton
    // sequence would collide with dim 0, so callers MUST stay in range.
    return primes[(dim < HALTON_NUM_DIMS) ? dim : (HALTON_NUM_DIMS - 1)];
}

/**
 * Encodes bounce number for the `depth` argument of makeRngState.
 * Pass depth = bounceNum * MAX_DRAWS_PER_BOUNCE so each bounce gets a
 * distinct bounceIndex value inside the hash.  The actual multiplier (8)
 * is arbitrary — any value works since bounceIndex goes into a chained hash.
 */
constexpr int MAX_DRAWS_PER_BOUNCE = 8;

// ============================================================================
// RNG mode selection  (RngMode defined in sceneStructs.h, shared with
// ShadingConfig and PathTracerOptions)
// ============================================================================

// ============================================================================
// Halton radical inverse
// ============================================================================

/**
 * Halton radical inverse: computes the n-th term of the Halton sequence
 * in the given prime base.
 *
 *   Phi_b(n) = sum_{k=0}^{m-1} d_k * b^{-(k+1)}
 *
 * where n = sum_{k=0}^{m-1} d_k * b^k is the base-b digit representation.
 *
 * Example (base 2):
 *   n=0 ->           -> 0.0_2     = 0.0
 *   n=1 -> "1"       -> 0.1_2     = 0.5
 *   n=2 -> "10"      -> 0.01_2    = 0.25
 *   n=3 -> "11"      -> 0.11_2    = 0.75
 *   n=4 -> "100"     -> 0.001_2   = 0.125
 *
 * @param base  Prime base (e.g. 2, 3, 5, 7, ...)
 * @param n     Sequence index (0-based)
 * @return      The n-th Halton sample in [0, 1)
 */
__host__ __device__ inline float radicalInverse(int base, unsigned int n)
{
    float invBase = 1.0f / (float)base;
    float invBaseN = invBase;
    float result = 0.0f;

    while (n > 0) {
        unsigned int digit = n % (unsigned int)base;
        result += (float)digit * invBaseN;
        invBaseN *= invBase;
        n /= (unsigned int)base;
    }
    return result;
}

// ============================================================================
// Cranley-Patterson rotation
// ============================================================================

/**
 * Cranley-Patterson rotation: shifts a Halton sample by a per-pixel,
 * per-dimension random offset, wrapped modulo 1.0.
 *
 *   result = (x + offset) mod 1.0
 *
 * The offset decorrelates different pixels' sequences while preserving
 * the low-discrepancy property within each pixel's own sequence.
 *
 * @param x       Raw Halton sample in [0, 1)
 * @param offset  Per-pixel per-dimension offset in [0, 1)
 * @return        Rotated sample in [0, 1)
 */
__host__ __device__ inline float cpRotate(float x, float offset)
{
    float val = x + offset;
    if (val >= 1.0f) val -= 1.0f;
    return val;
}

/**
 * Produces a hash-based starting offset for the Halton sequence, unique per
 * (pixelIndex, bounceIndex) pair.
 *
 * Uses CHAINED HASHING (not linear sum) to avoid any collision between the
 * two input components: each value passes through a full utilhash mix step
 * before combining.  Linear sum (a·p1 + b·p2) can collide when a·p1 = b·p2;
 * for typical render depths this doesn't arise, but the chained form is at
 * the correct end of the spectrum.
 *
 * The offset is then added to `iter` in makeRngState, so each pixel's Halton
 * index WALKS CONSECUTIVELY across iterations:
 *   iter=0: haltonIndex = baseOffset + 0
 *   iter=1: haltonIndex = baseOffset + 1   ← consecutive!
 *   iter=2: haltonIndex = baseOffset + 2   ← consecutive!
 *
 * Consecutive Halton indices are what give the sequence its O(log^d N / N)
 * low-discrepancy convergence.  The hash start eliminates the structured
 * aliasing from a linear offset (pixelIndex × stride), while the consecutive
 * walk preserves the filling property.
 *
 * Golden-ratio constants (0x9e3779b9, 0x85ebca6b) are Bob Jenkins'
 * proven mix additives — standard practice for chained hashing.
 */
__host__ __device__ inline unsigned int mixHaltonBaseOffset(
    unsigned int pixelIndex,
    unsigned int bounceIndex)
{
    unsigned int h = utilhash(pixelIndex + 0x9e3779b9u);
    h = utilhash(h ^ (bounceIndex + 0x85ebca6bu));
    return h;
}

// ============================================================================
// RngState — unified RNG interface (LCG or Halton)
// ============================================================================

/**
 * Unified RNG state.  Wraps both modes behind a uniform .next(dim) API.
 *
 * LCG mode (dim ignored):
 *   Delegates to thrust::default_random_engine, backward compatible.
 *
 * Halton mode:
 *   - All draws share haltonIndex (set per-bounce by makeRngState).
 *   - next(dim) selects the prime base by dim, computes radicalInverse,
 *     then applies Cranley-Patterson rotation with a seed derived from
 *     pixelIndex × iter × dim — this decorrelates adjacent pixels while
 *     varying per iteration to prevent coherent stripe accumulation.
 *   - The index does NOT advance — every draw in a bounce is a different
 *     dimension of the SAME multi-dimensional Halton point.
 *   - Each bounce gets a fresh RngState with a new index.
 *
 * The if-else on mode is warp-uniform (all threads read the same global
 * flag), so there is zero divergence cost.
 */
struct RngState {
    RngMode mode;

    // -- LCG branch (16 bytes) --
    thrust::default_random_engine lcgEngine;

    // -- Halton branch (16 bytes) --
    unsigned int haltonIndex;   // baseOffset(pixel, bounce) + iter — consecutive Halton index
    unsigned int pixelIndex;    // for CP offset decorrelation (per-pixel)
    int iter;                   // for CP offset per-iteration variation
    unsigned int bounceIndex;   // for CP offset / index decorrelation (per bounce)

    /** Returns a uniform random float in [0, 1) for the given dimension. */
    __host__ __device__ float next(int dim) {
        if (mode == RngMode::LCG) {
            thrust::uniform_real_distribution<float> u01(0, 1);
            return u01(lcgEngine);
        } else {
            // All dimensions within a bounce share the SAME haltonIndex.
            // This is proper multi-dimensional Halton: different prime bases
            // at the same index N form a well-distributed d-dimensional point.
            // (If we hashed dim into the index, each dim would be at a
            //  different pseudo-random position — losing the correlation
            //  structure that makes multi-dimensional Halton converge fast.)
            int base = getHaltonPrime(dim);
            float raw = radicalInverse(base, haltonIndex);

            // Cranley-Patterson rotation decorrelates adjacent pixels' raw
            // Halton values.  The CP seed combines pixelIndex (per-pixel),
            // iter (per-iteration), bounceIndex (per-bounce — prevents
            // identical offsets across bounces), and dim (per-dimension),
            // all with distinct prime multipliers.
            unsigned int h = utilhash(
                (unsigned int)pixelIndex * 131u
                + bounceIndex * 17u
                + (unsigned int)dim * 11u);
            float offset = (float)(h & 0xFFFFFFu) * (1.0f / 16777216.0f);
            return cpRotate(raw, offset);
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
 *   haltonIndex = baseOffset(pixelIndex, bounceIndex) + iter
 *     - baseOffset = chained_hash(pixelIndex, bounceIndex)
 *       gives each (pixel, bounce) pair a unique starting position,
 *       breaking the structured aliasing from pixel×stride formulas.
 *     - Adding iter makes the walk CONSECUTIVE across frames,
 *       preserving O(log^d N / N) low-discrepancy convergence.
 *     - Primary rays:  bounceIndex = 0
 *     - Bounce N:      bounceIndex = N * MAX_DRAWS_PER_BOUNCE
 *   pixelIndex, iter, and bounceIndex are stored separately for the
 *   CP offset seed in next(dim), providing per-pixel, per-iteration,
 *   per-bounce, and per-dimension decorrelation.
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
        // Replicated from makeSeededRandomEngine (pathtrace.cu):
        //   hash(depth, iter) ^ hash(pixelIndex)
        int h = utilhash((1u << 31) | ((unsigned int)depth << 22) | (unsigned int)iter)
                ^ utilhash((unsigned int)pixelIndex);
        state.lcgEngine = thrust::default_random_engine(h);
    } else {
        state.haltonIndex = mixHaltonBaseOffset(
            (unsigned int)pixelIndex,
            (unsigned int)depth)
            + (unsigned int)iter;
        state.pixelIndex  = (unsigned int)pixelIndex;
        state.iter        = iter;
        state.bounceIndex = (unsigned int)depth;
    }
    return state;
}
