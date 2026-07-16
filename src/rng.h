#pragma once

/**
 * @file rng.h
 * @brief Unified random number generation for GPU Monte Carlo path tracing.
 *
 * Provides two RNG modes with a uniform interface:
 *   LCG    — thrust::default_random_engine (backward compatible, default)
 *   HALTON — multi-dimensional Cranley-Patterson scrambled Halton
 *
 * Key design for Halton mode:
 *   rng.next(dim) uses HALTON_PRIMES[dim] as the prime base and derives
 *   the Cranley-Patterson offset from (pixelIndex, dim).  All calls within
 *   one bounce share the same haltonIndex, forming a proper multi-
 *   dimensional low-discrepancy point.  Each bounce gets its own unique
 *   index via depth = bounceNum * MAX_DRAWS_PER_BOUNCE in makeRngState,
 *   so no indices overlap between bounces.
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
 *
 * Originally defined in intersections.h (moved to rng.h for semantic
 * cohesion — hashing is an RNG concern, not an intersection concern).
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
// dimension index.  Dimensions 0-9 are currently allocated:
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
//   10+  31+   Available for extensions (MIS, light sampling, etc.)

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
    if (dim == 0) return 2;
    if (dim == 1) return 3;
    if (dim == 2) return 5;
    if (dim == 3) return 7;
    if (dim == 4) return 11;
    if (dim == 5) return 13;
    if (dim == 6) return 17;
    if (dim == 7) return 19;
    if (dim == 8) return 23;
    if (dim == 9) return 29;
    if (dim == 10) return 31;
    if (dim == 11) return 37;
    if (dim == 12) return 41;
    if (dim == 13) return 43;
    if (dim == 14) return 47;
    if (dim == 15) return 53;
    return 2; // fallback (should not be reached)
}

constexpr int HALTON_NUM_DIMS = 16;

/**
 * Inter-iteration stride for the Halton sequence index.
 *
 * haltonIndex = iter * HALTON_STRIDE + bounceNum * MAX_DRAWS_PER_BOUNCE
 *
 * Two overlapping constraints determine STRIDE:
 *
 *   1. Between consecutive bounces: the gap is MAX_DRAWS_PER_BOUNCE = 8.
 *      Since no bounce uses more than 8 draws (dims 4-9 = 6, plus 2 spare),
 *      each bounce occupies a unique non-overlapping index range.
 *
 *   2. Between consecutive iterations: STRIDE must exceed the total index
 *      span of all bounces in one iteration, i.e.
 *        STRIDE > traceDepth × MAX_DRAWS_PER_BOUNCE
 *      Otherwise iter N's last bounces and iter N+1's first bounces collide.
 *
 *   → STRIDE = 128 = 16 × 8, supports traceDepth up to 16.
 *   → Scene DEPTH is typically 4-12 (cornell.json uses 8), so this is
 *     generous.  If a scene somehow needs DEPTH > 16, bump STRIDE.
 */
constexpr unsigned int HALTON_STRIDE = 128;

/**
 * Halton index slots reserved per bounce.
 * Pass depth = bounceNum * MAX_DRAWS_PER_BOUNCE to makeRngState.
 */
constexpr int MAX_DRAWS_PER_BOUNCE = 8;

// ============================================================================
// RNG mode selection
// ============================================================================

enum class RngMode : int {
    LCG    = 0,  // thrust::default_random_engine (backward compatible)
    HALTON = 1   // Cranley-Patterson scrambled Halton
};

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
 *   - next(dim) selects base from HALTON_PRIMES[dim], computes
 *     radicalInverse, applies CP rotation with per-(pixelIndex, dim)
 *     offset.  The index does NOT advance — every draw in a bounce is
 *     a different dimension of the SAME multi-dimensional Halton point.
 *   - Each bounce gets a fresh RngState with a new index.
 *
 * The if-else on mode is warp-uniform (all threads read the same global
 * flag), so there is zero divergence cost.
 */
struct RngState {
    RngMode mode;

    // -- LCG branch (16 bytes) --
    thrust::default_random_engine lcgEngine;

    // -- Halton branch (8 bytes) --
    unsigned int haltonIndex;   // shared index across all dims for this bounce
    unsigned int pixelIndex;    // for per-dim CP offset computation

    /** Returns a uniform random float in [0, 1) for the given dimension. */
    __host__ __device__ float next(int dim) {
        if (mode == RngMode::LCG) {
            thrust::uniform_real_distribution<float> u01(0, 1);
            return u01(lcgEngine);
        } else {
            // Select prime base from the dimension index
            int base = getHaltonPrime(dim);
            // Halton sample at the shared index for this bounce
            float raw = radicalInverse(base, haltonIndex);
            // CP offset: per (pixelIndex, dim), deterministic, uniform in [0, 1)
            unsigned int h = utilhash(
                (unsigned int)pixelIndex * (unsigned int)HALTON_NUM_DIMS
                + (unsigned int)dim);
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
 *   Seeds a thrust::default_random_engine identically to the original
 *   makeSeededRandomEngine(iter, pixelIndex, depth).  Output is bit-
 *   identical.
 *
 * Halton mode:
 *   haltonIndex = iter * HALTON_STRIDE + depth
 *     - Primary rays:   depth = 0
 *     - Bounce N:       depth = N * MAX_DRAWS_PER_BOUNCE
 *   pixelIndex is stored for per-dimension CP offset computation in
 *   next(dim).
 *
 * @param iter        Current iteration (frame) counter
 * @param pixelIndex  Linear pixel index
 * @param depth       Depth offset (0 for primary, bounceNum * 8 for scatters)
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
        state.haltonIndex = (unsigned int)iter * HALTON_STRIDE + (unsigned int)depth;
        state.pixelIndex  = (unsigned int)pixelIndex;
    }
    return state;
}
