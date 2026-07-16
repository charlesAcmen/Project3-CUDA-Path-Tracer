# RNG Upgrade Plan: From LCG to Low-Discrepancy Halton Sequences

> **Feature:** `:three:` from INSTRUCTION.md — 3-point optional feature.
> **Reference:** `docs/CSE168_07_Random.pdf`
> **Date:** 2026-07-16

## Context

### Problem

The CUDA path tracer currently uses `thrust::default_random_engine` (a linear
congruential generator, LCG) for all Monte Carlo sampling.  While LCGs are fast,
they have known deficiencies for MC integration:

- **Short period** — 32-bit LCGs repeat after ~2³¹ samples, risking structured
  artifacts in high-sample-count progressive renders.
- **Poor equidistribution** — Marsaglia's theorem: LCG points fall on
  hyperplanes in high dimensions, introducing subtle correlation between
  successive sampling decisions (AA jitter ↔ lens DOF ↔ hemisphere direction).
- **No convergence acceleration** — LCG produces pseudo-random (white-noise)
  point sets, which converge at O(1/√N).  Low-discrepancy sequences converge at
  O(logᵈ N / N), meaning fewer samples for the same visual quality.

This is the `:three:` feature from INSTRUCTION.md (line 101, 3 points), with
reference material in `docs/CSE168_07_Random.pdf`.

### Current RNG Architecture (preserved — not to be deleted)

| Location | Item | Role |
|----------|------|------|
| `src/intersections.h:13-22` | `utilhash(unsigned int a)` | Jenkins-style bit-mixing hash, `__host__ __device__` |
| `src/pathtrace.cu:88-98` | `makeSeededRandomEngine(iter, index, depth)` | Combines iteration, pixel index, bounce depth via `utilhash()` into a `thrust::default_random_engine` seed |
| `src/pathtrace.cu:232` | Call site in `generateRayFromCamera` | 4 draws: AA jitter (x2) + lens DOF (x2, conditional), seeded `depth=0` |
| `src/pathtrace.cu:444` | Call site in `shadeMaterial` | Up to 5 draws/bounce: diffuse hemisphere (2), specular lobe (2), Fresnel RR (1), path RR (1), seeded `depth=remainingBounces` |
| `src/pathtrace.cu:538` | Call site in `shadeFakeMaterial` | 1 draw: debug noise, seeded `depth=0` |

**Key design property:** The engine is recreated on-the-fly per kernel launch
rather than stored in `PathSegment`.  This is a deliberate memory-bandwidth
optimisation (comment at `pathtrace.cu:92-93`).  The deterministic seed formula
`hash(depth, iter) ^ hash(pixelIndex)` guarantees:
- **Spatial decorrelation** — different pixels get different sequences.
- **Temporal decorrelation** — different iterations get different sequences.
- **Bounce-depth decorrelation** — different bounces get different sequences.

### Where Random Numbers Are Consumed

Per path per iteration (worst-case):

| Stage | Sampling Decision | Draws | Frequency |
|-------|------------------|-------|-----------|
| Primary ray | AA sub-pixel jitter (x, y) | 2 | Once per iteration |
| Primary ray | Lens aperture (u, v) for DoF | 2 | Once (conditional) |
| Diffuse bounce | Cosine-weighted hemisphere (θ, φ) | 2 | Per bounce |
| Glossy bounce | Phong lobe (cosⁿ, φ) | 2 | Per bounce |
| Refractive bounce | Fresnel Russian roulette | 1 | Per bounce |
| Termination | Path Russian roulette | 1 | Per bounce (after rrMinBounces) |

With `traceDepth=8`, a full path can consume up to ~44 random draws.

---

## Analysis: Is Halton the Best Choice?

### Option Matrix

| Method | Convergence | GPU Fit | Impl. Complexity | High-Dim Quality | Educational Value |
|--------|------------|---------|-----------------|------------------|-------------------|
| **LCG** (current) | O(1/√N) | ★★★★★ | Trivial | Poor | Baseline only |
| **Standard Halton** | O(logᵈ N / N) | ★★★★☆ | Simple | Poor (bases >7 correlate) | ★★★★★ |
| **Scrambled Halton** | O(logᵈ N / N) | ★★★★☆ | Moderate | Good | ★★★★☆ |
| **Sobol** | O(logᵈ N / N) | ★★★☆☆ | Complex | Excellent | ★★★☆☆ |
| **PCG / Xoshiro** | O(1/√N) | ★★★★★ | Moderate | N/A (PRNG) | ★★☆☆☆ |
| **CMJ** | O(1/N) for 2D | ★★★★★ | Simple | 2D only | ★★★☆☆ |

### Detailed Comparison

#### LCG (thrust::default_random_engine) — current
- **Pros:** Already implemented, zero change cost, adequate for interactive
  preview.
- **Cons:** White-noise convergence only; Marsaglia hyperplane correlations
  between successive dimensions can produce subtle structured noise in
  progressively-rendered images.

#### Standard Halton Sequence
- **How it works:** The radical inverse function reverses the digit
  representation of integer `n` in a prime base `b`.  For base 2:
  `H₂(0,1,2,3,4,5) = 0, 0.5, 0.25, 0.75, 0.125, 0.625`.  A d-dimensional
  Halton sequence uses the first d primes as independent bases.
- **Pros:** Low-discrepancy (every prefix of the sequence is well-distributed);
  O(log n) per sample; no state required; fits the current stateless pattern
  exactly; taught in the course reference PDF.
- **Cons:** **High-dimensional correlation** — dimensions with large prime bases
  (b ≥ 11) show visible stripe/streak patterns.  A path tracer with 8 bounces
  needs ~10+ independent sampling dimensions, requiring bases up to 29, where
  correlation is severe.
- **Verdict:** Standard Halton alone is **not sufficient** for this application.

#### Scrambled Halton (Recommended)
- **How it works:** Apply a random digit permutation (Owen scrambling) or a
  simple Cranley-Patterson rotation `(H_b(n) + offset) mod 1` to each Halton
  point.  The scramble breaks inter-dimensional correlation while preserving
  the low-discrepancy property.  Different pixels get different scrambles,
  producing independent sequences.
- **Pros:** Low-discrepancy + decorrelated; moderate implementation complexity;
  fits the current stateless GPU pattern (scramble per pixel, compute Halton
  point on-the-fly); well-studied in rendering literature (PBRT, Tungsten).
- **Cons:** Cranley-Patterson rotation is not as rigorous as Owen scrambling
  (stratification is only approximate).  Owen scrambling requires permutation
  tables or hash-based permutations per digit — more complex but feasible.
- **Verdict:** **This is the recommended approach.**  Cranley-Patterson
  rotation for the initial implementation; Owen scrambling as a future upgrade.

#### Sobol Sequence
- **Pros:** Best high-dimensional low-discrepancy properties; industry standard
  (RenderMan, Arnold, Cycles).
- **Cons:** Requires precomputed direction numbers (a set of bit-matrices) for
  each dimension; Gray-code iteration is needed for efficiency; significantly
  more complex to implement correctly.  Overkill for a graduate course project.
- **Verdict:** Not recommended for this project.  Halton is pedagogically
  simpler and the course reference focuses on Halton.

#### PCG / Xoshiro (Modern PRNGs)
- **Pros:** Excellent statistical quality, small state (128 bits for Xoshiro),
  fast, no inter-dimensional correlation issues.
- **Cons:** Still pseudo-random — no low-discrepancy property.  Convergence
  rate remains O(1/√N).  This is a *side-grade* (better randomness quality)
  rather than an *upgrade* (faster convergence).
- **Verdict:** Not recommended as the primary change.  Could be a useful
  fallback/alternative option.

### Decision: Scrambled Halton (Cranley-Patterson Rotation)

**Rationale:**
1. **Faster convergence** than LCG — low-discrepancy means fewer samples for
   the same noise level.
2. **Fits the existing architecture** — Halton's n-th-sample-is-just-a-function
   matches the current "recompute on-the-fly" GPU pattern.  No state to store.
3. **Simple to implement** — radical inverse is ~10 lines of CUDA C; CP
   rotation is a single addition + modulo.
4. **Educational** — Halton is the natural next step after LCG as taught in
   CSE168_07_Random.pdf.
5. **Extensible** — CP rotation can be upgraded to Owen scrambling later
   without changing the calling code.

**Future upgrade path:** If visual correlation artifacts appear at high sample
counts, upgrade CP rotation → Owen scrambling by replacing the simple offset
with a per-digit hash-based permutation, using `utilhash()` as the per-digit
randomizer.

---

## Software Engineering: Extract RNG to Its Own File

### Current Problem

RNG logic is scattered across files with poor semantic cohesion:

| File | RNG Content | Semantic Issue |
|------|------------|----------------|
| `src/intersections.h` | `utilhash()` | Hashing is not intersection logic |
| `src/pathtrace.cu` | `makeSeededRandomEngine()` | Mixed with path tracing pipeline |
| `src/interactions.h` | `#include <thrust/random.h>` | RNG dependency |
| `src/interactions.cu` | `samplePhongSpecularDir`, `calculateRandomDirectionInHemisphere` take `thrust::default_random_engine&` | Tight coupling to Thrust RNG type |

### Recommendation: Create `src/rng.h`

A single header that is the **single source of truth** for all random number
generation.  This follows the project's existing pattern of purpose-specific
headers (`constants.h`, `sceneStructs.h`, `intersections.h`).

**What goes into `src/rng.h`:**

```
src/rng.h
├── RngMode enum          { LCG, HALTON }
├── radicalInverse()        Halton radical inverse (base, n) → float in [0,1)
├── makeHaltonRng()         factory: creates a Halton state from (iter, index, dimension)
├── haltonSample()          convenience: one-shot Halton sample
├── (re-export)             #include guards to make existing utilhash + makeSeededRandomEngine
│                           accessible through this header for discoverability
└── RngState struct         wraps either mode behind a uniform interface
```

**What does NOT change (preserved exactly as-is):**
- `src/intersections.h` — `utilhash()` stays; a comment is added pointing to `rng.h` for
  new random-number functionality.
- `src/pathtrace.cu:88-98` — `makeSeededRandomEngine()` stays; a comment is added noting
  that `rng.h` provides Halton alternatives.
- All existing comments in Chinese and English are preserved.
- All existing call sites continue to work without modification.

**Why a header-only design?**
- CUDA device functions must be in headers or `.cuh` files for separable compilation.
- The project already uses header-only utilities (`constants.h`, `sceneStructs.h`).
- No `.cu` compilation unit needed — all functions are `__host__ __device__` and inline.
- Keeps the build system unchanged (no new CMakeLists entries).

### API Design: The `RngState` Wrapper

To support both LCG and Halton through a single interface without templates
polluting every function signature, we introduce a `RngState` struct that
type-erases the RNG mode behind a uniform `.next()` → float API:

```cuda
enum class RngMode : int {
    LCG    = 0,   // thrust::default_random_engine (backward compatible)
    HALTON = 1    // scrambled Halton sequence
};

struct RngState {
    RngMode mode;

    // -- LCG branch (preserved from existing code) --
    thrust::default_random_engine lcgEngine;

    // -- Halton branch (2 ints + 1 float) --
    unsigned int haltonIndex;     // current sample index in the Halton sequence
    int haltonBase;               // prime base for this sampling dimension
    float haltonOffset;           // Cranley-Patterson rotation offset, per-pixel per-dim in [0,1)

    // Returns the next uniform random float in [0, 1).
    // The if-else on `mode` is warp-uniform (all threads read the same
    // global flag), so there is zero divergence cost.
    __host__ __device__ float next() {
        if (mode == RngMode::LCG) {
            thrust::uniform_real_distribution<float> u01(0, 1);
            return u01(lcgEngine);
        } else {
            // Cranley-Patterson rotation: (Halton sample + per-pixel offset) mod 1.0.
            // The offset decorrelates different pixels while preserving
            // the low-discrepancy property within each pixel's sequence.
            float raw = radicalInverse(haltonBase, haltonIndex);
            float val = raw + haltonOffset;
            if (val >= 1.0f) val -= 1.0f;  // mod 1.0 (branch is uniform in warp)
            haltonIndex++;
            return val;
        }
    }
};

// Factory function — direct replacement for makeSeededRandomEngine.
// When mode==LCG: seeds lcgEngine exactly as before (bit-identical).
// When mode==HALTON: initialises haltonIndex, base, and per-pixel CP offset.
// The CP offset is derived from utilhash(pixelIndex * MAX_DIMENSIONS + dim)
// so each (pixel, dimension) pair gets its own fixed random offset in [0, 1).
__host__ __device__ RngState makeRngState(
    int iter, int pixelIndex, int depth, RngMode mode, int dim);
```

**Advantages of `RngState` over templates:**
1. **Single type** — all functions take `RngState&`, no `template<typename Rng>`
   proliferation in headers.
2. **Backward compatible** — `RngState` in LCG mode delegates to the same
   `thrust::default_random_engine` seeding logic (bit-identical output).
3. **Register-light** — Halton branch adds only 3 ints vs. LCG's 4-int engine
   state.  This is a small *improvement* for the register-pressure bottleneck
   noted in `shadeMaterial` (comment at `pathtrace.cu:435-437`).
4. **No pointer indirection** — both branches are inline in the struct, no
   heap allocation or virtual dispatch.

---

## Implementation Plan

### Phase 1: Create `src/rng.h` — The RNG Header

**New file:** `src/rng.h`

Contents:

1. **`radicalInverse(int base, unsigned int n)`**
   - Computes the radical inverse of `n` in the given prime base.
   - Returns a float in [0, 1).
   - O(log_base(n)) iterations — typically ≤ 16 for 32-bit n and base 2,
     ≤ 7 for base 7, etc.
   - `__host__ __device__` for both CPU precomputation and GPU use.
   - English comment explaining the Halton sequence and radical inverse.

2. **`cpRotate(float haltonSample, float offset)`**
   - Inline helper: returns `(haltonSample + offset) mod 1.0`.
   - Cranley-Patterson rotation breaks inter-dimensional correlation while
     preserving the low-discrepancy property of the base Halton sequence.
   - `__host__ __device__`, trivial — a single addition + conditional subtract.

3. **`HaltonRng` struct** (lightweight, GPU-friendly)
   ```cuda
   struct HaltonRng {
       unsigned int index;     // current position in the Halton sequence
       int base;               // prime base for this dimension (2, 3, 5, 7, ...)
       float offset;           // Cranley-Patterson offset in [0, 1), per-pixel per-dim

       __host__ __device__ float next() {
           float raw = radicalInverse(base, index);
           float val = raw + offset;
           if (val >= 1.0f) val -= 1.0f;  // mod 1.0
           index++;
           return val;
       }
   };
   ```
   - `next()` computes `radicalInverse(base, index)`, applies CP rotation with
     `offset`, then increments `index`.  Each call produces the next Halton
     point in this dimension's sequence, decorrelated per-pixel.

4. **`makeHaltonRng(int iter, int pixelIndex, int bounce, int dimension)`**
   - Factory function analogous to `makeSeededRandomEngine`.
   - `dimension` selects which prime base to use (0→2, 1→3, 2→5, ...).
   - `offset = float(utilhash(pixelIndex * 16 + dimension) & 0xFFFFFF) / float(0x1000000)`
     maps a hash to a uniform float in [0, 1) for Cranley-Patterson rotation.
     Each (pixel, dimension) pair gets its own fixed offset.
   - `index = iter * traceDepth + bounce` for per-bounce per-iteration
     progression along the sequence.
   - English comment explaining the CP offset seeding strategy.

5. **Prime base table** — a `__device__ constexpr int HALTON_PRIMES[]` with
   the first 16 primes: `{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
   43, 47, 53}`.  16 dimensions is enough for the current pipeline (max ~10
   needed).

6. **`RngMode` enum** — `{ LCG, HALTON }` for runtime or compile-time toggling.

### Phase 2: Assign Halton Dimensions

Map each sampling decision in the pipeline to a dedicated Halton dimension
(prime base):

| Dimension Index | Prime | Sampling Decision | Location |
|:---:|:---:|---|---|
| 0 | 2 | AA jitter x | `generateRayFromCamera` |
| 1 | 3 | AA jitter y | `generateRayFromCamera` |
| 2 | 5 | Lens aperture u | `generateRayFromCamera` (DoF only) |
| 3 | 7 | Lens aperture v | `generateRayFromCamera` (DoF only) |
| 4 | 11 | Diffuse hemisphere θ (up) | `calculateRandomDirectionInHemisphere` |
| 5 | 13 | Diffuse hemisphere φ (around) | `calculateRandomDirectionInHemisphere` |
| 6 | 17 | Specular lobe θ | `samplePhongSpecularDir` |
| 7 | 19 | Specular lobe φ | `samplePhongSpecularDir` |
| 8 | 23 | Fresnel roulette (refractive) | `scatterRay` |
| 9 | 29 | Path roulette (termination) | `russianRouletteTerminate` |

**Important design note:** Dimensions 4–9 are reused across bounces.  The
`bounce` parameter in `makeHaltonRng` ensures that different bounces get
different *sequence indices* within the same dimension's Halton sequence:
`index = iter * traceDepth + bounce`.  This means bounce 3 of iteration 50 in
dimension 4 (prime 11) is a different Halton point than bounce 2 of iteration
50 in dimension 4 — the indices differ (50*8+3=403 vs 50*8+2=402).

### Phase 3: Modify Call Sites

All call sites switch from `thrust::default_random_engine` to `RngState`,
using the factory `makeRngState()`.  The `RngState` struct handles mode
dispatch internally — call sites don't branch on RNG mode.

#### 3a. `generateRayFromCamera` (pathtrace.cu:232)

**Before (preserved as-is alongside new code):**
```cuda
thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
thrust::uniform_real_distribution<float> u01(0, 1);
float jitterX = u01(rng) - 0.5f;   // AA jitter x
float jitterY = u01(rng) - 0.5f;   // AA jitter y
float lensU   = u01(rng);           // DoF lens u
float lensV   = u01(rng);           // DoF lens v
```

**After (new Halton-capable code added, old code also retained for fallback):**
```cuda
RngState rngAAx = makeRngState(iter, index, 0, g_opts.rngMode, 0);  // dim 0, prime 2
RngState rngAAy = makeRngState(iter, index, 0, g_opts.rngMode, 1);  // dim 1, prime 3
float jitterX = rngAAx.next() - 0.5f;
float jitterY = rngAAy.next() - 0.5f;

RngState rngLensU = makeRngState(iter, index, 0, g_opts.rngMode, 2);  // dim 2, prime 5
RngState rngLensV = makeRngState(iter, index, 0, g_opts.rngMode, 3);  // dim 3, prime 7
float lensU = rngLensU.next();
float lensV = rngLensV.next();
```

Each sampling decision gets its own `RngState` with a dedicated dimension
index.  This ensures different sampling decisions use different Halton prime
bases, avoiding intra-pixel correlation.

#### 3b. `shadeMaterial` (pathtrace.cu:444) and callees

The `shadeMaterial` kernel creates `RngState` instances per sampling
dimension, then passes them to `scatterRay()`.  The function signatures
in `interactions.h` change from `thrust::default_random_engine&` to
`RngState&`:

```cuda
// interactions.h — updated signatures
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, RngState& rng);
__host__ __device__ void scatterRay(
    PathSegment& pathSegment, glm::vec3 intersect, glm::vec3 normal,
    const Material& m, RngState& rng, int fresnelMode);
__host__ __device__ void samplePhongSpecularDir(
    glm::vec3 reflectedDir, float exponent, RngState& rng);
```

Inside these functions, `thrust::uniform_real_distribution<float> u01(0,1)`
+ `u01(rng)` is replaced with `rng.next()`.

The callee doesn't know or care which RNG mode is active — it just calls
`next()`.  The dispatch happens inside `RngState`.

#### 3c. `russianRouletteTerminate` (pathtrace.cu:362)

Same pattern: change parameter from `thrust::default_random_engine&` to
`RngState&`, replace `u01(rng)` with `rng.next()`.

#### 3d. `shadeFakeMaterial` (pathtrace.cu:538)

Same pattern as `shadeMaterial` but only 1 dimension needed (debug noise).

### Phase 4: Add Toggle Mechanism

Add `rngMode` to `PathTracerOptions` in `src/pathtrace.h`, following the
existing pattern of `compactMethod`, `sortByMaterial`, and `debugMode`:

```cuda
struct PathTracerOptions {
    int  compactMethod  = 3;     // 0=off, 1=global scan, 2=Thrust, 3=shared-mem
    bool sortByMaterial = true;  // group paths by materialId before shading
    int  debugMode      = 0;     // 0=Hill ACES, 1=linear bypass, 2=Narkowicz ACES
    int  rngMode        = 0;     // 0=LCG (default, backward compat), 1=Halton
};
```

Add CLI flag `--rng=N` in `src/main.cpp`, following the `--fresnel=N` pattern:
```cpp
} else if (arg.rfind("--rng=", 0) == 0) {
    int v = std::stoi(arg.substr(6));
    setRngMode(v);
}
```

With setter/getter in `pathtrace.cu` (like `setCompactMethod`/`getCompactMethod`).

The RNG mode is **warp-uniform** (all threads in a kernel launch read the same
global flag), so the `if (mode == ...)` branch in `RngState::next()` has zero
divergence cost.

### Phase 5: Update `src/interactions.h` and `src/interactions.cu`

- Replace `#include <thrust/random.h>` with `#include "rng.h"`.
- Change function parameter types from `thrust::default_random_engine&` to
  `RngState&` in:
  - `calculateRandomDirectionInHemisphere()`
  - `scatterRay()`
  - `samplePhongSpecularDir()` (add declaration to `.h` if not already)
- Replace `thrust::uniform_real_distribution<float> u01(0,1)` + `u01(rng)`
  with `rng.next()` in all function bodies.

---

## Files to Modify (Summary)

| File | Action | Risk |
|------|--------|------|
| `src/rng.h` | **Create** — `RngState`, `RngMode`, Halton functions, prime table | None (new file) |
| `src/pathtrace.cu` | Replace RNG creation in 3 kernels + `russianRouletteTerminate`; add setter/getter; keep `makeSeededRandomEngine` as-is | Low — old code preserved |
| `src/pathtrace.h` | Add `rngMode` to `PathTracerOptions`; declare setter/getter | Low |
| `src/interactions.h` | Change `thrust::default_random_engine&` → `RngState&` in 3 signatures; swap include | Low — mechanical change |
| `src/interactions.cu` | Replace `u01(rng)` → `rng.next()` in 3 functions; swap include | Low — mechanical change |
| `src/main.cpp` | Parse `--rng=N` CLI flag | Low |
| `src/intersections.h` | Add comment pointing to `rng.h` (no code change) | None |
| `CMakeLists.txt` | Add `src/rng.h` to headers list | None |

---

## What Is NOT Changed (Preserved)

- `src/intersections.h:13-22` — `utilhash()` stays exactly as-is.
- `src/pathtrace.cu:88-98` — `makeSeededRandomEngine()` stays exactly as-is.
- All existing Chinese and English comments throughout the codebase are untouched.
- The default RNG mode is LCG — existing renders are bit-identical.
- No existing struct fields, kernel signatures, or memory layouts change.

---

## Verification

### Correctness
1. **Regression test:** Render `scenes/cornell.json` with `--rng=0` (LCG).
   The output must be pixel-identical to the pre-change render.
2. **Determinism test:** Render the same scene twice with `--rng=1`.  The
   outputs must be pixel-identical (no floating-point non-determinism from the
   Halton sequence).
3. **Progressive convergence:** Render with `--rng=1` at 100, 500, 2500,
   and 5000 iterations.  Verify that noise decreases monotonically and there
   are no grid-like/structured artifacts (a sign of Halton dimensional
   correlation).
4. **A/B comparison:** Render the same scene at the same iteration count with
   `--rng=0` (LCG) and `--rng=1` (Halton).  The Halton render should show
   comparable or lower perceived noise, especially in smooth diffuse regions.

### Performance
1. **Benchmark:** Profile `pathtrace` with `--benchmark` at 100 iterations
   for both LCG and Halton modes.  Compare iteration time, kernel time
   (`shadeMaterial`, `generateRayFromCamera`), and overall samples/second.
   Halton's `radicalInverse` has ~2–3× more arithmetic per sample than LCG's
   linear-congruential step, but this should be negligible compared to
   intersection testing and memory latency in the bounce loop.

### Visual Quality
1. **Noise floor:** At 100 iterations, compare RMSE of LCG vs Halton against
   a 5000-iteration reference (or analytic solution for Cornell box).
2. **Structured artifact check:** Render a scene with a large diffuse plane
   and inspect for streaks, grids, or moiré patterns — telltale signs of
   Halton dimensional correlation.  If present, the scrambling is insufficient
   and Owen scrambling should be implemented.

---

## Future Extensions (Out of Scope for This Plan)

1. **Owen scrambling** — Replace CP rotation with per-digit hash-based
   permutation for proper stratified sampling in all dimensions.
2. **Sobol sequence** — If dimensional correlation proves problematic at high
   sample counts, implement Sobol with Joe & Kuo direction numbers.
3. **Multiple Importance Sampling (MIS)** — Combine BSDF sampling with light
   sampling; requires additional random dimensions.
4. **Blue-noise / correlated multi-jitter** — For the 2D AA + lens sampling
   dimensions, use precomputed CMJ or blue-noise textures for even faster
   convergence.
