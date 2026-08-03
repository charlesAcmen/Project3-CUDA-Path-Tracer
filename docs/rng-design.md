# RNG Design: LCG + Owen-Scrambled Halton

> **Feature:** `:three:` from INSTRUCTION.md — 3-point optional feature.
> **Reference:** `docs/CSE168_07_Random.pdf`
> **Last updated:** 2026-08-03 (rewritten to match the implemented design)

## Overview

The path tracer exposes two RNG modes behind one uniform interface,
`RngState::next(dim)` in `src/rng/rng.h`:

| Mode | Implementation | Default |
|------|----------------|---------|
| `LCG` | `thrust::default_random_engine` seeded by `utilhash` (backward compatible) | **yes** |
| `HALTON` | True nested Owen-scrambled Halton (Owen 1997) | `--rng=1` |

The RNG state is stateless-per-iteration: every frame re-creates an
`RngState` from `makeRngState(iter, pixelIndex, depth, mode)`, so the GPU
bounce loop never carries mutable RNG state in `PathSegment`.  This matches
the original `makeSeededRandomEngine` pattern.

## Why Low-Discrepancy

LCG (and any PRNG) produces white-noise point sets converging at O(1/√N).
Low-discrepancy sequences converge at O(logᵈ N / N) — the same noise level in
far fewer samples.  For a Monte Carlo path tracer this directly reduces
render time to a target quality.

Measured on the actual implementation (`tests/rng_test`, 16 px × 16k iters,
averaged over pixels):

- 1D star discrepancy: **13.7× lower** than LCG (pooled per dimension).
- 2D π-integral error at N=16384: equals **~55× more LCG samples**
  (log-log slope −0.78 vs −0.59 for LCG; QMC ideal −1.0).
- 2D stratification: 16×16 grid max cell deviation **5.0** (LCG 22.0,
  random floor ~24–40) — Halton is net-level, not random-level.

## Option Matrix

| Method | Convergence | GPU Fit | Impl. Complexity | High-Dim Quality | Verdict |
|--------|------------|---------|-----------------|------------------|---------|
| **LCG** (baseline) | O(1/√N) | ★★★★★ | Trivial | Poor | default, backward compat |
| **Standard Halton** | O(logᵈ N/N) | ★★★★☆ | Simple | Poor (bases >7 correlate) | insufficient alone |
| **Owen-scrambled Halton** | O(logᵈ N/N) | ★★★★☆ | Moderate | Good | **implemented** |
| **Sobol** | O(logᵈ N/N) | ★★★☆☆ | Complex | Excellent | future option |
| **PCG / Xoshiro** | O(1/√N) | ★★★★★ | Moderate | N/A (PRNG) | side-grade only |
| **CMJ** | O(1/N) for 2D | ★★★★★ | Simple | 2D only | AA-only alternative |

Chosen: **true nested Owen-scrambled Halton** — the low-discrepancy property
comes from Halton's radical inverse; per-pixel decorrelation comes from
nested digit permutations plus a net-preserving toroidal shift.  See the
design-history section for why Cranley-Patterson rotation was rejected.

## Implemented Design (`src/rng/rng.h`)

### Halton dimension assignment

Each independent sampling decision in the pipeline gets a dedicated
dimension (prime base).  All draws within one bounce share the same
`haltonIndex`; different dims use different bases, so a bounce's draws form
a proper multi-dimensional Halton point.

| Dim | Prime | Usage | Location |
|:---:|:---:|---|---|
| 0 | 2 | AA jitter x | `ray_generation.cuh` |
| 1 | 3 | AA jitter y | `ray_generation.cuh` |
| 2 | 5 | Lens aperture u | `ray_generation.cuh` (DoF) |
| 3 | 7 | Lens aperture v | `ray_generation.cuh` (DoF) |
| 4 | 11 | Diffuse hemisphere θ | `interactions` |
| 5 | 13 | Diffuse hemisphere φ | `interactions` |
| 6 | 17 | Specular lobe θ | `interactions` |
| 7 | 19 | Specular lobe φ | `interactions` |
| 8 | 23 | Fresnel roulette | `scatterRay` (refractive) |
| 9 | 29 | Path Russian roulette | `shading.cuh` |

`getHaltonPrime(dim)` indexes the first 16 primes (dims 0–9 allocated,
10–15 reserved).  Dimensions 4–9 are reused across bounces; per-bounce
independence comes from the scramble seed, not the index.

### The index walk

`haltonIndex = iter` — every pixel walks the SAME consecutive index
(0, 1, 2, …) across frames.  Consecutive indices are exactly the ordering
that gives Halton its O((log N)ᵈ / N) guarantee.  Per-pixel decorrelation is
handled entirely by the scramble seed + toroidal shift, NOT by jumping each
pixel to a random index (an unmasked hash offset destroys the low-
discrepancy property — see design history, BUG 1).

### True nested Owen scrambling — `owenRadicalInverse(base, n, seed)`

For a prime base `b`, the radical inverse digit expansion of `n` is
`Phi_b(n) = Σ d_k b^-(k+1)`.  Owen scrambling permutes each digit with a
permutation that depends on the **prefix** (the higher digits already read):

    d'_k = pi_k(prefix_k)(d_k)

This prefix-dependence is the definition of *true* (nested) Owen scrambling,
as opposed to linear/digit scrambling (fixed permutation per level), which
keeps the net but leaves different pixels near-identical.

Here each per-level permutation is the affine map `d → (a·d + c) mod b`
(branch-free, bijective for prime `b`), with `(a, c)` derived from
`utilhash(prefix + level·0x9e3779b9 + 0x1f123bb5)`.  This is a restricted
instantiation of the full nested-Owen family, but it preserves the net in
every dimension AND jointly.

### Per-pixel / per-bounce / per-dim seed

The scramble seed is a chained hash of `(pixelIndex, bounceIndex, dim)`:

    seed = utilhash( utilhash(pixel + 0x9e3779b9)
                     ^ (bounce * 0x85ebca6b)
                     ^ (dim    * 0xc2b2ae35) )

Chained hashing (no linear-sum collisions) keeps different pixels / bounces /
dims decorrelated while leaving the index a clean consecutive walk.  Each
bounce gets a fresh scramble (bounce mixed into the seed), so the joint of a
path's bounces is a product of independent nets — standard per-bounce
randomized QMC, not a single high-dimensional net.

### Net-preserving toroidal shift — `cpRotate(s, rot)`

After Owen scrambling, `next()` applies a fixed per-(pixel, bounce, dim)
toroidal shift `rot = (seed & 0xFFFFFF) / 2^24`:

    return (s + rot) mod 1.0

This is **not** the legacy Cranley-Patterson rotation (which shifted the
raw Halton value with a linear-sum seed).  It is a randomized-QMC / CP-style
shift applied to the already-Owen-scrambled value, and it is net-preserving
(it cannot break the stratification).  It exists because the affine nested
permutations alone cannot fully decorrelate small bases (base 2 has only 2
affine permutations per level) — the shift closes most of that gap.

## Measured Behavior (2026-08, `tests/rng_test`)

### Convergence (2D π-integral, averaged over 16 pixels)

| N | LCG err | Owen err | effective LCG multiplier |
|:---:|:---:|:---:|:---:|
| 64 | 1.57e-1 | 8.62e-2 | 3× |
| 256 | 9.60e-2 | 2.15e-2 | 20× |
| 1024 | 4.20e-2 | 9.52e-3 | 19× |
| 4096 | 2.31e-2 | 4.09e-3 | 32× |
| 16384 | 8.21e-3 | 1.10e-3 | **55×** |

### Low-SPP behavior

At N ≲ a few hundred, Halton renders **noisier** than LCG.  Expected, not a
bug — two causes:

1. **Large-prime-base clustering.**  Dims 4–9 use bases 11..29; a base-`b`
   pair is only well-distributed once N ≫ b².  The diffuse pair (11, 13) is
   *worse* than LCG at N=16 (0.9×); the AA pair (2, 3) crosses over near
   N≈64, the diffuse pair near N≈256.  Specular/RR dims (17..29) are the
   worst low-SPP offenders (variance spikes / fireflies).
2. **Structured (non-white) error.**  Cross-pixel correlation makes the
   residual noise spatially correlated, which reads as "dirtier" than LCG's
   independent grain at the same RMS.

Practical crossover where Halton becomes visibly cleaner: simple scenes (low
effective dimension) ~ a few hundred iters; complex scenes (specular /
refractive / small lights) ~ a few thousand.

### Cross-pixel correlation (known limitation)

True per-pixel independence is impossible for mixed-prime Halton while
preserving the cross-dimensional net — small bases have too few digit
permutations, and a strong base-2 hash (Burley-style) destroys the joint net
(see design history, BUG 4).  Measured over 16 pixels:

| dim (base) | mean \|corr\| | worst pair |
|:---:|:---:|:---:|
| dim0 (2) | 0.278 | 0.876 |
| dim1 (3) | 0.286 | 0.832 |
| dim4 (11) | 0.221 | 0.867 |
| dim9 (29) | 0.127 | 0.920 |
| LCG reference | 0.028 | 0.338 |

Adjacent-pixel correlation (dim0): ~0.37 vs LCG ~0.01.  Consequence: at low
SPP the noise is "clumpier" than LCG's white grain.  Not a correctness bug,
and it does not break the within-pixel net.

## Future Path

- **Sobol + per-pixel Burley hash** is the only way to get *both* full
  per-pixel independence (corr ≈ 0.004) *and* a preserved net: all dims base
  2, so the hash scramble decorrelates without breaking the joint structure.
  This would eliminate the toroidal shift entirely (`next()` becomes a pure
  hash-scrambled Sobol sample).  Requires direction vectors + a rewrite of
  the dimension infrastructure; not needed for the current scene complexity.
- **Direct lighting / MIS** would add new Halton dimensions (10–15 are free).

## Testing (`tests/rng_test`)

Standalone, not wired into the root build.  Build with CMake + VS (see
`CLAUDE.md`):

- `rng_compare.cu` — host-only program including the real `rng.h`; writes
  `pixel,iter,bounce,dim,lcg,halton_owen` rows.  Deterministic.
- `rng_analyze.py` — star discrepancy, 2D scatter, π-convergence, grid
  stratification, pixel decorrelation, plots.
- `probe_owen.py` — independent net-structure probe (ports the algorithm in
  numpy; used to prove BUG 4 / BUG 5 during development).
- `verify_rng.py`, `verify_xpix.py`, `converge.py`, `low_spp.py`,
  `shift_test.py` — analysis helpers used for the measurements above.

## Design History

The implementation went through five documented design iterations (kept as
history in earlier `rng.h` headers):

- **BUG 1 — unmasked hash index offset.**  Adding a full-uint32 hash to `iter`
  pushed each pixel's walk into a random window (~2.7 B) where Halton is
  indistinguishable from noise.  Fix: `haltonIndex = iter`, decorrelation in
  the seed.
- **BUG 2 — linear-sum CP seed.**  `pixel·131 + bounce·17 + dim·11` is
  collision-prone.  Fix: chained `utilhash`.
- **BUG 3 — CP rotation alone.**  A pure float-domain toroidal shift
  introduced visible low-frequency patterns between nearby pixels.  Fix:
  Owen scrambling on the integer index before the radical-inverse map; the
  float shift kept only as a net-preserving decorrelation aid.
- **BUG 4 — base-2-only hash scramble.**  Applying a Burley-style hash
  scramble (`owenScramble`) to mixed bases kept each 1D marginal stratified
  but destroyed the cross-dimensional net (random-rate 2D convergence).
  Fix: `owenRadicalInverse` — prefix-dependent affine digit permutations in
  every base.
- **BUG 5 — linear digit scrambling.**  Fixed per-level permutations kept
  different pixels near-identical (corr ~1).  Fix: nested (prefix-dependent)
  permutations + fixed per-pixel toroidal shift.  Mean cross-pixel |corr|
  drops to ~0.1–0.3 (adjacent ~0.37, worst pairs ~0.9).
