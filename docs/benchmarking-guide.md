# Benchmarking & Experiments Guide

## Overview

The measurement framework instruments user-written GPU kernels and host-side
operations with **cudaEvent** (GPU) and **std::chrono** (CPU) timers.  Results
are written as CSV files to `profiler_output/<scene>_<timestamp>/` and can be
plotted with the companion Python scripts.  When you use
`scripts/benchmark_runner.py`, each full benchmark batch is archived under
`profiler_output/runs/<run-id>/` so old runs do not get overwritten.

Only user-authored or user-modified code and the post-processing pipeline are
measured.  `computeIntersections` is measured (it is user-written, mesh-based
intersection).  Primary ray generation (`generateRayFromCamera`) is the only
starter-code kernel excluded from per-operation timing, though it is included
in the frame-level wall-clock time recorded by `beginFrame` / `endFrame`.

## Quick Start

```
# Build
cd build && cmake --build . --config Release

# Normal render (no measurement overhead)
cis565_path_tracer.exe ../scenes/cornell.json

# Benchmark — CSVs appear in profiler_output/
cis565_path_tracer.exe ../scenes/cornell.json --benchmark

# Full automation (runs all configs, generates all plots)
python ../scripts/benchmark_runner.py bin/Release/cis565_path_tracer.exe ../scenes/cornell.json
```

---

## Command-Line Flags

| Flag | Values | Default | Effect |
|------|--------|---------|--------|
| `--benchmark` | (none) | off | Enables profiler. CSVs are written on shutdown to `profiler_output/<scene>_<timestamp>/`. |
| `--verbose` | (none) | off | Enables per-bounce path-count `printf` debug output to console. Can be used with or without `--benchmark`. |
| `--compact=N` | `0`, `1`, `2`, `3` | `3` | **Stream compaction method.** `0`=disabled, `1`=global-mem scan, `2`=Thrust `copy_if`, `3`=shared-mem scan. Other integers are accepted by the parser, but only these values have defined behavior. |
| `--sort=N` | `0`, `1` | `0` | **Material sorting.** `0`=disabled, `1`=enabled. Nonzero values are treated as enabled. |
| `--fresnel=N` | `0`, `1` | `0` | `0`=Schlick, `1`=accurate Fresnel (affects refractive materials). |
| `--rng=N` | `0`, `1` | `0` | `0`=LCG, `1`=scrambled Halton. |
| `--warmup=N` | any int | `3` | Warmup iterations excluded from summary statistics. |
| `--save` | (none) | off | Saves the rendered image when the app exits. Can be used with or without `--benchmark`. |
| `--save-at=N1,N2,...` | list of ints | off | Saves checkpoint images at the given iteration counts. |
| `--config=PATH` | path | `config.local.json` | Load runtime config from a JSON file (CLI flags still win). |

Flags are order-independent.  `--benchmark` must be present for CSV output;
`--warmup` only affects the profiler summary statistics (`--compact`, `--sort`,
`--fresnel`, `--rng` change rendering regardless of profiling).  Without
`--benchmark`, profiling overhead is zero — all `gpuStart` / `gpuStop` /
`cpuStart` / `cpuStop` calls are no-ops.

**Note:** `--verbose` and `--save` are independent of `--benchmark`. Use `--verbose` only when you need to debug path survival behavior, as it produces substantial console output (one line per bounce per iteration). `--save` writes the final image before shutdown.

### Examples

```
# Baseline: compaction ON (Thrust), sorting ON, no debug output
cis565_path_tracer.exe ../scenes/cornell.json --benchmark

# Compaction disabled, everything else default
cis565_path_tracer.exe ../scenes/cornell.json --benchmark --compact=0

# Both compaction and sorting disabled
cis565_path_tracer.exe ../scenes/cornell.json --benchmark --compact=0 --sort=0

# Run with a short warmup for quick comparisons
cis565_path_tracer.exe ../scenes/cornell.json --benchmark --warmup=1

# Enable debug output to see per-bounce path counts (verbose mode)
cis565_path_tracer.exe ../scenes/cornell.json --benchmark --verbose
```

---

## Control Variables

Two independent toggles define the experiment space:

### Stream Compaction (`--compact=N`)

Removes terminated paths from the active pool between bounces via
`gatherTerminatedPaths` + `compactActivePaths`.  Because path count shrinks
each bounce, **all downstream operations benefit** — most notably
`sortPathsByMaterial` (sorting fewer elements) and `computeIntersections`
(fewer ray-geometry tests, not measured by the profiler).  `shadeMaterial`
shows a smaller benefit because terminated paths early-return at the top of
the kernel anyway.

The net benefit is the sum of: reduced `sortPathsByMaterial` + reduced
`shadeMaterial` − `gatherTerminatedPaths` overhead − `compactPaths` overhead.
The unmeasured `computeIntersections` saving is additional.

When compaction is **disabled** (`--compact=0`), `compactPaths` is not
launched and `gatherTerminatedPaths` only runs as a single untimed tail call
after the bounce loop (to bank the remaining live paths' colors) — both are
absent from the CSV output (not present as zero-valued rows).  Paths terminate
via the `remainingBounces` guard in `shadeMaterial` and are collected by that
tail `gatherTerminatedPaths` call.

| Value | Meaning |
|-------|---------|
| `0` | **Disabled.** Terminated paths are guarded by `remainingBounces` in `shadeMaterial`. No compaction overhead. |
| `1` | Custom work-efficient scan-based compaction (from Project 2). |
| `2` | **Thrust `copy_if`** — reference implementation used in benchmarks. |
| `3` | Shared-memory multi-block scan-based compaction (GPU Gems 3, Ch. 39). **Default.** |

### Material Sorting (`--sort=N`)

Permutes `dev_paths` and `dev_intersections` before `shadeMaterial` so that
paths hitting the same material become contiguous.  This reduces warp divergence
(the emissive / diffuse / specular branch in `shadeMaterial`) and improves
memory coalescing for material lookups.

| Value | Meaning |
|-------|---------|
| `0` | **Disabled.** `sortPathsByMaterial` returns immediately. |
| `1` | **Enabled** (default). Thrust radix sort + double gather. |

### Scenes

| Scene | Type | Paths escape? |
|-------|------|---------------|
| `cornell.json` | Open Cornell Box (5 walls + diffuse sphere) | Yes — through the missing front wall |
| `cornellRefra.json` | Open Cornell Box + glass sphere | Yes — plus refraction/reflection chains keep some paths alive longer |
| `cornellGlossy.json` | Open Cornell Box + mirror/glossy spheres | Yes — specular chains keep many paths alive for deep bounces |

> **Note:** the old `cornell_closed.json` (6 walls, camera inside) was removed
> from the repo; there is currently no closed scene.  The path-survival
> contrast now comes from *scene complexity* — simple diffuse (`cornell.json`)
> vs. specular/refractive (`cornellRefra.json`, `cornellGlossy.json`), where
> fewer paths terminate each bounce.

**Hypothesis:** Compaction removes *more* paths per bounce in a scene where
paths terminate early (escape through the open front wall → miss) than in a
scene where specular/refractive chains keep paths bouncing.  Therefore the
performance benefit of compaction should be larger for simple diffuse scenes.

---

## Experiment Recipes

### Recipe A — Compaction ON vs OFF

**Purpose:** Quantify stream compaction benefit.  Answer: "How many paths does
compaction remove per bounce, and what is the benefit?"

**Commands:**
```
:: With compaction (baseline)
cis565_path_tracer.exe ../scenes/cornell.json --benchmark --compact=2

:: Without compaction
cis565_path_tracer.exe ../scenes/cornell.json --benchmark --compact=0
```

**Where the benefit shows up (not just shadeMaterial):**

Compaction removes terminated paths after each bounce, so **all** downstream
operations process fewer elements. The benefit is spread across:

| Operation | Measured? | Why it benefits |
|-----------|-----------|-----------------|
| `computeIntersections` | ❌ No (starter code) | Fewer ray-geometry tests — likely the largest absolute saving |
| `sortPathsByMaterial` | ✅ Yes | Sorting fewer elements — **largest measured benefit** |
| `shadeMaterial` | ✅ Yes | Fewer threads launched; terminated paths early-return anyway, so the per-path saving is modest |

The cost of compaction is:
| Operation | What it does |
|-----------|-------------|
| `gatherTerminatedPaths` | Banks dead-path colors before they are discarded |
| `compactPaths` | Thrust `copy_if` (or custom scan) to squeeze out terminated entries |

**Net benefit** = reduced `sortPathsByMaterial` + reduced `shadeMaterial` − `gatherTerminatedPaths` − `compactPaths`.  The unmeasured `computeIntersections` saving comes on top.

**When `--compact=0`:** `gatherTerminatedPaths` (per-bounce) and `compactPaths`
are **absent from the CSV entirely** (not zero — the kernels are never
launched inside the bounce loop). Path colors are banked by a single untimed
tail `gatherTerminatedPaths` call after the loop. All `pixelcount` paths stay
alive through every bounce, so `sortPathsByMaterial` and `shadeMaterial`
always process the full element count.

**Generate plots:**
```
python scripts/plot_comparison.py profiler_output/cornell_<ts>_*/timing.csv profiler_output/cornell_<ts2>_*/timing.csv --labels "Compaction ON" "Compaction OFF"
```

If you want a durable, code-versioned archive instead of ad-hoc plots, prefer:

```
python scripts/benchmark_runner.py build/bin/Release/cis565_path_tracer.exe scenes/cornell.json
```

That will leave the results in `profiler_output/runs/<run-id>/`, including the per-experiment PNGs and the comparison plots.

---

### Recipe B — Sorting ON vs OFF

**Purpose:** Quantify material sorting benefit.  Answer: "Does reduced warp
divergence in `shadeMaterial` outweigh the Thrust sort overhead?"

**Commands:**
```
:: With sorting
cis565_path_tracer.exe ../scenes/cornell.json --benchmark --sort=1

:: Without sorting (default)
cis565_path_tracer.exe ../scenes/cornell.json --benchmark --sort=0
```

**What to compare:**
- `shadeMaterial` time: should be lower with sorting (reduced warp divergence)
- `SortByMaterial` time: should be ~0 when `--sort=0`
- Total per-bounce time: `shadeMaterial + SortByMaterial` — is this sum lower with sorting?

**Generate plot:**
```
python scripts/plot_comparison.py profiler_output/cornell_<ts1>_*/timing.csv profiler_output/cornell_<ts2>_*/timing.csv --labels "With Sorting" "Without Sorting"
```

---

### Recipe C — Scene Complexity (Diffuse vs Specular/Refractive)

**Purpose:** Understand how scene geometry affects compaction efficiency.
Answer: "Does compaction help more in a simple diffuse scene or one with
specular/refractive chains that keep paths alive?"

**Commands:**
```
:: Simple diffuse scene
cis565_path_tracer.exe ../scenes/cornell.json --benchmark

:: Glass scene (refraction keeps paths bouncing)
cis565_path_tracer.exe ../scenes/cornellRefra.json --benchmark
```

**What to compare:**
- Path survival curves: the glass scene should have more survivors at deep bounces
- `gatherTerminatedPaths` time: should be higher in the diffuse scene (more paths terminate each bounce)
- `shadeMaterial` time: should be lower in the diffuse scene in later bounces (fewer active paths)

**Generate plot:**
```
python scripts/plot_comparison.py profiler_output/cornell_<ts1>_*/timing.csv profiler_output/cornellRefra_<ts2>_*/timing.csv --labels "Diffuse (Cornell)" "Glass (CornellRefra)"
```

---

### Recipe D — Full Matrix (Automated)

**Purpose:** Run all configurations and generate every comparison plot.
One command, hands-off.

**Command:**
```
python scripts/benchmark_runner.py build/bin/Release/cis565_path_tracer.exe scenes/cornell.json
```

**Configuration matrix run by the runner** (default `--configs quick` runs the
first 3; `--configs all` adds the "neither" row):

| # | Compact | Sort | Label |
|---|---------|------|-------|
| 1 | 3 | 1 | baseline (shared-mem scan) |
| 2 | 0 | 1 | no compaction |
| 3 | 3 | 0 | no sorting |
| 4 | 0 | 0 | neither (`--configs all`) |

The same matrix is repeated for a second scene if you pass `--closed-scene <scene>` and the file exists. The default `--closed-scene scenes/cornell_closed.json` no longer exists, so the runner prints a NOTE and skips it — pass e.g. `--closed-scene scenes/cornellRefra.json` to get the "open vs closed" pair.

**Plots generated** (inside `runs/<run-id>/`):

| Plot | Comparison |
|------|------------|
| `<experiment>/kernel_breakdown.png` | Per-bounce kernel breakdown, one per experiment |
| `<experiment>/path_survival.png` | Path survival curve, one per experiment |
| `comparisons/compare_compact_<type>.png` | Compaction ON vs OFF |
| `comparisons/compare_sort_<type>.png` | Sorting ON vs OFF |
| `comparisons/compare_open_vs_closed.png` | Primary vs second scene, both with compaction |
| `comparisons/fps_open.png` | FPS across all configs, primary scene |
| `comparisons/fps_open_vs_closed.png` | FPS comparison between the two scenes |

---

## CSV Output Format

Three files are written to `profiler_output/<scene>_<timestamp>/` on
the final iteration.  If you use `benchmark_runner.py`, the whole run is then
archived under `profiler_output/runs/<run-id>/experiments/` together with the
PNG plots.

### `timing.csv`

One row per measured operation per bounce per iteration.

| Column | Type | Description |
|--------|------|-------------|
| `iteration` | int | Frame number (0-based) |
| `bounce_depth` | int | Bounce index within this iteration |
| `operation` | string | `ShadeMaterial`, `GatherTerminatedPaths`, `SortByMaterial`, `CompactPaths`, `ComputeIntersections`, `BloomPass`, `PostProcessTail` |
| `time_ms` | float | Elapsed time in milliseconds |
| `num_active_paths` | int | Active path count at start of this bounce |
| `compact_method` | int | `0`, `1`, `2`, or `3` |
| `sort_by_material` | int | `0` or `1` |

### `path_survival.csv`

One row per bounce per iteration.

| Column | Type | Description |
|--------|------|-------------|
| `iteration` | int | Frame number |
| `bounce_depth` | int | Bounce index |
| `num_active_paths` | int | Active paths at the start of this bounce |
| `compact_method` | int | `0`, `1`, `2`, or `3` |
| `sort_by_material` | int | `0` or `1` |

### `summary.csv`

Per-operation aggregate statistics (warmup iterations excluded).

| Column | Type | Description |
|--------|------|-------------|
| `operation` | string | Operation name |
| `mean_ms` | float | Mean time across all bounces and non-warmup iterations |
| `std_ms` | float | Standard deviation |
| `min_ms` | float | Minimum observed time |
| `max_ms` | float | Maximum observed time |
| `num_samples` | int | Number of measurements (excluding warmup) |

---

## Measured Operations

| Operation | Timer | When | What it measures |
|-----------|-------|------|-----------------|
| `shadeMaterial` | GPU | Every bounce | BSDF evaluation + Russian roulette. Affected by material sorting (warp divergence) and path count (fewer threads when compaction is on). |
| `gatherTerminatedPaths` | GPU | Every bounce (inside `compactActivePaths`) | Banking dead-path colors into the accumulation buffer. **Absent from CSV when `--compact=0`** (kernel never launched). |
| `sortPathsByMaterial` | GPU | Every bounce | Thrust `sort_by_key` + double `gather`. Time is near-0 when `--sort=0` (early return). **This is typically the largest measured beneficiary of compaction** — fewer active paths → fewer elements to sort. |
| `compactActivePaths` | CPU | Every bounce | Thrust `copy_if` (or custom scan). Includes `cudaDeviceSynchronize` cost from internal Thrust calls. **Absent from CSV when `--compact=0`.** |
| `computeIntersections` | GPU | Every bounce | Ray-geometry intersection test (linear scene scan). Benefits from compaction (fewer active paths → fewer tests). |
| `BloomPass` | GPU | Once per frame (after bounce loop) | Full bloom pipeline: `thresholdExtract` + separable Gaussian blur (horizontal + vertical). Only recorded when bloom is enabled (`intensity > 0`). |
| `PostProcessTail` | GPU | Once per frame (after bounce loop) | Remaining display pipeline: `prepareDisplayKernel` + `tonemapKernel` + (optional) chromatic aberration + (optional) vignette + `sendImageToPBO`. Always recorded. |

Additionally, `num_active_paths` is recorded at the start of every bounce
(path survival metadata).

### NOT measured (starter code / trivial)

- `generateRayFromCamera` — primary ray generation (trivial, single kernel launch)
- `gatherTerminatedPaths` (tail call) — when compaction is OFF, remaining
  path colors are banked by one untimed `gatherTerminatedPaths` launch after
  the bounce loop (the same kernel IS timed per-bounce inside
  `compactActivePaths` when compaction is on).

These are not individually timed.
`sendImageToPBO` is now included inside `PostProcessTail`.

### Note on per-operation vs frame timing

The `beginFrame` / `endFrame` wall-clock timer in `main.cpp` wraps the entire
`pathtrace()` call, including all measured and unmeasured operations, the
post-processing pipeline, and the `cudaMemcpy` D2H of the accumulation buffer.
This is the best measure of real per-iteration cost and is reported in
`frame_times.csv`.  The per-operation timings in `timing.csv` sum to less than
the frame time — the difference is the unmeasured work (primary ray generation,
the tail `gatherTerminatedPaths` call when compaction is off, `cudaMemcpy`,
driver overhead).

---

## ImGui Overlay

When `--benchmark` is active, the "Path Tracer Analytics" window shows:

- Traced depth
- FPS (ImGui rolling average)
- Per-kernel timing for the most recent frame:
  - `ShadeMaterial`
  - `GatherTerminatedPaths`
  - `SortByMaterial`
  - `CompactPaths`
  - `BloomPass` (post-process: bloom pipeline)
  - `PostProcessTail` (post-process: tonemap + CA + vignette + PBO)
- Bounce count for the most recent frame

This is useful for spot-checking during development without waiting for CSV
output.

---

## Tips

1. **Use low iteration counts for quick experiments.**  `"ITERATIONS": 50`
   in the scene JSON gives enough data for a rough comparison and runs in
   seconds.

2. **Increase warmup for production runs.**  `--warmup=5` discards the first
   5 iterations where the GPU may still be thermally throttling or where CUDA
   driver overhead is elevated.

3. **Compare at the same iteration count.**  When comparing configs, use the
   same scene with the same `ITERATIONS` value.  The summary CSV excludes
   warmup iterations automatically.

4. **Nsight for micro-architecture.**  The cudaEvent framework measures
   kernel-level elapsed time.  For branch efficiency, memory coalescing, and
   occupancy analysis, use NVIDIA Nsight Compute.

5. **CSV naming.**  Timestamps prevent overwrites.  When running multiple
   experiments, note the timestamp or rename the files afterward for clarity.
   The `benchmark_runner.py` script tracks this automatically.

6. **Watch the scene camera.**  All shipped scenes place the camera at
   `[0, 5, 10.5]` looking into an open-front box.  When making your own scene,
   keep the camera outside the geometry and clear of the walls — a camera
   clipping into a wall produces a dark or black image.
