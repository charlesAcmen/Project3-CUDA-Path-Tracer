# Benchmarking & Experiments Guide

## Overview

The measurement framework instruments user-written GPU kernels and host-side
operations with **cudaEvent** (GPU) and **std::chrono** (CPU) timers.  Results
are written as CSV files to `profiler_output/<scene>_<timestamp>/` and can be
plotted with the companion Python scripts.

Only code you wrote or modified is measured.  Starter-code kernels
(`generateRayFromCamera`, `computeIntersections`, `finalGather`,
`sendImageToPBO`) are deliberately excluded.

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
| `--benchmark` | (none) | off | Enables profiler. CSVs written to `profiler_output/` after the final iteration. |
| `--verbose` | (none) | off | Enables per-bounce path-count `printf` debug output to console. Can be used with or without `--benchmark`. |
| `--compact=N` | `0`, `1`, `2`, `3` | `3` | **Stream compaction method.** `0`=disabled, `1`=global-mem scan, `2`=Thrust `copy_if`, `3`=shared-mem scan (default). |
| `--sort=N` | `0`, `1` | `1` | **Material sorting.** `0`=disabled, `1`=enabled. |
| `--warmup=N` | any int ≥ 0 | `3` | Warmup iterations excluded from summary statistics. |

Flags are order-independent.  `--benchmark` must be present for CSV output;
the other flags only take effect when `--benchmark` is active.  Without
`--benchmark`, profiling overhead is zero — all `gpuStart` / `gpuStop` /
`cpuStart` / `cpuStop` calls are no-ops.

**Note:** `--verbose` is independent of `--benchmark`. Use `--verbose` only when you need to debug path survival behavior, as it produces substantial console output (one line per bounce per iteration).

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

When compaction is **disabled** (`--compact=0`), `gatherTerminatedPaths` and
`compactPaths` are **not launched at all** — they will be absent from the CSV
output (not present as zero-valued rows).  Paths terminate via the
`remainingBounces` guard in `shadeMaterial` and are collected by `finalGather`.

| Value | Meaning |
|-------|---------|
| `0` | **Disabled.** Terminated paths are guarded by `remainingBounces` in `shadeMaterial`. No compaction overhead. |
| `1` | Custom work-efficient scan-based compaction (from Project 2). **Default.** |
| `2` | **Thrust `copy_if`** — reference implementation used in benchmarks. |
| `3` | Shared-memory multi-block scan-based compaction (GPU Gems 3, Ch. 39). |

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
| `cornell.json` | Open Cornell Box | Yes — through the missing front wall |
| `cornell_closed.json` | Closed Cornell Box | No — 6 walls, camera inside |
| `sphere.json` | Single emissive sphere | Yes — all paths miss after first hit |

**Hypothesis:** Compaction removes *more* paths per bounce in the open scene
(paths escape to the sky → miss → terminate) than in the closed scene (paths
hit walls and keep bouncing).  Therefore the performance benefit of compaction
should be larger in the open scene.

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

**When `--compact=0`:** `gatherTerminatedPaths` and `compactPaths` are
**absent from the CSV entirely** (not zero — the kernels are never launched).
Paths terminate via the `remainingBounces` guard in `shadeMaterial` and are
collected by `finalGather` at the end of the frame. All `pixelcount` paths
stay alive through every bounce, so `sortPathsByMaterial` and `shadeMaterial`
always process the full 640,000 elements.

**Generate plots:**
```
python scripts/plot_comparison.py profiler_output/cornell_<ts>_*/timing.csv profiler_output/cornell_<ts2>_*/timing.csv --labels "Compaction ON" "Compaction OFF"
```

---

### Recipe B — Sorting ON vs OFF

**Purpose:** Quantify material sorting benefit.  Answer: "Does reduced warp
divergence in `shadeMaterial` outweigh the Thrust sort overhead?"

**Commands:**
```
:: With sorting (baseline)
cis565_path_tracer.exe ../scenes/cornell.json --benchmark

:: Without sorting
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

### Recipe C — Open vs Closed Scene

**Purpose:** Understand how scene geometry affects compaction efficiency.
Answer: "Does compaction help more in open or closed scenes?"

**Commands:**
```
:: Open scene
cis565_path_tracer.exe ../scenes/cornell.json --benchmark

:: Closed scene
cis565_path_tracer.exe ../scenes/cornell_closed.json --benchmark
```

**What to compare:**
- Path survival curves: closed scene should have more survivors at deep bounces
- `gatherTerminatedPaths` time: should be higher in the open scene (more paths terminate)
- `shadeMaterial` time: should be lower in the open scene in later bounces (fewer active paths)

**Generate plot:**
```
python scripts/plot_comparison.py profiler_output/cornell_<ts1>_*/timing.csv profiler_output/cornell_closed_<ts2>_*/timing.csv --labels "Open (Cornell)" "Closed (Cornell)"
```

---

### Recipe D — Full Matrix (Automated)

**Purpose:** Run all 6 configurations and generate every comparison plot.
One command, hands-off.

**Command:**
```
python scripts/benchmark_runner.py build/bin/Release/cis565_path_tracer.exe scenes/cornell.json
```

**Configuration matrix run by the runner:**

| # | Scene | Compact | Sort | Label |
|---|-------|---------|------|-------|
| 1 | cornell.json | 2 | 1 | open baseline |
| 2 | cornell.json | 0 | 1 | open, no compaction |
| 3 | cornell.json | 2 | 0 | open, no sorting |
| 4 | cornell.json | 0 | 0 | open, neither |
| 5 | cornell_closed.json | 2 | 1 | closed baseline |
| 6 | cornell_closed.json | 0 | 1 | closed, no compaction |

**Plots generated:**

| Plot | Comparison |
|------|------------|
| `breakdown_open_compact2_sort1.png` | Per-bounce kernel breakdown (baseline) |
| `survival_open_compact2_sort1.png` | Path survival curve (baseline) |
| `compare_compact_open.png` | Compaction ON vs OFF, open scene |
| `compare_sort_open.png` | Sorting ON vs OFF, open scene |
| `compare_compact_closed.png` | Compaction ON vs OFF, closed scene |
| `compare_open_vs_closed.png` | Open vs closed, both with compaction |

Use `--configs all` to include the "neither" configuration (#4).

---

## CSV Output Format

Three files are written to `profiler_output/<scene>_<timestamp>/` on
the final iteration:

### `timing.csv`

One row per measured operation per bounce per iteration.

| Column | Type | Description |
|--------|------|-------------|
| `iteration` | int | Frame number (0-based) |
| `bounce_depth` | int | Bounce index within this iteration |
| `operation` | string | `ShadeMaterial`, `GatherTerminatedPaths`, `SortByMaterial`, `CompactPaths` |
| `time_ms` | float | Elapsed time in milliseconds |
| `num_active_paths` | int | Active path count at start of this bounce |
| `compact_method` | int | `0`, `1`, or `2` |
| `sort_by_material` | int | `0` or `1` |

### `path_survival.csv`

One row per bounce per iteration.

| Column | Type | Description |
|--------|------|-------------|
| `iteration` | int | Frame number |
| `bounce_depth` | int | Bounce index |
| `num_active_paths` | int | Active paths at the start of this bounce |
| `compact_method` | int | `0`, `1`, or `2` |
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
| `sortPathsByMaterial` | CPU | Every bounce | Thrust `sort_by_key` + double `gather`. Time is near-0 when `--sort=0` (early return). **This is typically the largest measured beneficiary of compaction** — fewer active paths → fewer elements to sort. |
| `compactActivePaths` | CPU | Every bounce | Thrust `copy_if` (or custom scan). Includes `cudaDeviceSynchronize` cost from internal Thrust calls. **Absent from CSV when `--compact=0`.** |

Additionally, `num_active_paths` is recorded at the start of every bounce
(path survival metadata).

### NOT measured (starter code)

- `generateRayFromCamera` — primary ray generation
- `computeIntersections` — ray-geometry intersection test
- `finalGather` — final path accumulation
- `sendImageToPBO` — tone-mapping and display

These were part of the original project skeleton and were not authored here.

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

6. **Closed scene camera.**  `cornell_closed.json` places the camera at
   `[0, 5, 3]` looking toward `[0, 5, -5]` — inside the box, near the front
   wall.  If the rendered image looks too dark, check that the camera isn't
   clipping through the front wall.
