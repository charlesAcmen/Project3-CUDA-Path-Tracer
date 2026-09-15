# Benchmarking and visualization guide

The profiling design separates three questions:

1. **Did the renderer get faster?** Use repeated `throughput` processes and
   end-to-end frame time.
2. **Where did time move?** Use `detail` processes and per-iteration stage
   contribution.
3. **Why did work change?** Use `counter` processes for path termination, BVH
   tests, and NEE visibility.

Detailed CUDA events and counter atomics are diagnostic overhead. Their frame
times are never included in the headline speedup chart.

Python 3.11+, NumPy, and Matplotlib are required for the runner/report suite:

```powershell
python -m pip install -r scripts/requirements.txt
```

## One-command single-run analysis

After any profiler-enabled render, pass either its directory or just its run
name to the public analysis entrypoint:

```powershell
python scripts/analyze_run.py stanford_dragon_refractive_20260911_035855Z
```

The command validates schema v2, reads the profiler role from `run.json`,
creates `<run>/analysis/`, generates every figure justified by the available
data, and writes `analysis.md`. A throughput run gets latency/throughput output,
a detail run gets stage/bounce/setup attribution, and a counter run gets path,
termination, BVH/NEE, and material-work figures. Memory is included whenever
the run records allocation estimates. Counter timing is deliberately excluded
from performance figures. If camera or setting changes created multiple
measured accumulation epochs, the command creates one subdirectory per epoch
and a top-level index instead of mixing incompatible samples. `--epoch N` can
restrict analysis to one epoch when desired.

The individual `plot_*.py` programs remain available for advanced composition,
but users do not need to invoke them for ordinary single-run analysis.

## One-command experiment matrix

Build the Release executable first, then run from the repository root:

```powershell
python scripts/benchmark_runner.py build/bin/Release/cis565_path_tracer.exe --spec scripts/experiments/project3.toml
```

The checked-in TOML matrix pins compaction, sorting, RNG, NEE, RR, save
checkpoints, post-processing, profiler role, and output root. It includes the
no-compaction baseline, all three compaction implementations, material sorting,
scrambled Halton, NEE-off, and RR-off variants. Jobs are shuffled independently
within each process-repetition round to reduce fixed-order thermal bias.
Report figures restore the experiment order declared in the TOML after the
randomized execution finishes.

Validate the matrix without rendering:

```powershell
python scripts/benchmark_runner.py build/bin/Release/cis565_path_tracer.exe --spec scripts/experiments/project3.toml --dry-run
```

Useful runner overrides are `--repetitions`, `--warmup`, `--seed`, `--timeout`,
`--output-dir`, and `--keep-going`. A successful batch produces
`profiler_output/benchmark_<UTC>/report.md`, figures, complete logs, generated
configs, and a JSON manifest with Git/executable/spec hashes.
Each job also records the exact scene JSON hash.

For a quick iteration, copy the TOML and restrict experiments or roles; do not
edit a scene's production sample count just for an undocumented benchmark.

## Direct executable controls

| Flag | Meaning |
|---|---|
| `--benchmark` | enable schema-v2 profiling |
| `--profile-mode=throughput\|detail` | whole-pipeline only or granular stages |
| `--profile-counters=0\|1` | exact work counters; use only for counter runs |
| `--warmup=N` | excludes 1-based iterations `1..N` everywhere |
| `--profiler-output=PATH` | result root |
| `--profile-tag=NAME` | stable batch tag in the directory name |
| `--compact=0..3` | off, global scan, Thrust, shared-memory scan |
| `--sort=0\|1` | material sorting |
| `--rng=0\|1` | LCG or scrambled Halton |
| `--direct-lighting=0\|1` | NEE/MIS toggle |
| `--rr-min-bounces=N` | guaranteed bounces; `N >= traceDepth` disables RR |

Configuration priority remains CLI, selected `--config`, local config, code
defaults. The runner uses a generated selected config with every experimental
variable explicit, so a developer's `config.local.json` does not silently
change the matrix.

## Timing boundaries

`end_to_end_ms` begins before CUDA-GL PBO map and ends after the required device
synchronization and PBO unmap. It includes the work the interactive renderer
must complete for one accumulation iteration. It excludes PNG readback,
encoding, checkpoint I/O, GUI draw, and swap/present.

`gpu_pipeline_ms` is a CUDA event around camera-ray generation, all bounce work,
accumulation, post-processing, and PBO write. Detail mode additionally records:

- setup: core/pipeline allocation, scene BVH and light-table construction,
  scene/texture upload;
- frame: camera rays, final gather, bloom stages, display preparation, tonemap,
  chromatic aberration, vignette, copies, PBO write;
- bounce: closest-hit traversal, optional material sort, shading, terminated
  path gathering, and compaction.

GPU ranges are resolved after the application's existing completion point. The
profiler does not synchronize after every operation. Stage stacks include GPU
ranges only; their `UninstrumentedGpuSpan` is the enclosing GPU timeline minus
the named ranges, so launch gaps or work without a child marker remain visible
without being falsely assigned to a kernel. CPU startup is shown separately,
so host wait time is never added on top of the GPU work it waited for.

## Correct aggregation

An operation may run once per frame or once per surviving bounce. Its
contribution is therefore:

```text
stage_ms(iteration, operation) = sum(time_ms of every call in that iteration)
```

Only after that sum do scripts calculate cross-frame means or process-repeat
confidence intervals. A direct unweighted mean across all bounce calls answers
only “average invocation latency”; it overweights shallow frames and cannot
explain total optimization benefit. The old `plot_comparison.py` was removed.

Missing deep bounces count as zero contribution for per-rendered-iteration
bounce plots. Warm-up rows are filtered centrally using the recorded
`is_warmup` bit. A run containing multiple measured accumulation epochs is
rejected by default instead of merging camera/setting resets silently.

## Canonical plots

`make_report.py` generates the normal batch report. Individual tools accept
schema-v2 run directories, not loose CSV paths:

```powershell
python scripts/plot_frame_time.py RUN_DIRS... -o frame_time.png
python scripts/plot_stage_breakdown.py DETAIL_RUN_DIRS... -o stages.png
python scripts/plot_stage_delta.py BASELINE_DETAIL VARIANT_DETAILS... -o delta.png
python scripts/plot_bounce_breakdown.py DETAIL_RUN_DIRS... -o bounce.png
python scripts/plot_path_survival.py COUNTER_RUN_DIRS... -o survival.png --termination-output termination.png --work-output work.png
python scripts/plot_setup_breakdown.py DETAIL_RUN_DIRS... -o setup.png
python scripts/plot_material_mix.py COUNTER_RUN_DIRS... -o material_mix.png
python scripts/plot_memory.py RUN_DIRS... -o memory.png
python scripts/plot_scaling.py THROUGHPUT_RUN_DIRS... -x triangles -o scaling.png
python scripts/plot_optimization_history.py THROUGHPUT_RUN_DIRS... --baseline baseline -o history.png
```

The figures answer distinct questions:

- frame time: repeated-process end-to-end/GPU-span latency, p95, and throughput;
- stage breakdown/delta: total stage contribution per rendered iteration;
- uninstrumented stage span: parent timer minus named, non-overlapping ranges;
- bounce breakdown: where deeper path work accumulates;
- path survival/termination: true live paths versus slots processed;
- work diagnostics: closest/shadow BVH tests and visible/occluded light samples;
- material mix: hit share and effective material diversity per bounce;
- memory: renderer-owned device allocation categories from compile-time layouts;
- setup: one-time SAH BVH, upload, and allocation cost;
- scaling: pixels/triangles/texture pixels versus latency;
- optimization history: end-to-end speedup relative to a declared baseline.

## Quality and external validation

Timing alone cannot justify an estimator change. Use same-camera, same
post-processing PNGs and a high-SPP reference:

```powershell
python scripts/plot_quality.py reference.png "LCG=lcg.png@8.4" "Halton=halton.png@8.6" -o quality.png
```

This reports display-space RGB RMSE/PSNR and, when `@milliseconds` is supplied,
a speed-quality plot. It is not linear-HDR error. Keep NEE/RR/RNG image quality
claims separate from pure implementation speedups.

For a second timing source, export `cuda_gpu_kern_sum` from Nsight Systems and
plot it with:

```powershell
python scripts/plot_nsight.py cuda_gpu_kern_sum.csv -o nsight_kernels.png
```

Use Nsight Compute separately for occupancy, register pressure, branch
efficiency, and memory behavior. Those tools validate microarchitecture; they
do not replace repeated application-level end-to-end timing.

## Extending profiling after a feature or refactor

Use this contract so source changes and plots cannot silently drift apart:

1. Add or rename the non-overlapping C++ range in `ProfilerOp` and
   `profilerOpName`. Generic stage plots discover operations from `timing.csv`;
   do not add another hard-coded operation list to a plot.
2. Put the range in the narrowest correct scope (`setup`, `frame`, or
   `bounce`) and record its work-item count. Keep an enclosing parent timer so
   uninstrumented time remains auditable.
3. If the feature changes work rather than only time, add a counter-mode field
   plus a loader invariant. Counter atomics/readbacks must remain absent from
   throughput and detail roles.
4. If an interactive switch can reset accumulation, add it to
   `ProfilerRuntimeConfig`/`epochs.csv`; otherwise samples from different
   settings may never share an epoch.
5. Add one TOML experiment that changes only the intended variable. Declare
   `compare_to`, `claim_class`, and whether it is safe for the headline
   implementation-speed chart. The report restores TOML declaration order
   after randomized execution.
6. When CSV meaning or required columns change, increment
   `kProfilerSchemaVersion`, update `profiler_utils.py`, its synthetic tests,
   and this document in the same change. Never coerce an old run into a new
   schema.
7. Estimator/sampling changes need same-camera image-quality evidence; kernel
   microarchitecture claims need a separate Nsight export.

Retain a plot only when its title can be phrased as a concrete question and
its metric answers that question without mixing evidence roles.

## Interpretation checklist

- Use Release, the same executable hash, scene, resolution, depth, sample count,
  post-processing, and GPU power/clock conditions.
- Use at least three independent throughput processes; confidence intervals
  over frames from one process are pseudo-replication.
- Treat p95 as a latency-tail descriptor, not a confidence interval.
- Compare compaction net benefit: traversal/shading work saved minus gather and
  scan overhead. Counter curves supply the causal work evidence.
- Compare material sorting using `SortByMaterial + ShadeMaterial` and final
  end-to-end time; lower shading time alone does not prove a net win.
- Never report counter-run timing as FPS, and never claim visual equivalence
  from CSV/static inspection alone.
