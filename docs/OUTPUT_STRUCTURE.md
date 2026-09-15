# Profiler schema v2 output

Each executable process creates one immutable run directory. The directory
name is `<scene>_<UTC timestamp>_<optional tag>`; collisions receive a numeric
suffix. The executable also prints `PROFILER_OUTPUT_DIR=<absolute path>`, which
the batch runner consumes instead of guessing the newest directory.

```text
profiler_output/
├── <single-run>/
│   ├── run.json                 # schema, scene, runtime, GPU, runner metadata
│   ├── epochs.csv               # settings for every accumulation epoch
│   ├── frame_times.csv          # end-to-end and GPU-pipeline time
│   ├── timing.csv               # GPU/CPU ranges (detail has all stages)
│   ├── bounce_counters.csv      # counter mode only
│   ├── material_hits.csv        # counter-only per-bounce material histogram
│   └── analysis/                # created by scripts/analyze_run.py
│       ├── analysis.md          # single-run summary and interpretation limits
│       ├── *.png                # role-safe figures for a one-epoch run
│       └── epoch-<N>/            # separate reports when multiple epochs exist
└── benchmark_<UTC timestamp>/
    ├── manifest.json            # commands, hashes, Git state, all exact run paths
    ├── configs/                 # generated configs that pin controlled variables
    ├── logs/                    # complete stdout/stderr per process
    ├── runs/                    # the immutable process run directories
    ├── plots/                   # canonical figures
    └── report.md                # table, figures, and interpretation boundaries
```

All CSVs carry `schema_version=2`, `run_id`, `epoch`, `iteration`, and
`is_warmup` where applicable. Python rejects missing columns and incompatible
schema versions rather than silently plotting stale data.

## Files

`run.json` is the provenance anchor. It records profiler policy, scene path and
size, render dimensions/depth/RR setting, initial renderer switches, and CUDA
device/runtime information. It also reports known renderer-owned device-memory
categories; CUDA/Thrust/runtime-internal allocations are explicitly excluded.
The compaction scan workspace is measured from its live allocation sizes; PBO
and display-texture storage is a nominal RGBA8 estimate because OpenGL driver
padding is not observable here.
The runner appends its experiment id, label, role,
repetition, baseline flag, and generated-config hash.
Runner metadata also carries the scene JSON hash and declared experiment order.

`epochs.csv` marks interactive accumulation resets and their settings. Python
aggregation refuses a run with more than one measured epoch unless the caller
selects exactly one; batch runs normally contain epoch 0 only.

`frame_times.csv` contains:

- `end_to_end_ms`: CUDA-GL map through required GPU completion and unmap;
- `gpu_pipeline_ms`: GPU event around ray generation, bounce pipeline, and
  post-processing.

Checkpoint PNG encoding and file I/O are deliberately outside both values.

`timing.csv` contains `scope` (`setup`, `frame`, `bounce`), optional
`bounce_depth`, operation, timer domain, elapsed milliseconds, and work-item
count. Stage plots first sum every operation invocation within one rendered
iteration. They never take a direct unweighted average across bounce calls.
The stacked frame view adds `UninstrumentedGpuSpan` as the enclosing
`FrameGpu` duration minus named non-overlapping GPU ranges, keeping launch gaps
visible without attributing them to a specific kernel.

`bounce_counters.csv` exists only in explicit counter runs. It records true
survivors and exclusive termination reasons, processed slots, closest/shadow
BVH node and triangle tests, and NEE selection/visibility outcomes. Counter
atomics and synchronous readback perturb timing, so these frame times are never
used for throughput claims.

`material_hits.csv` records the material id histogram for each bounce. It
supports material-mix/entropy plots that contextualize sorting; Nsight branch
metrics and end-to-end throughput are still required to prove a sorting win.

There is intentionally no C++ `summary.csv`. `scripts/profiler_utils.py` is the
single aggregation implementation, including warm-up filtering and bootstrap
confidence intervals.

## Three evidence roles

| Role | Profiler mode | Counters | Valid claim |
|---|---|---:|---|
| `throughput` | throughput | no | end-to-end latency, throughput, speedup |
| `detail` | detail | no | stage and bounce attribution |
| `counter` | detail | yes | path, BVH-work, and NEE causal explanation |

Do not merge samples between roles. For hardware-counter and kernel-level
validation, export Nsight data separately and use `plot_nsight.py`.
