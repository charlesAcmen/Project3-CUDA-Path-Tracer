# Profiler Output Structure

## Overview

The profiler outputs are organized in two layers:

1. Raw profiler CSVs written by the C++ executable to `profiler_output/<scene>_<timestamp>/`
2. Batch archives created by `scripts/benchmark_runner.py` under `profiler_output/runs/<run-id>/`

The batch archive is the durable unit to keep when you want to compare code changes over time.

## Directory Structure

```
build/profiler_output/
├── <scene>_<timestamp>/          # Raw profiler output for one executable run
│   ├── timing.csv                # Raw timing data (all operations × bounces × iterations)
│   ├── summary.csv               # Aggregated statistics (mean, std, min, max per operation)
│   ├── path_survival.csv         # Path count per bounce per iteration
│   ├── path_survival.png         # Path survival curve chart for this run
│   └── kernel_breakdown.png      # Stacked bar chart of kernel timing per bounce for this run
│
└── runs/                         # Durable benchmark batches
  └── <run-id>/                 # One full benchmark_runner invocation
    ├── manifest.txt          # Git hash, arguments, and archived experiment list
    ├── comparisons/          # Cross-experiment comparison charts for this batch
    └── experiments/          # Archived raw experiment directories with per-run PNGs
      └── <scene>_<timestamp>__<scene_type>_<config>/
        ├── timing.csv
        ├── summary.csv
        ├── path_survival.csv
        ├── frame_times.csv
        ├── path_survival.png
        └── kernel_breakdown.png
```

## Example

```
build/profiler_output/
├── cornell_20260708_085529/
│   ├── timing.csv
│   ├── summary.csv
│   ├── path_survival.csv
│   ├── path_survival.png
│   └── kernel_breakdown.png
│
├── cornellRefra_20260708_085541/
│   ├── timing.csv
│   ├── summary.csv
│   ├── path_survival.csv
│   ├── path_survival.png
│   └── kernel_breakdown.png
│
└── runs/
  └── 20260710_143455_g1a2b3c4-dirty/
    ├── manifest.txt
    ├── comparisons/
    │   └── compare_open_vs_closed.png
    └── experiments/
      └── cornell_20260710_143401__open_compact3_sort1/
        ├── timing.csv
        ├── summary.csv
        ├── path_survival.csv
        ├── frame_times.csv
        ├── path_survival.png
        └── kernel_breakdown.png
```

## Naming Convention

### Experiment Directory
Format: `<scene_name>_<timestamp>`
- `scene_name`: Extracted from the scene JSON file (e.g., "cornell" from "cornell.json")
- `timestamp`: YYYYMMDD_HHMMSS format (e.g., "20260708_085529")

### CSV Files
- `timing.csv` - Per-operation, per-bounce, per-iteration raw measurements
- `summary.csv` - Statistical summary (mean, std, min, max) for each operation
- `path_survival.csv` - Active path count at each bounce for each iteration

### PNG Files
- `path_survival.png` - Line chart showing how path count decreases with bounce depth
- `kernel_breakdown.png` - Stacked bar chart showing time breakdown by operation per bounce
- Comparison files in `runs/<run-id>/comparisons/` follow the pattern: `compare_*.png`

## Usage

### Running an Experiment

```bash
cd build
.\bin\Release\cis565_path_tracer.exe ..\scenes\cornell.json --benchmark
```

Output appears at:
```
build/profiler_output/cornell_<timestamp>/
```

### Generating Plots

The benchmark runner writes run-specific plots next to the CSVs in each experiment folder, then archives the whole batch into `runs/<run-id>/`:

```bash
# From the build directory
python ..\scripts\plot_path_survival.py .\profiler_output\cornell_20260708_085529\path_survival.csv

python ..\scripts\plot_kernel_breakdown.py .\profiler_output\cornell_20260708_085529\timing.csv
```

### Comparing Experiments

```bash
python ..\scripts\plot_comparison.py ^
  .\profiler_output\cornell_20260708_085529\timing.csv ^
  .\profiler_output\cornellRefra_20260708_085541\timing.csv ^
  --labels "Diffuse (Cornell)" "Glass (CornellRefra)"
```

Output appears at:
```
build/profiler_output/runs/<run-id>/comparisons/compare_open_vs_closed.png
```

## Benefits

1. **Self-contained experiments** - All data for one run is in one folder
2. **Easy to archive** - Just copy/move the experiment folder
3. **Clear organization** - CSV data and visualizations together
4. **Comparison separation** - Cross-experiment comparisons in dedicated folder
5. **No filename clutter** - Simple names (timing.csv, not cornell_20260708_085529_timing.csv)
