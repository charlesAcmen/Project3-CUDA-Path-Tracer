# Profiler Output Structure

## Overview

The profiler outputs are organized in a clean directory structure that keeps each experiment's data together.

## Directory Structure

```
build/profiler_output/
├── <scene>_<timestamp>/          # One directory per experiment run
│   ├── timing.csv                # Raw timing data (all operations × bounces × iterations)
│   ├── summary.csv               # Aggregated statistics (mean, std, min, max per operation)
│   ├── path_survival.csv         # Path count per bounce per iteration
│   ├── path_survival.png         # Path survival curve chart
│   └── kernel_breakdown.png      # Stacked bar chart of kernel timing per bounce
│
└── comparisons/                  # Cross-experiment comparisons
    └── expA_vs_expB.png          # Side-by-side comparison charts
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
├── cornell_closed_20260708_085541/
│   ├── timing.csv
│   ├── summary.csv
│   ├── path_survival.csv
│   ├── path_survival.png
│   └── kernel_breakdown.png
│
└── comparisons/
    └── cornell_20260708_085529_vs_cornell_closed_20260708_085541.png
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
- Comparison files in `comparisons/` follow the pattern: `<expA>_vs_<expB>.png`

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

Plots are generated automatically in the same directory as the CSV:

```bash
# From the build directory
python ..\scripts\plot_path_survival.py .\profiler_output\cornell_20260708_085529\path_survival.csv

python ..\scripts\plot_kernel_breakdown.py .\profiler_output\cornell_20260708_085529\timing.csv
```

### Comparing Experiments

```bash
python ..\scripts\plot_comparison.py ^
  .\profiler_output\cornell_20260708_085529\timing.csv ^
  .\profiler_output\cornell_closed_20260708_085541\timing.csv ^
  --labels "Open Scene" "Closed Scene"
```

Output appears at:
```
build/profiler_output/comparisons/cornell_20260708_085529_vs_cornell_closed_20260708_085541.png
```

## Benefits

1. **Self-contained experiments** - All data for one run is in one folder
2. **Easy to archive** - Just copy/move the experiment folder
3. **Clear organization** - CSV data and visualizations together
4. **Comparison separation** - Cross-experiment comparisons in dedicated folder
5. **No filename clutter** - Simple names (timing.csv, not cornell_20260708_085529_timing.csv)
