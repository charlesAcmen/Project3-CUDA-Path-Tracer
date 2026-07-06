# Build Variants for Performance Testing

## Overview

This project now builds **three** executables to facilitate performance comparison in Nsight Compute:

1. **cis565_path_tracer** - Default build (with material sorting enabled)
2. **cis565_path_tracer_sorted** - Explicitly enables `SORT_BY_MATERIAL=1`
3. **cis565_path_tracer_unsorted** - Explicitly disables `SORT_BY_MATERIAL=0`

## Building

```bash
cd build
cmake ..
cmake --build . --config Release
```

All three executables will be built simultaneously and placed in `build/bin/`.

## Executables Location

After building, you'll find:

- `build/bin/cis565_path_tracer.exe` (default, sorted)
- `build/bin/cis565_path_tracer_sorted.exe`
- `build/bin/cis565_path_tracer_unsorted.exe`

## Usage for Nsight Compute Analysis

### Method 1: Using Nsight Compute GUI

1. Open Nsight Compute
2. Create two separate profiling sessions:
   - Session 1: Point to `build/bin/cis565_path_tracer_sorted.exe`
   - Session 2: Point to `build/bin/cis565_path_tracer_unsorted.exe`
3. Run both sessions with the same scene and iteration count
4. Use Nsight's comparison features to analyze the performance difference

### Method 2: Using Command Line

Profile the sorted version:
```bash
ncu --set full -o profile_sorted build/bin/cis565_path_tracer_sorted.exe
```

Profile the unsorted version:
```bash
ncu --set full -o profile_unsorted build/bin/cis565_path_tracer_unsorted.exe
```

Then compare the two profile files in Nsight Compute GUI.

### Method 3: Kernel Replay for Specific Kernel

To profile just the `shadeMaterial` kernel:

```bash
# Sorted version
ncu --kernel-name shadeMaterial --set full -o shade_sorted build/bin/cis565_path_tracer_sorted.exe

# Unsorted version
ncu --kernel-name shadeMaterial --set full -o shade_unsorted build/bin/cis565_path_tracer_unsorted.exe
```

## Key Metrics to Compare

When analyzing the `shadeMaterial` kernel, focus on:

- **Warp execution efficiency** - Should be higher with sorting
- **Memory coalescing** (L1/L2 cache hit rates) - Should improve with sorting
- **Branch divergence** - Should be reduced with sorting
- **Overall execution time** - Compare total kernel duration

## Implementation Details

The `SORT_BY_MATERIAL` flag is now controlled via CMake compile definitions:
- Source code checks for the macro with `#ifndef SORT_BY_MATERIAL`
- CMake passes `-DSORT_BY_MATERIAL=1` or `-DSORT_BY_MATERIAL=0` during compilation
- This ensures complete recompilation when switching variants
- No need to manually edit source code

## Advantages of This Approach

1. **No manual code changes** - Switch between variants without editing source
2. **Parallel builds** - Both variants can coexist without recompilation
3. **Clean comparison** - Guaranteed identical code except for the flag
4. **Nsight-friendly** - Each executable can be profiled independently
5. **Version control** - No accidental commits of different flag values
