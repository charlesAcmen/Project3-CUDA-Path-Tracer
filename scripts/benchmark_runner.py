"""Automated benchmark runner.

Runs the path tracer binary with multiple configurations, collects CSVs,
and generates comparison plots. The executable is expected to run to
completion (ITERATIONS frames) and exit — CSVs are written by the
profiler on shutdown.

Usage:
    python benchmark_runner.py <path_to_exe> <scene_file>
                               [--configs all|quick]
                               [--closed-scene scenes/cornell_closed.json]
                               [--output-dir results]
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

# Import local plot scripts
import plot_kernel_breakdown
import plot_path_survival
import plot_comparison
import plot_fps
import profiler_utils as pu

# ---- Configuration matrix ----
# Each entry: (name_suffix, compact_method, sort_by_material)
CONFIGS_QUICK = [
    ("compact3_sort1", 3, 1),   # baseline (shared-mem scan)
    ("compact0_sort1", 0, 1),   # no compaction
    ("compact3_sort0", 3, 0),   # no sorting
]

CONFIGS_FULL = CONFIGS_QUICK + [
    ("compact0_sort0", 0, 0),   # neither
]


def find_latest_csvs(output_dir: str, scene_name: str) -> list[str]:
    """Find the most recently created CSV files for a scene.

    CSVs are written by the C++ profiler to:
        profiler_output/<scene_name>_<timestamp>/
    Each directory contains timing.csv, path_survival.csv, summary.csv.
    """
    # Match directory pattern: profiler_output/<scene>_<YYYYMMDD_HHMMSS>/
    pattern = f"{output_dir}/{scene_name}_*"
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not dirs:
        return []
    latest_dir = dirs[0]
    return [
        f"{latest_dir}/timing.csv",
        f"{latest_dir}/path_survival.csv",
        f"{latest_dir}/summary.csv",
        f"{latest_dir}/frame_times.csv",
    ]


def run_one(exe: str, scene: str, compact: int, sort_val: int, warmup: int = 3) -> bool:
    """Run one configuration. Returns True if CSVs were generated."""
    cmd = [
        exe, scene,
        "--benchmark",
        f"--compact={compact}",
        f"--sort={sort_val}",
        f"--warmup={warmup}",
    ]
    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  ERROR: exit code {result.returncode}")
            print(f"  stderr: {result.stderr[-500:]}")
            return False
        # Print last few lines of stdout for sanity
        for line in result.stdout.strip().split("\n")[-5:]:
            print(f"  [out] {line}")
    except subprocess.TimeoutExpired:
        print("  ERROR: timed out after 600s")
        return False
    except FileNotFoundError:
        print(f"  ERROR: executable not found: {exe}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Automated path tracer benchmark runner")
    parser.add_argument("exe", help="Path to the path tracer executable")
    parser.add_argument("scene", help="Path to the open scene (e.g. scenes/cornell.json)")
    parser.add_argument("--closed-scene", default="scenes/cornell_closed.json",
                        help="Path to closed scene")
    parser.add_argument("--configs", choices=["all", "quick"], default="quick",
                        help="Which configurations to run")
    parser.add_argument("--output-dir", "-d", default="profiler_output",
                        help="Directory where the C++ profiler writes CSV output")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations excluded from stats")
    args = parser.parse_args()

    configs = CONFIGS_FULL if args.configs == "all" else CONFIGS_QUICK
    scenes_to_run = [(args.scene, "open")]

    if os.path.exists(args.closed_scene):
        scenes_to_run.append((args.closed_scene, "closed"))
    else:
        print(f"NOTE: closed scene not found ({args.closed_scene}), skipping.")

    results = {}  # (scene_type, config_name) -> list of csv paths

    for scene_path, scene_type in scenes_to_run:
        # Extract scene name from path
        scene_name = Path(scene_path).stem
        for cfg_name, compact, sort_val in configs:
            key = f"{scene_type}_{cfg_name}"
            ok = run_one(args.exe, scene_path, compact, sort_val, args.warmup)
            if ok:
                csvs = find_latest_csvs(args.output_dir, scene_name)
                if csvs:
                    results[key] = csvs
                    print(f"  -> {len(csvs)} CSV(s) generated")
                else:
                    print(f"  -> WARNING: no CSVs found")
            else:
                print(f"  -> FAILED")
            time.sleep(1.0)  # let filesystem flush

    if not results:
        print("\nNo results generated. Aborting plots.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  Generating plots...")
    print(f"{'='*60}")

    for key, csvs in results.items():
        timing_csv = csvs[0]
        path_csv = csvs[1] if len(csvs) > 1 else None

        # Kernel breakdown per config
        plot_kernel_breakdown.main_raw(timing_csv, f"{args.output_dir}/breakdown_{key}.png")

        # Path survival per config
        if path_csv:
            plot_path_survival.main_raw(path_csv, f"{args.output_dir}/survival_{key}.png")

    # A/B comparisons: compaction on/off for open scene
    for scene_type in ["open", "closed"]:
        base_key = f"{scene_type}_compact3_sort1"
        nocomp_key = f"{scene_type}_compact0_sort1"
        nosort_key = f"{scene_type}_compact3_sort0"

        if base_key in results and nocomp_key in results:
            plot_comparison.main_raw(
                results[base_key][0], results[nocomp_key][0],
                f"{args.output_dir}/compare_compact_{scene_type}.png",
                ["With Compaction", "Without Compaction"],
            )

        if base_key in results and nosort_key in results:
            plot_comparison.main_raw(
                results[base_key][0], results[nosort_key][0],
                f"{args.output_dir}/compare_sort_{scene_type}.png",
                ["With Sorting", "Without Sorting"],
            )

    # Open vs closed comparison
    open_key = "open_compact3_sort1"
    closed_key = "closed_compact3_sort1"
    if open_key in results and closed_key in results:
        plot_comparison.main_raw(
            results[open_key][0], results[closed_key][0],
            f"{args.output_dir}/compare_open_vs_closed.png",
            ["Open (Cornell)", "Closed (Cornell)"],
        )

    # FPS comparison: all open-scene configs side by side
    fps_csvs = []
    fps_labels = []
    for cfg_name, compact, sort_val in configs:
        key = f"open_{cfg_name}"
        if key in results and len(results[key]) >= 4:
            fps_csvs.append(results[key][3])   # frame_times.csv
            fps_labels.append(f"compact={compact} sort={sort_val}")
    if len(fps_csvs) >= 2:
        plot_fps.main_raw(fps_csvs, fps_labels,
                          f"{args.output_dir}/fps_open.png")

    # FPS: open vs closed (baseline config only)
    if open_key in results and closed_key in results:
        if len(results[open_key]) >= 4 and len(results[closed_key]) >= 4:
            plot_fps.main_raw(
                [results[open_key][3], results[closed_key][3]],
                ["Open (Cornell)", "Closed (Cornell)"],
                f"{args.output_dir}/fps_open_vs_closed.png")

    print(f"\nAll done. Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
