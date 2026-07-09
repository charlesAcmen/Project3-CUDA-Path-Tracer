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
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

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


def find_latest_experiment_dir(output_dir: str, scene_name: str) -> Optional[Path]:
    """Find the most recently created experiment directory for a scene."""
    pattern = f"{output_dir}/{scene_name}_*"
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not dirs:
        return None
    return Path(dirs[0])


def find_latest_csvs(output_dir: str, scene_name: str) -> list[str]:
    """Find the most recently created CSV files for a scene.

    CSVs are written by the C++ profiler to:
        profiler_output/<scene_name>_<timestamp>/
    Each directory contains timing.csv, path_survival.csv, summary.csv.
    """
    latest_dir = find_latest_experiment_dir(output_dir, scene_name)
    if latest_dir is None:
        return []
    return [
        str(latest_dir / "timing.csv"),
        str(latest_dir / "path_survival.csv"),
        str(latest_dir / "summary.csv"),
        str(latest_dir / "frame_times.csv"),
    ]


def _git_short_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or "nogit"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "nogit"


def _git_dirty_suffix() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return "-dirty" if result.stdout.strip() else ""
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def create_run_root(output_dir: str) -> Path:
    runs_dir = Path(output_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_stamp}_g{_git_short_hash()}{_git_dirty_suffix()}"
    run_root = runs_dir / run_name

    suffix = 1
    while run_root.exists():
        run_root = runs_dir / f"{run_name}_{suffix}"
        suffix += 1

    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def archive_experiment_dir(experiment_dir: Path, archive_root: Path, tag: str) -> Path:
    archived_dir = archive_root / "experiments" / f"{experiment_dir.name}__{tag}"
    archived_dir.parent.mkdir(parents=True, exist_ok=True)

    if archived_dir.exists():
        shutil.rmtree(archived_dir)
    shutil.move(str(experiment_dir), str(archived_dir))
    return archived_dir


def write_run_manifest(run_root: Path, args: argparse.Namespace, results: dict) -> None:
    manifest = run_root / "manifest.txt"
    lines = [
        f"Run root: {run_root}",
        f"Git hash: {_git_short_hash()}{_git_dirty_suffix()}",
        f"Executable: {args.exe}",
        f"Primary scene: {args.scene}",
        f"Closed scene: {args.closed_scene}",
        f"Config set: {args.configs}",
        f"Warmup: {args.warmup}",
        "",
        "Experiments:",
    ]

    for key, (csvs, experiment_dir) in results.items():
        lines.append(f"- {key}: {experiment_dir}")
        for csv_path in csvs:
            lines.append(f"  - {csv_path}")

    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    run_root = create_run_root(args.output_dir)

    if os.path.exists(args.closed_scene):
        scenes_to_run.append((args.closed_scene, "closed"))
    else:
        print(f"NOTE: closed scene not found ({args.closed_scene}), skipping.")

    results = {}  # (scene_type, config_name) -> (list of csv paths, experiment dir)

    for scene_path, scene_type in scenes_to_run:
        # Extract scene name from path
        scene_name = Path(scene_path).stem
        for cfg_name, compact, sort_val in configs:
            key = f"{scene_type}_{cfg_name}"
            ok = run_one(args.exe, scene_path, compact, sort_val, args.warmup)
            if ok:
                latest_dir = find_latest_experiment_dir(args.output_dir, scene_name)
                csvs = find_latest_csvs(args.output_dir, scene_name)
                if csvs:
                    if latest_dir is None:
                        print(f"  -> WARNING: no experiment directory found")
                        continue
                    results[key] = (csvs, latest_dir)
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

    comparisons_dir = run_root / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    for key, (csvs, experiment_dir) in results.items():
        timing_csv = csvs[0]
        path_csv = csvs[1] if len(csvs) > 1 else None

        # Kernel breakdown per config
        plot_kernel_breakdown.main_raw(timing_csv, str(experiment_dir / "kernel_breakdown.png"))

        # Path survival per config
        if path_csv:
            plot_path_survival.main_raw(path_csv, str(experiment_dir / "path_survival.png"))

    # A/B comparisons: compaction on/off for open scene
    for scene_type in ["open", "closed"]:
        base_key = f"{scene_type}_compact3_sort1"
        nocomp_key = f"{scene_type}_compact0_sort1"
        nosort_key = f"{scene_type}_compact3_sort0"

        if base_key in results and nocomp_key in results:
            base_csvs, _ = results[base_key]
            nocomp_csvs, _ = results[nocomp_key]
            plot_comparison.main_raw(
                base_csvs[0], nocomp_csvs[0],
                str(comparisons_dir / f"compare_compact_{scene_type}.png"),
                ["With Compaction", "Without Compaction"],
            )

        if base_key in results and nosort_key in results:
            base_csvs, _ = results[base_key]
            nosort_csvs, _ = results[nosort_key]
            plot_comparison.main_raw(
                base_csvs[0], nosort_csvs[0],
                str(comparisons_dir / f"compare_sort_{scene_type}.png"),
                ["With Sorting", "Without Sorting"],
            )

    # Open vs closed comparison
    open_key = "open_compact3_sort1"
    closed_key = "closed_compact3_sort1"
    if open_key in results and closed_key in results:
        open_csvs, _ = results[open_key]
        closed_csvs, _ = results[closed_key]
        plot_comparison.main_raw(
            open_csvs[0], closed_csvs[0],
            str(comparisons_dir / "compare_open_vs_closed.png"),
            ["Open (Cornell)", "Closed (Cornell)"],
        )

    # FPS comparison: all open-scene configs side by side
    fps_csvs = []
    fps_labels = []
    for cfg_name, compact, sort_val in configs:
        key = f"open_{cfg_name}"
        if key in results:
            csvs, _ = results[key]
            if len(csvs) >= 4:
                fps_csvs.append(csvs[3])   # frame_times.csv
                fps_labels.append(f"compact={compact} sort={sort_val}")
    if len(fps_csvs) >= 2:
        plot_fps.main_raw(fps_csvs, fps_labels,
                          str(comparisons_dir / "fps_open.png"))

    # FPS: open vs closed (baseline config only)
    if open_key in results and closed_key in results:
        open_csvs, _ = results[open_key]
        closed_csvs, _ = results[closed_key]
        if len(open_csvs) >= 4 and len(closed_csvs) >= 4:
            plot_fps.main_raw(
                [open_csvs[3], closed_csvs[3]],
                ["Open (Cornell)", "Closed (Cornell)"],
                str(comparisons_dir / "fps_open_vs_closed.png"))

    archived_results = {}
    for key, (csvs, experiment_dir) in results.items():
        archived_results[key] = (csvs, archive_experiment_dir(experiment_dir, run_root, key))

    write_run_manifest(run_root, args, archived_results)

    print(f"\nAll done. Run archived to {run_root}/")


if __name__ == "__main__":
    main()
