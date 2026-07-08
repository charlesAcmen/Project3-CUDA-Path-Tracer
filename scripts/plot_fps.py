"""FPS comparison: mean iterations-per-second across configurations.

Reads frame_times.csv from one or more experiment directories and produces:
  - Grouped bar chart comparing mean FPS (1000 / frame_time_ms)
  - FPS = iterations per second of the core bounce loop
    (excludes finalGather / sendImageToPBO / cudaMemcpy).

Usage:
    python plot_fps.py <csv_a> [csv_b ...] [--labels "A" "B" ...] [-o out.png]

    # Called programmatically by benchmark_runner:
    plot_fps.main_raw(csv_paths, labels, output)
"""

import argparse
import sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def _read_frame_times(filepath: str) -> list[float]:
    """Parse frame_times.csv, return list of frame_time_ms values."""
    times = []
    with open(filepath, "r", newline="") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                times.append(float(parts[1]))
    return times


def _label_from_csv(csv_path: str) -> str:
    """Derive a compact label from the experiment directory name."""
    import os
    dname = os.path.basename(os.path.dirname(csv_path))
    parts = dname.split("_")
    if len(parts) >= 2:
        return parts[0]  # e.g. "cornell" or "cornell"
    return dname


def main_raw(csv_paths: list[str], labels: list[str], output: str) -> None:
    """Programmatic entry point (called by benchmark_runner)."""
    all_times = []
    for path in csv_paths:
        times = _read_frame_times(path)
        if not times:
            print(f"ERROR: no frame times in {path}", file=sys.stderr)
            return
        all_times.append(times)

    if labels is None or len(labels) != len(csv_paths):
        labels = [_label_from_csv(p) for p in csv_paths]

    means_fps = [1000.0 / np.mean(t) for t in all_times]
    stds_fps  = [1000.0 * np.std(t) / (np.mean(t) ** 2) for t in all_times]
    # ^ first-order propagation: std(FPS) ≈ 1000 * std(ms) / mean(ms)^2

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(labels))
    bars = ax.bar(x, means_fps, yerr=stds_fps, capsize=6,
                  color=["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#B279A2"][:len(labels)],
                  edgecolor="white", linewidth=0.5)

    # Annotate bars with FPS value
    for bar, fps in zip(bars, means_fps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds_fps) * 0.1,
                f"{fps:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Iterations / Second (FPS)")
    ax.set_title("Bounce-Loop Throughput by Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="FPS (iterations/sec) comparison from frame_times.csv")
    parser.add_argument("csvs", nargs="+", help="One or more frame_times.csv paths")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Labels for each CSV (same order)")
    parser.add_argument("--output", "-o", default="fps_comparison.png",
                        help="Output PNG path")
    args = parser.parse_args()

    main_raw(args.csvs, args.labels, args.output)


if __name__ == "__main__":
    main()
