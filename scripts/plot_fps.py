"""Render FPS comparison with stability metrics.

Reads frame_times.csv (wall time of full pathtrace() call per iteration)
and produces a grouped bar chart showing:
  - Mean render FPS (1000 / frame_time_ms) with ±1σ error bars
  - Numeric labels: "FPS ± σ"
  - CV% (coefficient of variation) below each bar for stability assessment

Usage:
    python plot_fps.py <csv> [csv ...] [--labels "A" "B"] [-o out.png]
    plot_fps.main_raw(csv_paths, labels, output)  # programmatic
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt


def _read_frame_times(filepath: str) -> list[float]:
    times = []
    with open(filepath, "r", newline="") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                times.append(float(parts[1]))
    return times


def main_raw(csv_paths: list[str], labels: list[str], output: str) -> None:
    all_times = []
    for path in csv_paths:
        times = _read_frame_times(path)
        if not times:
            print(f"ERROR: no frame times in {path}", file=sys.stderr)
            return
        all_times.append(times)

    if labels is None or len(labels) != len(csv_paths):
        labels = [f"Config {i}" for i in range(len(csv_paths))]

    means_ms = [np.mean(t) for t in all_times]
    stds_ms  = [np.std(t, ddof=1) for t in all_times]
    means_fps = [1000.0 / m for m in means_ms]
    # Propagation: σ_FPS ≈ (1000 / μ²) · σ_ms
    stds_fps  = [1000.0 * s / (m * m) for m, s in zip(means_ms, stds_ms)]
    cv_pct     = [100.0 * s / m for m, s in zip(means_ms, stds_ms)]

    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#B279A2"]

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(labels)), 6))

    x = np.arange(len(labels))
    bars = ax.bar(x, means_fps, yerr=stds_fps, capsize=8, width=0.55,
                  color=colors[:len(labels)],
                  edgecolor="white", linewidth=0.8,
                  error_kw={"linewidth": 1.5, "ecolor": "#333333"})

    # FPS value + σ above each bar
    y_offset = max(stds_fps) * 0.15 if max(stds_fps) > 0 else 1.0
    for bar, fps, std, cv in zip(bars, means_fps, stds_fps, cv_pct):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + std + y_offset,
                f"{fps:.1f} ± {std:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        # Stability metric below bar
        ax.text(bar.get_x() + bar.get_width() / 2, max(h * 0.02, 0.3),
                f"CV {cv:.1f}%",
                ha="center", va="bottom", fontsize=8, color="white",
                fontweight="bold")

    ax.set_ylabel("Render FPS  (iterations / second)")
    ax.set_title("Render Throughput by Configuration\n"
                 "(full frame: primary rays → bounce loop → finalGather → PBO → cudaMemcpy)\n"
                 "Error bars = ±1σ  |  CV% = std/mean — lower = more stable",
                 fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, max(means_fps) * 1.35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Render FPS comparison from frame_times.csv")
    parser.add_argument("csvs", nargs="+", help="frame_times.csv paths")
    parser.add_argument("--labels", nargs="*", default=None, help="Config labels")
    parser.add_argument("--output", "-o", default="fps_comparison.png")
    args = parser.parse_args()
    main_raw(args.csvs, args.labels, args.output)


if __name__ == "__main__":
    main()
