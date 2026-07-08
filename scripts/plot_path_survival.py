"""Path survival curve: active paths per bounce with value labels.

Usage:
    python plot_path_survival.py <path_survival_csv> [--output survival.png]
    plot_path_survival.main_raw(csv, output)  # programmatic
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import profiler_utils as pu


def main_raw(path_survival_csv: str, output: str) -> None:
    rows = pu.parse_path_survival_csv(path_survival_csv)
    if not rows:
        print(f"ERROR: no records in {path_survival_csv}", file=sys.stderr)
        return

    label = pu.scalar_to_label(rows[0]["compact_method"], rows[0]["sort_by_material"])

    bounce_counts = defaultdict(list)
    for r in rows:
        bounce_counts[r["bounce_depth"]].append(r["num_active_paths"])

    bounces = sorted(bounce_counts.keys())
    mean_counts = [np.mean(bounce_counts[b]) for b in bounces]
    std_counts  = [np.std(bounce_counts[b], ddof=1) for b in bounces]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(bounces,
                    [max(0, m - s) for m, s in zip(mean_counts, std_counts)],
                    [m + s for m, s in zip(mean_counts, std_counts)],
                    alpha=0.15, color="#4C78A8")
    ax.plot(bounces, mean_counts, color="#4C78A8", linewidth=2.5,
            marker='o', markersize=6, label="Active Paths")

    # Value labels at first, last, and every other point
    for i, (b, m) in enumerate(zip(bounces, mean_counts)):
        if i == 0 or i == len(bounces) - 1 or i % 2 == 1:
            ax.annotate(f"{m:,.0f}",
                        (b, m),
                        textcoords="offset points",
                        xytext=(0, 12 if i % 4 == 0 else -16),
                        ha="center", fontsize=8, fontweight="bold",
                        color="#4C78A8",
                        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5))

    ax.set_xlabel("Bounce Depth")
    ax.set_ylabel("Active Paths")
    ax.set_title(f"Path Survival Curve — {label}")
    ax.legend()
    ax.set_xticks(bounces)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Path survival line chart")
    parser.add_argument("path_survival_csv", help="Path to path_survival CSV")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()
    if args.output is None:
        csv_path = Path(args.path_survival_csv)
        args.output = str(csv_path.parent / "path_survival.png")
    main_raw(args.path_survival_csv, args.output)


if __name__ == "__main__":
    main()
