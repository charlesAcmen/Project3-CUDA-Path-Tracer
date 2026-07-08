"""Line chart: number of active paths per bounce (path survival curve).

Usage:
    python plot_path_survival.py <path_survival_csv> [--output path_survival.png]
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import profiler_utils as pu


def main_raw(path_survival_csv: str, output: str) -> None:
    """Programmatic entry point (called by benchmark_runner)."""
    rows = pu.parse_path_survival_csv(path_survival_csv)
    if not rows:
        print(f"ERROR: no records in {path_survival_csv}", file=sys.stderr)
        return

    label = pu.scalar_to_label(rows[0]["compact_method"], rows[0]["sort_by_material"])

    # Group path counts by bounce depth (across all iterations)
    bounce_counts = defaultdict(list)
    for r in rows:
        bounce_counts[r["bounce_depth"]].append(r["num_active_paths"])

    # Calculate mean path count for each bounce (averaged across iterations)
    bounces = sorted(bounce_counts.keys())
    mean_counts = [np.mean(bounce_counts[b]) for b in bounces]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean path survival curve
    ax.plot(bounces, mean_counts, color="#4C78A8", linewidth=2.5, marker='o', markersize=6, label="Active Paths")

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
    parser.add_argument("--output", "-o", default=None,
                        help="Output PNG path (default: same directory as CSV)")
    args = parser.parse_args()
    
    if args.output is None:
        # Output to the same directory as the CSV file
        csv_path = Path(args.path_survival_csv)
        args.output = str(csv_path.parent / "path_survival.png")
    
    main_raw(args.path_survival_csv, args.output)


if __name__ == "__main__":
    main()
