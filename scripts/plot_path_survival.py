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

    bounce_counts = defaultdict(list)
    for r in rows:
        bounce_counts[r["bounce_depth"]].append(r["num_active_paths"])

    bounces = sorted(bounce_counts.keys())
    mean_counts = [np.mean(bounce_counts[b]) for b in bounces]
    std_counts  = [np.std(bounce_counts[b])  for b in bounces]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Shaded band: mean ± std
    mean_arr = np.array(mean_counts)
    std_arr  = np.array(std_counts)
    ax.fill_between(bounces, mean_arr - std_arr, mean_arr + std_arr,
                    color="#4C78A8", alpha=0.20, label="Mean ± Std")

    # Mean line
    ax.plot(bounces, mean_counts, color="#4C78A8", linewidth=2.5, label="Mean")

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
                        help="Output PNG path (default: <CSV_stem>/path_survival.png)")
    args = parser.parse_args()
    if args.output is None:
        stem = Path(args.path_survival_csv).stem
        # Remove trailing _path_survival or _timing or _summary from stem
        for suffix in ("_path_survival", "_timing", "_summary"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        out_dir = Path(args.path_survival_csv).parent / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / "path_survival.png")
    main_raw(args.path_survival_csv, args.output)


if __name__ == "__main__":
    main()
