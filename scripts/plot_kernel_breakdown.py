"""Stacked bar chart: per-bounce kernel time breakdown.

Usage:
    python plot_kernel_breakdown.py <timing_csv> [--output kernel_breakdown.png]
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import profiler_utils as pu

# Fixed color per operation
OP_COLORS = {
    "ShadeMaterial": "#4C78A8",
    "GatherTerminatedPaths": "#F58518",
    "SortByMaterial": "#E45756",
    "CompactPaths": "#72B7B2",
}


def main_raw(timing_csv: str, output: str) -> None:
    """Programmatic entry point (called by benchmark_runner)."""
    rows = pu.parse_timing_csv(timing_csv)
    if not rows:
        print(f"ERROR: no timing records in {timing_csv}", file=sys.stderr)
        return

    label = pu.scalar_to_label(rows[0]["compact_method"], rows[0]["sort_by_material"])

    bounce_ops = defaultdict(lambda: defaultdict(list))
    for r in rows:
        bounce_ops[r["bounce_depth"]][r["operation"]].append(r["time_ms"])

    bounces = sorted(bounce_ops.keys())
    ops = ["SortByMaterial", "ShadeMaterial", "GatherTerminatedPaths", "CompactPaths"]
    ops_present = [op for op in ops if any(op in bounce_ops[b] for b in bounces)]

    means = {op: [] for op in ops_present}
    for b in bounces:
        for op in ops_present:
            vals = bounce_ops[b].get(op, [0.0])
            means[op].append(sum(vals) / len(vals))

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = [0.0] * len(bounces)

    for op in ops_present:
        color = OP_COLORS.get(op, "#888888")
        ax.bar(bounces, means[op], bottom=bottom, label=op, color=color,
               edgecolor="white", linewidth=0.5)
        for i in range(len(bounces)):
            bottom[i] += means[op][i]

    ax.set_xlabel("Bounce Depth")
    ax.set_ylabel("Mean Time (ms)")
    ax.set_title(f"Per-Bounce Kernel Breakdown — {label}")
    ax.legend(loc="upper right")
    ax.set_xticks(bounces)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Stacked bar chart of per-kernel timing per bounce")
    parser.add_argument("timing_csv", help="Path to timing CSV")
    parser.add_argument("--output", "-o", default=None,
                        help="Output PNG path (default: same directory as CSV)")
    args = parser.parse_args()
    
    if args.output is None:
        # Output to the same directory as the CSV file
        csv_path = Path(args.timing_csv)
        args.output = str(csv_path.parent / "kernel_breakdown.png")
    
    main_raw(args.timing_csv, args.output)


if __name__ == "__main__":
    main()
