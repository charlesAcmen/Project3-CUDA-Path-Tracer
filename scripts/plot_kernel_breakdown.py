"""Stacked bar chart: per-bounce kernel time breakdown with value labels.

Usage:
    python plot_kernel_breakdown.py <timing_csv> [--output kernel_breakdown.png]
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import profiler_utils as pu

OP_COLORS = {
    "ComputeIntersections":  "#B279A2",
    "ShadeMaterial":         "#4C78A8",
    "GatherTerminatedPaths": "#F58518",
    "SortByMaterial":        "#E45756",
    "CompactPaths":          "#72B7B2",
}


def main_raw(timing_csv: str, output: str) -> None:
    rows = pu.parse_timing_csv(timing_csv)
    if not rows:
        print(f"ERROR: no timing records in {timing_csv}", file=sys.stderr)
        return

    label = pu.scalar_to_label(rows[0]["compact_method"], rows[0]["sort_by_material"])

    bounce_ops = defaultdict(lambda: defaultdict(list))
    for r in rows:
        bounce_ops[r["bounce_depth"]][r["operation"]].append(r["time_ms"])

    bounces = sorted(bounce_ops.keys())
    ops = ["ComputeIntersections", "SortByMaterial", "ShadeMaterial",
           "GatherTerminatedPaths", "CompactPaths"]
    ops_present = [op for op in ops if any(op in bounce_ops[b] for b in bounces)]

    means = {op: [] for op in ops_present}
    for b in bounces:
        for op in ops_present:
            vals = bounce_ops[b].get(op, [0.0])
            means[op].append(sum(vals) / len(vals))

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = [0.0] * len(bounces)
    stack_total = [0.0] * len(bounces)

    for op in ops_present:
        color = OP_COLORS.get(op, "#888888")
        vals = means[op]
        bars = ax.bar(bounces, vals, bottom=bottom, label=op, color=color,
                      edgecolor="white", linewidth=0.5)

        # Value label on each segment that is tall enough
        for i, (bar, val, bt) in enumerate(zip(bars, vals, bottom)):
            if val > 0.04:  # only label segments > 0.04 ms
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bt + val / 2,
                        f"{val:.2f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white", fontweight="bold")
            bottom[i] += val
            stack_total[i] += val

    # Total per bounce above each stack
    for i, (b, total) in enumerate(zip(bounces, stack_total)):
        ax.text(b, total + max(stack_total) * 0.01,
                f"{total:.2f}",
                ha="center", va="bottom", fontsize=7.5,
                fontweight="bold", color="#333333")

    ax.set_xlabel("Bounce Depth (0 = Primary Rays)")
    ax.set_ylabel("Mean GPU Time (ms)")
    ax.set_title(f"Per-Bounce Kernel Breakdown — {label}\n"
                 "(values on bars = ms per operation  |  top numbers = total ms per bounce)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks(bounces)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Stacked bar chart of per-kernel timing per bounce")
    parser.add_argument("timing_csv", help="Path to timing CSV")
    parser.add_argument("--output", "-o", default=None,
                        help="Output PNG path")
    args = parser.parse_args()
    if args.output is None:
        csv_path = Path(args.timing_csv)
        args.output = str(csv_path.parent / "kernel_breakdown.png")
    main_raw(args.timing_csv, args.output)


if __name__ == "__main__":
    main()
