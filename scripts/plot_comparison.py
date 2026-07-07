"""Grouped bar chart: compare two configurations per-kernel.

Usage:
    python plot_comparison.py <timing_csv_A> <timing_csv_B> [--output comparison.png]
                             [--labels "With Compaction" "Without Compaction"]
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import profiler_utils as pu

OP_COLORS = {
    "ShadeMaterial": "#4C78A8",
    "GatherTerminatedPaths": "#F58518",
    "SortByMaterial": "#E45756",
    "CompactPaths": "#72B7B2",
}


def main_raw(csv_a: str, csv_b: str, output: str,
             labels: list = None) -> None:
    """Programmatic entry point (called by benchmark_runner)."""
    rows_a = pu.parse_timing_csv(csv_a)
    rows_b = pu.parse_timing_csv(csv_b)
    if not rows_a or not rows_b:
        print(f"ERROR: empty CSV(s): {csv_a} / {csv_b}", file=sys.stderr)
        return

    if labels:
        label_a, label_b = labels[0], labels[1]
    else:
        label_a = pu.scalar_to_label(rows_a[0]["compact_method"], rows_a[0]["sort_by_material"])
        label_b = pu.scalar_to_label(rows_b[0]["compact_method"], rows_b[0]["sort_by_material"])

    def collect_stats(rows):
        op_times = defaultdict(list)
        for r in rows:
            op_times[r["operation"]].append(r["time_ms"])
        ops = sorted(op_times.keys())
        means = [np.mean(op_times[op]) for op in ops]
        stds = [np.std(op_times[op]) for op in ops]
        return ops, means, stds

    ops_a, means_a, stds_a = collect_stats(rows_a)
    ops_b, means_b, stds_b = collect_stats(rows_b)
    all_ops = sorted(set(ops_a) | set(ops_b))

    def aligned(o_list, m_list, s_list):
        return [m_list[o_list.index(op)] if op in o_list else 0.0 for op in all_ops], \
               [s_list[o_list.index(op)] if op in o_list else 0.0 for op in all_ops]

    ma, sa = aligned(ops_a, means_a, stds_a)
    mb, sb = aligned(ops_b, means_b, stds_b)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_ops))
    width = 0.35

    ax.bar(x - width/2, ma, width, label=label_a, color="#4C78A8",
           edgecolor="white", yerr=sa, capsize=4)
    ax.bar(x + width/2, mb, width, label=label_b, color="#F58518",
           edgecolor="white", yerr=sb, capsize=4)

    ax.set_ylabel("Mean Time (ms)")
    ax.set_title("Per-Operation Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(all_ops, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="A/B comparison grouped bar chart")
    parser.add_argument("csv_a", help="Config A timing CSV")
    parser.add_argument("csv_b", help="Config B timing CSV")
    parser.add_argument("--output", "-o", default=None,
                        help="Output PNG path (default: derived from CSV names)")
    parser.add_argument("--labels", nargs=2, default=None, help="Labels for config A and B")
    args = parser.parse_args()

    if args.output is None:
        stem_a = Path(args.csv_a).stem
        stem_b = Path(args.csv_b).stem
        for suf in ("_timing", "_path_survival", "_summary"):
            if stem_a.endswith(suf): stem_a = stem_a[: -len(suf)]
            if stem_b.endswith(suf): stem_b = stem_b[: -len(suf)]
        out_dir = Path(args.csv_a).parent / f"{stem_a}_vs_{stem_b}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / "comparison.png")

    main_raw(args.csv_a, args.csv_b, args.output, list(args.labels) if args.labels else None)


if __name__ == "__main__":
    main()
