"""Grouped bar chart: compare two configurations per-operation with value labels.

Usage:
    python plot_comparison.py <csv_a> <csv_b> [--labels "A" "B"] [-o out.png]
    plot_comparison.main_raw(csv_a, csv_b, output, labels)
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import profiler_utils as pu

OP_COLORS = {
    "ComputeIntersections":  "#B279A2",
    "ShadeMaterial":         "#4C78A8",
    "GatherTerminatedPaths": "#F58518",
    "SortByMaterial":        "#E45756",
    "CompactPaths":          "#72B7B2",
    "BloomPass":             "#54A24B",
    "PostProcessTail":       "#Eeca3b",
}


def main_raw(csv_a: str, csv_b: str, output: str,
             labels: list = None) -> None:
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
        stds  = [np.std(op_times[op], ddof=1) for op in ops]
        return ops, means, stds

    ops_a, ma, sa = collect_stats(rows_a)
    ops_b, mb, sb = collect_stats(rows_b)
    all_ops = sorted(set(ops_a) | set(ops_b))

    def aligned(o_list, m_list, s_list):
        return ([m_list[o_list.index(op)] if op in o_list else 0.0 for op in all_ops],
                [s_list[o_list.index(op)] if op in o_list else 0.0 for op in all_ops])

    ma_al, sa_al = aligned(ops_a, ma, sa)
    mb_al, sb_al = aligned(ops_b, mb, sb)

    fig, ax = plt.subplots(figsize=(max(9, 1.5 * len(all_ops)), 6))
    x = np.arange(len(all_ops))
    width = 0.35

    bar_a = ax.bar(x - width / 2, ma_al, width, yerr=sa_al, capsize=4,
                   label=label_a, color="#4C78A8", edgecolor="white",
                   error_kw={"linewidth": 1.0})
    bar_b = ax.bar(x + width / 2, mb_al, width, yerr=sb_al, capsize=4,
                   label=label_b, color="#F58518", edgecolor="white",
                   error_kw={"linewidth": 1.0})

    # Value labels on each bar
    for bars, vals, stds in [(bar_a, ma_al, sa_al), (bar_b, mb_al, sb_al)]:
        for bar, val, std in zip(bars, vals, stds):
            if val > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + std + max(sa_al + sb_al) * 0.02,
                        f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7.5,
                        fontweight="bold")

    ax.set_ylabel("Mean Time (ms)")
    ax.set_title(f"Per-Operation Comparison\n{label_a}  vs  {label_b}")
    ax.set_xticks(x)
    ax.set_xticklabels(all_ops, rotation=15, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="A/B comparison grouped bar chart")
    parser.add_argument("csv_a", help="Config A timing CSV")
    parser.add_argument("csv_b", help="Config B timing CSV")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--labels", nargs=2, default=None)
    args = parser.parse_args()

    if args.output is None:
        path_a = Path(args.csv_a)
        path_b = Path(args.csv_b)
        comp_dir = Path("profiler_output/comparisons")
        comp_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(comp_dir / f"{path_a.parent.name}_vs_{path_b.parent.name}.png")

    main_raw(args.csv_a, args.csv_b, args.output,
             list(args.labels) if args.labels else None)


if __name__ == "__main__":
    main()
