"""Per-bounce material hit mix and effective material diversity."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import profiler_utils as pu
import plot_style


def main_raw(
    run_dirs: str | Path | pu.RunData | list[str | Path | pu.RunData],
    output: str | Path,
) -> None:
    paths = [run_dirs] if isinstance(run_dirs, (str, Path, pu.RunData)) else run_dirs
    runs = [pu.as_run(path) for path in paths]
    rows = [row for run in runs
            for row in pu.measured(run.material_hits, epoch=pu.selected_epoch(run))]
    if not rows:
        raise pu.SchemaError("material mix requires counter runs with material_hits.csv")
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in rows:
        buckets[(row["bounce_depth"], row["material_id"])].append(row["hit_count"])
    bounces = sorted({key[0] for key in buckets})
    materials = sorted({key[1] for key in buckets})
    values = np.asarray([
        [float(np.mean(buckets.get((bounce, material), [0])))
         for bounce in bounces]
        for material in materials
    ])
    totals = values.sum(axis=0)
    shares = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
    log_shares = np.zeros_like(shares)
    np.log(shares, out=log_shares, where=shares > 0)
    entropy = -np.sum(shares * log_shares, axis=0)
    effective = np.exp(entropy)

    plot_style.configure()
    fig, axis = plt.subplots(figsize=(max(10, len(bounces) * .9), 6))
    bottom = np.zeros(len(bounces))
    cmap = plt.get_cmap("tab20")
    for index, material in enumerate(materials):
        axis.bar(bounces, 100.0 * shares[index], bottom=bottom,
                 label=f"material {material}", color=cmap(index % 20))
        bottom += 100.0 * shares[index]
    axis.set_xlabel("Bounce depth")
    axis.set_ylabel("Surface-hit material share (%)")
    axis.set_title(f"Material mix before optional sorting — {pu.run_label(runs[0])}")
    axis.set_xticks(bounces)
    diversity = axis.twinx()
    diversity.plot(bounces, effective, color="black", marker="D",
                   label="effective material count")
    diversity.set_ylabel("exp(Shannon entropy)")
    handles, labels = axis.get_legend_handles_labels()
    handles2, labels2 = diversity.get_legend_handles_labels()
    axis.legend(handles + handles2, labels + labels2, fontsize=8,
                bbox_to_anchor=(1.1, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("-o", "--output", default="material_mix.png")
    args = parser.parse_args()
    main_raw(args.runs, args.output)


if __name__ == "__main__":
    main()
