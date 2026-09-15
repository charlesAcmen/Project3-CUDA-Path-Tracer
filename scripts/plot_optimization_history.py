"""End-to-end optimization history relative to a declared baseline."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import profiler_utils as pu
import plot_style


def main_raw(run_dirs: list[str | Path], output: str | Path,
             baseline_id: str | None = None) -> None:
    grouped: dict[str, list[pu.RunData]] = defaultdict(list)
    for path in run_dirs:
        run = pu.load_run(path)
        if (pu.dotted_get(run.metadata, "profiler.mode") == "throughput"
                and not pu.dotted_get(run.metadata, "profiler.collect_counters", False)
                and pu.dotted_get(run.metadata, "runner.headline", True)):
            key = str(pu.dotted_get(run.metadata, "runner.experiment_id", pu.run_label(run)))
            grouped[key].append(run)
    if len(grouped) < 2:
        raise pu.SchemaError("optimization history needs at least two throughput experiments")
    if baseline_id is None:
        baseline_id = next((key for key, runs in grouped.items()
                            if pu.dotted_get(runs[0].metadata, "runner.baseline", False)),
                           next(iter(grouped)))
    if baseline_id not in grouped:
        raise pu.SchemaError(f"baseline experiment not found: {baseline_id}")

    remaining = sorted(
        (key for key in grouped if key != baseline_id),
        key=lambda key: (
            int(pu.dotted_get(
                grouped[key][0].metadata, "runner.sequence", 1_000_000)),
            key,
        ))
    ordered = [baseline_id, *remaining]
    means = [float(np.mean([np.mean(pu.frame_values(run)) for run in grouped[key]]))
             for key in ordered]
    baseline = means[0]
    speedups = [baseline / value for value in means]
    labels = [str(pu.dotted_get(grouped[key][0].metadata, "runner.label", key))
              for key in ordered]

    plot_style.configure()
    fig, axis = plt.subplots(figsize=(max(10, len(labels) * 1.55), 6))
    x = np.arange(len(labels))
    colors = ["#9D9D9D", *["#4C78A8" if value >= 1 else "#E45756"
                              for value in speedups[1:]]]
    axis.bar(x, speedups, color=colors)
    axis.axhline(1.0, color="black", linewidth=0.8)
    for index, (speedup, latency) in enumerate(zip(speedups, means)):
        axis.text(index, speedup, f"{speedup:.2f}×\n{latency:.2f} ms",
                  ha="center", va="bottom", fontsize=8)
    axis.set_ylabel("End-to-end speedup vs baseline")
    axis.set_title("Optimization impact — process-repeat means")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--baseline")
    parser.add_argument("-o", "--output", default="optimization_history.png")
    args = parser.parse_args()
    main_raw(args.runs, args.output, args.baseline)


if __name__ == "__main__":
    main()
