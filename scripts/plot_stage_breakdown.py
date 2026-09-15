"""Stacked per-iteration stage contribution for detailed profiler runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import profiler_utils as pu
import plot_style


def main_raw(run_dirs: list[str | Path | pu.RunData], output: str | Path) -> None:
    runs = [pu.as_run(path) for path in run_dirs]
    runs = [run for run in runs if pu.dotted_get(run.metadata, "profiler.mode") == "detail"]
    if not runs:
        raise pu.SchemaError("stage breakdown requires at least one detail run")
    groups: dict[str, list[pu.RunData]] = defaultdict(list)
    for run in runs:
        groups[str(pu.dotted_get(
            run.metadata, "runner.experiment_id", pu.run_label(run)))].append(run)
    grouped_runs = list(groups.values())
    per_run_means = [[pu.stage_means(run) for run in group] for group in grouped_runs]
    means = []
    for values in per_run_means:
        operations = {op for stages in values for op in stages}
        means.append({op: float(np.mean([stages.get(op, 0.0) for stages in values]))
                      for op in operations})
    gpu_means = [float(np.mean([
        np.mean(pu.frame_values(run, "gpu_pipeline_ms")) for run in group
    ])) for group in grouped_runs]
    for stages, frame_gpu in zip(means, gpu_means):
        stages["UninstrumentedGpuSpan"] = max(
            0.0, frame_gpu - sum(stages.values()))
    operations = sorted({op for values in means for op in values})
    labels = [str(pu.dotted_get(group[0].metadata, "runner.label", pu.run_label(group[0])))
              for group in grouped_runs]

    plot_style.configure()
    experiment_count = len(grouped_runs)
    fig, axis = plt.subplots(figsize=(max(10, experiment_count * 2.2), 7))
    x = np.arange(experiment_count)
    bottom = np.zeros(experiment_count)
    for operation in operations:
        values = np.asarray([item.get(operation, 0.0) for item in means])
        axis.bar(x, values, bottom=bottom, label=operation,
                 color=plot_style.operation_color(operation), width=0.64)
        bottom += values
    axis.scatter(x, gpu_means, color="black", marker="D", s=30,
                 label="FrameGpu total", zorder=5)
    axis.set_ylabel("Mean contribution per rendered iteration (ms)")
    axis.set_title("Per-iteration stage breakdown\n"
                   "Every bounce invocation is summed before cross-frame statistics")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=18, ha="right")
    axis.grid(axis="x", visible=False)
    axis.legend(fontsize=8, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("-o", "--output", default="stage_breakdown.png")
    args = parser.parse_args()
    main_raw(args.runs, args.output)


if __name__ == "__main__":
    main()
