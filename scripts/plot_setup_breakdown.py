"""CPU startup cost split, including BVH construction and uploads."""

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
    stage_values_by_run: list[dict[str, float]] = []
    setup_totals_by_run: list[float] = []
    for run in runs:
        values: dict[str, float] = {}
        setup_total = 0.0
        for row in run.timings:
            if row["scope"] != "setup":
                continue
            if row["operation"] == "PathtraceInit":
                setup_total += row["time_ms"]
            else:
                values[row["operation"]] = values.get(row["operation"], 0.0) + row["time_ms"]
        stage_values_by_run.append(values)
        setup_totals_by_run.append(setup_total)
    groups: dict[str, list[tuple[pu.RunData, dict[str, float], float]]] = defaultdict(list)
    for run, values, total in zip(runs, stage_values_by_run, setup_totals_by_run):
        groups[str(pu.dotted_get(
            run.metadata, "runner.experiment_id", pu.run_label(run)))].append(
                (run, values, total))
    grouped = list(groups.values())
    stage_values: list[dict[str, float]] = []
    for group in grouped:
        operations = {op for _, values, _ in group for op in values}
        stage_values.append({
            op: float(np.mean([values.get(op, 0.0) for _, values, _ in group]))
            for op in operations
        })
    setup_totals = [float(np.mean([total for _, _, total in group]))
                    for group in grouped]
    for stages, total in zip(stage_values, setup_totals):
        stages["UninstrumentedSetup"] = max(0.0, total - sum(stages.values()))
    operations = sorted({op for values in stage_values for op in values})
    if not operations:
        raise pu.SchemaError("setup breakdown requires detail runs with setup timings")

    plot_style.configure()
    fig, axis = plt.subplots(figsize=(max(10, len(grouped) * 2.2), 6))
    x = np.arange(len(grouped))
    bottom = np.zeros(len(grouped))
    for operation in operations:
        values = np.asarray([stages.get(operation, 0.0) for stages in stage_values])
        axis.bar(x, values, bottom=bottom, label=operation,
                 color=plot_style.operation_color(operation))
        bottom += values
    axis.scatter(x, setup_totals, color="black", marker="D", s=30,
                 label="PathtraceInit total", zorder=5)
    axis.set_ylabel("One-time CPU setup (ms)")
    axis.set_title("Startup cost — scene acceleration, upload, and allocation")
    axis.set_xticks(x)
    axis.set_xticklabels([pu.run_label(group[0][0]) for group in grouped],
                         rotation=18, ha="right")
    axis.grid(axis="x", visible=False)
    axis.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("-o", "--output", default="setup_breakdown.png")
    args = parser.parse_args()
    main_raw(args.runs, args.output)


if __name__ == "__main__":
    main()
