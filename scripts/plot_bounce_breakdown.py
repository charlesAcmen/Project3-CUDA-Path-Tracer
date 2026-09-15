"""Per-bounce contribution where missing deep bounces count as zero."""

from __future__ import annotations

import argparse
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
    per_run = [pu.bounce_stage_means(run) for run in runs]
    if not any(per_run):
        raise pu.SchemaError("bounce breakdown requires a detail run with bounce timings")
    bounces = sorted({bounce for means in per_run for bounce in means})
    operations = sorted({op for means in per_run for values in means.values() for op in values})
    means = {
        bounce: {
            op: float(np.mean([values.get(bounce, {}).get(op, 0.0)
                               for values in per_run]))
            for op in operations
        }
        for bounce in bounces
    }
    plot_style.configure()
    fig, axis = plt.subplots(figsize=(max(10, len(bounces) * 0.9), 6))
    bottom = np.zeros(len(bounces))
    for operation in operations:
        values = np.asarray([means[bounce].get(operation, 0.0) for bounce in bounces])
        axis.bar(bounces, values, bottom=bottom, label=operation,
                 color=plot_style.operation_color(operation))
        bottom += values
    axis.set_xlabel("Bounce depth")
    axis.set_ylabel("Mean contribution per rendered iteration (ms)")
    axis.set_title(f"Bounce-stage contribution — {pu.run_label(runs[0])}\n"
                   "Frames that terminate before a deep bounce contribute zero")
    axis.set_xticks(bounces)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("-o", "--output", default="bounce_breakdown.png")
    args = parser.parse_args()
    main_raw(args.runs, args.output)


if __name__ == "__main__":
    main()
