"""Plot renderer scaling against pixels, triangles, or texture pixels."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import profiler_utils as pu
import plot_style


X_FIELDS = {
    "pixels": lambda run: int(pu.dotted_get(run.metadata, "scene.width", 0))
                          * int(pu.dotted_get(run.metadata, "scene.height", 0)),
    "triangles": lambda run: int(pu.dotted_get(run.metadata, "scene.triangles", 0)),
    "texture_pixels": lambda run: int(pu.dotted_get(
        run.metadata, "scene.texture_pixels", 0)),
}


def main_raw(run_dirs: list[str | Path], output: str | Path,
             x_field: str = "triangles") -> None:
    runs = [pu.load_run(path) for path in run_dirs]
    runs = [run for run in runs
            if pu.dotted_get(run.metadata, "profiler.mode") == "throughput"
            and not pu.dotted_get(run.metadata, "profiler.collect_counters", False)]
    if x_field not in X_FIELDS:
        raise ValueError(f"unsupported x field: {x_field}")
    grouped: dict[tuple[str, int], list[pu.RunData]] = defaultdict(list)
    for run in runs:
        experiment = str(pu.dotted_get(
            run.metadata, "runner.experiment_id", pu.run_label(run)))
        grouped[(experiment, X_FIELDS[x_field](run))].append(run)
    if len({x for _, x in grouped}) < 2:
        raise pu.SchemaError(f"scaling plot needs at least two distinct {x_field} values")

    points = []
    for (experiment, x_value), group in grouped.items():
        replicate_means = [float(np.mean(pu.frame_values(run))) for run in group]
        mean, low, high = pu.mean_ci95(replicate_means)
        points.append((x_value, mean, mean - low, high - mean,
                       pu.run_label(group[0]), experiment))
    points.sort(key=lambda point: (point[0], point[5]))

    plot_style.configure()
    fig, axis = plt.subplots(figsize=(9, 6))
    for index, (x_value, latency, low_error, high_error, label, _) in enumerate(points):
        color = plt.get_cmap("tab10")(index % 10)
        axis.errorbar(x_value, latency,
                      yerr=np.asarray([[low_error], [high_error]]),
                      fmt="o", markersize=7, capsize=4, color=color)
        axis.annotate(label, (x_value, latency), xytext=(5, 5),
                      textcoords="offset points", fontsize=8)
    axis.set_xlabel(x_field.replace("_", " ").title())
    axis.set_ylabel("Mean end-to-end latency (ms)")
    axis.set_title(
        f"Renderer scaling with {x_field.replace('_', ' ')}\n"
        "95% CI across process-repeat means; points are not fitted")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("-x", "--x-field", choices=sorted(X_FIELDS), default="triangles")
    parser.add_argument("-o", "--output", default="scaling.png")
    args = parser.parse_args()
    main_raw(args.runs, args.output, args.x_field)


if __name__ == "__main__":
    main()
