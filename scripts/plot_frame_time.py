"""End-to-end latency and throughput with confidence across process repeats."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import profiler_utils as pu
import plot_style


def _experiment_id(run: pu.RunData) -> str:
    return str(pu.dotted_get(run.metadata, "runner.experiment_id", pu.run_label(run)))


def main_raw(run_dirs: list[str | Path | pu.RunData], output: str | Path) -> None:
    runs = [pu.as_run(path) for path in run_dirs]
    groups: dict[str, list[pu.RunData]] = defaultdict(list)
    for run in runs:
        if pu.dotted_get(run.metadata, "profiler.collect_counters", False):
            continue
        if pu.dotted_get(run.metadata, "profiler.mode") != "throughput":
            continue
        groups[_experiment_id(run)].append(run)
    if not groups:
        raise pu.SchemaError("no throughput runs without counter overhead")

    labels: list[str] = []
    latency_means: list[float] = []
    latency_low: list[float] = []
    latency_high: list[float] = []
    gpu_means: list[float] = []
    gpu_low: list[float] = []
    gpu_high: list[float] = []
    throughput_means: list[float] = []
    throughput_low: list[float] = []
    throughput_high: list[float] = []
    p95_values: list[float] = []

    for experiment, experiment_runs in groups.items():
        labels.append(str(pu.dotted_get(experiment_runs[0].metadata, "runner.label", experiment)))
        replicate_means = [float(np.mean(pu.frame_values(run))) for run in experiment_runs]
        mean, low, high = pu.mean_ci95(replicate_means)
        latency_means.append(mean)
        latency_low.append(mean - low)
        latency_high.append(high - mean)

        replicate_gpu_means = [
            float(np.mean(pu.frame_values(run, "gpu_pipeline_ms")))
            for run in experiment_runs
        ]
        gpu_mean, gpu_ci_low, gpu_ci_high = pu.mean_ci95(replicate_gpu_means)
        gpu_means.append(gpu_mean)
        gpu_low.append(gpu_mean - gpu_ci_low)
        gpu_high.append(gpu_ci_high - gpu_mean)

        replicate_throughput = [1000.0 / value for value in replicate_means]
        t_mean, t_low, t_high = pu.mean_ci95(replicate_throughput)
        throughput_means.append(t_mean)
        throughput_low.append(t_mean - t_low)
        throughput_high.append(t_high - t_mean)
        all_frames = [value for run in experiment_runs for value in pu.frame_values(run)]
        p95_values.append(pu.percentile(all_frames, 95))

    plot_style.configure()
    fig, (latency_ax, throughput_ax) = plt.subplots(
        1, 2, figsize=(max(11, len(labels) * 2.4), 6))
    x = np.arange(len(labels))
    colors = [plt.get_cmap("Set2")(i % 8) for i in range(len(labels))]
    width = 0.36
    latency_ax.bar(x - width / 2, latency_means, width,
                   yerr=np.asarray([latency_low, latency_high]),
                   capsize=4, color=colors, label="end-to-end")
    latency_ax.bar(x + width / 2, gpu_means, width,
                   yerr=np.asarray([gpu_low, gpu_high]),
                   capsize=4, color=colors, alpha=0.42,
                   hatch="//", label="GPU pipeline span")
    throughput_ax.bar(x, throughput_means,
                      yerr=np.asarray([throughput_low, throughput_high]),
                      capsize=5, color=colors)
    for index, (mean, p95) in enumerate(zip(latency_means, p95_values)):
        latency_ax.text(index - width / 2, mean, f"{mean:.2f} ms\np95 {p95:.2f}",
                        ha="center", va="bottom", fontsize=8)
    for index, value in enumerate(throughput_means):
        throughput_ax.text(index, value, f"{value:.1f} iter/s",
                           ha="center", va="bottom", fontsize=8)
    latency_ax.set_ylabel("End-to-end frame latency (ms)")
    throughput_ax.set_ylabel("Render throughput (iterations/s)")
    latency_ax.set_title("Latency — end-to-end vs enclosed GPU timeline")
    throughput_ax.set_title("Throughput — 95% CI across process repeats")
    for axis in (latency_ax, throughput_ax):
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=18, ha="right")
        axis.grid(axis="x", visible=False)
    latency_ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="schema-v2 run directories")
    parser.add_argument("-o", "--output", default="frame_time.png")
    args = parser.parse_args()
    main_raw(args.runs, args.output)


if __name__ == "__main__":
    main()
