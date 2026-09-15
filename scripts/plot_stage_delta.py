"""Stage deltas against a baseline after per-iteration aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import profiler_utils as pu
import plot_style


def main_raw(baseline_dir: str | Path, variant_dirs: list[str | Path],
             output: str | Path) -> None:
    main_grouped([baseline_dir], [[path] for path in variant_dirs], output)


def _mean_stages(runs: list[pu.RunData]) -> dict[str, float]:
    values = [pu.stage_means(run) for run in runs]
    operations = {op for stages in values for op in stages}
    result = {op: float(np.mean([stages.get(op, 0.0) for stages in values]))
              for op in operations}
    frame_gpu = float(np.mean([
        np.mean(pu.frame_values(run, "gpu_pipeline_ms")) for run in runs
    ]))
    result["UninstrumentedGpuSpan"] = max(0.0, frame_gpu - sum(result.values()))
    return result


def main_grouped(baseline_dirs: list[str | Path],
                 variant_groups: list[list[str | Path]],
                 output: str | Path) -> None:
    baselines = [pu.load_run(path) for path in baseline_dirs]
    variants = [[pu.load_run(path) for path in group] for group in variant_groups]
    baseline_stages = _mean_stages(baselines)
    plot_style.configure()
    fig, axes = plt.subplots(len(variants), 1,
                             figsize=(12, max(4, 3.6 * len(variants))),
                             squeeze=False)
    for axis, variant_group in zip(axes[:, 0], variants):
        variant_stages = _mean_stages(variant_group)
        operations = sorted(set(baseline_stages) | set(variant_stages))
        deltas = np.asarray([
            variant_stages.get(op, 0.0) - baseline_stages.get(op, 0.0)
            for op in operations
        ])
        colors = ["#E45756" if value > 0 else "#54A24B" for value in deltas]
        y = np.arange(len(operations))
        axis.barh(y, deltas, color=colors)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_yticks(y)
        axis.set_yticklabels(operations)
        axis.set_xlabel("Variant − baseline stage contribution (ms/iteration)")
        axis.set_title(f"{pu.run_label(variant_group[0])} vs "
                       f"{pu.run_label(baselines[0])}  |  detail-mode attribution")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("variants", nargs="+")
    parser.add_argument("-o", "--output", default="stage_delta.png")
    args = parser.parse_args()
    main_raw(args.baseline, args.variants, args.output)


if __name__ == "__main__":
    main()
