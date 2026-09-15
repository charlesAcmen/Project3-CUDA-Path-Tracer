"""Exact active-path, processed-slot, and termination-reason plots."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import profiler_utils as pu
import plot_style


REASONS = [
    "misses",
    "emissive_terminations",
    "invalid_surface_terminations",
    "russian_roulette_terminations",
    "max_depth_terminations",
    "debug_terminations",
]


def _bounce_means(runs: pu.RunData | list[pu.RunData]) -> dict[int, dict[str, float]]:
    run_list = [runs] if isinstance(runs, pu.RunData) else runs
    rows = [row for run in run_list
            for row in pu.measured(run.counters, epoch=pu.selected_epoch(run))]
    if not rows:
        raise pu.SchemaError("no measured bounce counters")
    measured_frames = sum(len(pu.iteration_keys(run)) for run in run_list)
    if measured_frames == 0:
        raise pu.SchemaError("counter runs contain no measured frames")
    fields = [
        "processed_paths", "active_after_bounce", "already_terminated", *REASONS,
        "closest_bvh_node_tests", "closest_bvh_triangle_tests",
        "shadow_bvh_node_tests", "shadow_bvh_triangle_tests",
        "light_selections", "valid_light_samples", "shadow_rays",
        "visible_light_samples", "occluded_light_samples",
    ]
    # The denominator is every rendered iteration across all repetitions.
    # If all paths die before a deep bounce, that absent bounce contributes
    # zero instead of silently turning the result into a conditional mean.
    totals: dict[int, dict[str, float]] = defaultdict(
        lambda: {field: 0.0 for field in fields})
    for row in rows:
        for field in fields:
            totals[row["bounce_depth"]][field] += row[field]
    return {
        bounce: {field: value / measured_frames for field, value in values.items()}
        for bounce, values in sorted(totals.items())
    }


def main_raw(run_dirs: list[str | Path | pu.RunData], output: str | Path) -> None:
    runs = [pu.as_run(path) for path in run_dirs]
    groups: dict[str, list[pu.RunData]] = defaultdict(list)
    for run in runs:
        groups[str(pu.dotted_get(
            run.metadata, "runner.experiment_id", pu.run_label(run)))].append(run)
    plot_style.configure()
    fig, axis = plt.subplots(figsize=(10, 6))
    for index, group in enumerate(groups.values()):
        run = group[0]
        means = _bounce_means(group)
        bounces = sorted(means)
        pixel_count = int(pu.dotted_get(run.metadata, "scene.width", 1)) * int(
            pu.dotted_get(run.metadata, "scene.height", 1))
        active = [100.0 * means[b]["active_after_bounce"] / pixel_count for b in bounces]
        processed = [100.0 * means[b]["processed_paths"] / pixel_count for b in bounces]
        color = plt.get_cmap("tab10")(index % 10)
        axis.plot(bounces, active, marker="o", color=color,
                  label=f"{pu.run_label(run)} — true active")
        axis.plot(bounces, processed, linestyle="--", color=color, alpha=0.75,
                  label=f"{pu.run_label(run)} — processed slots")
    axis.set_xlabel("Bounce depth")
    axis.set_ylabel("Paths relative to primary rays (%)")
    axis.set_title("True path survival vs slots processed by the selected compaction mode")
    axis.set_ylim(bottom=0)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def termination_raw(
    run_dirs: str | Path | pu.RunData | list[str | Path | pu.RunData],
    output: str | Path,
) -> None:
    paths = [run_dirs] if isinstance(run_dirs, (str, Path, pu.RunData)) else run_dirs
    runs = [pu.as_run(path) for path in paths]
    run = runs[0]
    means = _bounce_means(runs)
    bounces = sorted(means)
    plot_style.configure()
    fig, axis = plt.subplots(figsize=(max(10, len(bounces) * 0.85), 6))
    bottom = np.zeros(len(bounces))
    for index, reason in enumerate(REASONS):
        values = np.asarray([means[b][reason] for b in bounces])
        axis.bar(bounces, values, bottom=bottom, label=reason.replace("_", " "),
                 color=plt.get_cmap("Set2")(index % 8))
        bottom += values
    axis.set_xlabel("Bounce depth")
    axis.set_ylabel("Mean newly terminated paths / iteration")
    axis.set_title(f"Termination reasons — {pu.run_label(run)}")
    axis.set_xticks(bounces)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def work_raw(
    run_dirs: str | Path | pu.RunData | list[str | Path | pu.RunData],
    output: str | Path,
) -> None:
    paths = [run_dirs] if isinstance(run_dirs, (str, Path, pu.RunData)) else run_dirs
    runs = [pu.as_run(path) for path in paths]
    run = runs[0]
    means = _bounce_means(runs)
    bounces = sorted(means)
    plot_style.configure()
    fig, (traversal, lighting) = plt.subplots(1, 2, figsize=(13, 5.5))
    closest_nodes = [means[b]["closest_bvh_node_tests"] for b in bounces]
    closest_tris = [means[b]["closest_bvh_triangle_tests"] for b in bounces]
    shadow_nodes = [means[b]["shadow_bvh_node_tests"] for b in bounces]
    shadow_tris = [means[b]["shadow_bvh_triangle_tests"] for b in bounces]
    traversal.plot(bounces, closest_nodes, marker="o", label="closest node tests")
    traversal.plot(bounces, closest_tris, marker="o", label="closest triangle tests")
    traversal.plot(bounces, shadow_nodes, marker="o", linestyle="--",
                   label="shadow node tests")
    traversal.plot(bounces, shadow_tris, marker="o", linestyle="--",
                   label="shadow triangle tests")
    traversal.set_xlabel("Bounce depth")
    traversal.set_ylabel("Mean tests / rendered iteration")
    traversal.set_title("BVH work pruned per bounce")
    traversal.legend(fontsize=8)

    visible = np.asarray([means[b]["visible_light_samples"] for b in bounces])
    occluded = np.asarray([means[b]["occluded_light_samples"] for b in bounces])
    lighting.bar(bounces, visible, label="visible", color="#54A24B")
    lighting.bar(bounces, occluded, bottom=visible, label="occluded", color="#E45756")
    lighting.plot(bounces, [means[b]["light_selections"] for b in bounces],
                  color="black", marker=".", label="light selections")
    lighting.set_xlabel("Bounce depth")
    lighting.set_ylabel("Mean samples / rendered iteration")
    lighting.set_title("NEE selection and visibility outcomes")
    lighting.legend(fontsize=8)
    fig.suptitle(f"Counter-mode work diagnostics — {pu.run_label(run)}")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("-o", "--output", default="path_survival.png")
    parser.add_argument("--termination-output")
    parser.add_argument("--work-output")
    args = parser.parse_args()
    main_raw(args.runs, args.output)
    if args.termination_output:
        termination_raw(args.runs, args.termination_output)
    if args.work_output:
        work_raw(args.runs, args.work_output)


if __name__ == "__main__":
    main()
