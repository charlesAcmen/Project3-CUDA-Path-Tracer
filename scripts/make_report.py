"""Generate the canonical Markdown summary and schema-v2 plot suite."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import plot_bounce_breakdown
import plot_frame_time
import plot_material_mix
import plot_memory
import plot_optimization_history
import plot_path_survival
import plot_setup_breakdown
import plot_stage_breakdown
import plot_stage_delta
import profiler_utils as pu


def _groups(runs: list[pu.RunData], role: str) -> dict[str, list[pu.RunData]]:
    result: dict[str, list[pu.RunData]] = defaultdict(list)
    for run in runs:
        if pu.dotted_get(run.metadata, "runner.role") == role:
            experiment = str(pu.dotted_get(
                run.metadata, "runner.experiment_id", pu.run_label(run)))
            result[experiment].append(run)
    return dict(sorted(
        result.items(),
        key=lambda item: (
            not bool(pu.dotted_get(item[1][0].metadata, "runner.baseline", False)),
            int(pu.dotted_get(item[1][0].metadata, "runner.sequence", 1_000_000)),
            item[0],
        )))


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def build_report(run_root: str | Path) -> Path:
    root = Path(run_root).resolve()
    runs = [pu.load_run(path) for path in pu.discover_runs(root / "runs")]
    if not runs:
        raise pu.SchemaError(f"{root}: no schema-v2 runs found")
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    generated: list[tuple[str, Path]] = []

    throughput = _groups(runs, "throughput")
    detail = _groups(runs, "detail")
    counter = _groups(runs, "counter")
    if throughput:
        path = plots / "frame_time.png"
        plot_frame_time.main_raw(
            [run.directory for group in throughput.values() for run in group], path)
        generated.append(("End-to-end latency and throughput", path))
        all_throughput = [run for group in throughput.values() for run in group]
        if any(pu.dotted_get(
                run.metadata, "estimated_device_memory_bytes.known_total", 0)
               for run in all_throughput):
            memory = plots / "memory.png"
            plot_memory.main_raw([run.directory for run in all_throughput], memory)
            generated.append(("Known device-memory footprint", memory))
    if len(throughput) >= 2:
        path = plots / "optimization_history.png"
        plot_optimization_history.main_raw(
            [run.directory for group in throughput.values() for run in group], path)
        generated.append(("Optimization speedup vs baseline", path))
    if detail:
        all_detail = [run.directory for group in detail.values() for run in group]
        path = plots / "stage_breakdown.png"
        plot_stage_breakdown.main_raw(all_detail, path)
        generated.append(("Per-iteration GPU stage contribution", path))
        setup = plots / "setup_breakdown.png"
        plot_setup_breakdown.main_raw(all_detail, setup)
        generated.append(("One-time setup cost", setup))
        comparisons: dict[str, list[list[pu.RunData]]] = defaultdict(list)
        for experiment, group in detail.items():
            compare_to = pu.dotted_get(group[0].metadata, "runner.compare_to")
            if compare_to in detail:
                comparisons[str(compare_to)].append(group)
        for reference, variants in comparisons.items():
            delta = plots / f"stage_delta_vs_{reference}.png"
            plot_stage_delta.main_grouped(
                [run.directory for run in detail[reference]],
                [[run.directory for run in group] for group in variants], delta)
            generated.append((f"Stage deltas vs {pu.run_label(detail[reference][0])}",
                              delta))
        for experiment, group in detail.items():
            bounce = plots / f"bounce_{experiment}.png"
            plot_bounce_breakdown.main_raw([run.directory for run in group], bounce)
            generated.append((f"Bounce contribution: {pu.run_label(group[0])}", bounce))
    if counter:
        survival = plots / "path_survival.png"
        plot_path_survival.main_raw(
            [run.directory for group in counter.values() for run in group], survival)
        generated.append(("True active paths vs processed slots", survival))
        for experiment, group in counter.items():
            directories = [run.directory for run in group]
            term = plots / f"termination_{experiment}.png"
            plot_path_survival.termination_raw(directories, term)
            generated.append((f"Termination reasons: {pu.run_label(group[0])}", term))
            work = plots / f"work_{experiment}.png"
            plot_path_survival.work_raw(directories, work)
            generated.append((f"BVH and NEE work: {pu.run_label(group[0])}", work))
            if any(run.material_hits for run in group):
                material = plots / f"material_mix_{experiment}.png"
                plot_material_mix.main_raw(directories, material)
                generated.append((f"Material mix: {pu.run_label(group[0])}", material))

    rows = []
    experiment_means = {
        experiment: float(np.mean([
            np.mean(pu.frame_values(run)) for run in group]))
        for experiment, group in throughput.items()
    }
    for experiment, group in throughput.items():
        replicate_means = [float(np.mean(pu.frame_values(run))) for run in group]
        latency, low, high = pu.mean_ci95(replicate_means)
        all_frames = [value for run in group for value in pu.frame_values(run)]
        p95 = pu.percentile(all_frames, 95)
        compare_to = pu.dotted_get(group[0].metadata, "runner.compare_to")
        reference_mean = experiment_means.get(str(compare_to)) if compare_to else latency
        speedup = reference_mean / latency if reference_mean and latency > 0 else float("nan")
        claim_class = str(pu.dotted_get(
            group[0].metadata, "runner.claim_class", "implementation"))
        rows.append((pu.run_label(group[0]), claim_class,
                     str(compare_to) if compare_to else "self", len(group), latency,
                     low, high, p95, 1000.0 / latency, speedup))

    lines = [
        "# Project 3 profiling report", "",
        "The headline timing table uses throughput-mode runs only. Detailed-event "
        "and counter-mode runs are diagnostic evidence and are intentionally excluded "
        "from speedup/FPS claims.", "",
        "## Throughput summary", "",
        "| Experiment | Claim class | Compared with | Repeats | Mean ms | 95% CI | p95 ms | iter/s | Speedup |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, claim_class, compare_to, count, mean, low, high, p95, rate, speedup in rows:
        speedup_text = f"{speedup:.3f}x" if np.isfinite(speedup) else "n/a"
        lines.append(f"| {label} | {claim_class} | {compare_to} | {count} | "
                     f"{mean:.3f} | [{low:.3f}, {high:.3f}] "
                     f"| {p95:.3f} | {rate:.2f} | {speedup_text} |")
    lines.extend(["", "## Figures", ""])
    for title, path in generated:
        lines.extend([f"### {title}", "", f"![{title}]({_relative(path, root)})", ""])
    lines.extend([
        "## Interpretation boundary", "",
        "- End-to-end time covers CUDA-GL map, the CUDA pipeline, the required device "
        "synchronization, and unmap. PNG encoding/checkpoint saves are outside the sample.",
        "- Every bounce-stage bar first sums all invocations in one iteration; it never "
        "uses an unweighted mean of bounce calls.",
        "- Counter plots show causal work (active paths, termination, BVH tests, shadow "
        "visibility) but counter-run timings are not performance results.",
        "- Image-quality claims require plot_quality.py with a same-camera, same-pipeline "
        "high-SPP reference. Nsight evidence is generated separately with plot_nsight.py.",
        "",
    ])
    report = root / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", help="benchmark_* directory containing runs/")
    args = parser.parse_args()
    print(build_report(args.run_root))


if __name__ == "__main__":
    main()
