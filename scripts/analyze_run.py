"""Generate every valid figure and a short report for one profiler run.

Usage from the repository root:

    python scripts/analyze_run.py <run-directory-or-run-name>

A bare run name is resolved below ``profiler_output/``. The profiler role is
read from run.json, so users do not need to choose individual plot scripts.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import plot_bounce_breakdown
    import plot_frame_time
    import plot_material_mix
    import plot_memory
    import plot_path_survival
    import plot_setup_breakdown
    import plot_stage_breakdown
    import profiler_utils as pu
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing analysis dependency. Run once: "
        "python -m pip install -r scripts/requirements.txt"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_run_directory(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate.resolve()
    fallback = REPO_ROOT / "profiler_output" / candidate
    if fallback.exists():
        return fallback.resolve()
    raise pu.SchemaError(
        f"profiler run not found: {value!s}; also checked {fallback}")


def _add_figure(
    generated: list[tuple[str, Path, str]],
    title: str,
    path: Path,
    explanation: str,
) -> None:
    generated.append((title, path, explanation))


def _write_report(
    run: pu.RunData,
    role: str,
    output_dir: Path,
    generated: list[tuple[str, Path, str]],
) -> Path:
    scene = run.metadata.get("scene", {})
    epoch = pu.selected_epoch(run)
    frames = pu.frame_values(run)
    end_to_end_mean = statistics.fmean(frames)
    gpu_frames = pu.frame_values(run, "gpu_pipeline_ms")
    gpu_mean = statistics.fmean(gpu_frames)
    lines = [
        "# Single-run profiling analysis",
        "",
        f"- Run: `{run.directory.name}`",
        f"- Scene: `{scene.get('name', 'unknown')}`",
        f"- Evidence role: `{role}`",
        f"- Accumulation epoch: {epoch}",
        f"- Measured iterations after warmup: {len(frames)}",
        f"- Mean end-to-end time: {end_to_end_mean:.3f} ms",
        f"- Mean enclosed GPU pipeline: {gpu_mean:.3f} ms",
        f"- Approximate iteration rate: {1000.0 / end_to_end_mean:.2f} iter/s",
        "",
    ]

    if role == "detail":
        stages = pu.stage_means(run)
        lines.extend(["## Largest measured GPU stages", ""])
        for operation, milliseconds in sorted(
                stages.items(), key=lambda item: item[1], reverse=True)[:8]:
            lines.append(f"- `{operation}`: {milliseconds:.3f} ms/iteration")
        lines.append("")

    memory = pu.dotted_get(
        run.metadata, "estimated_device_memory_bytes.known_total", 0)
    if memory:
        lines.extend([
            "## Known device memory",
            "",
            f"Renderer-owned estimate: {float(memory) / (1024.0 ** 2):.2f} MiB. "
            "CUDA, Thrust, OpenGL, and driver-internal allocations are excluded.",
            "",
        ])

    lines.extend(["## Figures", ""])
    for title, path, explanation in generated:
        lines.extend([
            f"### {title}",
            "",
            explanation,
            "",
            f"![{title}]({path.relative_to(output_dir).as_posix()})",
            "",
        ])

    lines.extend(["## Interpretation boundary", ""])
    if role == "throughput":
        lines.extend([
            "This run is suitable for latency and throughput measurement. A single "
            "process is still exploratory evidence; final speedup claims should use "
            "multiple independent baseline and variant runs.",
            "",
        ])
    elif role == "detail":
        lines.extend([
            "Detailed CUDA events explain where GPU time is spent. Their instrumentation "
            "overhead means this run must not be used as the headline FPS or speedup result.",
            "",
        ])
    else:
        lines.extend([
            "Counter atomics, per-bounce clears, and device-to-host readbacks deliberately "
            "perturb timing. Use these figures to explain work and causality, never FPS or "
            "speedup.",
            "",
        ])

    report = output_dir / "analysis.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _analyze_selected_run(
    run: pu.RunData,
    output_dir: Path,
    role: str,
) -> tuple[Path, list[Path]]:
    run_dir = run.directory
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[tuple[str, Path, str]] = []

    if role == "throughput":
        path = output_dir / "frame_time.png"
        plot_frame_time.main_raw([run], path)
        _add_figure(
            generated, "Frame latency and throughput", path,
            "End-to-end latency includes CUDA-GL map, required completion, and unmap. "
            "The GPU span encloses the CUDA rendering pipeline.")
    elif role == "detail":
        if any(row["scope"] in {"frame", "bounce"}
               and row["operation"] != "FrameGpu" for row in run.timings):
            path = output_dir / "stage_breakdown.png"
            plot_stage_breakdown.main_raw([run], path)
            _add_figure(
                generated, "GPU stage contribution", path,
                "All calls are summed inside each iteration before cross-iteration "
                "averaging; the grey residual is uninstrumented GPU span.")
        if any(row["scope"] == "bounce" for row in run.timings):
            path = output_dir / "bounce_breakdown.png"
            plot_bounce_breakdown.main_raw(run, path)
            _add_figure(
                generated, "Per-bounce GPU contribution", path,
                "Missing deep bounces count as zero, so bars represent contribution to "
                "an average rendered iteration rather than conditional call latency.")
        if any(row["scope"] == "setup" for row in run.timings):
            path = output_dir / "setup_breakdown.png"
            plot_setup_breakdown.main_raw([run], path)
            _add_figure(
                generated, "One-time setup cost", path,
                "Separates BVH construction, scene and texture upload, light-table build, "
                "and allocation from steady-state rendering.")
    else:
        if not run.counters:
            raise pu.SchemaError(
                "run.json enables counters but bounce_counters.csv is missing or empty")
        path = output_dir / "path_survival.png"
        plot_path_survival.main_raw([run], path)
        _add_figure(
            generated, "Active paths and processed slots", path,
            "Shows true surviving paths alongside the number of slots processed by the "
            "selected compaction method.")
        path = output_dir / "termination_reasons.png"
        plot_path_survival.termination_raw(run, path)
        _add_figure(
            generated, "Termination reasons", path,
            "Exclusive path outcomes identify misses, emitter hits, Russian roulette, "
            "maximum depth, invalid surfaces, and debug termination.")
        path = output_dir / "bvh_nee_work.png"
        plot_path_survival.work_raw(run, path)
        _add_figure(
            generated, "BVH and direct-lighting work", path,
            "Counts actual closest-hit and shadow traversal tests together with NEE "
            "selection and visibility outcomes.")
        if run.material_hits:
            path = output_dir / "material_mix.png"
            plot_material_mix.main_raw(run, path)
            _add_figure(
                generated, "Material hit mix", path,
                "Material shares and effective material diversity indicate when sorting "
                "may reduce divergent shading work.")

    if pu.dotted_get(run.metadata, "estimated_device_memory_bytes.known_total", 0):
        path = output_dir / "memory.png"
        plot_memory.main_raw([run], path)
        _add_figure(
            generated, "Known device-memory footprint", path,
            "Reports renderer-owned allocations; the display-interoperability component "
            "is nominal and driver-internal memory is excluded.")

    if not generated:
        raise pu.SchemaError("the run contains no data supported by the analysis entrypoint")
    report = _write_report(run, role, output_dir, generated)
    return report, [path for _, path, _ in generated]


def run_analysis(
    run_value: str | Path,
    output: str | Path | None = None,
    epoch: int | None = None,
) -> tuple[Path, list[Path]]:
    run_dir = resolve_run_directory(run_value)
    run = pu.load_run(run_dir)
    mode = str(pu.dotted_get(run.metadata, "profiler.mode", ""))
    counters_enabled = bool(pu.dotted_get(
        run.metadata, "profiler.collect_counters", False))
    if mode not in {"throughput", "detail"}:
        raise pu.SchemaError(f"unsupported profiler mode in run.json: {mode!r}")
    role = "counter" if counters_enabled else mode
    output_dir = Path(output).resolve() if output else run_dir / "analysis"

    available_epochs = sorted({
        row["epoch"] for row in pu.measured(run.frames)
    })
    if not available_epochs:
        raise pu.SchemaError("no measured frames remain after warmup")
    if epoch is not None:
        selected_epochs = [pu.selected_epoch(run, epoch)]
    else:
        selected_epochs = available_epochs

    if len(selected_epochs) == 1:
        selected = pu.select_epoch_data(run, selected_epochs[0])
        return _analyze_selected_run(selected, output_dir, role)

    all_figures: list[Path] = []
    epoch_reports: list[tuple[int, Path, int]] = []
    for selected_epoch in selected_epochs:
        selected = pu.select_epoch_data(run, selected_epoch)
        epoch_dir = output_dir / f"epoch-{selected_epoch}"
        report, figures = _analyze_selected_run(selected, epoch_dir, role)
        all_figures.extend(figures)
        epoch_reports.append((selected_epoch, report, len(pu.frame_values(selected))))

    lines = [
        "# Multi-epoch profiling analysis",
        "",
        f"Run `{run.directory.name}` contains multiple measured accumulation epochs. "
        "They were analyzed separately and were never averaged together.",
        "",
        "## Epoch reports",
        "",
    ]
    for selected_epoch, report, frame_count in epoch_reports:
        lines.append(
            f"- [Epoch {selected_epoch}]({report.relative_to(output_dir).as_posix()}): "
            f"{frame_count} measured iterations")
    lines.extend([
        "",
        "An epoch starts whenever camera or live renderer settings reset accumulation. "
        "Use `--epoch N` to regenerate only one epoch.",
        "",
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    index = output_dir / "analysis.md"
    index.write_text("\n".join(lines), encoding="utf-8")
    return index, all_figures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all valid plots for one schema-v2 profiler run.")
    parser.add_argument(
        "run", help="run directory, or its name below profiler_output/")
    parser.add_argument(
        "-o", "--output", help="output directory; defaults to RUN/analysis")
    parser.add_argument(
        "--epoch", type=int,
        help="analyze one accumulation epoch; default analyzes all separately")
    args = parser.parse_args()
    try:
        report, figures = run_analysis(args.run, args.output, args.epoch)
    except (OSError, ValueError, pu.SchemaError) as exc:
        raise SystemExit(f"Analysis failed: {exc}") from exc
    print(f"Generated {len(figures)} figure(s)")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
