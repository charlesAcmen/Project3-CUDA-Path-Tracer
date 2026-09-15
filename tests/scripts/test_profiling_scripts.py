from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import benchmark_runner
import profiler_utils as pu

try:
    import matplotlib
    matplotlib.use("Agg")
    import analyze_run
    import make_report
    import plot_path_survival
    import plot_quality
    import plot_scaling
    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False


TIMING_FIELDS = [
    "schema_version", "run_id", "epoch", "iteration", "is_warmup", "scope",
    "bounce_depth", "operation", "timer_domain", "time_ms", "work_items",
]
COUNTER_FIELDS = [
    "schema_version", "run_id", "epoch", "iteration", "is_warmup",
    "bounce_depth", "processed_paths", "active_after_bounce",
    "already_terminated", "surface_hits", "misses", "emissive_terminations",
    "invalid_surface_terminations", "russian_roulette_terminations",
    "max_depth_terminations", "debug_terminations", "closest_bvh_node_tests",
    "closest_bvh_triangle_tests", "shadow_bvh_node_tests",
    "shadow_bvh_triangle_tests", "light_selections", "valid_light_samples",
    "shadow_rays", "visible_light_samples", "occluded_light_samples",
]


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _create_run(root: Path, experiment: str, role: str, latency: float,
                triangles: int = 12, repetition: int = 1) -> Path:
    run_id = f"{experiment}_{role}_r{repetition}"
    directory = root / run_id
    directory.mkdir(parents=True)
    metadata = {
        "schema_version": 2,
        "run_id": run_id,
        "profiler": {
            "mode": "throughput" if role == "throughput" else "detail",
            "warmup_iterations": 1,
            "collect_counters": role == "counter",
            "counter_overhead_in_frame_times": role == "counter",
        },
        "scene": {
            "name": "synthetic", "file": "synthetic.json", "width": 10,
            "height": 10, "iterations": 3, "trace_depth": 2,
            "rr_min_bounces": 1, "objects": 1, "meshes": 1,
            "materials": 2, "triangles": triangles, "textures": 0,
            "texture_pixels": 0,
        },
        "initial_runtime": {
            "compact_method": 0 if experiment == "baseline" else 3,
            "compact_method_name": "off" if experiment == "baseline" else "shared_mem",
            "sort_by_material": False, "rng_mode": 0, "rng_mode_name": "lcg",
            "direct_lighting": True,
        },
        "estimated_device_memory_bytes": {
            "path_buffers": 1024, "scene_buffers": 2048, "bvh": 512,
            "light_sampling": 256, "textures": 0, "post_process": 1024,
            "compaction_workspace": 256, "material_sort_workspace": 0,
            "counter_only": 128 if role == "counter" else 0,
            "display_interop_nominal": 800,
            "known_total": 5920 + (128 if role == "counter" else 0),
        },
        "runner": {
            "sequence": 0 if experiment == "baseline" else 1,
            "experiment_id": experiment,
            "label": "Baseline" if experiment == "baseline" else "Optimized",
            "role": role, "repetition": repetition,
            "baseline": experiment == "baseline",
            "compare_to": None if experiment == "baseline" else "baseline",
            "claim_class": "implementation", "headline": True,
        },
    }
    (directory / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    _write_csv(directory / "epochs.csv", [
        "schema_version", "run_id", "epoch", "compact_method",
        "compact_method_name", "sort_by_material", "rng_mode", "rng_mode_name",
        "direct_lighting", "bloom_enabled", "bloom_threshold", "bloom_intensity",
        "bloom_radius", "bloom_sigma", "chromatic_aberration_enabled",
        "chromatic_aberration_intensity", "vignette_enabled", "vignette_intensity",
        "vignette_exponent",
    ], [{
        "schema_version": 2, "run_id": run_id, "epoch": 0,
        "compact_method": 0, "compact_method_name": "off",
        "sort_by_material": 0, "rng_mode": 0, "rng_mode_name": "lcg",
        "direct_lighting": 1, "bloom_enabled": 0, "bloom_threshold": 1,
        "bloom_intensity": .5, "bloom_radius": 10, "bloom_sigma": 5,
        "chromatic_aberration_enabled": 0, "chromatic_aberration_intensity": .003,
        "vignette_enabled": 0, "vignette_intensity": .5, "vignette_exponent": 2,
    }])
    frames = []
    timings = []
    for iteration in (1, 2, 3):
        warmup = int(iteration == 1)
        frames.append({
            "schema_version": 2, "run_id": run_id, "epoch": 0,
            "iteration": iteration, "is_warmup": warmup,
            "end_to_end_ms": latency + iteration * .1,
            "gpu_pipeline_ms": latency - .5 + iteration * .1,
        })
        timings.append({
            "schema_version": 2, "run_id": run_id, "epoch": 0,
            "iteration": iteration, "is_warmup": warmup, "scope": "frame",
            "bounce_depth": "", "operation": "FrameGpu", "timer_domain": "gpu",
            "time_ms": latency - .5, "work_items": 100,
        })
        if role != "throughput":
            for bounce, shade in ((0, 2.0), (1, 1.0)):
                timings.extend([
                    {"schema_version": 2, "run_id": run_id, "epoch": 0,
                     "iteration": iteration, "is_warmup": warmup, "scope": "bounce",
                     "bounce_depth": bounce, "operation": "ComputeIntersections",
                     "timer_domain": "gpu", "time_ms": shade * .5,
                     "work_items": 100 if bounce == 0 else 70},
                    {"schema_version": 2, "run_id": run_id, "epoch": 0,
                     "iteration": iteration, "is_warmup": warmup, "scope": "bounce",
                     "bounce_depth": bounce, "operation": "ShadeMaterial",
                     "timer_domain": "gpu", "time_ms": shade,
                     "work_items": 100 if bounce == 0 else 70},
                ])
    timings.insert(0, {
        "schema_version": 2, "run_id": run_id, "epoch": 0, "iteration": 0,
        "is_warmup": 0, "scope": "setup", "bounce_depth": "",
        "operation": "BuildSceneBvh", "timer_domain": "cpu", "time_ms": 4.0,
        "work_items": triangles,
    })
    _write_csv(directory / "frame_times.csv", [
        "schema_version", "run_id", "epoch", "iteration", "is_warmup",
        "end_to_end_ms", "gpu_pipeline_ms"], frames)
    _write_csv(directory / "timing.csv", TIMING_FIELDS, timings)
    if role == "counter":
        rows = []
        material_rows = []
        for iteration in (1, 2, 3):
            for bounce in (0, 1):
                if bounce == 0:
                    survivors, already, misses, surface_hits = 70, 0, 30, 70
                    light_selections, valid_samples = 50, 30
                    visible_samples, occluded_samples = 20, 10
                    material_counts = ((0, 40), (1, 30))
                else:
                    survivors, already, misses, surface_hits = 40, 30, 30, 40
                    light_selections, valid_samples = 30, 20
                    visible_samples, occluded_samples = 12, 8
                    material_counts = ((0, 25), (1, 15))
                rows.append({
                    "schema_version": 2, "run_id": run_id, "epoch": 0,
                    "iteration": iteration, "is_warmup": int(iteration == 1),
                    "bounce_depth": bounce, "processed_paths": 100,
                    "active_after_bounce": survivors, "already_terminated": already,
                    "surface_hits": surface_hits, "misses": misses,
                    "emissive_terminations": 0, "invalid_surface_terminations": 0,
                    "russian_roulette_terminations": 0, "max_depth_terminations": 0,
                    "debug_terminations": 0, "closest_bvh_node_tests": 500,
                    "closest_bvh_triangle_tests": 120, "shadow_bvh_node_tests": 100,
                    "shadow_bvh_triangle_tests": 20,
                    "light_selections": light_selections,
                    "valid_light_samples": valid_samples,
                    "shadow_rays": valid_samples,
                    "visible_light_samples": visible_samples,
                    "occluded_light_samples": occluded_samples,
                })
                for material_id, hit_count in material_counts:
                    material_rows.append({
                        "schema_version": 2, "run_id": run_id, "epoch": 0,
                        "iteration": iteration, "is_warmup": int(iteration == 1),
                        "bounce_depth": bounce, "material_id": material_id,
                        "hit_count": hit_count,
                    })
        _write_csv(directory / "bounce_counters.csv", COUNTER_FIELDS, rows)
        _write_csv(directory / "material_hits.csv", [
            "schema_version", "run_id", "epoch", "iteration", "is_warmup",
            "bounce_depth", "material_id", "hit_count"], material_rows)
    return directory


class ProfilingScriptsTest(unittest.TestCase):
    def test_warmup_and_per_iteration_sum(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = pu.load_run(_create_run(Path(temp), "baseline", "detail", 10.0))
            self.assertEqual(len(pu.frame_values(run)), 2)
            self.assertAlmostEqual(pu.stage_means(run)["ShadeMaterial"], 3.0)

    def test_multiple_measured_epochs_are_not_merged(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = _create_run(Path(temp), "baseline", "detail", 10.0)
            path = directory / "frame_times.csv"
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = list(reader.fieldnames or [])
                rows = list(reader)
            second_epoch = dict(rows[-1])
            second_epoch["epoch"] = "1"
            rows.append(second_epoch)
            _write_csv(path, fields, rows)
            run = pu.load_run(directory)
            with self.assertRaisesRegex(pu.SchemaError, "spans epochs"):
                pu.frame_values(run)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
    def test_report_renders_all_core_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs = root / "runs"
            for experiment, latency in (("baseline", 10.0), ("optimized", 8.0)):
                for repetition in (1, 2):
                    for role in ("throughput", "detail", "counter"):
                        _create_run(runs, experiment, role, latency,
                                    repetition=repetition)
            report = make_report.build_report(root)
            self.assertTrue(report.is_file())
            self.assertIn("1.242x", report.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(list((root / "plots").glob("*.png"))), 10)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
    def test_missing_deep_counter_bounce_counts_as_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = _create_run(Path(temp), "baseline", "counter", 10.0)
            for filename in ("bounce_counters.csv", "material_hits.csv"):
                path = directory / filename
                with path.open("r", newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    fields = list(reader.fieldnames or [])
                    rows = [row for row in reader if not (
                        row["iteration"] == "3" and row["bounce_depth"] == "1")]
                _write_csv(path, fields, rows)
            means = plot_path_survival._bounce_means(pu.load_run(directory))
            self.assertAlmostEqual(means[1]["processed_paths"], 50.0)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
    def test_scaling_and_quality(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = _create_run(root, "baseline", "throughput", 10.0, triangles=10)
            second = _create_run(root, "optimized", "throughput", 12.0, triangles=100)
            plot_scaling.main_raw([first, second], root / "scaling.png", "triangles")
            self.assertTrue((root / "scaling.png").is_file())
            import numpy as np
            reference = np.zeros((2, 2, 3), dtype=float)
            candidate = np.full((2, 2, 3), .25, dtype=float)
            rmse, psnr = plot_quality.quality_metrics(reference, candidate)
            self.assertAlmostEqual(rmse, .25)
            self.assertAlmostEqual(psnr, 12.041199826559248)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
    def test_single_run_entrypoint_selects_role_safe_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = {
                "throughput": {"frame_time.png", "memory.png"},
                "detail": {
                    "stage_breakdown.png", "bounce_breakdown.png",
                    "setup_breakdown.png", "memory.png",
                },
                "counter": {
                    "path_survival.png", "termination_reasons.png",
                    "bvh_nee_work.png", "material_mix.png", "memory.png",
                },
            }
            for role, filenames in expected.items():
                run = _create_run(root, "baseline", role, 10.0)
                report, figures = analyze_run.run_analysis(run)
                self.assertTrue(report.is_file())
                self.assertEqual({path.name for path in figures}, filenames)
                text = report.read_text(encoding="utf-8")
                self.assertIn(f"Evidence role: `{role}`", text)
                self.assertEqual(
                    {path.name for path in (run / "analysis").glob("*.png")},
                    filenames)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
    def test_single_run_entrypoint_splits_multiple_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = _create_run(Path(temp), "baseline", "detail", 10.0)
            for filename in ("frame_times.csv", "timing.csv"):
                path = run / filename
                with path.open("r", newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    fields = list(reader.fieldnames or [])
                    rows = list(reader)
                copied = []
                for row in rows:
                    if row["is_warmup"] != "0":
                        continue
                    if filename == "timing.csv" and row["scope"] == "setup":
                        continue
                    duplicate = dict(row)
                    duplicate["epoch"] = "2"
                    copied.append(duplicate)
                _write_csv(path, fields, [*rows, *copied])

            report, figures = analyze_run.run_analysis(run)
            self.assertEqual(len(figures), 8)
            self.assertTrue((run / "analysis" / "epoch-0" / "analysis.md").is_file())
            self.assertTrue((run / "analysis" / "epoch-2" / "analysis.md").is_file())
            self.assertIn(
                "never averaged together", report.read_text(encoding="utf-8"))

    def test_benchmark_spec(self) -> None:
        runner, experiments = benchmark_runner.load_spec(
            SCRIPTS / "experiments" / "project3.toml")
        self.assertEqual(len(experiments), 8)
        self.assertEqual(len(benchmark_runner.make_jobs(experiments, 3, 5650)), 72)
        self.assertEqual(runner["warmup"], 16)


if __name__ == "__main__":
    unittest.main()
