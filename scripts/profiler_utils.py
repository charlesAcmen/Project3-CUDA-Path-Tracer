"""Schema-v2 loading, validation, aggregation, and statistics helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCHEMA_VERSION = 2


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class RunData:
    directory: Path
    metadata: dict[str, Any]
    epochs: list[dict[str, Any]]
    timings: list[dict[str, Any]]
    frames: list[dict[str, Any]]
    counters: list[dict[str, Any]]
    material_hits: list[dict[str, Any]]


def as_run(value: str | Path | RunData) -> RunData:
    return value if isinstance(value, RunData) else load_run(value)


def _require_columns(path: Path, fieldnames: Sequence[str] | None,
                     required: set[str]) -> None:
    actual = set(fieldnames or [])
    missing = sorted(required - actual)
    if missing:
        raise SchemaError(f"{path}: missing columns: {', '.join(missing)}")


def _read_csv(path: Path, required: set[str], *, optional: bool = False) -> list[dict[str, str]]:
    if not path.exists():
        if optional:
            return []
        raise SchemaError(f"missing required profiler artifact: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(path, reader.fieldnames, required)
        rows = list(reader)
    for row_number, row in enumerate(rows, start=2):
        if int(row.get("schema_version", -1)) != SCHEMA_VERSION:
            raise SchemaError(
                f"{path}:{row_number}: expected schema {SCHEMA_VERSION}, "
                f"got {row.get('schema_version')!r}")
    return rows


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    return default if value == "" else int(value)


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return default if value == "" else float(value)


def load_run(path: str | Path) -> RunData:
    directory = Path(path).resolve()
    metadata_path = directory / "run.json"
    if not metadata_path.exists():
        raise SchemaError(f"{directory}: run.json not found")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(
            f"{metadata_path}: expected schema {SCHEMA_VERSION}, "
            f"got {metadata.get('schema_version')!r}")

    epoch_rows = _read_csv(
        directory / "epochs.csv",
        {"schema_version", "run_id", "epoch", "compact_method",
         "compact_method_name", "sort_by_material", "rng_mode",
         "rng_mode_name", "direct_lighting"})
    epochs = [{**row, "epoch": _int(row, "epoch")} for row in epoch_rows]

    timing_rows = _read_csv(
        directory / "timing.csv",
        {"schema_version", "run_id", "epoch", "iteration", "is_warmup",
         "scope", "bounce_depth", "operation", "timer_domain", "time_ms",
         "work_items"}, optional=True)
    timings = [{
        **row,
        "epoch": _int(row, "epoch"),
        "iteration": _int(row, "iteration"),
        "is_warmup": bool(_int(row, "is_warmup")),
        "bounce_depth": None if row.get("bounce_depth", "") == ""
                         else _int(row, "bounce_depth"),
        "time_ms": _float(row, "time_ms"),
        "work_items": _int(row, "work_items"),
    } for row in timing_rows]

    frame_rows = _read_csv(
        directory / "frame_times.csv",
        {"schema_version", "run_id", "epoch", "iteration", "is_warmup",
         "end_to_end_ms", "gpu_pipeline_ms"}, optional=True)
    frames = [{
        **row,
        "epoch": _int(row, "epoch"),
        "iteration": _int(row, "iteration"),
        "is_warmup": bool(_int(row, "is_warmup")),
        "end_to_end_ms": _float(row, "end_to_end_ms"),
        "gpu_pipeline_ms": _float(row, "gpu_pipeline_ms"),
    } for row in frame_rows]

    counter_rows = _read_csv(
        directory / "bounce_counters.csv",
        {"schema_version", "run_id", "epoch", "iteration", "is_warmup",
         "bounce_depth", "processed_paths", "active_after_bounce",
         "already_terminated", "surface_hits", "misses",
         "emissive_terminations", "invalid_surface_terminations",
         "russian_roulette_terminations", "max_depth_terminations",
         "debug_terminations", "closest_bvh_node_tests",
         "closest_bvh_triangle_tests", "shadow_bvh_node_tests",
         "shadow_bvh_triangle_tests", "light_selections",
         "valid_light_samples", "shadow_rays", "visible_light_samples",
         "occluded_light_samples"}, optional=True)
    counter_int_fields = {
        "epoch", "iteration", "bounce_depth", "processed_paths",
        "active_after_bounce", "already_terminated", "surface_hits", "misses",
        "emissive_terminations", "invalid_surface_terminations",
        "russian_roulette_terminations", "max_depth_terminations",
        "debug_terminations", "closest_bvh_node_tests",
        "closest_bvh_triangle_tests", "shadow_bvh_node_tests",
        "shadow_bvh_triangle_tests", "light_selections",
        "valid_light_samples", "shadow_rays", "visible_light_samples",
        "occluded_light_samples",
    }
    counters = []
    for row in counter_rows:
        typed: dict[str, Any] = dict(row)
        for field in counter_int_fields:
            typed[field] = _int(row, field)
        typed["is_warmup"] = bool(_int(row, "is_warmup"))
        exclusive_total = (
            typed["active_after_bounce"] + typed["already_terminated"] +
            typed["misses"] + typed["emissive_terminations"] +
            typed["invalid_surface_terminations"] +
            typed["russian_roulette_terminations"] +
            typed["max_depth_terminations"] + typed["debug_terminations"])
        if exclusive_total != typed["processed_paths"]:
            raise SchemaError(
                f"{directory / 'bounce_counters.csv'}: outcome total "
                f"{exclusive_total} != processed_paths {typed['processed_paths']} "
                f"at epoch={typed['epoch']} iteration={typed['iteration']} "
                f"bounce={typed['bounce_depth']}")
        classified_hits = (
            typed["surface_hits"] + typed["misses"] +
            typed["already_terminated"])
        if classified_hits != typed["processed_paths"]:
            raise SchemaError(
                f"{directory / 'bounce_counters.csv'}: hit classification total "
                f"{classified_hits} != processed_paths {typed['processed_paths']} "
                f"at epoch={typed['epoch']} iteration={typed['iteration']} "
                f"bounce={typed['bounce_depth']}")
        if (typed["visible_light_samples"] + typed["occluded_light_samples"]
                != typed["shadow_rays"]):
            raise SchemaError(
                f"{directory / 'bounce_counters.csv'}: visibility total "
                f"does not equal shadow_rays at epoch={typed['epoch']} "
                f"iteration={typed['iteration']} bounce={typed['bounce_depth']}")
        if not (typed["shadow_rays"] <= typed["valid_light_samples"]
                <= typed["light_selections"] <= typed["surface_hits"]):
            raise SchemaError(
                f"{directory / 'bounce_counters.csv'}: expected shadow_rays <= "
                "valid_light_samples <= light_selections <= surface_hits at "
                f"epoch={typed['epoch']} iteration={typed['iteration']} "
                f"bounce={typed['bounce_depth']}")
        counters.append(typed)

    material_rows = _read_csv(
        directory / "material_hits.csv",
        {"schema_version", "run_id", "epoch", "iteration", "is_warmup",
         "bounce_depth", "material_id", "hit_count"}, optional=True)
    material_hits = [{
        **row,
        "epoch": _int(row, "epoch"),
        "iteration": _int(row, "iteration"),
        "is_warmup": bool(_int(row, "is_warmup")),
        "bounce_depth": _int(row, "bounce_depth"),
        "material_id": _int(row, "material_id"),
        "hit_count": _int(row, "hit_count"),
    } for row in material_rows]

    if material_hits and counters:
        hit_totals: dict[tuple[int, int, int], int] = {}
        for row in material_hits:
            key = (row["epoch"], row["iteration"], row["bounce_depth"])
            hit_totals[key] = hit_totals.get(key, 0) + row["hit_count"]
        for row in counters:
            key = (row["epoch"], row["iteration"], row["bounce_depth"])
            if hit_totals.get(key, 0) != row["surface_hits"]:
                raise SchemaError(
                    f"{directory / 'material_hits.csv'}: material hit total "
                    f"{hit_totals.get(key, 0)} != surface_hits {row['surface_hits']} "
                    f"at epoch={key[0]} iteration={key[1]} bounce={key[2]}")

    return RunData(directory, metadata, epochs, timings, frames, counters,
                   material_hits)


def measured(rows: Iterable[dict[str, Any]], *, epoch: int | None = None) -> list[dict[str, Any]]:
    return [row for row in rows
            if not row.get("is_warmup", False)
            and (epoch is None or row.get("epoch") == epoch)]


def selected_epoch(run: RunData, epoch: int | None = None) -> int:
    available = sorted({row["epoch"] for row in measured(run.frames)})
    if epoch is not None:
        if epoch not in available:
            raise SchemaError(
                f"{run.directory}: epoch {epoch} has no measured frames; "
                f"available={available}")
        return epoch
    if not available:
        raise SchemaError(f"{run.directory}: no measured frames remain after warmup")
    if len(available) != 1:
        raise SchemaError(
            f"{run.directory}: measured data spans epochs {available}; "
            "select exactly one epoch before aggregation")
    return available[0]


def select_epoch_data(run: RunData, epoch: int) -> RunData:
    """Return a non-mutating one-epoch view suitable for plot composition.

    Setup records are process-wide and remain available in every view. All
    steady-state timing and counter rows are restricted to the chosen epoch.
    """
    chosen = selected_epoch(run, epoch)
    return RunData(
        directory=run.directory,
        metadata=run.metadata,
        epochs=[row for row in run.epochs if row["epoch"] == chosen],
        timings=[row for row in run.timings
                 if row["scope"] == "setup" or row["epoch"] == chosen],
        frames=[row for row in run.frames if row["epoch"] == chosen],
        counters=[row for row in run.counters if row["epoch"] == chosen],
        material_hits=[row for row in run.material_hits if row["epoch"] == chosen],
    )


def iteration_keys(run: RunData, *, epoch: int | None = None) -> list[tuple[int, int]]:
    epoch = selected_epoch(run, epoch)
    return sorted({(row["epoch"], row["iteration"])
                   for row in measured(run.frames, epoch=epoch)})


def aggregate_stages_per_iteration(
    run: RunData,
    *,
    epoch: int | None = None,
    scopes: set[str] | None = None,
    timer_domains: set[str] | None = None,
    exclude_operations: set[str] | None = None,
) -> dict[tuple[int, int], dict[str, float]]:
    """Sum every invocation of an operation within each measured iteration."""
    epoch = selected_epoch(run, epoch)
    scopes = scopes or {"frame", "bounce"}
    timer_domains = timer_domains or {"gpu"}
    excluded = exclude_operations or {"FrameGpu"}
    keys = iteration_keys(run, epoch=epoch)
    result = {key: {} for key in keys}
    for row in measured(run.timings, epoch=epoch):
        if (row["scope"] not in scopes or row["timer_domain"] not in timer_domains
                or row["operation"] in excluded):
            continue
        key = (row["epoch"], row["iteration"])
        if key not in result:
            result[key] = {}
        op = row["operation"]
        result[key][op] = result[key].get(op, 0.0) + row["time_ms"]
    return result


def stage_means(run: RunData, *, epoch: int | None = None) -> dict[str, float]:
    per_iteration = aggregate_stages_per_iteration(run, epoch=epoch)
    if not per_iteration:
        return {}
    operations = sorted({op for values in per_iteration.values() for op in values})
    return {
        op: statistics.fmean(values.get(op, 0.0) for values in per_iteration.values())
        for op in operations
    }


def bounce_stage_means(run: RunData, *, epoch: int | None = None) -> dict[int, dict[str, float]]:
    """Mean contribution per rendered iteration; missing deep bounces count as zero."""
    epoch = selected_epoch(run, epoch)
    keys = iteration_keys(run, epoch=epoch)
    if not keys:
        return {}
    values: dict[tuple[int, int], dict[int, dict[str, float]]] = {
        key: {} for key in keys
    }
    for row in measured(run.timings, epoch=epoch):
        if (row["scope"] != "bounce" or row["timer_domain"] != "gpu"
                or row["operation"] == "FrameGpu"):
            continue
        key = (row["epoch"], row["iteration"])
        bounce = int(row["bounce_depth"])
        stage = values.setdefault(key, {}).setdefault(bounce, {})
        stage[row["operation"]] = stage.get(row["operation"], 0.0) + row["time_ms"]
    bounces = sorted({bounce for frame in values.values() for bounce in frame})
    operations = sorted({op for frame in values.values() for bounce in frame.values() for op in bounce})
    return {
        bounce: {
            op: statistics.fmean(
                frame.get(bounce, {}).get(op, 0.0) for frame in values.values())
            for op in operations
        }
        for bounce in bounces
    }


def frame_values(run: RunData, field: str = "end_to_end_ms",
                 *, epoch: int | None = None) -> list[float]:
    if field not in {"end_to_end_ms", "gpu_pipeline_ms"}:
        raise ValueError(f"unsupported frame field: {field}")
    epoch = selected_epoch(run, epoch)
    return [float(row[field]) for row in measured(run.frames, epoch=epoch)]


def mean_ci95(values: Sequence[float], *, seed: int = 5650,
              samples: int = 10000) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    boot_means = array[indices].mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return mean, float(low), float(high)


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def dotted_get(data: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = data
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            return default
        value = value[component]
    return value


def run_label(run: RunData) -> str:
    label = dotted_get(run.metadata, "runner.label")
    if label:
        return str(label)
    runtime = run.metadata.get("initial_runtime", {})
    compact = runtime.get("compact_method_name", "?")
    sort = "sort" if runtime.get("sort_by_material") else "no-sort"
    rng = runtime.get("rng_mode_name", "?")
    nee = "NEE" if runtime.get("direct_lighting") else "no-NEE"
    return f"{compact}, {sort}, {rng}, {nee}"


def discover_runs(root: str | Path) -> list[Path]:
    path = Path(root)
    if (path / "run.json").exists():
        return [path.resolve()]
    return sorted(item.parent.resolve() for item in path.rglob("run.json"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_runner_metadata(run_dir: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(run_dir) / "run.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["runner"] = metadata
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
