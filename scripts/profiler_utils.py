"""Shared utilities for reading profiler CSV output."""
import csv
from pathlib import Path


def parse_timing_csv(filepath: str) -> list[dict]:
    """Read timing CSV, return list of row dicts with typed values."""
    rows = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "iteration": int(r["iteration"]),
                "bounce_depth": int(r["bounce_depth"]),
                "operation": r["operation"],
                "time_ms": float(r["time_ms"]),
                "num_active_paths": int(r["num_active_paths"]),
                "compact_method": int(r["compact_method"]),
                "sort_by_material": int(r["sort_by_material"]),
            })
    return rows


def parse_path_survival_csv(filepath: str) -> list[dict]:
    """Read path survival CSV, return list of row dicts."""
    rows = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "iteration": int(r["iteration"]),
                "bounce_depth": int(r["bounce_depth"]),
                "num_active_paths": int(r["num_active_paths"]),
                "compact_method": int(r["compact_method"]),
                "sort_by_material": int(r["sort_by_material"]),
            })
    return rows


def parse_summary_csv(filepath: str) -> list[dict]:
    """Read summary CSV, return list of row dicts."""
    rows = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "operation": r["operation"],
                "mean_ms": float(r["mean_ms"]),
                "std_ms": float(r["std_ms"]),
                "min_ms": float(r["min_ms"]),
                "max_ms": float(r["max_ms"]),
                "num_samples": int(r["num_samples"]),
            })
    return rows


def scalar_to_label(compact_method: int, sort_by_material: int) -> str:
    """Human-readable label from the CSV metadata columns."""
    parts = []
    if compact_method == 0:
        parts.append("no-compact")
    elif compact_method == 1:
        parts.append("custom-compact")
    else:
        parts.append("thrust-compact")

    parts.append("sorted" if sort_by_material else "unsorted")
    return ", ".join(parts)
