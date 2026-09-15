"""Top CUDA kernels from an exported Nsight Systems stats CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_style


def _normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _column(fieldnames: list[str], candidates: set[str]) -> str:
    for field in fieldnames:
        if _normalized(field) in candidates:
            return field
    raise ValueError(f"missing one of columns {sorted(candidates)}; got {fieldnames}")


def load_kernel_totals(path: str | Path) -> list[tuple[str, float]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        name_col = _column(fields, {"name", "kernelname"})
        total_col = _column(fields, {"totaltime", "totaltimens", "totaltimems",
                                     "totaltimeus"})
        normalized = _normalized(total_col)
        scale = 1.0
        if normalized.endswith("ns"):
            scale = 1e-6
        elif normalized.endswith("us"):
            scale = 1e-3
        rows = []
        for row in reader:
            raw = row[total_col].replace(",", "").strip()
            if raw:
                rows.append((row[name_col], float(raw) * scale))
    return sorted(rows, key=lambda item: item[1], reverse=True)


def main_raw(csv_path: str | Path, output: str | Path, top: int = 15) -> None:
    values = load_kernel_totals(csv_path)[:top]
    if not values:
        raise ValueError("Nsight CSV contains no kernel rows")
    plot_style.configure()
    fig, axis = plt.subplots(figsize=(11, max(5, len(values) * 0.42)))
    names = [item[0] for item in reversed(values)]
    times = np.asarray([item[1] for item in reversed(values)])
    axis.barh(np.arange(len(names)), times, color="#4C78A8")
    axis.set_yticks(np.arange(len(names)))
    axis.set_yticklabels(names, fontsize=8)
    axis.set_xlabel("Total GPU time (ms)")
    axis.set_title("Nsight Systems CUDA kernel totals\n"
                   "Independent validation; not mixed with in-app profiler timing")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("-n", "--top", type=int, default=15)
    parser.add_argument("-o", "--output", default="nsight_kernels.png")
    args = parser.parse_args()
    main_raw(args.csv, args.output, args.top)


if __name__ == "__main__":
    main()
