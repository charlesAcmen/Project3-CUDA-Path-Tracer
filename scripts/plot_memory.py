"""Known renderer-owned device allocation breakdown from run.json."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import profiler_utils as pu
import plot_style


CATEGORIES = [
    "path_buffers", "scene_buffers", "bvh", "light_sampling", "textures",
    "post_process", "compaction_workspace", "material_sort_workspace",
    "counter_only", "display_interop_nominal",
]


def main_raw(run_dirs: list[str | Path | pu.RunData], output: str | Path) -> None:
    groups: dict[str, list[pu.RunData]] = defaultdict(list)
    for path in run_dirs:
        run = pu.as_run(path)
        key = str(pu.dotted_get(run.metadata, "runner.experiment_id", pu.run_label(run)))
        groups[key].append(run)
    values = []
    labels = []
    for key, group in groups.items():
        labels.append(str(pu.dotted_get(group[0].metadata, "runner.label", key)))
        values.append({category: float(np.mean([
            pu.dotted_get(run.metadata, f"estimated_device_memory_bytes.{category}", 0)
            for run in group])) / (1024.0 * 1024.0) for category in CATEGORIES})
    if not any(sum(item.values()) for item in values):
        raise pu.SchemaError("run metadata contains no device allocation estimates")

    plot_style.configure()
    fig, axis = plt.subplots(figsize=(max(10, len(labels) * 2.1), 6))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for category in CATEGORIES:
        amount = np.asarray([item[category] for item in values])
        if np.any(amount):
            axis.bar(x, amount, bottom=bottom, label=category.replace("_", " "))
            bottom += amount
    axis.set_ylabel("Known renderer-owned device allocations (MiB)")
    axis.set_title("Device-memory footprint\n"
                   "Interop is nominal RGBA8; driver-internal allocations excluded")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=18, ha="right")
    axis.grid(axis="x", visible=False)
    axis.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("-o", "--output", default="memory.png")
    args = parser.parse_args()
    main_raw(args.runs, args.output)


if __name__ == "__main__":
    main()
