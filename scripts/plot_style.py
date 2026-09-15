"""One visual vocabulary shared by every profiler chart."""

from __future__ import annotations

import hashlib

import matplotlib.pyplot as plt


KNOWN_COLORS = {
    "GenerateCameraRays": "#4C78A8",
    "ComputeIntersections": "#B279A2",
    "SortByMaterial": "#E45756",
    "ShadeMaterial": "#F58518",
    "GatherTerminatedPaths": "#72B7B2",
    "CompactPaths": "#54A24B",
    "FinalGather": "#9D755D",
    "BloomWeights": "#BAB0AC",
    "BloomThreshold": "#FF9DA6",
    "BloomBlurHorizontal": "#D37295",
    "BloomBlurVertical": "#A05195",
    "PrepareDisplay": "#EDC948",
    "Tonemap": "#59A14F",
    "ChromaticAberration": "#76B7B2",
    "Vignette": "#AF7AA1",
    "PostProcessCopy": "#FFBE7D",
    "SendImageToPbo": "#79706E",
    "UninstrumentedGpuSpan": "#D3D3D3",
    "UninstrumentedSetup": "#D3D3D3",
}


def configure() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def operation_color(name: str):
    if name in KNOWN_COLORS:
        return KNOWN_COLORS[name]
    palette = plt.get_cmap("tab20").colors
    index = int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16) % len(palette)
    return palette[index]
