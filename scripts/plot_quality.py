"""Display-space image error and speed-quality trade-off plots.

This script evaluates saved PNGs after the renderer's ACES/sRGB pipeline. It
does not claim linear-HDR error. Use a high-SPP image from the same camera and
post-processing configuration as the reference.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import plot_style


def _rgb(path: str | Path) -> np.ndarray:
    image = np.asarray(mpimg.imread(path), dtype=np.float64)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"expected RGB/RGBA image: {path}")
    image = image[:, :, :3]
    if image.max(initial=0.0) > 1.0:
        image /= 255.0
    return image


def quality_metrics(reference: np.ndarray,
                    candidate: np.ndarray) -> tuple[float, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"image shapes differ: {reference.shape} vs {candidate.shape}")
    rmse = float(np.sqrt(np.mean(np.square(reference - candidate))))
    psnr = math.inf if rmse == 0.0 else float(20.0 * math.log10(1.0 / rmse))
    return rmse, psnr


def _parse_candidate(value: str) -> tuple[str, Path, float | None]:
    # LABEL=IMAGE or LABEL=IMAGE@MILLISECONDS
    if "=" not in value:
        raise ValueError("candidate must be LABEL=IMAGE or LABEL=IMAGE@MILLISECONDS")
    label, payload = value.split("=", 1)
    if "@" in payload:
        image, milliseconds = payload.rsplit("@", 1)
        return label, Path(image), float(milliseconds)
    return label, Path(payload), None


def main_raw(reference_path: str | Path,
             candidates: list[tuple[str, Path, float | None]],
             output: str | Path) -> None:
    reference = _rgb(reference_path)
    metrics = [(label, *quality_metrics(reference, _rgb(path)), milliseconds)
               for label, path, milliseconds in candidates]
    plot_style.configure()
    has_speed = all(item[3] is not None for item in metrics)
    fig, axes = plt.subplots(1, 2 if has_speed else 1,
                             figsize=(12 if has_speed else 8, 6), squeeze=False)
    axis = axes[0, 0]
    x = np.arange(len(metrics))
    rmse = [item[1] for item in metrics]
    axis.bar(x, rmse, color=[plt.get_cmap("Set2")(i % 8) for i in range(len(metrics))])
    axis.set_xticks(x)
    axis.set_xticklabels([item[0] for item in metrics], rotation=18, ha="right")
    axis.set_ylabel("Display-space RGB RMSE (lower is better)")
    axis.set_title("Image convergence against high-SPP reference")
    axis.grid(axis="x", visible=False)
    for index, item in enumerate(metrics):
        psnr = "∞" if math.isinf(item[2]) else f"{item[2]:.1f} dB"
        axis.text(index, item[1], psnr, ha="center", va="bottom", fontsize=8)
    if has_speed:
        tradeoff = axes[0, 1]
        for index, (label, error, _, milliseconds) in enumerate(metrics):
            tradeoff.scatter(milliseconds, error, s=65,
                             color=plt.get_cmap("Set2")(index % 8))
            tradeoff.annotate(label, (milliseconds, error), xytext=(5, 5),
                              textcoords="offset points", fontsize=8)
        tradeoff.set_xlabel("End-to-end latency (ms)")
        tradeoff.set_ylabel("Display-space RGB RMSE")
        tradeoff.set_title("Speed-quality frontier (lower-left is better)")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidates", nargs="+")
    parser.add_argument("-o", "--output", default="quality.png")
    args = parser.parse_args()
    main_raw(args.reference, [_parse_candidate(value) for value in args.candidates],
             args.output)


if __name__ == "__main__":
    main()
