#!/usr/bin/env python3
"""
analyze_rng.py — Statistical comparison of LCG vs Halton RNG sequences.

Reads the CSV produced by rng_compare.cu (tests/rng_test/rng_compare.cu)
and generates diagnostic visualizations in profiler_output/rng_test/.

Usage:
  python scripts/analyze_rng.py [--csv path/to/data.csv] [--out dir/]

Dependencies: numpy, matplotlib (same as existing scripts/ scripts)
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CSV  = "profiler_output/rng_test/rng_data.csv"
DEFAULT_OUT  = "profiler_output/rng_test"
DIMS         = 10            # match HALTON_NUM_DIMS
DIM_LABELS   = [
    "AA jitter x  (b=2)",
    "AA jitter y  (b=3)",
    "Lens u       (b=5)",
    "Lens v       (b=7)",
    "Diffuse θ    (b=11)",
    "Diffuse φ    (b=13)",
    "Specular θ   (b=17)",
    "Specular φ   (b=19)",
    "Fresnel RR   (b=23)",
    "Path RR      (b=29)",
]
DIM_PRIMES    = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

# Matplotlib style — match existing scripts/ style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})
COLOR_LCG    = "#e74c3c"   # red
COLOR_HALTON = "#2980b9"   # blue


# ============================================================================
# Data loading
# ============================================================================

def load_csv(path):
    """Returns list of dicts from CSV."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["pixel"]  = int(row["pixel"])
            row["iter"]   = int(row["iter"])
            row["bounce"] = int(row["bounce"])
            row["dim"]    = int(row["dim"])
            row["lcg"]    = float(row["lcg"])
            row["halton"] = float(row["halton"])
            rows.append(row)
    return rows


def group_data(rows):
    """
    Group rows by (pixel, bounce) → dict of (dim, iters, lcg_vals, halton_vals).
    Returns a list of groups, one per (pixel, bounce) combination.
    """
    buckets = defaultdict(lambda: defaultdict(lambda: {"iter": [], "lcg": [], "halton": []}))

    for r in rows:
        key = (r["pixel"], r["bounce"])
        buckets[key][r["dim"]]["iter"].append(r["iter"])
        buckets[key][r["dim"]]["lcg"].append(r["lcg"])
        buckets[key][r["dim"]]["halton"].append(r["halton"])

    # Sort by iter within each dim (data is already ordered, but be safe)
    groups = []
    for (pixel, bounce), dims in sorted(buckets.items()):
        sorted_dims = {}
        for dim, data in dims.items():
            idx = np.argsort(data["iter"])
            sorted_dims[dim] = {
                "iter":   np.array(data["iter"])[idx],
                "lcg":    np.array(data["lcg"])[idx],
                "halton": np.array(data["halton"])[idx],
            }
        groups.append({
            "pixel": pixel,
            "bounce": bounce,
            "dims": sorted_dims,
            "num_iters": len(sorted_dims[0]["iter"]) if 0 in sorted_dims else 0,
        })
    return groups


# ============================================================================
# Figure 1: 2D Scatter — dim 0 vs dim 1
# ============================================================================

def plot_2d_scatter(groups, out_dir):
    """Side-by-side 2D scatter: LCG vs Halton for dimensions (0,1)."""
    g = groups[0]  # pixel 0, bounce 0
    N = min(256, g["num_iters"])
    x_lcg    = g["dims"][0]["lcg"][:N]
    y_lcg    = g["dims"][1]["lcg"][:N]
    x_halton = g["dims"][0]["halton"][:N]
    y_halton = g["dims"][1]["halton"][:N]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    ax1.scatter(x_lcg, y_lcg, s=8, alpha=0.6, c=COLOR_LCG, edgecolors="none")
    ax1.set_title(f"LCG  ({N} samples)")
    ax1.set_xlabel("Dim 0 (b=2)")
    ax1.set_ylabel("Dim 1 (b=3)")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")

    ax2.scatter(x_halton, y_halton, s=8, alpha=0.6, c=COLOR_HALTON, edgecolors="none")
    ax2.set_title(f"Halton ({N} samples)")
    ax2.set_xlabel("Dim 0 (b=2)")
    ax2.set_ylabel("Dim 1 (b=3)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect("equal")

    plt.tight_layout()
    path = os.path.join(out_dir, "rng_2d_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ============================================================================
# Figure 2: 2D Bin Heatmap
# ============================================================================

def plot_2d_heatmap(groups, out_dir, bins=32):
    """2D histogram / heatmap of sample density."""
    g = groups[0]
    N = min(4096, g["num_iters"])

    x_lcg    = g["dims"][0]["lcg"][:N]
    y_lcg    = g["dims"][1]["lcg"][:N]
    x_halton = g["dims"][0]["halton"][:N]
    y_halton = g["dims"][1]["halton"][:N]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    for ax, xs, ys, label, cmap in [
        (ax1, x_lcg, y_lcg, "LCG", "Reds"),
        (ax2, x_halton, y_halton, "Halton", "Blues"),
    ]:
        counts, _, _ = np.histogram2d(xs, ys, bins=bins, range=[[0,1],[0,1]])
        im = ax.imshow(counts.T, origin="lower", extent=[0,1,0,1],
                       cmap=cmap, aspect="equal")
        ax.set_title(f"{label}  ({N} samples, {bins}×{bins} bins)")
        ax.set_xlabel("Dim 0 (b=2)")
        ax.set_ylabel("Dim 1 (b=3)")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Samples / bin")

    plt.tight_layout()
    path = os.path.join(out_dir, "rng_2d_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# Figure 3: Convergence plots
# ============================================================================

def test_func_1d(x):
    """∫₀¹ x² dx = 1/3"""
    return x * x

def test_func_2d(x, y):
    """∫₀¹∫₀¹ (x + y²) dx dy = 1/2 + 1/3 = 5/6"""
    return x + y * y

TRUE_1D = 1.0 / 3.0
TRUE_2D = 5.0 / 6.0


def compute_convergence(values_1d, values_2d_x, values_2d_y):
    """
    Compute running MC estimate error as N increases.
    Returns (ns, lcg_err_1d, halton_err_1d, lcg_err_2d, halton_err_2d)
    """
    N = len(values_1d["lcg"])

    # Log-spaced evaluation points (plus dense at small N)
    log_ns = np.unique(np.logspace(0, np.log10(N), 200).astype(int))
    dense_ns = np.arange(1, min(101, N+1))
    ns = np.unique(np.concatenate([dense_ns, log_ns]))
    ns = ns[ns <= N]

    def running_error(vals, true_val):
        cumsum = np.cumsum(vals)
        estimates = cumsum[ns - 1] / ns
        return np.abs(estimates - true_val)

    return (
        ns,
        running_error(values_1d["lcg"], TRUE_1D),
        running_error(values_1d["halton"], TRUE_1D),
        running_error(values_2d_x["lcg"] + values_2d_y["lcg"]**2, TRUE_2D),
        running_error(values_2d_x["halton"] + values_2d_y["halton"]**2, TRUE_2D),
    )


def plot_convergence(groups, out_dir):
    """Log-log convergence: error vs N for 1D and 2D test integrals."""
    g = groups[0]
    N = min(65536, g["num_iters"])

    values_1d = {k: g["dims"][4]["lcg"][:N] if k == "lcg" else g["dims"][4]["halton"][:N]
                 for k in ["lcg", "halton"]}
    values_2d_x = {k: g["dims"][4]["lcg"][:N] if k == "lcg" else g["dims"][4]["halton"][:N]
                   for k in ["lcg", "halton"]}
    values_2d_y = {k: g["dims"][5]["lcg"][:N] if k == "lcg" else g["dims"][5]["halton"][:N]
                   for k in ["lcg", "halton"]}

    ns, e1_lcg, e1_hal, e2_lcg, e2_hal = compute_convergence(values_1d, values_2d_x, values_2d_y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # 1D integral
    ax1.loglog(ns, e1_lcg,  color=COLOR_LCG,    label="LCG",    linewidth=1)
    ax1.loglog(ns, e1_hal,  color=COLOR_HALTON,  label="Halton", linewidth=1)
    # Reference slopes
    ref_n = ns[ns > 10]
    ax1.loglog(ref_n, 0.5 / np.sqrt(ref_n), "k--",  alpha=0.3, label="O(1/√N)")
    ax1.loglog(ref_n, 1.0 / ref_n,           "k:",   alpha=0.3, label="O(1/N)")
    ax1.set_title("1D: ∫ x² dx = 1/3")
    ax1.set_xlabel("Samples (N)")
    ax1.set_ylabel("|Error|")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2D integral
    ax2.loglog(ns, e2_lcg,  color=COLOR_LCG,    label="LCG",    linewidth=1)
    ax2.loglog(ns, e2_hal,  color=COLOR_HALTON,  label="Halton", linewidth=1)
    ax2.loglog(ref_n, 0.5 / np.sqrt(ref_n), "k--", alpha=0.3, label="O(1/√N)")
    ax2.loglog(ref_n, 1.0 / ref_n,           "k:",  alpha=0.3, label="O(1/N)")
    ax2.set_title("2D: ∫∫ (x + y²) dx dy = 5/6")
    ax2.set_xlabel("Samples (N)")
    ax2.set_ylabel("|Error|")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "rng_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# Figure 4: 1D Marginal Histograms
# ============================================================================

def plot_histograms(groups, out_dir):
    """Per-dimension histogram comparison."""
    g = groups[0]
    N = min(10000, g["num_iters"])

    ncols = 5
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6))
    axes = axes.flatten()

    for dim in range(min(DIMS, ncols * nrows)):
        ax = axes[dim]
        data_lcg    = g["dims"][dim]["lcg"][:N]
        data_halton = g["dims"][dim]["halton"][:N]

        ax.hist(data_lcg,    bins=50, alpha=0.5, color=COLOR_LCG,    label="LCG",    density=True)
        ax.hist(data_halton, bins=50, alpha=0.5, color=COLOR_HALTON, label="Halton", density=True)
        ax.set_title(DIM_LABELS[dim] if dim < len(DIM_LABELS) else f"Dim {dim}")
        ax.set_xlim(0, 1)
        ax.set_ylabel("Density")
        if dim >= ncols * (nrows - 1):
            ax.set_xlabel("Value")
        ax.legend(fontsize=6)

    # Hide unused subplots
    for dim in range(min(DIMS, ncols * nrows), ncols * nrows):
        axes[dim].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "rng_histograms.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# Figure 5: Autocorrelation
# ============================================================================

def autocorr(x, max_lag=100):
    """Compute sample autocorrelation for lags 0..max_lag."""
    n = len(x)
    if n <= max_lag + 1:
        return None
    mu = np.mean(x)
    var = np.var(x)
    if var == 0:
        return None
    result = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        result[k] = np.mean((x[:n-k] - mu) * (x[k:] - mu)) / var
    return result


def plot_autocorrelation(groups, out_dir, max_lag=100):
    """ACF for dim 0 (base 2) — up to max_lag."""
    g = groups[0]
    N = min(10000, g["num_iters"])

    acf_lcg    = autocorr(g["dims"][0]["lcg"][:N], max_lag)
    acf_halton = autocorr(g["dims"][0]["halton"][:N], max_lag)

    if acf_lcg is None or acf_halton is None:
        print("  Skipping autocorrelation: not enough samples")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    lags = np.arange(max_lag + 1)
    ax1.bar(lags, acf_lcg, width=0.8, color=COLOR_LCG, alpha=0.7)
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.axhline(1.96 / np.sqrt(N), color="gray", linestyle="--", linewidth=0.5)
    ax1.axhline(-1.96 / np.sqrt(N), color="gray", linestyle="--", linewidth=0.5)
    ax1.set_title("LCG — Autocorrelation (dim 0, b=2)")
    ax1.set_ylabel("ACF")

    ax2.bar(lags, acf_halton, width=0.8, color=COLOR_HALTON, alpha=0.7)
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.axhline(1.96 / np.sqrt(N), color="gray", linestyle="--", linewidth=0.5)
    ax2.axhline(-1.96 / np.sqrt(N), color="gray", linestyle="--", linewidth=0.5)
    ax2.set_title("Halton — Autocorrelation (dim 0, b=2)")
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("ACF")

    plt.tight_layout()
    path = os.path.join(out_dir, "rng_autocorrelation.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# Figure 6: CP rotation decorrelation (multi-pixel scatter)
# ============================================================================

def plot_pixel_decorrelation(groups, out_dir):
    """
    Show how CP rotation decorrelates adjacent pixels.
    Scatter plot of dim 0 values from pixel 0 vs pixel 1.
    LCG: independently seeded → no structure.
    Halton: same raw Halton with different CP offsets → decorrelated but
    each is individually low-discrepancy.
    """
    # Group by pixel
    pix_groups = {}
    for g in groups:
        pix_groups[g["pixel"]] = g

    if len(pix_groups) < 2:
        print("  Skipping pixel decorrelation: only 1 pixel in data")
        return

    p0 = pix_groups[0]
    p1 = pix_groups[1]
    N = min(1024, p0["num_iters"], p1["num_iters"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    # LCG: pixel 0 vs pixel 1
    ax1.scatter(
        p0["dims"][0]["lcg"][:N],
        p1["dims"][0]["lcg"][:N],
        s=4, alpha=0.5, color=COLOR_LCG, edgecolors="none"
    )
    ax1.set_title("LCG: Pixel 0 vs Pixel 1 (dim 0)")
    ax1.set_xlabel("Pixel 0")
    ax1.set_ylabel("Pixel 1")
    ax1.set_aspect("equal")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Halton: pixel 0 vs pixel 1
    ax2.scatter(
        p0["dims"][0]["halton"][:N],
        p1["dims"][0]["halton"][:N],
        s=4, alpha=0.5, color=COLOR_HALTON, edgecolors="none"
    )
    ax2.set_title("Halton: Pixel 0 vs Pixel 1 (dim 0)")
    ax2.set_xlabel("Pixel 0")
    ax2.set_ylabel("Pixel 1")
    ax2.set_aspect("equal")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(out_dir, "rng_pixel_decorrelation.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# Figure 7: Cross-frame consistency check
# ============================================================================

def plot_frame_consistency(groups, out_dir):
    """
    Check whether Halton samples from consecutive iterations preserve
    low-discrepancy structure across frames.

    Plot: Halton dim 0 value vs iteration number for first 256 iters.
    LCG should look like random scatter; Halton should show structured
    filling (alternating pattern for base 2: 0, 0.5, 0.25, 0.75, ...).
    """
    g = groups[0]
    N = min(256, g["num_iters"])
    iters = np.arange(N)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(iters, g["dims"][0]["lcg"][:N], "o", markersize=2,
             color=COLOR_LCG, alpha=0.5, label="LCG")
    ax1.set_ylabel("Dim 0 value")
    ax1.set_title("LCG: Sampled value vs Iteration")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(iters, g["dims"][0]["halton"][:N], "o", markersize=2,
             color=COLOR_HALTON, alpha=0.5, label="Halton")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Dim 0 value")
    ax2.set_title("Halton (base 2): Sampled value vs Iteration")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "rng_frame_consistency.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# Main
# ============================================================================

def print_summary(groups):
    """Print a text summary of key statistics."""
    g = groups[0]
    N = g["num_iters"]

    print("\n=== Summary (pixel 0, bounce 0) ===\n")
    print(f"  Samples per dimension: {N}")
    print()

    for dim in range(DIMS):
        lcg_vals    = g["dims"][dim]["lcg"]
        halton_vals = g["dims"][dim]["halton"]
        label = DIM_LABELS[dim] if dim < len(DIM_LABELS) else f"Dim {dim}"

        # Mean should be ~0.5 for both
        # Std should be ~0.289 (1/√12) for uniform
        lcg_mean    = np.mean(lcg_vals[:N])
        halton_mean = np.mean(halton_vals[:N])
        lcg_std     = np.std(lcg_vals[:N])
        halton_std  = np.std(halton_vals[:N])
        expected_std = 1.0 / np.sqrt(12.0)

        print(f"  {label:25s}  "
              f"LCG mean={lcg_mean:.4f}  Halton mean={halton_mean:.4f}  "
              f"LCG std={lcg_std:.4f}  Halton std={halton_std:.4f}  "
              f"(expected σ={expected_std:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RNG sequences from rng_compare.cu output")
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help=f"Path to CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"Output directory (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    csv_path = args.csv
    out_dir  = args.out

    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        print("Run tests/rng_test/rng_compare first to generate data, or specify --csv")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {csv_path} ...")
    rows = load_csv(csv_path)
    print(f"  {len(rows)} rows loaded")

    groups = group_data(rows)
    print(f"  {len(groups)} (pixel, bounce) groups")
    for g in groups:
        print(f"    pixel={g['pixel']}, bounce={g['bounce']}, "
              f"samples={g['num_iters']}")

    print_summary(groups)

    print("\nGenerating plots ...")
    plot_2d_scatter(groups, out_dir)
    plot_2d_heatmap(groups, out_dir)
    plot_convergence(groups, out_dir)
    plot_histograms(groups, out_dir)
    plot_autocorrelation(groups, out_dir)
    plot_pixel_decorrelation(groups, out_dir)
    plot_frame_consistency(groups, out_dir)

    print(f"\nAll plots saved to {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
