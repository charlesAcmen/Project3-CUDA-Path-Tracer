"""
rng_analyze.py — Analysis and visualization for rng_compare output.

Usage:
    python tests/rng_test/rng_analyze.py [--csv PATH] [--out-dir DIR]

Produces:
    1. Console table: 1D star discrepancy per method per dimension
    2. 2D scatter plots: dim0 vs dim1, one subplot per method (pixel 0, bounce 0)
    3. Pi convergence curve: Monte Carlo estimate of pi via dim0^2 + dim1^2 < 1
    4. Variance reduction factor table (LCG baseline = 1.0)
    5. 2D grid-stratification check (regression guard for BUG 4 in rng.h)
    6. Pixel-decorrelation check

Expected results:
    - Owen Halton discrepancy: lowest of the two (best stratification)
    - Pi MSE:                  Owen ~ 10-100x lower than LCG and net-like in
                               the 2D grid check (NOT random-rate).
    - 2D scatter:              Owen fills unit square evenly; LCG has gaps.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyze RNG compare CSV output")
    p.add_argument("--csv",     default="profiler_output/rng_test/rng_data.csv",
                   help="Path to rng_data.csv produced by rng_compare")
    p.add_argument("--out-dir", default="profiler_output/rng_test",
                   help="Directory for output plots")
    p.add_argument("--pixel",   type=int, default=0,
                   help="Which pixel to use for 2D scatter / convergence (default 0)")
    p.add_argument("--bounce",  type=int, default=0,
                   help="Which bounce to use for scatter / convergence (default 0)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# 1D Star Discrepancy  D*_N = max_x | F_N(x) - x |  for uniform on [0,1)
# ---------------------------------------------------------------------------

def star_discrepancy_1d(samples: np.ndarray) -> float:
    """
    Compute the 1D star discrepancy of a sample set in [0, 1).
    D*_N = max over all x in [0,1) of |empirical_CDF(x) - x|
    This equals the Kolmogorov-Smirnov statistic against Uniform(0,1).
    Lower is better.  Perfect stratification would give D* ~ 1/(2N).
    """
    n = len(samples)
    if n == 0:
        return float("nan")
    s = np.sort(samples)
    # KS statistic: D+ and D-
    idx = np.arange(1, n + 1, dtype=float)
    d_plus  = np.max(idx / n - s)
    d_minus = np.max(s - (idx - 1) / n)
    return float(max(d_plus, d_minus))

# ---------------------------------------------------------------------------
# Pi convergence
# ---------------------------------------------------------------------------

def pi_convergence(x_samples: np.ndarray, y_samples: np.ndarray) -> np.ndarray:
    """
    Given N pairs (x, y) in [0,1)^2, compute the running Monte Carlo
    estimate of pi/4 = P(x^2 + y^2 < 1) and return the absolute error
    |estimate * 4 - pi| at each sample count.
    """
    inside = (x_samples**2 + y_samples**2) < 1.0
    cumsum = np.cumsum(inside.astype(float))
    n = np.arange(1, len(x_samples) + 1, dtype=float)
    estimate = (cumsum / n) * 4.0
    return np.abs(estimate - np.pi)


def grid_stratification_deviation(x: np.ndarray, y: np.ndarray, g: int) -> float:
    """
    Max cell-count deviation from uniform on a g x g grid of [0,1)^2.

    A low-discrepancy net keeps every cell within a few points of uniform
    (deviation ~ O(1)); random points have Poisson fluctuations
    (deviation ~ several x sqrt(mean)).  This is the regression guard for
    BUG 4 in rng.h: the base-2-only Owen scramble made Owen's 2D points
    random-like (deviation ~ LCG) despite good 1D marginals.
    """
    n = len(x)
    if n == 0:
        return float("nan")
    j = (np.floor(x * g).astype(int) * g + np.floor(y * g).astype(int))
    counts = np.bincount(j, minlength=g * g)
    return float(counts.max() - n / (g * g))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.csv} …")
    df = pd.read_csv(args.csv)
    print(f"  {len(df):,} rows loaded.")

    METHODS = {
        "lcg":         ("LCG (baseline)",           "#4e79a7"),
        "halton_owen": ("Halton Owen (new)",          "#59a14f"),
    }

    # -----------------------------------------------------------------------
    # 1. 1D Star Discrepancy table
    # -----------------------------------------------------------------------
    print("\n── 1D Star Discrepancy (lower = better) ──")
    print(f"{'Dim':>4}  {'Prime':>5}  ", end="")
    for col, (label, _) in METHODS.items():
        print(f"  {label[:22]:>22}", end="")
    print()

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    all_dims = sorted(df["dim"].unique())
    disc_table = {col: [] for col in METHODS}

    for d in all_dims:
        sub = df[df["dim"] == d]
        prime = primes[d] if d < len(primes) else "?"
        print(f"{d:>4}  {prime:>5}  ", end="")
        for col in METHODS:
            vals = sub[col].values
            D = star_discrepancy_1d(vals)
            disc_table[col].append(D)
            print(f"  {D:>22.6f}", end="")
        print()

    # Summary row
    print(f"{'mean':>4}  {'':>5}  ", end="")
    for col in METHODS:
        mean_D = float(np.mean(disc_table[col]))
        print(f"  {mean_D:>22.6f}", end="")
    print()

    # Variance-reduction factor vs LCG
    print("\nVariance-reduction factor (LCG / method, higher = better):")
    lcg_mean = float(np.mean(disc_table["lcg"]))
    for col, (label, _) in METHODS.items():
        m = float(np.mean(disc_table[col]))
        factor = lcg_mean / m if m > 0 else float("inf")
        print(f"  {label:<28}: {factor:.2f}x")

    # -----------------------------------------------------------------------
    # 2. 2D Scatter plot — dim0 vs dim1 for one pixel, one bounce
    # -----------------------------------------------------------------------
    pixel_mask  = df["pixel"]  == args.pixel
    bounce_mask = df["bounce"] == args.bounce
    dim0_mask   = df["dim"]    == 0
    dim1_mask   = df["dim"]    == 1

    x_data = {}
    y_data = {}
    for col in METHODS:
        x_data[col] = df[pixel_mask & bounce_mask & dim0_mask][col].values
        y_data[col] = df[pixel_mask & bounce_mask & dim1_mask][col].values

    # Use min length (should all be equal)
    N_scatter = min(len(x_data[col]) for col in METHODS)
    N_show = min(N_scatter, 1024)   # cap scatter at 1024 for clarity

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        f"2D Sample Distribution  (pixel={args.pixel}, bounce={args.bounce}, "
        f"dim0 vs dim1, first {N_show} samples)",
        fontsize=12, fontweight="bold"
    )

    for ax, (col, (label, color)) in zip(axes, METHODS.items()):
        ax.scatter(x_data[col][:N_show], y_data[col][:N_show],
                   s=3, alpha=0.5, color=color, rasterized=True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("dim 0 (base 2)")
        ax.set_ylabel("dim 1 (base 3)")
        # Draw unit circle for pi estimation reference
        theta = np.linspace(0, np.pi / 2, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.4)

    plt.tight_layout()
    scatter_path = os.path.join(args.out_dir, "scatter_2d.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"\nSaved 2D scatter → {scatter_path}")

    # -----------------------------------------------------------------------
    # 3. Pi convergence curve
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Monte Carlo π Estimation Convergence  "
        f"(pixel={args.pixel}, bounce={args.bounce})",
        fontsize=12, fontweight="bold"
    )

    ax_lin, ax_log = axes
    n_vals = np.arange(1, N_scatter + 1)

    for col, (label, color) in METHODS.items():
        x = x_data[col][:N_scatter]
        y = y_data[col][:N_scatter]
        err = pi_convergence(x, y)
        ax_lin.plot(n_vals, err, color=color, lw=0.8, alpha=0.85, label=label)
        ax_log.loglog(n_vals, err, color=color, lw=0.8, alpha=0.85, label=label)

    # Reference lines O(1/sqrt(N)) and O((log N)/N)
    ref_n = np.logspace(0, np.log10(N_scatter), 300)
    ax_log.loglog(ref_n, 3.0 / np.sqrt(ref_n),        "k:",  lw=1.0, label="O(1/√N) LCG theory")
    ax_log.loglog(ref_n, 3.0 * np.log(ref_n) / ref_n, "k--", lw=1.0, label="O(log N / N) QMC theory")

    for ax in axes:
        ax.set_xlabel("Number of samples N")
        ax.set_ylabel("|estimate - π|")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls=":", alpha=0.5)

    ax_lin.set_title("Linear scale")
    ax_log.set_title("Log-log scale")

    plt.tight_layout()
    conv_path = os.path.join(args.out_dir, "pi_convergence.png")
    plt.savefig(conv_path, dpi=150)
    plt.close()
    print(f"Saved π convergence → {conv_path}")

    # -----------------------------------------------------------------------
    # 4. 2D grid-stratification check (dim0 x dim1) — regression guard for BUG 4
    #    (rng.h's base-2-only Owen scramble made Owen's 2D points random-like
    #    despite good 1D marginals; the proper digit scramble is net-like.)
    # -----------------------------------------------------------------------
    g2d = 16
    random_std = np.sqrt(len(x_data["lcg"]) / (g2d * g2d))
    print(f"\n── 2D stratification (dim0×dim1, {g2d}x{g2d} grid, max cell deviation) ──")
    print(f"    (net ~ a few; random ~ {3*random_std:.0f}-{5*random_std:.0f}; "
          f"per-cell mean = {len(x_data['lcg'])/(g2d*g2d):.1f})")
    grid_devs = {}
    for col, (label, _) in METHODS.items():
        dev = grid_stratification_deviation(x_data[col], y_data[col], g2d)
        grid_devs[col] = dev
        print(f"  {label:<28}: {dev:>7.1f}")

    # Pixel decorrelation check (pixel 0 vs pixel 1, same dim/bounce)
    print("\n── Pixel decorrelation (pixel0 vs pixel1, dim0, sample corr; ~0 = independent) ──")
    p0 = df[(df["pixel"] == 0) & (df["bounce"] == args.bounce) & (df["dim"] == 0)]
    p1 = df[(df["pixel"] == 1) & (df["bounce"] == args.bounce) & (df["dim"] == 0)]
    if len(p0) and len(p1):
        n_corr = min(len(p0), len(p1))
        for col, (label, _) in METHODS.items():
            r = np.corrcoef(p0[col].values[:n_corr], p1[col].values[:n_corr])[0, 1]
            print(f"  {label:<28}: corr = {r:+.3f}")
    else:
        print("  (need >= 2 pixels in the CSV to check decorrelation)")

    # -----------------------------------------------------------------------
    # 5. Per-dimension discrepancy bar chart
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    x_pos = np.arange(len(all_dims))
    bar_w = 0.25

    for i, (col, (label, color)) in enumerate(METHODS.items()):
        ax.bar(x_pos + i * bar_w, disc_table[col],
               width=bar_w, color=color, label=label, alpha=0.85)

    ax.set_xticks(x_pos + bar_w)
    ax.set_xticklabels([f"dim {d}\n(p={primes[d] if d < len(primes) else '?'})"
                        for d in all_dims], fontsize=8)
    ax.set_ylabel("1D Star Discrepancy D*")
    ax.set_title("Per-dimension 1D Star Discrepancy (lower = better)")
    ax.legend()
    ax.grid(axis="y", ls=":", alpha=0.5)

    plt.tight_layout()
    disc_path = os.path.join(args.out_dir, "discrepancy_bars.png")
    plt.savefig(disc_path, dpi=150)
    plt.close()
    print(f"Saved discrepancy bars → {disc_path}")

    # -----------------------------------------------------------------------
    # Summary verdict
    # -----------------------------------------------------------------------
    print("\n── Summary ──")
    owen_mean = float(np.mean(disc_table["halton_owen"]))
    lcg_mean2 = float(np.mean(disc_table["lcg"]))
    owen_dev  = grid_devs["halton_owen"]
    lcg_dev   = grid_devs["lcg"]

    best1d = min(METHODS.keys(), key=lambda c: float(np.mean(disc_table[c])))
    print(f"  1D best:            {METHODS[best1d][0]}")
    print(f"  1D Owen vs LCG:     {lcg_mean2/owen_mean:.2f}x lower (expected > 2x)")
    print(f"  2D grid dev (g={g2d}):  Owen {owen_dev:.1f} | LCG {lcg_dev:.1f} "
          f"(net ~ few, random ~ {3*random_std:.0f}+)")

    pass_1d  = owen_mean < lcg_mean2                    # Owen beats LCG in 1D
    pass_2d  = owen_dev < lcg_dev                       # Owen 2D is net-like, not random
    if pass_1d and pass_2d:
        print("  [PASS] Owen beats LCG in 1D and is net-like in 2D.")
    elif pass_1d:
        print("  [FAIL] Owen beats LCG in 1D but 2D is random -- check the net.")
    else:
        print("  [FAIL] Unexpected result -- check CSV for data integrity issues.")

    print("\nDone.")


if __name__ == "__main__":
    main()
