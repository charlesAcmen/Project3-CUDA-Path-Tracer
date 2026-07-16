#!/usr/bin/env python3
"""
Convergence comparison: LCG vs Halton for the CIS565 CUDA Path Tracer.

Measures RMSE against a high-iteration Halton reference and plots the
convergence curve for both modes.

Methodology:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ Reference = Halton @ high iterations (closest available ground truth,   │
  │ since Halton's low-discrepancy walk converges faster than white noise). │
  │                                                                         │
  │ Both LCG and Halton checkpoints are compared against the SAME reference │
  │ in LINEAR RGB (sRGB decoded), so error corresponds to physical          │
  │ brightness, not gamma-encoded values.                                   │
  └──────────────────────────────────────────────────────────────────────────┘

Usage:
  # Run full experiment (render + analyze):
  python scripts/compare_convergence.py --run --scene scenes/cornell.json

  # Analyze existing renders (skip render step):
  python scripts/compare_convergence.py --ref "ref/*.png" \\
      --lcg "lcg/*.png" --halton "halton/*.png" --plot conv.png --csv conv.csv
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zlib

# =========================================================================
#  sRGB → linear conversion
# =========================================================================

def srgb_to_linear(c):
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


# =========================================================================
#  PNG reader
# =========================================================================

def read_png_linear(filename):
    """Read a PNG, return (w, h, linear_RGB_floats)."""
    with open(filename, "rb") as f:
        sig = f.read(8)
        assert sig == b"\x89PNG\r\n\x1a\n", f"{filename}: not a PNG"
        chunks = []
        while True:
            length = struct.unpack(">I", f.read(4))[0]
            ctype = f.read(4)
            data = f.read(length)
            f.read(4)  # crc
            chunks.append((ctype, data))
            if ctype == b"IEND":
                break

    idat = b""
    for ctype, data in chunks:
        if ctype == b"IDAT":
            idat += data
    raw = zlib.decompress(idat)

    w = h = 0
    channels = 3
    for ctype, data in chunks:
        if ctype == b"IHDR":
            w, h = struct.unpack(">II", data[:8])
            channels = 3 if data[9] == 2 else 4
            break

    stride = w * channels + 1  # +1 filter byte per row
    pixels = []
    for y in range(h):
        row_start = y * stride + 1
        for x in range(w):
            for c in range(3):
                pixels.append(srgb_to_linear(
                    raw[row_start + x * channels + c] / 255.0))
    return w, h, pixels


# =========================================================================
#  Metrics
# =========================================================================

def rmse_linear(ref, test):
    """Per-pixel RMSE in linear RGB."""
    n = len(ref) // 3
    total = 0.0
    for i in range(n):
        dr = ref[3*i]     - test[3*i]
        dg = ref[3*i + 1] - test[3*i + 1]
        db = ref[3*i + 2] - test[3*i + 2]
        total += dr*dr + dg*dg + db*db
    return math.sqrt(total / (3 * n))

def relmae_linear(ref, test):
    """Relative Mean Absolute Error (per-pixel, linear RGB)."""
    n = len(ref) // 3
    total_abs = 0.0
    total_ref = 0.0
    for i in range(n):
        total_abs += abs(ref[3*i] - test[3*i])
        total_abs += abs(ref[3*i+1] - test[3*i+1])
        total_abs += abs(ref[3*i+2] - test[3*i+2])
        total_ref += ref[3*i] + ref[3*i+1] + ref[3*i+2]
    return total_abs / total_ref if total_ref > 0 else 0.0


# =========================================================================
#  File name helpers
# =========================================================================

SAMPLE_RE = re.compile(r'(\d+)samp', re.IGNORECASE)

def guess_samples(filename):
    m = SAMPLE_RE.search(filename)
    return int(m.group(1)) if m else None


# =========================================================================
#  Render runner
# =========================================================================

def run_render(binary, scene_json, rng_mode, save_at, description):
    """Run the path tracer.  Returns list of output PNG paths."""
    save_str = ",".join(str(x) for x in save_at)
    cmd = [binary, scene_json,
           f"--rng={rng_mode}",
           f"--save-at={save_str}",
           "--save",
           "--benchmark"]
    tag = f"{description} (rng={rng_mode}, save-at={save_str})"
    print(f"  [{tag}]")
    sys.stdout.flush()
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    elapsed = time.time() - t0
    # Print renderer stdout / stderr
    for line in result.stdout.splitlines():
        print(f"    {line}")
    if result.stderr:
        for line in result.stderr.splitlines():
            print(f"    stderr: {line}")
    if result.returncode != 0:
        print(f"    ⚠  exited with code {result.returncode}")
        return []
    print(f"    ✓  {elapsed:.1f}s")
    return []  # We'll glob for files instead


# =========================================================================
#  Scene file helpers
# =========================================================================

def load_scene_config(path):
    with open(path) as f:
        return json.load(f)

def write_scene_config(config, path):
    with open(path, "w") as f:
        json.dump(config, f, indent=4)


# =========================================================================
#  find_image_by_suffix
# =========================================================================

def find_image_by_suffix(directory, file_prefix, iter_count):
    """Find a PNG in directory whose filename matches <prefix>.<anything>.<iter>samp.png"""
    pattern = os.path.join(directory, f"{file_prefix}.*.{iter_count}samp.png")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    # Broader fallback
    pattern2 = os.path.join(directory, f"*{file_prefix}*{iter_count}samp*")
    matches2 = glob.glob(pattern2)
    return matches2[0] if matches2 else None


# =========================================================================
#  Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare convergence of LCG vs Halton renders")

    # Analysis options
    parser.add_argument("--ref", default=None,
                        help="Reference PNG (default: auto-detect from run)")
    parser.add_argument("--lcg", nargs="*", default=[],
                        help="LCG PNG files (or glob)")
    parser.add_argument("--halton", nargs="*", default=[],
                        help="Halton PNG files (or glob)")
    parser.add_argument("--plot", default="convergence.png",
                        help="Save convergence plot (default: convergence.png)")
    parser.add_argument("--csv", default="convergence.csv",
                        help="Write CSV (default: convergence.csv)")
    parser.add_argument("--no-display", action="store_true",
                        help="Do not print the table (useful for batch)")

    # Run mode
    parser.add_argument("--run", action="store_true",
                        help="Run the experiment (render + analyze)")
    parser.add_argument("--scene", default="scenes/cornell.json",
                        help="Scene file for checkpoint renders")
    parser.add_argument("--bin", default="build/bin/Release/cis565_path_tracer.exe",
                        help="Path tracer binary")
    parser.add_argument("--ref-iters", type=int, default=5000,
                        help="Reference iteration count")
    parser.add_argument("--checkpoints", default="50,200,500,1000",
                        help="Comma-separated checkpoint iterations")
    parser.add_argument("--out", default="outputs/convergence",
                        help="Output directory")

    args = parser.parse_args()

    # ──────────────────────────────────────────────────────────────────
    #  RUN MODE: render first
    # ──────────────────────────────────────────────────────────────────
    if args.run:
        out_dir = args.out
        ref_iters = args.ref_iters
        ckpt_iters = [int(x) for x in args.checkpoints.split(",")]

        if not os.path.isfile(args.bin):
            print(f"Error: binary not found: {args.bin}")
            sys.exit(1)
        if not os.path.isfile(args.scene):
            print(f"Error: scene not found: {args.scene}")
            sys.exit(1)

        os.makedirs(out_dir, exist_ok=True)

        scene_name = os.path.splitext(os.path.basename(args.scene))[0]

        # ---- Step 1: Reference scene ----
        print("─── Step 1: Reference scene ───")
        ref_scene_path = os.path.join(out_dir, f"{scene_name}_ref{ref_iters}.json")
        ref_config = load_scene_config(args.scene)
        ref_config["Camera"]["ITERATIONS"] = ref_iters
        ref_file = f"{scene_name}_ref{ref_iters}"
        ref_config["Camera"]["FILE"] = ref_file
        write_scene_config(ref_config, ref_scene_path)

        # ---- Step 2: Render reference (Halton) ----
        print("─── Step 2: Halton reference ───")
        run_render(args.bin, ref_scene_path, rng_mode=1,
                   save_at=[ref_iters],
                   description=f"Halton ref @ {ref_iters}")

        # Find reference file
        ref_glob = os.path.join(".", f"{ref_file}.*.{ref_iters}samp.png")
        ref_matches = sorted(glob.glob(ref_glob))
        if not ref_matches:
            print(f"Error: reference output not found ({ref_glob})")
            sys.exit(1)
        ref_path = ref_matches[0]
        ref_dest = os.path.join(out_dir, os.path.basename(ref_path))
        shutil.move(ref_path, ref_dest)
        print(f"  Reference: {ref_dest}")

        # ---- Step 3: Render checkpoints ----
        print("─── Step 3: Halton checkpoints ───")
        run_render(args.bin, args.scene, rng_mode=1,
                   save_at=ckpt_iters,
                   description="Halton checkpoints")

        print("─── Step 4: LCG checkpoints ───")
        run_render(args.bin, args.scene, rng_mode=0,
                   save_at=ckpt_iters,
                   description="LCG checkpoints")

        # ---- Step 5: Collect files ----
        print("─── Step 5: Collect outputs ───")
        lcg_dir = os.path.join(out_dir, "lcg")
        hal_dir = os.path.join(out_dir, "halton")
        os.makedirs(lcg_dir, exist_ok=True)
        os.makedirs(hal_dir, exist_ok=True)

        # Move checkpoint files
        for it in ckpt_iters:
            pattern = f"{scene_name}.*.{it}samp.png"
            for f in glob.glob(pattern):
                basename = os.path.basename(f)
                # Check if it's Halton or LCG from the current directory
                # (we can't tell from filename alone, so we rely on run order:
                #  first Halton run, then LCG run → both produce same filenames
                #  but in the same dir, later overwrites earlier. We need a
                #  smarter approach.)
                # Instead: rename by rng mode.
                pass

        print("  (file collection logic in --run mode needs refinement; "
              "use separate output dirs per mode for now)")

        # For now, just set up the analysis from the output directory
        args.ref = ref_dest

        # Find checkpoint files - they're still in CWD with timestamp names
        # Better approach: just glob for them
        lcg_files = sorted(glob.glob(f"{scene_name}.*.??samp.png"))
        # Can't distinguish LCG from Halton by name in current setup
        print("  ⚠  --run mode currently puts both modes' outputs in CWD.")
        print("  Run --rng=0 and --rng=1 separately with different output dirs.")
        print("  For now, use --lcg/--halton to specify files manually.\n")
        sys.exit(0)

    # ──────────────────────────────────────────────────────────────────
    #  ANALYSIS MODE
    # ──────────────────────────────────────────────────────────────────

    # Resolve globs
    def resolve_glob(patterns):
        files = []
        for p in patterns:
            expanded = sorted(glob.glob(p))
            if expanded:
                files.extend(expanded)
            elif os.path.isfile(p):
                files.append(p)
        return files

    ref_path = args.ref
    lcg_files = resolve_glob(args.lcg)
    hal_files = resolve_glob(args.halton)

    if ref_path and "*" in ref_path:
        ref_matches = sorted(glob.glob(ref_path))
        if ref_matches:
            ref_path = ref_matches[0]

    if not ref_path or not os.path.isfile(ref_path):
        print("Error: reference file not found.")
        sys.exit(1)

    # Read reference
    _, _, ref_pixels = read_png_linear(ref_path)
    n_pix = len(ref_pixels) // 3
    print(f"\nReference: {os.path.basename(ref_path)} ({n_pix} px, linear RGB)")

    # Group test files
    groups = {}
    if lcg_files:
        groups["LCG"] = lcg_files
    if hal_files:
        groups["Halton"] = hal_files
    if not groups:
        print("Error: no test images (use --lcg and/or --halton)")
        sys.exit(1)

    # Compute errors
    results = []
    for gname, files in groups.items():
        for f in files:
            if not os.path.isfile(f):
                continue
            try:
                _, _, px = read_png_linear(f)
            except Exception:
                continue
            if len(px) != len(ref_pixels):
                print(f"  Skip {f}: size mismatch")
                continue
            err_rmse = rmse_linear(ref_pixels, px)
            err_rmae = relmae_linear(ref_pixels, px)
            samples = guess_samples(f) or 0
            results.append((samples, gname, os.path.basename(f), err_rmse, err_rmae))
    results.sort()

    if not results:
        print("No results.")
        sys.exit(1)

    # ── Print table ──
    if not args.no_display:
        print()
        print(f"  {'Samples':>8}  {'Mode':<8}  {'RMSE':>10}  {'RMAE':>10}  {'File'}")
        print(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*40}")
        for s, g, fn, rmse_v, rmae_v in results:
            print(f"  {s:>8}  {g:<8}  {rmse_v:.6f}  {rmae_v:.6f}  {fn}")

        # Cross-mode comparison
        print()
        print("  ── Cross-mode at matching iterations ──")
        print(f"  {'Iters':>8}  {'LCG RMSE':>10}  {'Halton RMSE':>12}  {'Ratio':>8}  {'Better?'}")
        print(f"  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*10}")
        by_s = {}
        for s, g, _, rmse_v, _ in results:
            by_s.setdefault(s, {})[g] = rmse_v
        for s in sorted(by_s):
            l = by_s[s].get("LCG")
            h = by_s[s].get("Halton")
            if l is not None and h is not None:
                ratio = h / l if l > 0 else float("inf")
                better = "<-- Halton" if h < l else "<-- LCG" if l < h else "tie"
                print(f"  {s:>8}  {l:.6f}  {h:.6f}      {ratio:.4f}  {better}")
            elif l is not None:
                print(f"  {s:>8}  {l:.6f}  {'─':>12}  {'─':>8}  {'─':>10}")
            elif h is not None:
                print(f"  {s:>8}  {'─':>10}  {h:.6f}  {'─':>8}  {'─':>10}")

    # ── Plot ──
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            for g in ["LCG", "Halton"]:
                pts = sorted((s, rmse_v) for s, grp, _, rmse_v, _ in results if grp == g)
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, "o-", linewidth=2, label=g)
            ax.set_xlabel("Iterations")
            ax.set_ylabel("RMSE (linear RGB vs Halton reference)")
            ax.set_title("Convergence: LCG vs Halton")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(args.plot, dpi=150)
            print(f"\n  Plot: {args.plot}")
        except ImportError:
            print("\n  (matplotlib not available, skip plot)")

    # ── CSV ──
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["samples", "mode", "file", "rmse_linear_rgb", "rmae_linear_rgb"])
            w.writerows(results)
        print(f"  CSV:  {args.csv}")

    print()


if __name__ == "__main__":
    main()
