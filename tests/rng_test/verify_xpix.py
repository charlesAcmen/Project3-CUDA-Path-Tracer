"""verify_xpix.py — full cross-pixel correlation + CSV composition."""
import numpy as np
import pandas as pd

import sys
csv = sys.argv[1] if len(sys.argv) > 1 else "profiler_output/rng_test/rng_data_current.csv"

# Composition of the file
df = pd.read_csv(csv, usecols=["pixel", "iter", "bounce", "dim"])
print("composition:",
      f"pixels={df.pixel.nunique()} iters={df.iter.nunique()} "
      f"bounces={df.bounce.nunique()} dims={df.dim.nunique()} rows={len(df):,}")

# Load a subsample spanning all pixels (iters <= 4000) for correlation
big = pd.read_csv(csv, usecols=["pixel", "iter", "bounce", "dim",
                                "lcg", "halton_cp", "halton_owen"])
big = big[big.iter <= 4000]
print("subsample rows:", f"{len(big):,}")

pix = sorted(big.pixel.unique())
for dim in [0, 1, 4, 9]:
    rows = big[big.bounce == 0]
    corrs = []
    for i, p in enumerate(pix):
        a = rows[(rows.pixel == p) & (rows.dim == dim)].sort_values("iter").halton_owen.values
        for q in pix[i + 1:]:
            b = rows[(rows.pixel == q) & (rows.dim == dim)].sort_values("iter").halton_owen.values
            n = min(len(a), len(b))
            corrs.append(abs(float(np.corrcoef(a[:n], b[:n])[0, 1])))
    corrs = np.array(corrs)
    print(f"dim{dim}: cross-pixel |corr|  mean={corrs.mean():+.3f} max={corrs.max():.3f} "
          f"pairs={len(corrs)}")

# LCG reference for the same pairs
rows = big[big.bounce == 0]
for col in ["lcg", "halton_cp"]:
    corrs = []
    for i, p in enumerate(pix):
        a = rows[(rows.pixel == p) & (rows.dim == 0)].sort_values("iter")[col].values
        for q in pix[i + 1:]:
            b = rows[(rows.pixel == q) & (rows.dim == 0)].sort_values("iter")[col].values
            n = min(len(a), len(b))
            corrs.append(abs(float(np.corrcoef(a[:n], b[:n])[0, 1])))
    corrs = np.array(corrs)
    print(f"{col} (dim0):      cross-pixel |corr|  mean={corrs.mean():+.3f} max={corrs.max():.3f}")

# Adjacent-pixel pairs specifically (structured-noise risk)
for col in ["lcg", "halton_owen"]:
    r_adj = []
    for p in pix:
        if p + 1 not in pix:
            continue
        a = rows[(rows.pixel == p) & (rows.dim == 0)].sort_values("iter")[col].values
        b = rows[(rows.pixel == p + 1) & (rows.dim == 0)].sort_values("iter")[col].values
        n = min(len(a), len(b))
        r_adj.append(abs(float(np.corrcoef(a[:n], b[:n])[0, 1])))
    print(f"adjacent-pixel corr ({col}, dim0): {np.mean(r_adj):+.3f}")

# At iter=0, every pixel's halton_owen sample should equal its fixed float rot
# (owenRadicalInverse of 0 == 0).  Verify it's a per-pixel fixed value in [0,1).
it0 = big[(big.iter == 0) & (big.bounce == 0) & (big.dim == 0)]
print("\niter=0, dim0 (bounce0): halton_owen per pixel =",
      it0.sort_values("pixel").halton_owen.round(6).tolist()[:8], "... (fixed per-pixel rot)")
