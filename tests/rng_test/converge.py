"""converge.py — averaged pi-convergence, LCG vs Owen, over all pixels."""
import sys
import numpy as np
import pandas as pd

csv = sys.argv[1] if len(sys.argv) > 1 else "profiler_output/rng_test/rng_data_current.csv"
df = pd.read_csv(csv, usecols=["pixel", "iter", "bounce", "dim", "lcg", "halton_owen"])
sub = df[df.bounce == 0]
pix = sorted(sub.pixel.unique())
print(f"pixels={len(pix)}")

# iters x pixels matrices per (col, dim)
M = {}
for col in ["lcg", "halton_owen"]:
    M[col] = {}
    for d in (0, 1):
        P = sub[sub.dim == d].pivot(index="iter", columns="pixel", values=col)
        P = P.sort_index()
        M[col][d] = P.values  # shape (iters, pixels), columns sorted by pixel

NMAX = M["lcg"][0].shape[0]
err = {c: {N: [] for N in (64, 256, 1024, 4096, 16384)} for c in M}
for c in M:
    for j in range(len(pix)):
        x = M[c][0][:NMAX, j]
        y = M[c][1][:NMAX, j]
        inside = np.cumsum((x**2 + y**2) < 1.0)
        nn = np.arange(1, NMAX + 1, dtype=float)
        e = np.abs((inside / nn) * 4.0 - np.pi)
        for N in err[c]:
            err[c][N].append(e[N - 1])

print("mean |pi-err| over all pixels (bounce0):")
print(f"{'N':>6}{'LCG':>12}{'Owen':>12}{'eff_mult':>10}")
for N in (64, 256, 1024, 4096, 16384):
    l = float(np.mean(err["lcg"][N]))
    o = float(np.mean(err["halton_owen"][N]))
    print(f"{N:>6}{l:>12.3e}{o:>12.3e}{(l / o) ** 2:>9.0f}x")

print("asymptotic slope on averaged curves (N=1024..16384):")
for c in M:
    a = np.array([np.mean(err[c][N]) for N in (1024, 4096, 16384)])
    slope = np.polyfit(np.log(np.array((1024, 4096, 16384), dtype=float)), np.log(a), 1)[0]
    print(f"  {c:<14} slope={slope:+.3f}   (random -0.5, QMC ~ -1.0)")
