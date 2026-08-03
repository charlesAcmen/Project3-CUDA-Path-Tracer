"""low_spp.py — why Halton looks noisier at low SPP.

Compares 2D integration error for the AA pair (dim0,1 / bases 2,3) vs the
diffuse pair (dim4,5 / bases 11,13), LCG vs Owen, averaged over 16 pixels.
If the diffuse pair is much worse at low N, that confirms the large-prime-base
clustering problem (the reason low-SPP Halton looks noisy in real renders).
"""
import sys
import numpy as np
import pandas as pd

csv = sys.argv[1] if len(sys.argv) > 1 else "profiler_output/rng_test/rng_data_current.csv"
df = pd.read_csv(csv, usecols=["pixel", "iter", "bounce", "dim", "lcg", "halton_owen"])
sub = df[df.bounce == 0]
pix = sorted(sub.pixel.unique())

def build(col, d):
    P = sub[sub.dim == d].pivot(index="iter", columns="pixel", values=col).sort_index()
    return P.values

PAIRS = {
    "AA   (dim0,1 b2,3)":  (0, 1),
    "diff (dim4,5 b11,13)": (4, 5),
}
NS = (16, 64, 256, 1024, 4096, 16384)

def pi_err(N, col, pair):
    x = build(col, pair[0])[:N]
    y = build(col, pair[1])[:N]
    e = []
    for j in range(len(pix)):
        xx = x[:N, j]
        yy = y[:N, j]
        inside = np.cumsum((xx**2 + yy**2) < 1.0)
        nn = np.arange(1, N + 1, dtype=float)
        e.append(np.abs((inside / nn) * 4.0 - np.pi)[-1])
    return float(np.mean(e))

print(f"{'N':>6}  {'pair':<20}{'LCG':>12}{'Owen':>12}  ratio(L/O)")
for pair_name, (d0, d1) in PAIRS.items():
    print(f"\n{pair_name}  (bases {[2,3,5,7,11,13,17,19,23,29][d0]} , {[2,3,5,7,11,13,17,19,23,29][d1]})")
    for N in NS:
        l = pi_err(N, "lcg", (d0, d1))
        o = pi_err(N, "halton_owen", (d0, d1))
        print(f"{N:>6}  {'':<20}{l:>12.3e}{o:>12.3e}  {l/o:>6.1f}x")
