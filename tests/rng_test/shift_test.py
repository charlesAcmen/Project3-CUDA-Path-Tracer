"""shift_test.py — how should next() write the per-pixel decorrelator?

Compares three variants of the Halton next() tail:
  no-rot      : return s                        (pure Owen)
  per-dim-rot : return (s + rot(seed)) % 1      (CURRENT rng.h; rot depends on dim)
  common-rot  : return (s + rot(seedBase)) % 1  (PROPOSED; one rot per pixel+bounce,
                shared across dims — a common toroidal shift of a net is a net)

For each: 2D net quality (grid16), pi-integral error, max cross-pixel |corr|.
"""
import numpy as np

M32 = 0xFFFFFFFF


def utilhash(a):
    a = np.uint64(a) & M32
    a = ((a + 0x7ed55d16) + (a << 12)) & M32
    a = ((a ^ 0xc761c23c) ^ (a >> 19)) & M32
    a = ((a + 0x165667b1) + (a << 5)) & M32
    a = ((a + 0xd3a2646c) ^ (a << 9)) & M32
    a = ((a + 0xfd7046c5) + (a << 3)) & M32
    a = ((a ^ 0xb55a4f09) ^ (a >> 16)) & M32
    return a


def owen_seed(pixel, bounce, dim):
    return utilhash(utilhash(np.uint64(pixel) + 0x9e3779b9)
                    ^ (np.uint64(bounce) * 0x85ebca6b)
                    ^ (np.uint64(dim) * 0xc2b2ae35))


def seed_base(pixel, bounce):
    return utilhash(utilhash(np.uint64(pixel) + 0x9e3779b9)
                    ^ (np.uint64(bounce) * 0x85ebca6b))


def nested_affine_vec(base, n, seed):
    """Exact port of rng.h owenRadicalInverse, vectorized across points."""
    n = n.astype(np.uint64).copy()
    prefix = np.full(len(n), np.uint64(seed & M32), dtype=np.uint64)
    result = np.zeros(len(n), dtype=np.float64)
    iv = 1.0 / base
    level = 0
    while n.any():
        d = (n % base).astype(np.uint64)
        h = utilhash(prefix + np.uint64(level * 0x9e3779b9 + 0x1f123bb5))
        a = (h >> 8) % base
        c = (h >> 16) % base
        a = np.where(a == 0, base - 1, a)
        result += ((a * d + c) % base).astype(np.float64) * iv
        iv /= base
        n //= base
        prefix = utilhash(prefix ^ (d * np.uint64(0x85ebca6b)))
        level += 1
    return result


def grid_max_dev(x, y, g, N):
    j = (np.floor(x * g).astype(np.int64) * g + np.floor(y * g).astype(np.int64))
    counts = np.bincount(j, minlength=g * g)
    return counts.max() - N / (g * g)


def pi_err(x, y):
    inside = np.cumsum((x**2 + y**2) < 1.0)
    n = np.arange(1, len(x) + 1, dtype=float)
    return np.abs((inside / n) * 4.0 - np.pi)[-1]


PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
N = 8192
idx = np.arange(N, dtype=np.uint64)
NPIX = 8
BOUNCE = 0

def build(variant, pixel, dim):
    base = PRIMES[dim]
    s = nested_affine_vec(base, idx, owen_seed(pixel, BOUNCE, dim))
    if variant == "no-rot":
        return s
    if variant == "per-dim-rot":
        rot = float(owen_seed(pixel, BOUNCE, dim) & 0xFFFFFF) / 16777216.0
    else:  # common-rot
        rot = float(seed_base(pixel, BOUNCE) & 0xFFFFFF) / 16777216.0
    return (s + rot) % 1.0

print(f"N={N}, pixels={NPIX}, bounce={BOUNCE}")
print(f"\n{'variant':<12}{'grid16':>8}{'pi-err':>12}{'max|cross-px corr|':>20}")
for variant in ["no-rot", "per-dim-rot", "common-rot"]:
    x = build(variant, 0, 0)
    y = build(variant, 0, 1)
    g = grid_max_dev(x, y, 16, N)
    pe = pi_err(x, y)
    # cross-pixel corr (dim0) over all pixel pairs
    seqs = [build(variant, p, 0) for p in range(NPIX)]
    corrs = []
    for i in range(NPIX):
        for j in range(i + 1, NPIX):
            corrs.append(abs(float(np.corrcoef(seqs[i], seqs[j])[0, 1])))
    corrs = np.array(corrs)
    print(f"{variant:<12}{g:>8.1f}{pe:>12.3e}{corrs.max():>16.3f}  (mean {corrs.mean():+.3f})")
