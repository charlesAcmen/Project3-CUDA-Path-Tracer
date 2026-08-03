"""
probe_owen.py — Is the bad 2D behavior inherent to Owen+Halton, or a bug
in rng.h's owenScramble?

Compares three sequences on (base 2, base 3) Halton, N=120000:
  1. PLAIN Halton            — Halton itself (2D net baseline)
  2. PROPER digit-scrambled  — level-wise fixed digit permutations per base
  3. BURRLEY (rng.h impl)    — current owenScramble (bit-reverse + hash mix)

Decisive if (1) and (2) are net-like in 2D while (3) is random-like:
=> the algorithm (Owen/digit scrambling of Halton) is fine; the current
   implementation is the problem.
"""

import numpy as np
import pandas as pd

M32 = 0xFFFFFFFF


def utilhash(a):
    a = np.uint64(a)
    a = ((a + 0x7ed55d16) + (a << 12)) & M32
    a = ((a ^ 0xc761c23c) ^ (a >> 19)) & M32
    a = ((a + 0x165667b1) + (a << 5)) & M32
    a = ((a + 0xd3a2646c) ^ (a << 9)) & M32
    a = ((a + 0xfd7046c5) + (a << 3)) & M32
    a = ((a ^ 0xb55a4f09) ^ (a >> 16)) & M32
    return int(a)


def bit_reverse_vec(x, bits=32):
    x = x.astype(np.uint64)
    r = np.zeros_like(x)
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def radical_inverse_vec(base, n):
    inv_base = 1.0 / base
    result = np.zeros(len(n), dtype=np.float64)
    inv = inv_base
    x = n.astype(np.uint64)
    while x.any():
        d = (x % base).astype(np.float64)
        result += d * inv
        inv *= inv_base
        x //= base
    return result


# --- 3. Burley scramble (exact port of rng.h owenScramble + owenSeed) ---
def owen_scramble_vec(n, seed):
    x = bit_reverse_vec(n)
    x = (x ^ ((x * np.uint64(0x3d20adea)) & M32)) & M32
    x = (x + np.uint64(seed)) & M32
    x = (x * np.uint64((seed >> 16) | 1)) & M32
    x = (x ^ ((x * np.uint64(0x05526c56)) & M32)) & M32
    x = (x ^ ((x * np.uint64(0x53a22864)) & M32)) & M32
    return bit_reverse_vec(x)


def owen_seed(pixel, bounce, dim):
    return utilhash(utilhash(pixel + 0x9e3779b9) ^ (bounce * 0x85ebca6b) ^ (dim * 0xc2b2ae35))


# --- 2. proper digit scramble: fixed permutation per base-b digit level ---
def make_perm(seed, b):
    rng = np.random.default_rng(seed)
    perm = np.arange(b)
    for i in range(b - 1, 0, -1):
        j = int(rng.integers(0, i + 1))
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def digit_scramble_vec(base, n, seed, levels=24):
    perms = [make_perm(utilhash(seed ^ (k * 0x9E3779B9)), base) for k in range(levels)]
    inv_base = 1.0 / base
    result = np.zeros(len(n), dtype=np.float64)
    inv = inv_base
    x = n.astype(np.uint64)
    k = 0
    while x.any():
        d = (x % base).astype(np.int64)
        result += perms[k][d].astype(np.float64) * inv
        inv *= inv_base
        x //= base
        k += 1
    return result


def affine_scramble_vec(base, n, seed, levels=24):
    """Branch-free digit scramble: per-level affine permutation d -> (a*d+c) mod b.
    b prime => bijection for any a in [1,b-1], c in [0,b-1].
    Mirrors scrambledRadicalInverse() in rng.h exactly (same constants)."""
    inv_base = 1.0 / base
    result = np.zeros(len(n), dtype=np.float64)
    inv = inv_base
    x = n.astype(np.uint64)
    level = 0
    while x.any():
        d = (x % base).astype(np.int64)
        h = utilhash((int(seed) + level * 0x9E3779B9 + 0x1F123BB5) & M32)
        a = (h >> 8) % base
        c = (h >> 16) % base
        if a == 0:
            a = base - 1
        dd = (a * d + c) % base
        result += dd.astype(np.float64) * inv
        inv *= inv_base
        x //= base
        level += 1
    return result


def grid_max_dev(x, y, g, N):
    """max cell-count deviation from uniform on a g x g grid."""
    j = (np.floor(x * g).astype(np.int64) * g + np.floor(y * g).astype(np.int64))
    counts = np.bincount(j, minlength=g * g)
    return counts.max() - N / (g * g)


def pi_err(x, y):
    inside = np.cumsum((x**2 + y**2) < 1.0)
    n = np.arange(1, len(x) + 1, dtype=float)
    return np.abs((inside / n) * 4.0 - np.pi)


def getHaltonPrime(dim):
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53][dim]


N = 120000
idx = np.arange(N, dtype=np.uint64)

# validate Burley replication against the CSV (if present)
import os
if os.path.exists("profiler_output/rng_test/rng_data_large.csv"):
    df = pd.read_csv("profiler_output/rng_test/rng_data_large.csv", nrows=20000)
    df = df[(df.pixel == 0) & (df.bounce == 0)]
    probe = df[df.iter < 8]
    errs = []
    for _, r in probe.iterrows():
        mine = radical_inverse_vec(getHaltonPrime(int(r.dim)),
                                   owen_scramble_vec(np.array([np.uint64(int(r.iter))], dtype=np.uint64),
                                                     owen_seed(0, 0, int(r.dim))))[0]
        errs.append(abs(mine - r.halton_owen))
    print(f"[validate] max |python - CSV| over 8 iters x dims: {max(errs):.2e}  (should be < 1e-6)")
else:
    df = None
    print("[validate] CSV not present — skipping Burley replication check")

# build the three (dim0, dim1) point sets
sets = {}
# 1. plain Halton
sets["plain Halton"] = (radical_inverse_vec(2, idx), radical_inverse_vec(3, idx))
# 2. proper digit scramble (independent seeds per dim)
sets["digit-scrambled (proper)"] = (
    digit_scramble_vec(2, idx, owen_seed(0, 0, 0)),
    digit_scramble_vec(3, idx, owen_seed(0, 0, 1)),
)
# 3. current rng.h impl
sets["Burley (rng.h)"] = (
    radical_inverse_vec(2, owen_scramble_vec(idx, owen_seed(0, 0, 0))),
    radical_inverse_vec(3, owen_scramble_vec(idx, owen_seed(0, 0, 1))),
)
# 4. affine digit scramble (candidate fix)
sets["affine digit-scr (fix?)"] = (
    affine_scramble_vec(2, idx, owen_seed(0, 0, 0)),
    affine_scramble_vec(3, idx, owen_seed(0, 0, 1)),
)

print("\n2D grid test: max cell deviation |count - N/cells|  (net ~ 0-10, random ~ 20-100+)")
print(f"{'method':<24}" + "".join(f"{f'{g}x{g}':>10}" for g in [4, 16, 64]) + f"{'pi err@N':>12}")
for name, (x, y) in sets.items():
    row = f"{name:<24}"
    for g in [4, 16, 64]:
        row += f"{grid_max_dev(x, y, g, N):>10.1f}"
    row += f"{pi_err(x, y)[-1]:>12.2e}"
    print(row)

# ===========================================================================
# Pixel-decorrelation study — follow-up to BUG 4 fix.
# Linear digit scrambling keeps all pixels as digit-permutations of the same
# van der Corput walk, so cross-pixel correlation is high (dim0/base2 ~ +/-0.6
# .. 0.99).  Candidate fix: a MASKED random start index per (pixel, dim) so
# each pixel walks a different contiguous segment, combined with the digit
# scramble (which preserves the net).
# ===========================================================================
START_MASK = (1 << 22) - 1          # modest offset window (not the huge-window bug)


def start_of(pixel, dim, bounce=0):
    return utilhash(owen_seed(pixel, bounce, dim)) & START_MASK


def pixel_corr(f, n_pix=4):
    """max |sample corr| between all pixel pairs; f(pixel) -> dim0 sequence."""
    seqs = [f(p) for p in range(n_pix)]
    best = 0.0
    for p in range(n_pix):
        for q in range(p + 1, n_pix):
            best = max(best, abs(float(np.corrcoef(seqs[p], seqs[q])[0, 1])))
    return best


def mk_shared(p, dim):
    return radical_inverse_vec(getHaltonPrime(dim), idx)


def mk_digit(p, dim):
    return affine_scramble_vec(getHaltonPrime(dim), idx, owen_seed(p, 0, dim))


def mk_floatrot(p, dim):
    """digit scramble + per-pixel float toroidal shift (decorrelator)."""
    s = affine_scramble_vec(getHaltonPrime(dim), idx, owen_seed(p, 0, dim))
    rot = (owen_seed(p, 0, dim) & 0xFFFFFF) / float(1 << 24)
    return (s + rot) % 1.0


def mk_burley(p, dim):
    """old rng.h impl (base-2 hash scramble) — decorrelation ceiling."""
    return radical_inverse_vec(getHaltonPrime(dim), owen_scramble_vec(idx, owen_seed(p, 0, dim)))


def mk_combo(p, dim):
    """digit-scr + masked random start + float rot (full combo)."""
    base = getHaltonPrime(dim)
    n = (start_of(p, dim) + idx) % (1 << 32)
    s = affine_scramble_vec(base, n, owen_seed(p, 0, dim))
    rot = (owen_seed(p, 0, dim) & 0xFFFFFF) / float(1 << 24)
    return (s + rot) % 1.0


def mk_cp(p, dim):
    """corrected legacy CP: plain Halton + masked random start + float rot."""
    base = getHaltonPrime(dim)
    n = (start_of(p, dim) + idx) % (1 << 32)
    s = radical_inverse_vec(base, n)
    rot = (owen_seed(p, 0, dim) & 0xFFFFFF) / float(1 << 24)
    return (s + rot) % 1.0


print("\n── Pixel decorrelation vs 2D net (dim0 = base 2) ──")
print(f"{'approach':<28}{'max|corr|':>10}{'grid16':>9}{'pi err':>11}")
rows = [
    ("shared Halton (no per-pix)", mk_shared),
    ("old Burley (base-2 hash)", mk_burley),
    ("digit-scr only (current)", mk_digit),
    ("digit-scr + float rot", mk_floatrot),
    ("digit-scr+start+rot", mk_combo),
    ("corrected CP (start+rot)", mk_cp),
]
for name, mk in rows:
    mc = pixel_corr(lambda p: mk(p, 0))
    g = grid_max_dev(mk(0, 0), mk(0, 1), 16, N)
    x = mk(0, 0)
    y = mk(0, 1)
    pe = pi_err(x, y)[-1]
    print(f"{name:<28}{mc:>10.3f}{g:>9.1f}{pe:>11.2e}")

# hybrid: base-2 dim uses Burley hash (perfect decorrelation there), other
# bases use the digit scramble — does the JOINT (dim0,dim1) stay a net?
def mk_hybrid(p, dim):
    base = getHaltonPrime(dim)
    if base == 2:
        return radical_inverse_vec(base, owen_scramble_vec(idx, owen_seed(p, 0, dim)))
    return affine_scramble_vec(base, idx, owen_seed(p, 0, dim))


mc = pixel_corr(lambda p: mk_hybrid(p, 0))
g = grid_max_dev(mk_hybrid(0, 0), mk_hybrid(0, 1), 16, N)
pe = pi_err(mk_hybrid(0, 0), mk_hybrid(0, 1))[-1]
print(f"hybrid (Burley b2, digit b3+):  {mc:>10.3f}{g:>9.1f}{pe:>11.2e}")

# nested (prefix-dependent) Owen scramble — the theoretically correct version.
# At each digit level the permutation depends on the higher digits already read
# (the prefix), which is what makes it decorrelate per pixel.  Scalar, so use a
# smaller N here.
N2 = 30000
idx2 = np.arange(N2, dtype=np.uint64)


def nested_affine_vec(base, n, seed):
    out = np.empty(len(n), dtype=np.float64)
    inv = 1.0 / base
    for i, nn in enumerate(n):
        x = int(nn)
        prefix = seed
        val = 0.0
        iv = inv
        level = 0
        while x > 0:
            d = x % base
            h = utilhash((prefix + level * 0x9E3779B9 + 0x1F123BB5) & M32)
            a = (h >> 8) % base
            c = (h >> 16) % base
            if a == 0:
                a = base - 1
            val += ((a * d + c) % base) * iv
            iv *= inv
            x //= base
            prefix = utilhash((prefix ^ (d * 0x85EBCA6B)) & M32)
            level += 1
        out[i] = val
    return out


def mk_nested(p, dim):
    return nested_affine_vec(getHaltonPrime(dim), idx2, owen_seed(p, 0, dim))


def mk_nested_rot(p, dim):
    s = nested_affine_vec(getHaltonPrime(dim), idx2, owen_seed(p, 0, dim))
    rot = (owen_seed(p, 0, dim) & 0xFFFFFF) / float(1 << 24)
    return (s + rot) % 1.0


# final-design comparison (true nested Owen; index jump removed)
print(f"\n── final design (true nested Owen, no index jump, N={N2}) ──")
print(f"{'approach':<28}{'max|corr|':>10}{'grid16':>9}{'pi err':>11}")
for name, mk in [("true nested Owen", mk_nested),
                 ("true nested + float rot", mk_nested_rot)]:
    mc = pixel_corr(lambda p: mk(p, 0))
    g = grid_max_dev(mk(0, 0), mk(0, 1), 16, N2)
    pe = pi_err(mk(0, 0), mk(0, 1))[-1]
    print(f"{name:<28}{mc:>10.3f}{g:>9.1f}{pe:>11.2e}")
