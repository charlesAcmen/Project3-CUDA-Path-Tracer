#!/usr/bin/env python3
"""Generate a high-tessellation UV sphere OBJ for the black-edge diagnostic.

Output: sphere_fine.obj  (radius 0.5, origin-centered, smooth per-vertex
normals, same v/vn + 'f i//i j//j k//k' layout as sphere.obj)

Used by scenes/debug_black_2_tessellation.json to compare silhouette
black-edge width against the coarse sphere.obj (24x12 = 576 tris).
"""

import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = 0.5          # unit radius, matches sphere.obj
STACKS = 96      # rings
SEGMENTS = 96    # points per ring
OUT = os.path.join(HERE, "sphere_fine.obj")


def main():
    lines = []
    lines.append(f"# Fine UV sphere, radius {R}, origin-centered")
    lines.append(f"# {SEGMENTS} segments x {STACKS} rings = {SEGMENTS*STACKS*2} triangles")
    lines.append("# Smooth per-vertex normals")
    lines.append("")

    # Vertices: ring-major, index = ring*SEGMENTS + j  (+1 for 1-based).
    for i in range(STACKS + 1):
        phi = math.pi * i / STACKS
        y = R * math.cos(phi)
        rad = R * math.sin(phi)
        ny = math.cos(phi)
        for j in range(SEGMENTS):
            th = 2.0 * math.pi * j / SEGMENTS
            cth, sth = math.cos(th), math.sin(th)
            lines.append(f"v {rad*cth:.9f} {y:.9f} {rad*sth:.9f}")
            lines.append(f"vn {math.sin(phi)*cth:.9f} {ny:.9f} {math.sin(phi)*sth:.9f}")

    # Faces: two triangles per quad (ring i, ring i+1).
    for i in range(STACKS):
        base = i * SEGMENTS + 1          # 1-based index of ring i, j=0
        for j in range(SEGMENTS):
            jn = (j + 1) % SEGMENTS
            top0, top1 = base + j, base + jn
            bot0, bot1 = top0 + SEGMENTS, top1 + SEGMENTS
            lines.append(f"f {top0}//{top0} {top1}//{top1} {bot1}//{bot1}")
            lines.append(f"f {top0}//{top0} {bot1}//{bot1} {bot0}//{bot0}")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")

    tris = SEGMENTS * STACKS * 2
    verts = (STACKS + 1) * SEGMENTS
    print(f"wrote {OUT}: {tris} tris, {verts} verts")


if __name__ == "__main__":
    main()
