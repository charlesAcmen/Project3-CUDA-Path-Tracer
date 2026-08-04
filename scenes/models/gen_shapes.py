#!/usr/bin/env python3
"""Generate shape assets (glTF + GLB) for the path tracer's scenes/ folder.

Writes into this directory (scenes/models/):
  sphere.gltf/.glb     UV sphere, radius 1 (12 x 24 rings/segments)
  cylinder.gltf/.glb   capped cylinder, radius 1, height 2 (y in [-1, 1])
  cone.gltf/.glb       cone, base radius 1, height 2 (base y=-1, apex y=+1)
  torus.gltf/.glb      flat torus in the XZ plane, major R=1, minor r=0.3
  capsule.gltf/.glb    capsule (pill), radius 0.5, total height 2 (y in [-1, 1])

All meshes are centered at the origin, outward-wound with smooth per-vertex
normals, and use the same interleaved [pos.xyz | nrm.xyz] + uint16 index
layout as cube.gltf.  The scene JSON places them purely via TRANS/SCALE/ROTAT.

The winding of every emitted triangle is forced consistent with its vertex
normals (face normal must point along the average vertex normal), so the
resulting solids are closed and watertight-looking for the double-sided
intersection test / refraction in the renderer.

Python 3 stdlib only.  Run:  python gen_shapes.py
"""

import base64
import json
import math
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))

# glTF component types
T_FLOAT = 5126
T_U16 = 5123


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def pack_f3s(flat):
    return struct.pack("<" + "f" * len(flat), *flat)


def pack_u16(vals):
    return struct.pack("<" + "H" * len(vals), *vals)


def interleave(pos, nrm):
    """Interleaved [pos.xyz, nrm.xyz] float32, stride 24 B."""
    out = bytearray()
    for i in range(len(pos) // 3):
        out += struct.pack("<fff", *pos[3 * i:3 * i + 3])
        out += struct.pack("<fff", *nrm[3 * i:3 * i + 3])
    return bytes(out)


class Mesh:
    """Vertex + triangle accumulator with winding-fix.

    Vertices are deduplicated by exact (pos, normal) so shared corners
    (poles, seams, caps) reuse one index.  `triangle()` flips the winding
    whenever the geometric face normal opposes the average vertex normal,
    guaranteeing outward-facing normals on every emitted triangle.
    """

    def __init__(self):
        self.pos = []       # flat [x, y, z, ...]
        self.nrm = []       # flat [nx, ny, nz, ...]
        self.idx = []       # triangle vertex indices (uint16)
        self._vmap = {}     # (pos-tuple, nrm-tuple) -> vertex index

    def vertex(self, p, n):
        key = (p[0], p[1], p[2], n[0], n[1], n[2])
        i = self._vmap.get(key)
        if i is None:
            i = len(self.pos) // 3
            self.pos.extend(p)
            self.nrm.extend(n)
            self._vmap[key] = i
        return i

    def triangle(self, a, b, c):
        pa = self.pos[3 * a:3 * a + 3]
        pb = self.pos[3 * b:3 * b + 3]
        pc = self.pos[3 * c:3 * c + 3]
        e1 = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
        e2 = (pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2])
        fx = e1[1] * e2[2] - e1[2] * e2[1]
        fy = e1[2] * e2[0] - e1[0] * e2[2]
        fz = e1[0] * e2[1] - e1[1] * e2[0]
        flen = math.sqrt(fx * fx + fy * fy + fz * fz)
        if flen == 0.0:
            return  # degenerate (e.g. a pole fan already covered)
        na = self.nrm[3 * a:3 * a + 3]
        nb = self.nrm[3 * b:3 * b + 3]
        nc = self.nrm[3 * c:3 * c + 3]
        avg = (na[0] + nb[0] + nc[0],
               na[1] + nb[1] + nc[1],
               na[2] + nb[2] + nc[2])
        if fx * avg[0] + fy * avg[1] + fz * avg[2] < 0.0:
            b, c = c, b
        self.idx += [a, b, c]

    def quad(self, a, b, c, d):
        """Two triangles over corners (a,b,c,d) in ring order."""
        self.triangle(a, b, c)
        self.triangle(a, c, d)


# ---------------------------------------------------------------------
# Shape builders (all unit-sized, centered at origin)
# ---------------------------------------------------------------------

def build_sphere(mesh, stacks=12, segments=24, r=1.0):
    rings = []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        y = r * math.cos(phi)
        rad = r * math.sin(phi)
        ring = []
        for j in range(segments):
            th = 2.0 * math.pi * j / segments
            cth, sth = math.cos(th), math.sin(th)
            ring.append(mesh.vertex(
                (rad * cth, y, rad * sth),
                (math.sin(phi) * cth, math.cos(phi), math.sin(phi) * sth)))
        rings.append(ring)
    for i in range(stacks):
        top, bot = rings[i], rings[i + 1]
        for j in range(segments):
            jn = (j + 1) % segments
            mesh.quad(top[j], top[jn], bot[jn], bot[j])


def build_cylinder(mesh, segments=24, r=1.0, half=1.0):
    bot_c = mesh.vertex((0.0, -half, 0.0), (0.0, -1.0, 0.0))
    top_c = mesh.vertex((0.0, half, 0.0), (0.0, 1.0, 0.0))
    bot_ring, top_ring = [], []
    for j in range(segments):
        th = 2.0 * math.pi * j / segments
        cth, sth = math.cos(th), math.sin(th)
        bot_ring.append(mesh.vertex((r * cth, -half, r * sth), (cth, 0.0, sth)))
        top_ring.append(mesh.vertex((r * cth, half, r * sth), (cth, 0.0, sth)))
    for j in range(segments):
        jn = (j + 1) % segments
        # side wall
        mesh.quad(top_ring[j], top_ring[jn], bot_ring[jn], bot_ring[j])
        # top cap fan (facing +Y)
        mesh.triangle(top_c, top_ring[jn], top_ring[j])
        # bottom cap fan (facing -Y)
        mesh.triangle(bot_c, bot_ring[j], bot_ring[jn])


def build_cone(mesh, segments=24, r=1.0, half=1.0):
    """Apex at y=+half, base at y=-half."""
    apex = mesh.vertex((0.0, half, 0.0), (0.0, 1.0, 0.0))
    base_c = mesh.vertex((0.0, -half, 0.0), (0.0, -1.0, 0.0))
    base_ring = []
    # Side slope vector is (cos, 1/2, sin) in unit cone coords -> normalized.
    s = 1.0 / math.sqrt(1.0 + 0.5 * 0.5)
    for j in range(segments):
        th = 2.0 * math.pi * j / segments
        cth, sth = math.cos(th), math.sin(th)
        side_n = (cth * s, 0.5 * s, sth * s)
        base_ring.append(mesh.vertex((r * cth, -half, r * sth), side_n))
    for j in range(segments):
        jn = (j + 1) % segments
        mesh.triangle(apex, base_ring[jn], base_ring[j])  # side
        mesh.triangle(base_c, base_ring[j], base_ring[jn])  # base cap


def build_torus(mesh, u_seg=24, v_seg=16, R=1.0, r=0.3):
    """Flat torus in the XZ plane (donut lying flat)."""
    rings = []
    for ui in range(u_seg):
        u = 2.0 * math.pi * ui / u_seg
        cu, su = math.cos(u), math.sin(u)
        ring = []
        for vi in range(v_seg):
            v = 2.0 * math.pi * vi / v_seg
            cv, sv = math.cos(v), math.sin(v)
            rad = R + r * cv
            ring.append(mesh.vertex(
                (rad * cu, r * sv, rad * su),
                (cv * cu, sv, cv * su)))
        rings.append(ring)
    for ui in range(u_seg):
        ui2 = (ui + 1) % u_seg
        top, bot = rings[ui], rings[ui2]
        for vi in range(v_seg):
            vin = (vi + 1) % v_seg
            mesh.quad(top[vi], top[vin], bot[vin], bot[vi])


def build_capsule(mesh, segments=24, stacks=6, r=0.5, half=0.5):
    """Pill: cylinder from y=-half..half with hemispherical caps, total
    height 2*half + 2*r = 2.  Smooth normals everywhere."""
    mid_bot, mid_top = [], []
    for j in range(segments):
        th = 2.0 * math.pi * j / segments
        cth, sth = math.cos(th), math.sin(th)
        mid_bot.append(mesh.vertex((r * cth, -half, r * sth), (cth, 0.0, sth)))
        mid_top.append(mesh.vertex((r * cth, half, r * sth), (cth, 0.0, sth)))
    for j in range(segments):
        jn = (j + 1) % segments
        mesh.quad(mid_top[j], mid_top[jn], mid_bot[jn], mid_bot[j])

    def hemi(y0, sign):
        """Caps rings from a pole to the equator at height y0 (sign=+1 top,
        sign=-1 bottom)."""
        rings = []
        for i in range(stacks + 1):
            phi = (math.pi / 2.0) * i / stacks
            cp = math.cos(phi)
            sp = math.sin(phi)
            y = y0 + sign * r * cp
            ring = []
            for j in range(segments):
                th = 2.0 * math.pi * j / segments
                cth, sth = math.cos(th), math.sin(th)
                ring.append(mesh.vertex(
                    (r * sp * cth, y, r * sp * sth),
                    (sp * cth, sign * cp, sp * sth)))
            rings.append(ring)
        for i in range(stacks):
            top, bot = rings[i], rings[i + 1]
            for j in range(segments):
                jn = (j + 1) % segments
                mesh.quad(top[j], top[jn], bot[jn], bot[j])
        return rings[-1]

    hemi(half, +1.0)
    hemi(-half, -1.0)


# ---------------------------------------------------------------------
# glTF / GLB serialization
# ---------------------------------------------------------------------

def gltf_spec(mesh):
    vdata = interleave(mesh.pos, mesh.nrm)
    idata = pack_u16(mesh.idx)
    nv = len(mesh.pos) // 3
    ni = len(mesh.idx)
    mn = [min(mesh.pos[0::3]), min(mesh.pos[1::3]), min(mesh.pos[2::3])]
    mx = [max(mesh.pos[0::3]), max(mesh.pos[1::3]), max(mesh.pos[2::3])]
    return {
        "asset": {"version": "2.0"},
        "buffers": [
            {"byteLength": len(vdata),
             "uri": "data:application/octet-stream;base64," + b64(vdata)},
            {"byteLength": len(idata),
             "uri": "data:application/octet-stream;base64," + b64(idata)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(vdata),
             "byteStride": 24, "target": 34962},
            {"buffer": 1, "byteOffset": 0, "byteLength": len(idata),
             "target": 34963}],
        "accessors": [
            {"bufferView": 0, "byteOffset": 0, "componentType": T_FLOAT,
             "count": nv, "type": "VEC3", "min": mn, "max": mx},
            {"bufferView": 0, "byteOffset": 12, "componentType": T_FLOAT,
             "count": nv, "type": "VEC3"},
            {"bufferView": 1, "componentType": T_U16, "count": ni,
             "type": "SCALAR"}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2,
            "mode": 4}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }


def glb_spec(mesh):
    """Same glTF JSON but with one buffer (no uri -> GLB BIN chunk)."""
    vdata = interleave(mesh.pos, mesh.nrm)
    idata = pack_u16(mesh.idx)
    nv = len(mesh.pos) // 3
    ni = len(mesh.idx)
    mn = [min(mesh.pos[0::3]), min(mesh.pos[1::3]), min(mesh.pos[2::3])]
    mx = [max(mesh.pos[0::3]), max(mesh.pos[1::3]), max(mesh.pos[2::3])]
    return {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(vdata) + len(idata)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(vdata),
             "byteStride": 24, "target": 34962},
            {"buffer": 0, "byteOffset": len(vdata), "byteLength": len(idata),
             "target": 34963}],
        "accessors": [
            {"bufferView": 0, "byteOffset": 0, "componentType": T_FLOAT,
             "count": nv, "type": "VEC3", "min": mn, "max": mx},
            {"bufferView": 0, "byteOffset": 12, "componentType": T_FLOAT,
             "count": nv, "type": "VEC3"},
            {"bufferView": 1, "componentType": T_U16, "count": ni,
             "type": "SCALAR"}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2,
            "mode": 4}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }


def write_gltf(name, mesh):
    spec = gltf_spec(mesh)
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(spec, f, indent=2, separators=(",", ": "))


def write_glb(name, mesh):
    """Single-buffer GLB. JSON chunk padded with 0x20, BIN with 0x00."""
    vdata = interleave(mesh.pos, mesh.nrm)
    idata = pack_u16(mesh.idx)
    bin_data = vdata + idata
    json_bytes = json.dumps(glb_spec(mesh), separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_data += b"\x00" * ((4 - len(bin_data) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    glb = struct.pack("<III", 0x46546C67, 2, total)                    # 'glTF', v2
    glb += struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes  # JSON
    glb += struct.pack("<II", len(bin_data), 0x004E4942) + bin_data    # BIN\0
    with open(os.path.join(HERE, name), "wb") as f:
        f.write(glb)


def main():
    shapes = {
        "sphere": build_sphere,
        "cylinder": build_cylinder,
        "cone": build_cone,
        "torus": build_torus,
        "capsule": build_capsule,
    }
    for name, builder in shapes.items():
        mesh = Mesh()
        builder(mesh)
        assert all(0 <= i < 65536 for i in mesh.idx), name + " exceeds uint16"
        write_gltf(name + ".gltf", mesh)
        write_glb(name + ".glb", mesh)
        tris = len(mesh.idx) // 3
        print(f"{name}: {tris} tris, {len(mesh.pos)//3} verts -> {name}.gltf/.glb")


if __name__ == "__main__":
    main()
