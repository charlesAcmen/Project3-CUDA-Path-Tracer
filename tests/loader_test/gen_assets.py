#!/usr/bin/env python3
"""Generate edge-case mesh assets for the scene-loader test suite.

Writes into tests/loader_test/assets/:
  cube_embed.gltf     indexed unit cube, embedded base64 buffers      -> 12 tris
  cube_glb.glb        same cube as a GLB binary container            -> 12 tris
  cube_nonindexed.gltf 36 corner verts, NO indices                   -> 12 tris
  cube_nonormal.gltf   indexed cube, NO NORMAL accessor              -> 12 tris (face-normal fallback)
  mode_line.gltf       points primitive (mode 0) + cube (mode 4)     -> 12 tris (points skipped)
  no_position.gltf     primitive with only NORMAL                     -> 0 tris (skipped)
  no_mode.gltf         triangle with no "mode" (cgltf defaults)      -> 1 tri
  multi_mesh.gltf      2 meshes sharing buffers, byte offsets        -> 3 tris (2 + 1)
  oob_index.gltf       cube + triangle referencing vertex 999        -> 12 tris (OOB skipped)
  index_not_mult3.gltf 8 indices (not a multiple of 3)               -> 0 tris (skipped)
  degenerate.gltf      3 identical vertices, no NORMAL               -> 1 tri (no crash)
  zero_normal.gltf     triangle with all-zero NORMAL accessor        -> 1 tri (face-normal fallback)
  external_bin.gltf + external.bin  triangle via external .bin       -> 1 tri
  node_transform.gltf  triangle under a translated node              -> 1 tri (verts stay raw)
  norm_i8.gltf         triangle, normalized int8 POSITION            -> 1 tri
  normal_mismatch.gltf NORMAL count 2 != POSITION count 4            -> {-1,0} (cgltf rejects)
  tri_vn.obj           3 v + 3 vn, one face                          -> 1 tri
  tri_no_vn.obj        3 v, no vn                                    -> 1 tri (face-normal fallback)
  quad.obj             one quad (tinyobj triangulates)               -> 2 tris
  degenerate.obj       3 identical v                                 -> 1 tri (no crash)
  oob.obj              valid tri + quad referencing OOB verts        -> 1 tri (tinyobj skips quad)

"Expected" counts are what the loader must produce AFTER the robustness
fixes (see src/scene/scene_loader.cpp loadGLTF).  Python 3 stdlib only.
"""

import base64
import json
import os
import struct

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)

# glTF component types
T_FLOAT = 5126
T_U16 = 5123
T_I8 = 5120


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def write_gltf(name: str, spec: dict) -> None:
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(spec, f, indent=2, separators=(",", ": "))


def write_glb(name: str, buffers, spec: dict) -> None:
    """Single-buffer GLB.  JSON chunk padded with 0x20, BIN with 0x00;
    each chunk's length field includes its pad bytes (per the spec)."""
    bin_data = b"".join(buffers)
    json_bytes = json.dumps(spec, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_data += b"\x00" * ((4 - len(bin_data) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    glb = struct.pack("<III", 0x46546C67, 2, total)                  # 'glTF', v2
    glb += struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes  # JSON
    glb += struct.pack("<II", len(bin_data), 0x004E4942) + bin_data  # BIN\0
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(glb)


def pack_f3s(flat: list) -> bytes:
    """flat = [x,y,z, ...] floats -> big blob of float32s."""
    return struct.pack("<" + "f" * len(flat), *flat)


def pack_u16(vals) -> bytes:
    return struct.pack("<" + "H" * len(vals), *vals)


def interleave(pos: list, nrm: list) -> bytes:
    """Interleaved [pos.xyz, nrm.xyz] float32, stride 24 B."""
    out = bytearray()
    for i in range(len(pos) // 3):
        out += struct.pack("<fff", pos[3 * i], pos[3 * i + 1], pos[3 * i + 2])
        out += struct.pack("<fff", nrm[3 * i], nrm[3 * i + 1], nrm[3 * i + 2])
    return bytes(out)


# ---------------------------------------------------------------------
# Unit cube: 24 unique verts (per-face normals) + 36 uint16 indices.
# ---------------------------------------------------------------------
faces = [
    ([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)], (0, 0, -1)),  # -Z
    ([(-1, -1, 1), (-1, 1, 1), (1, 1, 1), (1, -1, 1)], (0, 0, 1)),       # +Z
    ([(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)], (-1, 0, 0)),  # -X
    ([(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)], (1, 0, 0)),       # +X
    ([(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)], (0, -1, 0)),  # -Y
    ([(-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)], (0, 1, 0)),       # +Y
]

cube_pos = []   # 24 x 3
cube_nrm = []   # 24 x 3
cube_idx = []   # 36 uint16
base = 0
for verts, n in faces:
    for v in verts:
        cube_pos.extend(v)
        cube_nrm.extend(n)
    cube_idx += [base + 0, base + 1, base + 2, base + 0, base + 2, base + 3]
    base += 4

assert len(cube_pos) == 72 and len(cube_nrm) == 72 and len(cube_idx) == 36


def cube_corners():
    """36 corner vertices (flat pos/nrm lists) for a NON-indexed cube."""
    pos, nrm = [], []
    for verts, n in faces:
        for vi in (0, 1, 2, 0, 2, 3):
            pos.extend(verts[vi])
            nrm.extend(n)
    assert len(pos) == 108 and len(nrm) == 108
    return pos, nrm


def cube_json(buffer_specs, buffer_views, with_normals=True):
    """JSON body of the indexed cube.  buffer_specs: [{byteLength, uri?}]."""
    accessors = [
        {"bufferView": 0, "byteOffset": 0, "componentType": T_FLOAT, "count": 24,
         "type": "VEC3", "min": [-1, -1, -1], "max": [1, 1, 1]},   # POSITION
    ]
    if with_normals:
        accessors.append({"bufferView": 0, "byteOffset": 12, "componentType": T_FLOAT,
                          "count": 24, "type": "VEC3"})           # NORMAL
    accessors.append({"bufferView": 1, "componentType": T_U16, "count": 36,
                      "type": "SCALAR"})                          # INDICES
    attrs = {"POSITION": 0}
    if with_normals:
        attrs["NORMAL"] = 1
    # The INDICES accessor lands AFTER any NORMAL, so its index depends on
    # with_normals: (pos, nrm, idx) -> 2, (pos, idx) -> 1.
    idx_slot = 2 if with_normals else 1
    return {
        "asset": {"version": "2.0"},
        "buffers": buffer_specs,
        "bufferViews": buffer_views,
        "accessors": accessors,
        "meshes": [{"primitives": [{"attributes": attrs,
                                    "indices": idx_slot,
                                    "mode": 4}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }


def simple_tri(pos, with_glb=False):
    """A single XY triangle (no normals): (0,0,0),(1,0,0),(0,1,0)."""
    pdata = pack_f3s([x for v in pos for x in v])
    spec = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(pdata), "uri": "data:application/octet-stream;base64," + b64(pdata)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pdata), "target": 34962}],
        "accessors": [{"bufferView": 0, "byteOffset": 0, "componentType": T_FLOAT,
                       "count": len(pos), "type": "VEC3"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": 4}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    return spec, pdata


# ---------------------------------------------------------------------
# 1. cube_embed.gltf — indexed cube, 2 embedded base64 buffers
# ---------------------------------------------------------------------
cube_vdata = interleave(cube_pos, cube_nrm)          # 576 B
cube_idata = pack_u16(cube_idx)                       # 72 B
write_gltf("cube_embed.gltf", cube_json(
    [{"byteLength": len(cube_vdata), "uri": "data:application/octet-stream;base64," + b64(cube_vdata)},
     {"byteLength": len(cube_idata), "uri": "data:application/octet-stream;base64," + b64(cube_idata)}],
    [{"buffer": 0, "byteOffset": 0, "byteLength": len(cube_vdata), "byteStride": 24, "target": 34962},
     {"buffer": 1, "byteOffset": 0, "byteLength": len(cube_idata), "target": 34963}]))

# ---------------------------------------------------------------------
# 2. cube_glb.glb — same cube as a GLB binary container
# ---------------------------------------------------------------------
write_glb("cube_glb.glb", [cube_vdata, cube_idata], cube_json(
    [{"byteLength": len(cube_vdata) + len(cube_idata)}],  # no uri -> GLB BIN chunk
    [{"buffer": 0, "byteOffset": 0, "byteLength": len(cube_vdata), "byteStride": 24, "target": 34962},
     {"buffer": 0, "byteOffset": len(cube_vdata), "byteLength": len(cube_idata), "target": 34963}]))

# ---------------------------------------------------------------------
# 3. cube_nonindexed.gltf — 36 corner verts, no indices
# ---------------------------------------------------------------------
npos, nnrm = cube_corners()
nvdata = interleave(npos, nnrm)                        # 864 B
write_gltf("cube_nonindexed.gltf", {
    "asset": {"version": "2.0"},
    "buffers": [{"byteLength": len(nvdata), "uri": "data:application/octet-stream;base64," + b64(nvdata)}],
    "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(nvdata), "byteStride": 24, "target": 34962}],
    "accessors": [
        {"bufferView": 0, "byteOffset": 0, "componentType": T_FLOAT, "count": 36, "type": "VEC3"},
        {"bufferView": 0, "byteOffset": 12, "componentType": T_FLOAT, "count": 36, "type": "VEC3"}],
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "mode": 4}]}],
})

# ---------------------------------------------------------------------
# 4. cube_nonormal.gltf — indexed cube, NO NORMAL accessor
# ---------------------------------------------------------------------
ppos = pack_f3s(cube_pos)                               # 288 B
write_gltf("cube_nonormal.gltf", {
    "asset": {"version": "2.0"},
    "buffers": [{"byteLength": len(ppos), "uri": "data:application/octet-stream;base64," + b64(ppos)},
                {"byteLength": len(cube_idata), "uri": "data:application/octet-stream;base64," + b64(cube_idata)}],
    "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(ppos), "byteStride": 12, "target": 34962},
                    {"buffer": 1, "byteOffset": 0, "byteLength": len(cube_idata), "target": 34963}],
    "accessors": [
        {"bufferView": 0, "byteOffset": 0, "componentType": T_FLOAT, "count": 24, "type": "VEC3"},
        {"bufferView": 1, "componentType": T_U16, "count": 36, "type": "SCALAR"}],
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
})

# ---------------------------------------------------------------------
# 5. mode_line.gltf — points primitive (mode 1) + cube (mode 4)
# ---------------------------------------------------------------------
pts = pack_f3s([0, 0, 0, 1, 1, 1])                      # 2 verts, 24 B
write_gltf("mode_line.gltf", {
    "asset": {"version": "2.0"},
    "buffers": [
        {"byteLength": len(cube_vdata), "uri": "data:application/octet-stream;base64," + b64(cube_vdata)},
        {"byteLength": len(pts), "uri": "data:application/octet-stream;base64," + b64(pts)},
        {"byteLength": len(cube_idata), "uri": "data:application/octet-stream;base64," + b64(cube_idata)}],
    "bufferViews": [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(cube_vdata), "byteStride": 24, "target": 34962},
        {"buffer": 1, "byteOffset": 0, "byteLength": len(pts), "target": 34962},
        {"buffer": 2, "byteOffset": 0, "byteLength": len(cube_idata), "target": 34963}],
    "accessors": [
        {"bufferView": 0, "byteOffset": 0, "componentType": T_FLOAT, "count": 24, "type": "VEC3"},
        {"bufferView": 0, "byteOffset": 12, "componentType": T_FLOAT, "count": 24, "type": "VEC3"},
        {"bufferView": 1, "componentType": T_FLOAT, "count": 2, "type": "VEC3"},
        {"bufferView": 2, "componentType": T_U16, "count": 36, "type": "SCALAR"}],
    "meshes": [{"primitives": [
        {"attributes": {"POSITION": 2}, "mode": 0},   # 0 = POINTS
        {"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 3, "mode": 4}]}],
})

# ---------------------------------------------------------------------
# 6. no_position.gltf — primitive with only a NORMAL attribute
# ---------------------------------------------------------------------
nn = pack_f3s([0, 0, 1] * 3)                            # 36 B
write_gltf("no_position.gltf", {
    "asset": {"version": "2.0"},
    "buffers": [{"byteLength": len(nn), "uri": "data:application/octet-stream;base64," + b64(nn)}],
    "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(nn), "target": 34962}],
    "accessors": [{"bufferView": 0, "componentType": T_FLOAT, "count": 3, "type": "VEC3"}],
    "meshes": [{"primitives": [{"attributes": {"NORMAL": 0}, "mode": 4}]}],
})

# ---------------------------------------------------------------------
# 7. no_mode.gltf — triangle with no "mode" (cgltf defaults to triangles)
# ---------------------------------------------------------------------
spec, _ = simple_tri([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
del spec["meshes"][0]["primitives"][0]["mode"]
write_gltf("no_mode.gltf", spec)

# ---------------------------------------------------------------------
# 8. multi_mesh.gltf — 2 meshes sharing buffers (byteOffset-indexed)
# ---------------------------------------------------------------------
mpos = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]             # XY quad, 4 verts
mnrm = [0, 0, 1] * 4
mvdata = interleave(mpos, mnrm)                         # 96 B
midx = pack_u16([0, 1, 2, 0, 2, 3, 0, 1, 3])            # 9 indices, 18 B
write_gltf("multi_mesh.gltf", {
    "asset": {"version": "2.0"},
    "buffers": [
        {"byteLength": len(mvdata), "uri": "data:application/octet-stream;base64," + b64(mvdata)},
        {"byteLength": len(midx), "uri": "data:application/octet-stream;base64," + b64(midx)}],
    "bufferViews": [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(mvdata), "byteStride": 24, "target": 34962},
        {"buffer": 1, "byteOffset": 0, "byteLength": len(midx), "target": 34963}],
    "accessors": [
        {"bufferView": 0, "byteOffset": 0, "componentType": T_FLOAT, "count": 4, "type": "VEC3"},
        {"bufferView": 0, "byteOffset": 12, "componentType": T_FLOAT, "count": 4, "type": "VEC3"},
        {"bufferView": 1, "byteOffset": 0, "componentType": T_U16, "count": 6, "type": "SCALAR"},
        {"bufferView": 1, "byteOffset": 12, "componentType": T_U16, "count": 3, "type": "SCALAR"}],
    "meshes": [
        {"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "mode": 4}]},
        {"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 3, "mode": 4}]}],
})

# ---------------------------------------------------------------------
# 9. oob_index.gltf — cube + a triangle referencing vertex 999 (OOB)
# ---------------------------------------------------------------------
oob_idx = pack_u16(cube_idx + [999, 1, 2])              # 39 indices, 78 B
write_gltf("oob_index.gltf", {
    "asset": {"version": "2.0"},
    "buffers": [
        {"byteLength": len(cube_vdata), "uri": "data:application/octet-stream;base64," + b64(cube_vdata)},
        {"byteLength": len(oob_idx), "uri": "data:application/octet-stream;base64," + b64(oob_idx)}],
    "bufferViews": [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(cube_vdata), "byteStride": 24, "target": 34962},
        {"buffer": 1, "byteOffset": 0, "byteLength": len(oob_idx), "target": 34963}],
    "accessors": [
        {"bufferView": 0, "byteOffset": 0, "componentType": T_FLOAT, "count": 24, "type": "VEC3"},
        {"bufferView": 0, "byteOffset": 12, "componentType": T_FLOAT, "count": 24, "type": "VEC3"},
        {"bufferView": 1, "componentType": T_U16, "count": 39, "type": "SCALAR"}],
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "mode": 4}]}],
})

# ---------------------------------------------------------------------
# 10. index_not_mult3.gltf — 8 indices (not a multiple of 3)
# ---------------------------------------------------------------------
bad8 = pack_u16([0, 1, 2, 0, 1, 2, 0, 1])               # 8 indices, 16 B
tpos = pack_f3s([0, 0, 0, 1, 0, 0, 0, 1, 0])            # 36 B
write_gltf("index_not_mult3.gltf", {
    "asset": {"version": "2.0"},
    "buffers": [
        {"byteLength": len(tpos), "uri": "data:application/octet-stream;base64," + b64(tpos)},
        {"byteLength": len(bad8), "uri": "data:application/octet-stream;base64," + b64(bad8)}],
    "bufferViews": [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(tpos), "target": 34962},
        {"buffer": 1, "byteOffset": 0, "byteLength": len(bad8), "target": 34963}],
    "accessors": [
        {"bufferView": 0, "componentType": T_FLOAT, "count": 3, "type": "VEC3"},
        {"bufferView": 1, "componentType": T_U16, "count": 8, "type": "SCALAR"}],
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
})

# ---------------------------------------------------------------------
# 11. degenerate.gltf — 3 identical vertices, no NORMAL
# ---------------------------------------------------------------------
spec, _ = simple_tri([(0, 0, 0), (0, 0, 0), (0, 0, 0)])
write_gltf("degenerate.gltf", spec)

# ---------------------------------------------------------------------
# 12. zero_normal.gltf — triangle, all-zero NORMAL accessor
# ---------------------------------------------------------------------
zpos = [0, 0, 0, 1, 0, 0, 0, 1, 0]
znrm = [0, 0, 0] * 3
zvdata = interleave(zpos, znrm)                         # 72 B
write_gltf("zero_normal.gltf", {
    "asset": {"version": "2.0"},
    "buffers": [{"byteLength": len(zvdata), "uri": "data:application/octet-stream;base64," + b64(zvdata)}],
    "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(zvdata), "byteStride": 24, "target": 34962}],
    "accessors": [
        {"bufferView": 0, "byteOffset": 0, "componentType": T_FLOAT, "count": 3, "type": "VEC3"},
        {"bufferView": 0, "byteOffset": 12, "componentType": T_FLOAT, "count": 3, "type": "VEC3"}],
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "mode": 4}]}],
})

# ---------------------------------------------------------------------
# 13. external_bin.gltf + external.bin — buffer via external file
# ---------------------------------------------------------------------
ext_pos = pack_f3s([0, 0, 0, 1, 0, 0, 0, 1, 0])         # 36 B
with open(os.path.join(OUT, "external.bin"), "wb") as f:
    f.write(ext_pos)
write_gltf("external_bin.gltf", {
    "asset": {"version": "2.0"},
    "buffers": [{"byteLength": len(ext_pos), "uri": "external.bin"}],
    "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(ext_pos), "target": 34962}],
    "accessors": [{"bufferView": 0, "componentType": T_FLOAT, "count": 3, "type": "VEC3"}],
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": 4}]}],
})

# ---------------------------------------------------------------------
# 14. node_transform.gltf — triangle under a translated node
# ---------------------------------------------------------------------
spec, _ = simple_tri([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
# Column-major translation by (5,5,5).  The loader iterates data->meshes
# directly and ignores nodes, so loaded vertices must stay RAW.
spec["nodes"] = [{"mesh": 0, "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 5, 5, 1]}]
write_gltf("node_transform.gltf", spec)

# ---------------------------------------------------------------------
# 15. norm_i8.gltf — POSITION as normalized int8 (0,0,0),(127,0,0),(0,127,0)
# ---------------------------------------------------------------------
i8data = bytes([0, 0, 0, 127, 0, 0, 0, 127, 0])         # 9 B
write_gltf("norm_i8.gltf", {
    "asset": {"version": "2.0"},
    "buffers": [{"byteLength": 9, "uri": "data:application/octet-stream;base64," + b64(i8data)}],
    "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 9, "target": 34962}],
    "accessors": [{"bufferView": 0, "byteOffset": 0, "componentType": T_I8, "count": 3,
                   "type": "VEC3", "normalized": True}],
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": 4}]}],
})

# ---------------------------------------------------------------------
# 16. normal_mismatch.gltf — NORMAL count (2) != POSITION count (4).
#     cgltf_validate requires every attribute to share attributes[0]'s count,
#     so the whole file is rejected before the loader unpacks anything.
# ---------------------------------------------------------------------
mm_pos = pack_f3s([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0])   # 4 verts, 48 B
mm_nrm = pack_f3s([0, 0, 1, 0, 0, 1])                      # 2 verts, 24 B
write_gltf("normal_mismatch.gltf", {
    "asset": {"version": "2.0"},
    "buffers": [
        {"byteLength": len(mm_pos), "uri": "data:application/octet-stream;base64," + b64(mm_pos)},
        {"byteLength": len(mm_nrm), "uri": "data:application/octet-stream;base64," + b64(mm_nrm)}],
    "bufferViews": [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(mm_pos), "target": 34962},
        {"buffer": 1, "byteOffset": 0, "byteLength": len(mm_nrm), "target": 34962}],
    "accessors": [
        {"bufferView": 0, "componentType": T_FLOAT, "count": 4, "type": "VEC3"},
        {"bufferView": 1, "componentType": T_FLOAT, "count": 2, "type": "VEC3"}],
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "mode": 4}]}],
})

# ---------------------------------------------------------------------
# OBJ assets (hand-written content, tinyobjloader)
# ---------------------------------------------------------------------
OBJ = {
    "tri_vn.obj": """v 0 0 0
v 1 0 0
v 0 1 0
vn 0 0 1
vn 0 0 1
vn 0 0 1
f 1//1 2//2 3//3
""",
    "tri_no_vn.obj": """v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
""",
    # tinyobjloader triangulates the quad (default triangulate=true).
    "quad.obj": """v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
""",
    "degenerate.obj": """v 0 0 0
v 0 0 0
v 0 0 0
f 1 2 3
""",
    # The quad references vertices 4..7 (only 3 exist) — tinyobj warns+skips.
    "oob.obj": """v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
f 5 6 7 8
""",
}
for name, content in OBJ.items():
    with open(os.path.join(OUT, name), "w") as f:
        f.write(content)

print("wrote %d assets to %s" % (len(os.listdir(OUT)), OUT))
