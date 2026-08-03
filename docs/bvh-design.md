# BVH Acceleration Design

> Status: **planned** — implementation deferred until the glTF mesh loader lands.
> This document records the agreed design so it survives the pause.

## Scope & Requirements

`INSTRUCTION.md` (Part 2, Performance, :six:) requires a hierarchical spatial data
structure (BVH/Octree) with:

- **CPU-side construction is sufficient** — "GPU-side construction was a final project".
- **GPU traversal must be iterative** ("Note that traversal on the GPU must be coded iteratively!").
- **Toggleable** for performance comparisons ("Make sure this is toggleable").
- **Maximum tree depth configurable from the start**.

Decisions locked with the user:

| Decision | Choice |
|----------|--------|
| Scope | CPU build + GPU iterative traversal, wired into the renderer |
| Granularity | **Per-mesh BVH in object space** (reuse the existing per-geom ray transform) |
| Baseline | **Existing O(N) kernel (`computeIntersections`) is kept untouched as the default/toggle-off path** |
| Toggle | `--bvh=0/1` (default OFF), `--bvh-depth=N`, `--bvh-leaf=N` |

Current state: the only intersection path is the GPU `computeIntersections` kernel
(`src/kernels/intersection.cuh:49`) — per-path thread, outer loop over all Geoms,
inner linear scan over `meshTriangleOffset..+count`. Triangles live in one flat
device array `g_dev.deviceTriangles`. There is no AABB/BVH code anywhere.

---

## Data Structures (new `src/bvh/`)

### `src/bvh/aabb.h` (header-only, `__host__ __device__`)

```cpp
struct AABB {
    glm::vec3 min{ FLT_MAX }; glm::vec3 max{ -FLT_MAX };
    void expand(const glm::vec3& p);
    void expand(const Triangle& t);        // v0,v1,v2
    void expand(const AABB& b);
    float surfaceArea() const;             // degenerate -> 1.0 (SAH divide-by-zero guard)
    glm::vec3 centroid() const;
};
__host__ __device__ bool intersectRayAABB(const glm::vec3& o, const glm::vec3& invDir,
    const AABB& box, float tNear, float tFar);
```

`intersectRayAABB` = sign-based slab test (min/max swapped by `invDir < 0`, immune to
`0·inf = NaN` on zero direction components). Hit iff the ray interval `[tmin, tmax]`
**overlaps** `[tNear, tFar]` — must correctly handle a ray origin *inside* the box
(`tmin < 0` but `tmax >= tNear` still hits). Caller passes `tNear = RAY_EPSILON`,
`tFar = current best closestT` (far-plane pruning).

### `src/bvh/bvh.h` (header-only)

```cpp
inline constexpr int kMaxBvhStackDepth = 64;   // explicit-stack capacity (compile-time)

struct BvhNode {
    AABB bounds;      // object space
    int  left;        // internal: left child node index; leaf: absolute triangle offset
    int  right;       // internal: right child node index; leaf: triangle count
    int  isLeaf;
};
struct BvhMeta { int rootNodeIndex = -1; int nodeCount = 0; };  // -1 = empty mesh

// Tree buffers + dedicated GPU memory management (INSTRUCTION.md).
struct BvhBuffers {
    BvhNode* deviceNodes = nullptr;  int numNodes = 0;
    BvhMeta* deviceBvhMeta = nullptr; int numGeoms = 0;
    std::vector<BvhNode>  hostNodes;      // construction output
    std::vector<BvhMeta>  hostBvhMeta;
    std::vector<Triangle> hostTriangles;  // REORDERED flat triangles (uploaded as deviceTriangles)
};

// Shared iterative closest-hit traversal — the EXACT algorithm the GPU kernel and the
// host test both use, so correctness is validated once.
__host__ __device__ bool traverseBvhClosest(const Ray& objRay, const BvhNode* nodes,
    int rootNodeIndex, const Triangle* tris, float& closestT, glm::vec3& objNormal);
```

`traverseBvhClosest` (explicit stack `int stack[kMaxBvhStackDepth]`):
1. Pop node; AABB test against `[RAY_EPSILON, closestT]`; miss -> skip subtree.
2. Leaf -> test `tris[node.left + j]` (`j < node.right`) with the **same**
   `triangleIntersectionTest`, update on `t < closestT` (strict).
3. Internal -> push children (with `sp < kMaxBvhStackDepth` guard).

`bvhMaxDepth` is clamped to `[1, 63]` so tree height can never overflow the stack.

### `src/bvh/bvh.cu` (compiled TU — pure host build + GPU memory management)

```cpp
namespace bvh {
int  buildMeshBvh(std::vector<BvhNode>& out, const std::vector<Triangle>& hostTris,
                  int triOffset, int triCount, int maxDepth, int leafSize);
void buildSceneBvh(BvhBuffers& out, const std::vector<Triangle>& hostTris,
                   const std::vector<Geom>& geoms, int maxDepth, int leafSize);
void uploadToDevice(BvhBuffers& b);   // cudaMalloc node+meta, H2D
void freeDevice(BvhBuffers& b);
}
```

---

## Build Algorithm (CPU, pure host)

**Exhaustive SAH** (Surface Area Heuristic) — scenes are ≤ ~1248 triangles, build is
a few ms. Recursive:

```
build(bounds, primIdx, depth):
  if primIdx.size() <= leafSize or depth >= maxDepth:  -> leaf
  for each axis (longest first):
      sort primIdx by triangle centroid along axis
      build prefix/suffix AABBs
      for each split k: cost = 1 + (areaL*leftCost + areaR*rightCost)/areaNode
  keep best (axis, split);  if bestCost >= leafCost (= primIdx.size()): -> leaf
  else partition and recurse left/right
```

- `surfaceArea()` is the probability weight (a random ray is more likely to hit a
  larger box, so that subtree deserves finer splits).
- Depth capped at `maxDepth` (the INSTRUCTION-tunable knob; also bounds GPU stack).
- Degenerate/planar AABBs guarded by `surfaceArea() == 1.0`.

**Flattening pass** (inside `buildSceneBvh`): after building each mesh's tree, a
post-order DFS writes each leaf's triangles into a contiguous chunk of the per-mesh
reordered copy and sets `triOffset = meshBaseOffset + chunkStart`, `triCount =
chunkSize`. Meshes concatenated -> `BvhBuffers::hostTriangles`, uploaded as
`deviceTriangles`.

The reorder is **within each mesh only**, so `Geom::meshTriangleOffset/Count` still
slice the same triangle set. The O(N) kernel computes closest-hit independently of
order, so it stays 100% correct on the reordered array — both paths share one
`deviceTriangles`.

---

## GPU Traversal Kernel (new `src/kernels/bvh_traversal.cuh`)

```cpp
__global__ void bvhTraverse(
    int depth, int num_paths,
    PathSegment* pathSegments, Geom* geoms, int geoms_size,
    ShadeableIntersection* intersections,
    Triangle* deviceTriangles,
    BvhNode* deviceBvhNodes, BvhMeta* deviceBvhMeta);
```

Structure mirrors `computeIntersections`: per-path thread; outer `for i in geoms`;
skip `deviceBvhMeta[i].rootNodeIndex < 0`; transform ray to object space; call
`traverseBvhClosest`; on hit nearer than current best record `hit_geom_index` and the
world-space normal. Write-out identical (miss `t = -1`; else `t`, `materialId =
geoms[i].materialid`, `surfaceNormal`).

Shared helpers (added to `src/intersection/intersections.h`, `__device__ inline`):
`transformRayToObjectSpace` and `recordWorldNormal` — the latter replicates the O(N)
normal path exactly: `multiplyMV(geom.invTranspose, vec4(objNormal,0))`, normalize,
fallback `(0,1,0)` if NaN / `len2 < RAY_EPSILON`.

**Invariant for bit-identical `--bvh=0` vs `--bvh=1` renders**: same object-space
transform, same `triangleIntersectionTest`, same `t` (object-space parameter, no
renormalization), same closest-hit bookkeeping (a new geom must be *strictly* nearer),
same normal fallback. AABB pruning never changes the set of hit triangles — the only
possible difference is an exact-tie on a shared edge (measure-zero, invisible).

---

## Wiring (`src/pathtrace.h/.cu`)

- `DeviceBuffers` gains `BvhBuffers bvh;`.
- `pathtraceInit` (inside the triangle-upload block, `pathtrace.cu:139-147`): always
  `bvh::buildSceneBvh(...)` (cheap), upload `g_dev.bvh.hostTriangles` as
  `deviceTriangles`, then `bvh::uploadToDevice(g_dev.bvh)`.
- `pathtraceFree`: `bvh::freeDevice(g_dev.bvh)`.
- Bounce loop (`pathtrace.cu:292-297`):
  ```cpp
  const bool useBvh = g_opts.bvh.enabled && g_dev.bvh.deviceNodes != nullptr;
  if (useBvh) { LAUNCH_KERNEL_AUTO(bvhTraverse, ...); }
  else        { LAUNCH_KERNEL_AUTO(computeIntersections, ...); }
  ```
  Both share `ProfilerOp::ComputeIntersections` so the benchmark CSV column is
  unchanged and A/B compares directly.

---

## Configuration (`src/config/config.h/.cpp`)

`BvhConfig` (pattern of `BloomConfig`: `clamp()` + kMin/kMax, `#include "bvh/bvh.h"`
for `kMaxBvhStackDepth`):

```cpp
struct BvhConfig {
    bool enabled = false;    // default OFF -> O(N) baseline
    int  maxDepth = 24;      // clamp [1, 63]
    int  leafSize = 4;       // clamp [1, 64]
    void clamp();
};
```

CLI: `--bvh=0/1`, `--bvh-depth=N`, `--bvh-leaf=N`. JSON: nested `"bvh": {...}`.
Runtime setters/getters `setBvhEnabled/getBvhEnabled` etc. in `pathtrace.h/.cu`
(mutate `g_opts`, live kernel-switch). ImGui section: `Enable BVH traversal` checkbox
(live); `Max depth / Leaf size` shown as read-only (build-time, restart to change).

---

## Files

**New**: `src/bvh/aabb.h`, `src/bvh/bvh.h`, `src/bvh/bvh.cu`,
`src/kernels/bvh_traversal.cuh`, `tests/bvh_test/` (host build+traverse vs brute-force),
`docs/bvh-design.md` (this file).

**Modified**: `src/intersection/triangle.h` (`__device__` -> `__host__ __device__`),
`src/intersection/intersections.h` (two helpers), `src/kernels/intersection.cuh`
(minimal refactor to call the helpers — extraction, not deletion),
`src/pathtrace.h/.cu`, `src/config/config.h/.cpp`, `src/main.cpp` (ImGui),
`CMakeLists.txt` (`sources` += `src/bvh/bvh.cu`; `headers` += the two bvh headers).

---

## Verification

1. `tests/bvh_test` (host): build BVH for synthetic meshes (axis-aligned cube, random
   cloud, degenerate set) over `maxDepth ∈ {8,24} × leafSize ∈ {1,4}`; deterministic ray
   batch; assert `traverseBvhClosest` == brute-force O(N) `t`/normal per ray.
2. **GPU A/B**: `--rng=0 --sort=0 --compact=0 --bvh=0` vs `--bvh=1` on the same scene
   must produce byte-identical saved PNGs (intersection is bit-identical).
3. `--benchmark --warmup=2` with/without `--bvh`; compare `ComputeIntersections` column.
   **Honest expectation**: at ≤1248 triangles the O(N) scan may already be fast enough;
   the win appears on large meshes. That is a legitimate analysis result.

---

## Edge Cases

- Empty mesh / no triangles: `rootNodeIndex = -1`, kernel skips; null node buffer falls
  back to O(N).
- Degenerate AABBs (planar/zero-volume): `surfaceArea()==1.0` SAH guard.
- Zero direction components: sign-based slab test.
- Double-sided triangles: no back-face culling added; AABBs are orientation-free.
- Normal output: exact copy of O(N) fallback path.
- Exact ties: measure-zero, not constructed in tests.
- Stack overflow: build-time depth clamp `<= 63`; runtime push guard.
