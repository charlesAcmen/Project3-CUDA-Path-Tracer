# BVH Acceleration Design

> Status: **implemented** — single world-space BVH, CPU build + GPU iterative
> traversal, wired into the renderer as the **only** intersection path.
>
> History: the first implementation was a **per-mesh, object-space** BVH (each
> `Geom` got its own subtree, the kernel looped over geoms, transformed each ray
> to object space, and per-hit transformed the normal back with
> `recordWorldNormal`).  That design and its `BvhMeta` table are gone.  This
> document describes the current single-tree design; see the git log
> (`f1cbd69` "overhaul bvh") for the removed one.

## Design Overview

**One BVH over ALL mesh triangles, in world space.**  At build time every mesh's
triangles are **baked** from object space to world space (vertices via the geom
`transform`, vertex normals via `invTranspose`, each tagged with its
`materialId`), and a single tree is built over the combined array.  The GPU
kernel therefore runs **one closest-hit traversal per ray** with no per-mesh
loop and no ray or normal transformation — the transform cost moved from
per-ray-per-bounce to once-per-triangle-at-build.

Key decisions:

| Decision | Choice | Why |
|----------|--------|-----|
| Granularity | Single tree over all meshes | Ray does 1 traversal, not G; far-plane pruning is scene-global. Win scales with mesh count (glTF models). |
| Space | World space, baked at build | Runtime needs zero transforms; normals get one `invTranspose` per triangle instead of per hit. |
| Build | Exhaustive SAH, pure host | Scenes ≤ ~1248 triangles → build is a few ms; no binning needed. |
| Traversal | Iterative, explicit stack, near-child-first | GPU has no recursion; near-first + far-plane pruning tightens `result.t` early. |
| Depth / leaf | Compile-time `kBvhMaxDepth`/`kBvhLeafSize` | Not runtime-configurable (see Configuration). |
| Toggle | None — BVH is the only path | The O(N) fallback (`computeIntersections`) was deleted with the refactor. |

## Data Structures

### `src/bvh/aabb.h` (header-only, `__host__ __device__`)

```cpp
struct AABB {
    glm::vec3 min{ FLT_MAX }; glm::vec3 max{ -FLT_MAX };
    void expand(const glm::vec3& p);
    void expand(const Triangle& t);   // v0, v1, v2
    void expand(const AABB& b);
    float surfaceArea() const;        // degenerate/NaN -> 1.0 (SAH divide guard)
};
__host__ __device__ bool intersectRayAABB(const glm::vec3& o, const glm::vec3& invDir,
    const AABB& box, float tNear, float tFar);
__host__ __device__ bool intersectRayAABBEntry(const glm::vec3& o, const glm::vec3& invDir,
    const AABB& box, float tNear, float tFar, float& tEntry);
```

- Sign-based slab test: min/max swapped per negative `invDir` component; zero
  direction → ±inf slab (immune to `0·inf = NaN`).
- `intersectRayAABBEntry` also reports `tEntry` = the near clip (`max(RAY_EPSILON, near)`),
  the **near-first ordering metric** (see Traversal).
- `intersectRayAABB` is a thin wrapper that discards `tEntry` — one slab-test
  implementation.

### `src/bvh/bvh.h` (header-only, shared host/device)

```cpp
struct BvhNode {
    AABB bounds;
    int  left;          // internal: left child index;  leaf: triangle-run offset
    int  right;         // internal: right child index; leaf: triangle count
    bool isLeaf = false;
    // accessors: childL()/childR() (internal), leafTriOffset()/leafTriCount() (leaf)
};
struct BvhBuffers {
    BvhNode*              deviceNodes   = nullptr;   // uploadToDevice/freeDevice
    std::vector<BvhNode>  hostNodes;                 // construction output
    std::vector<Triangle> hostTriangles;             // REORDERED world-space triangles
};
struct BvhHit { bool hit = false; float t = LARGE_T; glm::vec3 normal; int triIndex = -1; };

__host__ __device__ inline BvhHit traverseBvhClosest(
    const Ray& objRay, const BvhNode* nodes, const Triangle* tris, float maxT);
```

`traverseBvhClosest` — the **exact** algorithm the GPU kernel and the host test
run, validated once:

1. Loop over an explicit stack (`int stack[kMaxBvhStackDepth]`); root is always
   node 0 (`nodes == nullptr` → clean miss).
2. AABB test clipped to `[RAY_EPSILON, result.t]` — near bound skips self-hits,
   far bound is the current best distance (**far-plane pruning**).
3. Leaf → sequential scan `tris[triBase + j]` over the leaf's contiguous run
   (see Flatten); update `result` on `t < result.t` (strict).
4. Internal → compute both children's entry distance (`intersectRayAABBEntry`);
   if both hit, **push the farther child, descend the nearer immediately**.
   Near-first finds close hits early so `result.t` shrinks sooner and popped
   subtrees are pruned against it.

### `src/bvh/bvh.cu` (compiled TU — pure host build + GPU memory management)

```cpp
namespace bvh {
void buildMeshBvh(BvhBuffers& out, const std::vector<Triangle>& hostTris);
void buildSceneBvh(BvhBuffers& out,
                   const std::vector<Triangle>& hostTris,
                   const std::vector<Geom>& geoms);
void uploadToDevice(BvhBuffers& b);   // cudaMalloc + H2D copy of hostNodes
void freeDevice(BvhBuffers& b);
}
```

## Build Algorithm (CPU, pure host)

**World-space bake** (`buildSceneBvh`):

- Vertices: `bakePoint(transform, p)` = `transform * vec4(p, 1)`.
- Normals: `bakeNormal(invTranspose, n)` = `invTranspose * vec4(n, 0)` —
  **deliberately NOT re-normalized**.  `triangleIntersectionTest` interpolates the
  baked normals and normalizes the result; by linearity of the matrix
  (`invTranspose * lerp(n0,n1,n2) == lerp(invTranspose*n0, ...)`), this reproduces
  the old per-hit `recordWorldNormal` (invTranspose + normalize) exactly, with one
  transform per triangle instead of one per hit.
- Each triangle is tagged with the geom's `materialId` → **per-triangle materials**
  (a mesh can carry several, e.g. glTF).

**Exhaustive SAH build** (`buildMeshBvh`) — a few ms at ≤1248 triangles:

```
build(begin, end, depth):                       # over a per-mesh index array
  if n <= kBvhLeafSize or depth >= kBvhMaxDepth: -> leaf
  for each axis (longest extent first):
      sort the range by triangle centroid along the axis
      build prefix/suffix AABBs
      for each split k: cost = 1 + (areaL*k + areaR*(n-k)) / areaNode
  keep best (axis, split); if bestCost >= leafCost (= n): -> leaf
  else re-sort along the winning axis, partition, recurse
```

The recursion **permutes an index array in place** so a leaf's triangles always
occupy a contiguous run of it.

**Flatten pass** (post-order DFS, `flattenRecursive`): writes each leaf's
triangles into a contiguous chunk of `hostTriangles` and rewrites the leaf's
`left` to that chunk's offset.  Leaves are visited exactly once and their counts
sum to the total, so the chunks **tile `[0, N)` with no overlap and no gap** —
each leaf's triangles are a sequential memory run (cache-friendly reads).
Verified by `testBuildStructure`.

## GPU Traversal Kernel (`src/kernels/bvh_traversal.cuh` / `.cu`)

```cpp
__global__ void bvhTraverse(
    int num_paths,
    PathSegment* pathSegments,
    ShadeableIntersection* intersections,
    Triangle* deviceTriangles,
    BvhNode* deviceBvhNodes);
```

One thread per active path.  Guard: null nodes/triangles (empty scene) → miss.
Otherwise a single `traverseBvhClosest(ray, deviceBvhNodes, deviceTriangles,
LARGE_T)` over the whole tree — no `Geom*`, no `BvhMeta*`, no per-mesh loop.
On hit, writes world-space `t`, `surfaceNormal` (baked), and the **hit
triangle's** `materialId` (`deviceTriangles[hit.triIndex].materialId`).

## Wiring (`src/pathtrace.h/.cu`)

- `DeviceBuffers` holds `BvhBuffers bvh` plus a separate `deviceTriangles`.
- `pathtraceInit`:
  ```cpp
  bvh::buildSceneBvh(g_dev.bvh, scene->hostTriangles, scene->geoms);   // bake + build + flatten
  cudaMemcpy(g_dev.deviceTriangles, g_dev.bvh.hostTriangles.data(),
             n * sizeof(Triangle), cudaMemcpyHostToDevice);            // reordered world tris
  bvh::uploadToDevice(g_dev.bvh);                                      // nodes only
  ```
  `deviceTriangles` is allocated/copied once, freed only at shutdown — the leaf
  chunk offsets stay valid across the whole render.
- Bounce loop: `bvhTraverse` is the **only** intersection path (there is no O(N)
  fallback anymore).
- `pathtraceFree`: `bvh::freeDevice(g_dev.bvh)`.

## Configuration

`kBvhMaxDepth = 24`, `kBvhLeafSize = 4`, `kMaxBvhStackDepth = 64` are
compile-time constants in `src/constants.h`.  They are **not** runtime-tunable:
tree height ≤ 24 < 64 is safely below the stack capacity, so the push guard never
silently drops a node.

## Files

- `src/bvh/aabb.h`, `src/bvh/bvh.h`, `src/bvh/bvh.cu` — the acceleration structure.
- `src/kernels/bvh_traversal.cuh` / `.cu` — the only intersection kernel.
- `tests/bvh_test/` — host validation (compiles the production `bvh.cu`).

## Verification (`tests/bvh_test`)

1. **World-space bake** (`testWorldBake`, `testMultiGeomBake`) — baked vertices /
   normals / materialId match per-geom transform math.
2. **Build structure + flatten tiling** (`testBuildStructure`) — root is node 0,
   DFS reaches every node once (no cycles/orphans), and leaf chunks **partition**
   the triangle array exactly: contiguous, non-overlapping, gap-free.
3. **Traversal vs brute force** (`testTraversalVsBrute`) — `traverseBvhClosest`
   matches a brute-force O(N) scan on the baked array for hit flag, `t`, normal,
   and `materialId`, across mesh kinds and transforms (incl. non-uniform scale).
4. **AABB entry + near-first metric** (`testAabbEntry`) — entry-distance
   correctness, two-box ordering under +x/−x, origin-inside-box → `RAY_EPSILON`.
5. **Empty scene** (`testEmptyScene`) — no triangles → no tree; traversal with
   null buffers misses without touching memory.

```
bvh_test: ALL PASS
```

The main renderer build (CUDA + GL) is validated by visual inspection, per the
project's convention (no test suite in the main build).

## Edge Cases

- Empty scene / no triangles → `hostNodes` empty → `nodes == nullptr` → miss.
- Degenerate / planar AABBs → `surfaceArea() == 1.0` SAH guard (no divide-by-zero).
- Zero direction components → sign-based slab test (no NaN poison).
- Degenerate triangles → the triangle test's barycentric / normal fallbacks.
- Double-sided triangles → no back-face culling; AABBs are orientation-free.
- World-space normal = `invTranspose` at bake, normalized at hit — matches the old
  per-hit transform exactly by linearity (winding preserved for refraction).
- Near-first tie (`entryL == entryR`, e.g. ray inside both children) →
  deterministic: right child descended, left pushed.
