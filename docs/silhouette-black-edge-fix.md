# Silhouette Black-Edge Fix — Adaptive Facet-Sagitta Ray Offset

> Recorded 2026-08-07. This document captures a set of working-tree changes that were
> deliberately **discarded** so the renderer could be re-built from clean source and
> compared against the "with fix" result. Use this doc to re-apply the changes (or to
> understand exactly what was different).

**Status at record time:** the fix builds (Release), `bvh_test` ALL PASS, `refraction_test`
ALL PASSED (Schlick + Accurate). Visual result on `scenes/test_self_intersection.json`:
the thick black rim on the two spheres (glass + mirror) is gone; at extreme zoom a
sub-pixel dark fringe remains at the silhouette (see [Results](#results) for the open
question of whether that fringe is physical or numerical).

---

## 1. TL;DR

`scatterRay` previously offset every secondary ray origin by a flat `EPSILON`
(~1e-4). On a **tessellated sphere**, the facet planes are chords that dip up to
`≈ L²/8R` below the smooth surface, and `EPSILON` is far smaller than that dip. A
silhouette (grazing) ray starts *inside* the mesh → re-intersects → ping-pongs until
the bounce budget dies → **black rim**.

The fix computes, per triangle at BVH build time, how far the facet plane sits below
the smooth surface — the **sagitta** — and threads it (plus a stable **frontFace**
flag) down to `scatterRay`, which offsets by `EPSILON·(1+t) + 2·sagitta`. Flat facets
get sagitta ≈ 0 and behave exactly as before.

---

## 2. Problem

`scenes/test_self_intersection.json`: camera at (0,5,10.5) looking at (0,5,0); a glass
sphere (r=1.5) at (−2,4,0) and a mirror sphere (r=0.75) at (2,2,0) in a Cornell-style
room. Both spheres show a **black band around their silhouette** where the ray and the
surface normal are nearly perpendicular (grazing angles). At 800×800 it is clearly
visible without zooming.

## 3. Root cause analysis

### 3.1 Facet sagitta trap

- The intersection point lies on the **facet plane**, which for a tessellated sphere
  is a chord of the true curved surface.
- The smooth (Phong) surface bulges *above* the facet plane by up to
  `sagitta ≈ L²/8R` at the facet center (L = facet edge length, R = local curvature
  radius). For `sphere.obj` (24×12 UV sphere, Δ=15°):
  - glass (r=1.5): worst edge = quad **diagonal**, `sagitta ≈ 0.0253`
  - mirror (r=0.75): `sagitta ≈ 0.0127`
- A ray offset `EPSILON ≈ 1e-4` leaves the new origin *below* the smooth surface when
  the hit point sits in the facet dip → the reflected/refracted ray re-intersects the
  mesh → ping-pong → bounce budget exhausted → black.
- A hand-tuned constant (`eps = 0.02f`) was tried and worked for this scene but is
  scene-scale/tessellation dependent and fails silently elsewhere (smaller spheres,
  coarser meshes, thin geometry).

### 3.2 Degenerate `dot(normal, ray)` at grazing angles

- Old code oriented the shading normal and classified refraction enter/exit with
  `dot(normal, ray)`. At grazing incidence that dot → 0 and its **sign** is decided by
  floating-point noise, which flips the reflection into the surface and traps the path
  inside a closed mesh (silhouette acne).
- The intersection test itself has a stable side signal: the sign of the
  Möller–Trumbore determinant `a`. Because the test only accepts `|a| ≥ RAY_EPSILON`,
  that sign is exact.

## 4. The fix — overview

Two independent changes:

1. **Adaptive ray-origin offset** — a per-triangle `sagitta` (computed once at BVH
   build, in world space) is threaded through `BvhHit → ShadeableIntersection →
   shading.cu`, which assembles `rayOffset = EPSILON·(1+t) + 2·sagitta`.
2. **Stable front/back orientation** — `triangleIntersectionTest` reports
   `outFrontFace = (a > 0)`, threaded to `scatterRay`, which uses it (never
   `dot(normal, ray)`) for the shading-normal orientation and refraction enter/exit.

Data flow:

```
bvh.cu            facetSagitta()          →  Triangle.sagitta            (build-time bake)
triangle.h        outFrontFace = (a > 0)  →  from Möller–Trumbore test
bvh.h             BvhHit.frontFace/sagitta                                (leaf loop fills)
bvh_traversal.cu  → ShadeableIntersection.frontFace/sagitta
shading.cu        rayOffset = EPSILON*(1+t) + 2*sagitta  →  scatterRay(...)
interactions.cu   scatterRay uses frontFace for orientation, rayOffset for magnitude
```

---

## 5. File-by-file changes

### `src/sceneStructs.h`

`Triangle` gains a per-triangle, world-space sagitta field (computed by
`buildSceneBvh`):

```cpp
struct Triangle {
    glm::vec3 v0, v1, v2;  // three vertex positions
    glm::vec3 n0, n1, n2;  // vertex normals (smooth shading interpolation)
    int materialId = -1;   // material index; set during the world-space bake
    float sagitta = 0.0f;  // world-space facet dip below the smooth Phong surface
};
```

`ShadeableIntersection` gains:

```cpp
bool frontFace = false;   // true on front-face hit (a > 0), stable at grazing
float sagitta  = 0.0f;    // hit triangle's world-space facet dip
```

The now-dead `enum class HitSide { Outside, Inside }` was removed.

### `src/intersection/triangle.h`

`triangleIntersectionTest` signature extended with an output front-face flag, set just
after the hit `t`:

```cpp
outFrontFace = (a > 0.0f);
```

Header comment updated: `outFrontFace` is the test's own stable sign, well-conditioned
where `dot(normal, ray) → 0`.

### `src/bvh/bvh.h`

`BvhHit` gains `frontFace` and `sagitta`; the leaf loop now:

```cpp
bool triFrontFace;
if (triangleIntersectionTest(objRay, tris[triBase + j], t, triNormal, triFrontFace))
{
    if (t < result.t)
    {
        result.t         = t;
        result.normal    = triNormal;
        result.frontFace = triFrontFace;
        result.sagitta   = tris[triBase + j].sagitta;
        result.hit       = true;
        result.triIndex  = triBase + j;
    }
}
```

### `src/bvh/bvh.cu`

New helper `facetSagitta(const Triangle&)`:

```cpp
float facetSagitta(const Triangle& t)
{
    const glm::vec3 v[3] = { t.v0, t.v1, t.v2 };
    const glm::vec3 n[3] = { t.n0, t.n1, t.n2 };
    float sagitta = 0.0f;
    for (int i = 0; i < 3; i++)
    {
        const int j = (i + 1) % 3;
        const float len = glm::length(v[j] - v[i]);
        const float ni2 = glm::dot(n[i], n[i]);
        const float nj2 = glm::dot(n[j], n[j]);
        if (len < 1e-6f || ni2 < 1e-6f || nj2 < 1e-6f) continue;  // flat / degenerate
        const float cosTheta = glm::clamp(
            glm::dot(n[i], n[j]) * glm::inversesqrt(ni2) * glm::inversesqrt(nj2),
            -1.0f, 1.0f);
        sagitta = fmaxf(sagitta, len * acosf(cosTheta) * (1.0f / 8.0f));
    }
    return sagitta;
}
```

`buildSceneBvh` bake loop sets `dst.sagitta = facetSagitta(dst);` after the world-space
bake (vertices via `transform`, normals via `invTranspose`). Reasoning: for an edge of
length L between vertices whose unit normals differ by angle θ, the smooth surface
curves through the chord with local radius R ≈ L/θ, so the dip below the chord is
≈ L·θ/8. Max over the three edges bounds the dip anywhere on the facet; on a sphere this
reproduces `r(1−cos(Δ/2))` including the quad diagonals.

### `src/kernels/bvh_traversal.cu`

```cpp
intersections[path_index].t             = hit.t;
intersections[path_index].surfaceNormal = hit.normal;
intersections[path_index].frontFace     = hit.frontFace;
intersections[path_index].sagitta       = hit.sagitta;
intersections[path_index].materialId    = deviceTriangles[hit.triIndex].materialId;
```

### `src/kernels/shading.cu`

The `scatterRay` call site now computes and passes the offset + flag:

```cpp
const float rayOffset =
    EPSILON * (1.0f + intersection.t) + 2.0f * intersection.sagitta;
scatterRay(pathSegment, intersectionPoint, intersection.surfaceNormal, material,
           rngScatter, config.fresnelMode, rayOffset, intersection.frontFace);
```

(1) covers hit-point FP error (grows with travel distance); (2) covers the facet dip;
×2 gives clearance for off-center hits and non-radial interpolated normals. Flat
surfaces → the old `EPSILON` behavior.

### `src/interactions/interactions.h`

`classifyRefraction` declaration removed. `scatterRay` signature extended:

```cpp
void scatterRay(PathSegment&, glm::vec3 intersect, glm::vec3 normal,
                const Material&, RngState&, FresnelMode fresnelMode,
                float rayOffset, bool frontFace);
```

### `src/interactions/interactions.cu`

- `classifyRefraction` definition removed (dead code).
- Shading-normal orientation: `const glm::vec3 shadingNormal = frontFace ? normal : -normal;`
- Refraction enter/exit: `const bool entering = frontFace;` and
  `cosThetaI = fabsf(glm::dot(dir, normal));` (feeds only Fresnel magnitudes — fabsf is
  safe there).
- All four offset assignments changed from `EPSILON` to `rayOffset` (direction logic
  unchanged): refractive reflect `entering ? +1 : −1`, refractive refract
  `entering ? −1 : +1`, reflective `sign(dot(scatterDir, shadingNormal))`, diffuse
  `+shadingNormal * rayOffset`.

### `src/pathtrace.cu`

Include comment updated (`scatterRay, fresnel*` — `classifyRefraction` no longer exists).

### `tests/bvh_test/bvh_test.cu`

`bruteForceClosest` records `frontFace` and `sagitta` of the closest hit; the
traversal-vs-brute comparison now also checks `frontFace` equality and `sagitta`
within `1e-6`. (This test compiles the production `src/bvh/bvh.cu` directly.)

### `tests/refraction_test/refraction_test.cu`

Updated the `scatterRay` call to the new signature
(`..., 0.02f, /*frontFace=*/false`) and fixed the comment: exiting rays meet the
geometric **back** face → `frontFace == false`.

### `AGENTS.md` / `CLAUDE.md`

Intersection Testing + Scattering sections updated: `outFrontFace` flag, refraction
enter/exit off `frontFace`, new "Ray-origin offset" bullet describing
`rayOffset = EPSILON·(1+t) + 2·sagitta`.

---

## 6. Scene experiment changes (`scenes/test_self_intersection.json`)

Separate from the source fix — an experiment to determine whether the residual black
fringe is caused by rays reflecting out of the open scene. The file is **untracked**
(`??`), so `git checkout`/discard will NOT touch it. Changes:

- **New front wall** at `TRANS [0,5,13]`, `ROTAT (0,−90,0)`, `SCALE (0.01,10,10)`,
  `MATERIAL "diffuse_white"` — placed **behind** the camera (z=10.5) so the view is
  unchanged but the room is closed (a full wall at z=+5 would block the camera).
- **Room extended** to enclose the camera: floor, ceiling, left/right walls moved to
  `z = 4` and scaled to `z`-extent 18 (span z∈[−5,13]) so the z∈[5,13] band is sealed.
- **Color / white-balance adjustment** (Cornell style):
  - `diffuse_white` RGB `0.98 → 0.73`
  - left wall → new `diffuse_red` (0.63, 0.07, 0.05)
  - right wall → new `diffuse_green` (0.14, 0.45, 0.09)

Numerical trace (Python) of the sphere reflections **before** the change:

| pixel class | open room | closed room (z=13 front wall) |
|---|---|---|
| mirror/glass **silhouette** reflections | all hit back/floor/side walls, 0 MISS | same, 0 MISS |
| mirror/glass **center** reflection | **MISS (empty space) → black** | **front wall → white** |

So the front wall removes the black blob at the sphere **center** (retro-reflection of
the photographer position = empty space), but the silhouette fringe already reflects
white walls in both cases — the wall is not expected to change it.

## 7. Build & test

```
cmake --build build --config Release
build/bin/Release/cis565_path_tracer.exe scenes/test_self_intersection.json
```

Standalone tests (not wired into root build):

```
cmake -G "Visual Studio 17 2022" -A x64 -B tests/bvh_test/build tests/bvh_test
cmake --build tests/bvh_test/build --config Release
tests/bvh_test/build/Release/bvh_test.exe          # expect: ALL PASS
cmake -G "Visual Studio 17 2022" -A x64 -B tests/refraction_test/build tests/refraction_test
cmake --build tests/refraction_test/build --config Release
tests/refraction_test/build/Release/refraction_test.exe   # expect: ALL PASSED
```

## 8. Results

- `bvh_test`: ALL PASS (traversal == brute force for t, normal, materialId, frontFace,
  sagitta).
- `refraction_test`: ALL PASSED (L1/L2/L3, both Schlick and Accurate; no NaN; TIR
  deterministic reflection).
- Render of `test_self_intersection.json` (2000 iters): thick black rim **gone**; from
  a normal viewing distance no black edge. Only at extreme zoom a ~1 px dark fringe
  remains at the silhouette.
- Open question at record time: whether that fringe is (a) the physical mirror/Fresnel
  grazing reflection of the open scene, or (b) a residual numerical artifact of
  near-degenerate hits at the exact silhouette. The scene experiment (front wall,
  §6) was designed to discriminate: silhouette reflections already hit white walls in
  both configs, so if the fringe persists it is numerical, not scene-related.

## 9. Reverting / re-applying

- **Revert:** `git checkout -- src/ tests/ AGENTS.md CLAUDE.md` (the tracked files).
  `scenes/test_self_intersection.json` is untracked and unaffected.
- **Re-apply:** follow §5 file by file, or re-derive from this document's snippets.
  The two tests (§7) guard the exact behavior: traversal must match brute force on the
  new fields, and `scatterRay` must never emit a NaN for a refractive material.
