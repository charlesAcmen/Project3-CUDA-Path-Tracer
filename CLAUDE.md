# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

- **Windows (Visual Studio):** Generate with `cmake -B build`, then build with `cmake --build build --config Release` (or open the generated `.sln`).
- **Linux/WSL (Make):** `make` or `make Release` builds with CMake into `build/`.
- **Run:** `build/bin/cis565_path_tracer <scenefile.json>` — e.g. `build/bin/cis565_path_tracer scenes/cornell.json`. In VS, set `Debugging > Command Arguments` to `../scenes/cornell.json`.
- **Clean:** `make clean`
- **No test suite in the main build** — validation is visual inspection of the rendered output. Standalone tests live under `tests/` (`bvh_test`, `config_test`, `loader_test`, `rng_test`, `refraction_test`) but are NOT wired into the root CMakeLists. Each has its own CMake project; build with `cmake -G "Visual Studio 17 2022" -A x64 -B <test>/build <test>` then `cmake --build <test>/build --config Release`. `bvh_test` compiles the production `src/bvh/bvh.cu` directly (no duplicate copies) and validates the world-space bake, single-tree structure, near-first `traverseBvhClosest` vs brute force, and `intersectRayAABBEntry`.

## Runtime Configuration (three-layer priority)

`CLI flags > config.local.json > code defaults`, handled by `src/config/config.cpp`/`config.h` through the `appConfig()` singleton. Defaults: `compactMethod = SharedMem`, `sortByMaterial = false`, `rngMode = LCG`, `autoSave = true`. All runtime renderer settings live in the `AppConfig` singleton. BVH depth/leaf size are **compile-time constants** `kBvhMaxDepth`/`kBvhLeafSize` in `src/constants.h` — not runtime-configurable.

In `config.local.json`, enum fields (`compactMethod`, `rngMode`) accept either the **case-insensitive name** (`"SharedMem"`, `"lcg"`) or the **legacy integer** (`3`, `0`); unknown names fall back to the current value with a warning. CLI flags stay numeric (`--compact=N`, `--rng=N`).

| Flag | Meaning |
|------|---------|
| `--compact=N` | Compaction: 0=off, 1=global-mem scan, 2=Thrust `copy_if`, 3=shared-mem scan (default) |
| `--sort=N` | Material sorting 0/1 (default off) |
| `--rng=N` | 0=LCG (default), 1=scrambled Halton |
| `--benchmark` | Enable profiler CSV output to `profiler_output/<scene>_<timestamp>/` |
| `--warmup=N` | Profiler warmup iterations (default 3) |
| `--save` / `--save-at=N1,N2,...` | Save final / checkpoint images |
| `--config=PATH` | Load config JSON (default `config.local.json` in CWD) |
| `-h`, `--help` | Help text |

Many settings can also be toggled live via the ImGui overlay (`RenderImGui` in `main.cpp`); those mutate the same `g_opts` singleton through setters declared in `pathtrace.h`.

## Project Architecture

CUDA-based Monte Carlo path tracer (CIS 565 at UPenn). **All geometry is triangulated at load time** — there are no sphere/cube primitives anymore; every `Geom` is a mesh referencing a slice of a flat triangle array. Rendering is CUDA-GL interop with a live ImGui overlay.

### Source Layout

```
src/
├── main.cpp                  # GLFW/GL window, camera controls, ImGui, render loop → pathtrace()
├── pathtrace.cu / .h         # GPU pipeline + runtime getters/setters; includes kernels/ + pipeline/
├── config/config.cpp / .h    # AppConfig singleton, JSON + CLI merge, startup help/summary
├── scene/scene_loader.cpp / .h    # JSON scene dispatch (Materials/Objects/Camera); the only public entry
├── scene/obj_loader.cpp           # OBJ mesh loading (tinyobjloader impl) + shared makeTri
├── scene/gltf_loader.cpp          # glTF 2.0 mesh loading (cgltf impl): scene-graph walk + texture auto-load
├── scene/texture_loader.cpp       # PNG/JPG decode (stb_image header only) into Scene::textures
├── scene/loader_internal.h        # cross-TU loader helpers (makeTri/loadOBJ/loadGLTF/loadTexture…)
├── scene/scene.cpp / .h           # Scene container + computeSceneStats
├── sceneStructs.h            # All shared data structures (Ray, Geom, Material, PathSegment, …)
├── constants.h               # PI, EPSILON, RAY_EPSILON, RR_P_MIN/MAX, LARGE_T, …
├── utils/utilities.h / .cu   # buildTransformationMatrix, checkCUDAErrorFn
├── utils/logger.h            # tagged stdout/stderr logging (Log::info/warn/error/raw)
├── kernels/kernel_config.h   # LAUNCH_KERNEL_AUTO macros, KernelConfig / OccupancyConfig / DeviceInfo
├── image.h / .cpp            # PNG/HDR output (stb_image)
├── glslUtility.* / window_setup.h   # GLFW/GL/CUDA-interop init, PBO registration & per-frame mapping
├── bvh/                      # single world-space BVH: AABB + node structs + host SAH build + flatten
│   ├── aabb.h                #   AABB + sign-based slab ray test + entry-distance variant (__host__ __device__)
│   ├── bvh.h                 #   BvhNode/BvhBuffers/BvhHit + shared traverseBvhClosest (near-first)
│   └── bvh.cu                #   buildSceneBvh (world-space bake + SAH) → flatten → uploadToDevice/freeDevice
├── kernels/                  # __global__ kernels (pure GPU, data passed as parameters):
│   ├── ray_generation.cuh    #   primary rays + AA sub-pixel jitter + thin-lens DoF
│   ├── bvh_traversal.cuh     #   single world-space BVH closest-hit traversal (only intersection path)
│   ├── shading.cuh           #   BSDF eval + scatterRay + Russian roulette
│   └── accumulation.cuh      #   gatherTerminatedPaths, sendImageToPBO
├── pipeline/                 # host-side orchestration (references g_opts / g_dev globals):
│   ├── sort.cuh              #   material sorting (thrust sort_by_key + gather)
│   ├── compact.cuh           #   stream-compaction dispatch (gather + compact per bounce)
│   └── postprocess.cuh       #   bloom → prepareDisplay → ACES/sRGB → CA → vignette → PBO
├── postprocess/              # bloom, tonemap (ACES + sRGB), chromatic_aberration, vignette kernels
├── interactions/             # scatterRay + Fresnel (Schlick/Accurate) + hemisphere/GGX surface sampling
├── intersection/             # intersections.h (ray utils, concentricSampleDisk), triangle.h (Möller–Trumbore)
├── rng/rng.h                 # RngState: LCG + scrambled Halton, utilhash
├── profiler/                 # Profiler singleton: cudaEvent GPU + chrono CPU timing, CSV export
├── stream_compaction/        # efficient.cu/h: global-scan + shared-mem hierarchical-scan compaction
├── ImGui/                    # Dear ImGui source
```
(json.hpp lives in `external/include/` alongside the other vendored headers — see Dependencies.)

### Path Tracing Pipeline (one iteration = one sample per pixel)

`pathtrace()` in `pathtrace.cu` runs once per frame:

1. **`generateRayFromCamera`** — primary rays per pixel (AA jitter; thin-lens DoF if `lensRadius > 0`). All `pixelcount` paths start with `remainingBounces = traceDepth`.
2. **Bounce loop** (until every path terminates or `depth ≥ traceDepth`):
   - **Intersection** — `bvhTraverse` (single world-space BVH closest-hit traversal, the only path). Triangles were baked to world space at build time (and tagged with `materialId`), so each active path runs ONE `traverseBvhClosest` over the whole tree — no per-mesh loop, no ray transform. Near-child-first traversal orders children by AABB entry distance to tighten the far plane early. Double-sided Möller–Trumbore `triangleIntersectionTest` with interpolated vertex normals; records closest `t`, `materialId` (from the hit triangle), world-space `surfaceNormal`.
   - **`sortPathsByMaterial`** *(optional)* — thrust sort_by_key on `materialId`, reorders paths + intersections so same-material paths are contiguous (less warp divergence).
   - **`shadeMaterial`** — emissive hit: multiply by emittance, terminate. Miss: terminate black. Surface hit: `scatterRay()` (diffuse/glossy/mirror/refractive) then Russian roulette.
   - **`compactActivePaths`** *(optional, 4 methods)* — first `gatherTerminatedPaths` banks dead-path colors into the HDR accumulation image, then stream-compacts survivors to the front of a ping-pong buffer.
3. **`runPostProcess`** — bloom (linear HDR) → `prepareDisplayKernel` (÷iter, composite bloom) → ACES + sRGB tonemap → chromatic aberration → vignette → PBO.
4. Copy the **tonemapped** display buffer (`g_dev.imageDisplay`) to host `state.image` so `saveImage()` matches the on-screen preview (raw HDR lives in `g_dev.image`).

### Key Data Structures (all in `sceneStructs.h`)

- **`Triangle`** — 3 vertices + 3 vertex normals + `materialId`. Object space in `Scene::hostTriangles`; baked to world space (vertices via `transform`, normals via `invTranspose`) by `buildSceneBvh`.
- **`Geom`** — `materialid`, transform/inverse/invTranspose (used only by the world-space bake at build time; no longer read by any kernel), plus `meshTriangleOffset`/`meshTriangleCount` (slice into `Scene::hostTriangles`; `-1,0` for none).
- **`Material`** — `color`, `specular { color, roughness }` (raw JSON `ROUGHNESS` ∈ [0,1]; -1 = unspecified), `metallic` (raw JSON `METALLIC` ∈ [0,1]; -1 = unspecified), `type` (`MaterialType` enum: Diffuse/Reflective/Pbr/Refractive/Emissive), `indexOfRefraction` + `invIndexOfRefraction`, `emittance`.
- **`Camera`** — resolution, position/lookAt/view/up/right, fov, pixelLength, `lensRadius` (0 = pinhole), `focalDistance`.
- **`PathSegment`** — ray + accumulated color + pixelIndex + remainingBounces.
- **`ShadeableIntersection`** — `t` (<0 = miss), `surfaceNormal`, `materialId`.
- **`RenderState`** — camera + iterations + traceDepth + rrMinBounces + host `image` buffer + `DebugConfig`. (Renderer settings like `rngMode`/compaction live in `AppConfig`, not here.)
- **`AppConfig`** — runtime config singleton (see above).

### Random Number Generation (`src/rng/rng.h`)

`RngState` exposes a uniform `.next(dim)` API for both modes. `makeRngState(iter, pixelIndex, bounceNum * MAX_DRAWS_PER_BOUNCE, mode)` creates the per-bounce state (bounce encoding via `MAX_DRAWS_PER_BOUNCE = 8`).

- **LCG** — `thrust::default_random_engine` seeded by `utilhash` (backward compatible).
- **Halton** — `haltonIndex = hash(pixel, bounce) + iter` (consecutive walk across frames → low-discrepancy convergence). `next(dim)` picks a prime base per dimension (`HaltonDim`: AA jitter, lens, diffuse θ/φ, GGX half-vector θ/φ, Fresnel roulette, path RR, PBR diffuse/specular split) and applies a Cranley-Patterson rotation with a **fixed** per-(pixel, bounce, dim) offset. `iter` is deliberately excluded from the CP seed (see comments in `rng.h`).
- Dimensions 0–10 are allocated (10 = PBR diffuse/specular split); 11–15 reserved.

### Intersection Testing

Mesh-only. Every `Geom` is a triangulated mesh; non-mesh geoms silently miss. Triangles are baked to world space at BVH build time, so the traversal kernel works directly in world space (no per-mesh ray transform). Triangle test in `intersection/triangle.h` is **double-sided** (accepts back faces — required for rays inside refractive objects) and reports the model's **true** shading normal (winding preserved — the bake's `invTranspose` normal transform preserves it) plus an `outFrontFace` flag (the Möller–Trumbore determinant's sign). Opaque materials orient the normal toward the ray in `scatterRay` off that stable flag; refraction keys enter vs exit off it too (never `dot(normal, ray)`, which degenerates at grazing angles). `t` is the world-space distance along the normalized world ray.

### Scattering (`interactions/interactions.cu`)

- **Diffuse** — cosine-weighted hemisphere sampling; `color *= albedo` (the `cosθ/pdf` factor cancels the `1/π` in the Lambert BRDF).
- **Reflective / Pbr** — unified metallic-roughness GGX surface (`resolvePbrSurfaceParams` resolves per-hit `r`, `α = r²`, conductor `F0 = mix(0.04, baseColor, metallic)`, `diffuseColor = baseColor·(1−metallic)`). `r < ROUGHNESS_THRESHOLD` collapses to a single mirror lobe weighted by `Fresnel(N·V)`; otherwise a Fresnel-weighted probabilistic split, `specProb = clamp(lum(Fresnel(N·V)), 0.05, 0.95)`:
  - **Specular branch** (prob `specProb`) — importance-sample the GGX half-vector (`cosθh² = (1−ξ₁)/(1+(α²−1)ξ₁)`), reflect the view about it, weight `F·G·(V·H)/((N·V)(N·H)) ÷ specProb` (D and N·L cancel out of the estimator). Below-surface samples (N·L ≤ 0) get weight 0 and scatter the mirror reflection so the path survives for Russian roulette.
  - **Diffuse branch** (prob `1−specProb`) — cosine hemisphere, weight `diffuseColor·(1−lum(F_view)) ÷ (1−specProb)`. The `(1−lum(F_view))` energy compensation prevents specular+diffuse from summing past 1 (grazing fireflies).
  - Legacy JSON `Specular` → `Reflective` is the metallic=1 chrome case: F0 = `specular.color`, no diffuse.
- **Refractive** — enter/exit keyed off the intersection's `frontFace` flag, Fresnel hardcoded to **Accurate** (Schlick is kept in `interactions.cu` for a one-line switch — no runtime mode dispatch), Russian-roulette split between reflection and refraction with probability-compensated throughput; ray origin offset by `EPSILON` into the correct side of the surface (keyed off the enter/exit state, stable near grazing).
- **Russian roulette** (`shading.cuh`) — after `rrMinBounces`, survival probability = clamp(max RGB, `RR_P_MIN`, `RR_P_MAX`); survivors divide color by p.
- **Ray-origin offset** — every scatter pushes the new origin off the surface by `EPSILON` along the shading normal (sign per branch: refraction off the enter/exit side, GGX/diffuse off the scatter-direction orientation), and the traversal's `RAY_EPSILON` near-clip skips the freshly-scattered surface.

### Post-Processing

Bloom runs in linear HDR space (threshold → separable Gaussian blur with shared-memory tiling). ACES filmic (Hill fit) + sRGB gamma in `tonemap.cuh`. Chromatic aberration and vignette run in sRGB space. The display buffer `g_dev.imageDisplay` is separate from the raw HDR accumulation `g_dev.image`.

### Scene Files

- **Materials** — `TYPE`: `Diffuse` / `Emitting` / `Specular` / `PBR` / `Refractive`. `Specular` (legacy → `Reflective` chrome) supports `SPECULAR_COLOR` and `ROUGHNESS`; `PBR` is the unified metallic-roughness GGX surface; `Refractive` uses `IOR`. Roughness/metallic/baseColor are resolved by **source chains** (first hit wins, see `resolvePbrSurfaceParams` / `resolveBaseColor` in `interactions.cu`):
  - **Roughness** — a bound glTF `metallicRoughness` texture overrides everything **per-texel** (ORM G channel — a data map, raw bytes → already linear); else explicit JSON `ROUGHNESS`; else the glTF `roughnessFactor` (into `TextureBinding::roughnessFactor`, `-1` for OBJ); else fixed default `r = 0.5` (an incomplete material must not silently become a mirror). `r < ROUGHNESS_THRESHOLD` (or `ROUGHNESS:0` / G=0 / factor 0) is a true mirror.
  - **Metallic** — ORM B channel per-texel; else explicit JSON `METALLIC`; else glTF `metallicFactor`; else type default: `Specular`→Reflective = 1.0 (chrome, no diffuse), `PBR` = 0.0 (dielectric).
  - **Base color** — `PBR` resolves the albedo via `resolveBaseColor` (glTF `baseColor` texture × `baseColorFactor`, else JSON `TEXTURE`, else `RGB`); `Specular` uses `RGB`/`SPECULAR_COLOR` as the metal tint; `Diffuse` stays pure Lambert (`RGB`/`TEXTURE`, unchanged).
- **Camera** — `RES`, `FOVY`, `ITERATIONS`, `DEPTH`, `RR_DEPTH`, `FILE`, `EYE`, `LOOKAT`, `UP`; optional `LENS_RADIUS` / `FOCAL_DISTANCE` (DoF).
- **Objects** — no `TYPE` field (mesh is the only object type; the loader ignores it entirely, and the old JSONs' `"TYPE":"mesh"` still loads as a harmless unknown key). Each entry: `FILE` (mesh path relative to the scene file; `.obj` via tinyobjloader, `.gltf`/`.glb` via cgltf), `MATERIAL`, `TRANS`, `ROTAT`, `SCALE`. Models live in `scenes/models/` (cube.obj, sphere.obj, sphere_inv.obj, light.obj, pyramid.obj, diamond.obj, plus `gen_shapes.py` outputs and downloaded glTF under `glTF-Sample-Assets/`).
- **glTF node transforms are applied** — `gltf_loader.cpp` walks the scene graph and emits each `(node, mesh)` instance under the accumulated TRS/matrix transform (`nodeLocalMatrix` → `walkNode`), so multi-part models assemble exactly as the file specifies. Caveat: some Sketchfab/asset exports are **baked** — the vertices already carry the world-space transform and the node matrices are redundant. Applying both double-transforms the model. For such files, zero out the redundant node matrices (see `scenes/models/glTF-Sample-Assets/johnmarston.gltf` — nodes 0/1/3 set to identity — vs `johnmarston_original.gltf`, the untouched copy), or compensate with the scene `TRANS`/`ROTAT`/`SCALE`. The loader reads POSITION/NORMAL/TEXCOORD_0; glTF skinning/morphs/Draco are ignored — the scene JSON's `MATERIAL` governs the shading model (see the Textures bullet for what the asset's own maps contribute).
- **Textures** — auto-loaded from the model file itself. glTF `baseColor` / `normal` / `metallicRoughness` / `occlusion` / `emissive` slots load from external PNG/JPG URIs (relative to the `.gltf`/`.glb`) **and** from images packed into the `.glb` binary buffer (bufferView — decoded from memory); `data:` URIs fall back to the material color. OBJ companion `.mtl` maps load too (`map_Kd` → baseColor, `map_Bump`/`map_bump` → normal, `map_Ke` → emissive), resolved relative to the OBJ file (tinyobjloader is handed the OBJ's dir as its MTL base — otherwise it looks in the CWD). MTL flat colors (`Kd`/`Ks`/`Ns`) are **not** applied — the scene JSON's `MATERIAL` still governs type/color. An explicit JSON `TEXTURE` on a material overrides the asset's own baseColor (see `parseObjects`).
- **Winding / normals** — the renderer **trusts the model's winding and normal direction**. The intersection reports the true shading normal; `scatterRay` orients it toward the ray only for opaque materials, and refraction reads its sign (dot with the ray) to classify enter vs exit. For solid glass use an **outward-wound** mesh (`sphere.obj` — smooth `vn`, CCW). `sphere_inv.obj` is **inward-wound and flat-shaded** (no `vn`): it renders as inside-out glass — Fresnel wall reflections still visible, but the entry ray is misclassified as "exit", so there is **no lensing/caustics** (that's the correct winding-respecting behavior, not a bug).

### Known Performance Notes

- `checkCUDAError` (`utilities.h`) forces a `cudaDeviceSynchronize()` after every kernel (~25 call sites; more per frame inside the compaction sweep loops). Fine for correctness; consider disabling `ERRORCHECK` for pure benchmark runs.
- `LAUNCH_KERNEL_AUTO` calls `cudaGetDeviceProperties` on every launch (`kernel_config.h`).
- Intersection is always the single world-space BVH closest-hit traversal (`bvhTraverse` + `traverseBvhClosest`). One traversal per ray (no per-mesh loop or transform) with near-child-first ordering; the win over a linear scan appears on large meshes. The exhaustive SAH build + world-space bake is a few ms at a few thousand triangles.
- **Fast math** — `CMakeLists.txt` compiles CUDA with `-use_fast_math` (rcp.approx division, fast sqrt/trig/pow). Errors are ~2 ulp, invisible in a path tracer, and it makes the hot `1.0f / a` in `intersection/triangle.h` cheap. Consequence: **results differ from a precise-math build in the last few bits** — use the same flags when diffing renders or benchmarking.
- **GPU division avoidance** — reciprocals and ratios that are constant per frame / per material are precomputed on the host: per-pixel `÷iter` became `*invIter` (`postprocess.cuh` → `tonemap.cuh`/`bloom.cuh`), Fresnel takes precomputed `eta` = n1/n2 (`interactions.cu`, from `invIndexOfRefraction`/`indexOfRefraction`), and `sendImageToPBO` no longer divides (display buffer is pre-averaged). The old Phong `exponent`/`invExponentPlusOne` precomputes are gone — the GGX surface derives `α = r²` per-hit in `resolvePbrSurfaceParams` (roughness/metallic can be per-texel or glTF factors, so a host-side exponent would be wrong), and the GGX branch divides only by its own branch probability. `fresnelSchlickConductor`'s `(1−cosθ)⁵` is explicit multiplies, never `powf` (fast-math friendly).

## Controls (Runtime)

| Key | Action |
|-----|--------|
| Esc | Save image and exit |
| P   | Save image |
| R   | Re-center camera to original position + orientation |
| W / A / S / D | Fly forward / left / backward / right along camera axes |
| Space / Shift | Fly up / down along camera up |
| Left mouse drag | Rotate camera **in place** (yaw/pitch; position unchanged) |
| Right mouse drag (vertical) / wheel | Dolly along the view axis (fly toward/away from the focused point) |
| Middle mouse drag | Pan the camera along its right/up axes |

The camera is a free-fly camera: `cam.position` is independent state, translated by WASD / middle-pan / scroll-dolly, while left-drag only changes the view orientation (`theta`/`phi`) — the camera turns in place and never moves. `cam.lookAt` is a derived reference point `zoom` units ahead along the view axis (bookkeeping only; the renderer uses position/view/up/right). Fly speed scales with `zoom` (the reference distance).

## Open TODO Items

- **Motion blur** — jitter rays "in time" (`src/kernels/ray_generation.cuh`).
- **Wire `tests/` into the root CMake build.**

## Dependencies

CUDA (with Thrust), OpenGL, GLFW, GLEW, GLM (header-only, in `external/`), nlohmann/json (`external/include/json.hpp`), stb (`src/stb.cpp`), tinyobjloader (`external/include/tiny_obj_loader.h`), cgltf (`external/include/cgltf.h`). All vendored headers are `#include`d with angle brackets (`<glm/…>`, `<json.hpp>`, `<stb_image.h>`, `<cgltf.h>`, `<tiny_obj_loader.h>`) and resolve via the `external/include` include path; project-local headers under `src/` use quotes. Compiled with C++17 / CUDA 17 standard, `CUDA_SEPARABLE_COMPILATION ON`, targeting `native` architecture.
