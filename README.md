# CUDA Path Tracer

University of Pennsylvania CIS 565 Project 3 CUDA path tracer. The renderer is
a progressive CUDA Monte Carlo path tracer with CUDA–OpenGL interop and a live
ImGui overlay: each displayed frame adds one sample per pixel.

This README is a source-level implementation inventory. It distinguishes code
that exists from requirements that still need user-supplied render images or
performance measurements; it does not claim an unverified visual or benchmark
result.

## What is implemented

All scene geometry is triangulated at load time. The renderer loads OBJ through
tinyobjloader and glTF 2.0 / GLB through cgltf; it walks the glTF scene graph
and applies accumulated node transforms, so multi-node assets assemble in their
authored layout. The runtime uses one world-space BVH over the baked triangles,
not a per-object intersection loop.

### Project requirement inventory

| Instruction feature | Source status | Notes |
|---|---|---|
| Cosine-weighted diffuse BSDF | Implemented | `calculateRandomDirectionInHemisphere` and `scatterRay` sample a cosine-weighted hemisphere. |
| Material-path sorting | Implemented and runtime-toggleable | `--sort=1` uses Thrust `sort_by_key` plus gathers to group active paths by resolved material before shading. |
| Stochastic anti-aliasing | Implemented | Primary rays jitter sub-pixel positions each iteration. |
| Refraction with Fresnel | Implemented | Uses `glm::refract`, total-internal-reflection handling, and accurate dielectric Fresnel roulette. |
| Thin-lens depth of field | Implemented | `LENS_RADIUS` and `FOCAL_DISTANCE` drive aperture sampling and focal-plane convergence. |
| OBJ loading | Implemented | Triangle loading plus companion MTL color, bump/normal, and emissive maps. |
| glTF / GLB loading | Implemented | POSITION, NORMAL, TEXCOORD_0, COLOR_0, TANGENT, node transforms, external images, and embedded bufferView images. |
| Shared-memory stream compaction | Implemented and runtime-toggleable | The shared-memory hierarchical scan is `--compact=3`; global scan and Thrust variants are also retained. |
| Russian roulette termination | Implemented | After `RR_DEPTH`, survival probability is derived from path throughput and clamped to configured bounds. |
| Hierarchical acceleration structure | Implemented | CPU-built, GPU-traversed world-space BVH with iterative closest-hit traversal. It is always enabled; there is no linear-intersection fallback toggle for an A/B comparison. |
| Better random sequence | Implemented | LCG and scrambled Halton modes share the `RngState::next(dim)` interface. |
| Direct lighting / next-event estimation | Implemented; render validation pending | One emissive triangle is selected per eligible non-delta surface hit through an area-and-emission-weighted alias table. A bounded BVH any-hit shadow query tests visibility, and power-heuristic MIS combines this estimator with BSDF paths that hit a light. |
| Final-ray post-processing | Implemented | Bloom in linear HDR, then ACES/sRGB, optional chromatic aberration, vignette, and PBO output. |
| Metallic-roughness PBR | Implemented extension | GGX/Smith/Fresnel surface with glTF ORM factors and tangent-space normal maps. |

### Partial or unsupported instruction features

| Feature | Current boundary |
|---|---|
| Texture mapping and bump mapping | File-loaded base-color, normal, ORM, and emissive textures are implemented. The instruction's required basic procedural texture and a file-vs-procedural performance comparison are not implemented. |
| Procedural shapes and textures | `scenes/models/gen_shapes.py` provides multiple generated mesh shapes. There is no procedural texture shader, so this is not presented as the complete combined feature. |
| Motion blur | Not implemented. Primary-ray code explicitly reserves it as future time jitter. |
| Subsurface scattering, denoising, CUDA–Vulkan interop | Not implemented. |
| Restartable path tracing | Not implemented as persistent save/resume. `--save-at` saves images only; it does not serialize accumulation or BVH state. |
| glTF alpha, skinning, morph targets, Draco, extra UV/color sets | Not implemented. |
| Occlusion texture | Loaded and carried in the surface binding, but not sampled by shading. |

## Rendering pipeline

For each iteration:

1. `generateRayFromCamera` creates one primary path per pixel with AA jitter
   and optional thin-lens sampling.
2. Every active path traverses the single world-space BVH. Optional material
   sorting then groups path and hit buffers before `shadeMaterial` resolves
   the surface. At eligible non-delta hits, next-event estimation samples one
   emissive triangle and tests its shadow ray against the same BVH; MIS avoids
   double-counting light paths reached through BSDF sampling. The BSDF then
   scatters the next ray.
3. Terminated paths are accumulated. Optional stream compaction gathers their
   radiance before removing them from the active queue.
4. The display pipeline averages HDR radiance, optionally composites bloom,
   applies ACES filmic tone mapping and sRGB transfer, then optional chromatic
   aberration and vignette before writing the OpenGL PBO.

`pathtraceCopyDisplayToHost()` reads back the post-processed display buffer, so
saved PNGs match the on-screen preview rather than the raw HDR sum.

## Materials and textures

JSON materials select the BSDF: `Diffuse`, `Emitting`, `Specular`, `PBR`, or
`Refractive`.

- `Diffuse` uses cosine-weighted Lambert scattering.
- `Specular` is the chrome-like reflective case of the unified GGX path;
  `ROUGHNESS` controls its lobe. It is not the older Phong implementation.
- `PBR` uses metallic-roughness GGX. ORM texture channels are G = roughness
  and B = metallic, multiplied by their glTF factors.
- `Refractive` uses the material `IOR` (default 1.5), Fresnel roulette, and
  winding-aware entry/exit classification.
- `Emitting` terminates after adding its radiance. A nonzero glTF/MTL emissive
  binding on another BSDF is additive auto-glow, so the path continues.

Direct-light sampling includes both JSON emitters and asset-driven auto-glow
surfaces. It samples world-space emissive triangles with a Walker/Vose alias
table weighted by area and estimated emitted power, then evaluates the exact
texture and directional emission at the sampled point. `EMISSION_SIDEDNESS`
accepts `OneSided`; the default `TwoSided` preserves the renderer's original
double-sided emission behavior.

Texture ownership is asset-driven: JSON scene materials do not provide a
separate `TEXTURE` or UV-scale override. glTF contributes base-color, normal,
metallic-roughness, occlusion, and emissive slots; OBJ MTL contributes
`map_Kd`, `map_Bump` / `map_bump`, and `map_Ke`. Color maps decode from sRGB to
linear values; normal, ORM, and occlusion maps retain raw linear byte values.

## Build and run

### Windows / Visual Studio

```powershell
cmake -B build
cmake --build build --config Release
build\bin\Release\cis565_path_tracer.exe scenes\cornell_box.json
```

### Linux / WSL

```bash
make Release
./build/bin/cis565_path_tracer scenes/cornell_box.json
```

The standalone tests under `tests/` are not included by the root CMake target.
Generate and build each test project separately when validating its subsystem.
For example, with CUDA 12.8 and Visual Studio 2022 x64 installed:

```powershell
cmake -S tests/bvh_test -B tests/bvh_test/build -G "Visual Studio 17 2022" -A x64 -T cuda=12.8
cmake --build tests/bvh_test/build --config Release
tests\bvh_test\build\Release\bvh_test.exe
```

## Runtime configuration

Configuration priority is:

```text
CLI flags > explicitly selected --config file > config.local.json > code defaults
```

| Flag | Meaning |
|---|---|
| `--compact=N` | `0` off, `1` global-memory scan, `2` Thrust `copy_if`, `3` shared-memory scan (default) |
| `--sort=N` | Material sorting; nonzero enables it (default off) |
| `--rng=N` | `0` LCG (default), `1` scrambled Halton |
| `--benchmark`, `--warmup=N`, `--verbose` | Enable profiler CSV output, choose warm-up iterations, print per-bounce counts |
| `--save`, `--save-at=N1,N2,...` | Save the final image or checkpoint images |
| `--config=PATH`, `-h`, `--help` | Select configuration, show help |

Bloom, chromatic aberration, vignette, compaction, sorting, and RNG are also
available through ImGui. Camera and the relevant live renderer setting changes
restart accumulation.

## Controls

| Input | Action |
|---|---|
| `Esc` | Save image and exit |
| `P` | Save image |
| `R` | Restore loaded position, orientation, and reference distance |
| `W` / `A` / `S` / `D` | Fly forward / left / backward / right on camera axes |
| `Space` / `Left Shift` | Fly up / down |
| Left drag | Rotate in place |
| Right drag (vertical) or wheel | Dolly along the viewing axis |
| Middle drag | Pan in the camera image plane |

The camera is free-fly: `cam.position` is authoritative. `lookAt` is a derived
reference point at the current zoom distance along the view direction.

## Source layout

```text
src/
├── main.cpp                  # startup, configuration, scene loading, loop assembly
├── app/                      # AppState, window setup, input, ImGui, frame loop
├── pathtrace.cu / .h         # GPU resource lifetime and path-tracing pipeline orchestration
├── sceneStructs.h            # shared CPU/GPU layouts and runtime enums
├── config/                   # CLI + JSON merge and AppConfig singleton
├── scene/                    # JSON, OBJ, glTF/GLB, texture loading
├── bvh/                      # world-space BVH construction, flattening, traversal helper
├── lighting/                  # emissive-triangle alias-table construction for next-event estimation
├── kernels/                  # ray generation, BVH traversal, shading, accumulation
├── interactions/             # texture sampling, GGX, diffuse and refractive scattering
├── pipeline/                 # optional sort, compaction, post-process dispatch
├── postprocess/              # bloom, tone map, chromatic aberration, vignette kernels
├── rng/                      # LCG and scrambled Halton RNG
├── profiler/                 # GPU/CPU phase timing and CSV export
└── stream_compaction/        # global and shared-memory compaction implementations
```

## Validation and submission evidence

The project instructions require render images and measured analysis for each
claimed optional feature. Add only user-validated evidence here:

- before/after images for visual features;
- profiler CSV or chart comparisons for sorting, compaction, Russian roulette,
  and BVH claims;
- a direct-lighting comparison that shows the NEE scenes converge without
  light leaks or double-counted emission; `scenes/nee_small_light.json`,
  `scenes/nee_occluded_light.json`, `scenes/nee_deep_room.json`, and
  `scenes/nee_rough_pbr.json` are focused inputs for that check;
- open versus closed-scene path-survival and compaction comparisons;
- file-texture versus procedural-texture comparison only after a procedural
  texture implementation exists.

The built-in profiler writes CSV files under
`profiler_output/<scene>_<timestamp>/` when `--benchmark` is enabled. See
`docs/benchmarking-guide.md` for experiment recipes and CSV fields.

## References

- [Project instructions](INSTRUCTION.md)
- [Direct-lighting design](docs/direct-lighting-design.md)
- [PBRT v4](https://pbr-book.org/4ed/contents)
- [PBRT v3](https://www.pbr-book.org/3ed-2018/contents)
- [GPU Gems 3, Chapter 39: Parallel Prefix Sum](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
- [Antialiasing and Raytracing — Paul Bourke](https://paulbourke.net/miscellaneous/raytracing/)
