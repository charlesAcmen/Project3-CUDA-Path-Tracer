# CUDA Path Tracer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

A CUDA-based Monte Carlo path tracer with CUDA–OpenGL interop and a live ImGui overlay. It renders globally illuminated scenes progressively, one sample per pixel per frame.

All geometry is **triangulated at load time** — there are no sphere/cube primitives. Every object in a scene is a mesh loaded from `.obj` (tinyobjloader) or `.gltf` / `.glb` (cgltf), with smooth per-vertex normals. glTF scene-graph node transforms (TRS/matrix) are applied at load, so multi-part models scattered across nodes assemble automatically.

## Features

- **Mesh-only rendering** — OBJ + glTF/GLB loading, glTF scene-graph transforms (multi-part models auto-assemble), double-sided Möller–Trumbore intersection, interpolated vertex normals
- **BVH acceleration** — single world-space BVH over all baked triangles, near-child-first closest-hit traversal
- **Materials** — diffuse, emissive, specular (perfect mirror + glossy Phong via `ROUGHNESS`), refractive (glass, IOR, accurate Fresnel), Russian roulette path termination
- **Stream compaction** — 4 methods (off / global-mem scan / Thrust `copy_if` / shared-mem scan), toggleable at runtime
- **Material sorting** — Thrust `sort_by_key` + gather so same-material paths are contiguous, toggleable
- **RNG** — LCG (default) and true nested Owen-scrambled Halton (`--rng=1`) behind one `RngState::next(dim)` API
- **Camera** — per-pixel AA jitter and thin-lens depth-of-field
- **Post-processing** — HDR bloom, ACES filmic + sRGB tone mapping, chromatic aberration, vignette
- **Profiler** — per-kernel `cudaEvent` / CPU `chrono` timing with CSV export and companion Python plot scripts
- **ImGui overlay** — live toggles for compact method, sorting, RNG, bloom, and more

## Build & run

### Windows / Visual Studio

```powershell
cmake -B build
cmake --build build --config Release
```

Then run with a scene file:

```powershell
build\bin\Release\cis565_path_tracer.exe scenes\cornell.json
```

### Linux / WSL

```bash
make
# or
make Release
```

```bash
./build/bin/cis565_path_tracer scenes/cornell.json
```

## Runtime configuration

Settings are resolved with three-layer priority **CLI flags > `config.local.json` > code defaults**, handled by `src/config/` through the `appConfig()` singleton.

| Flag | Meaning |
|------|---------|
| `--compact=N` | Compaction: `0` off, `1` global-mem scan, `2` Thrust `copy_if`, `3` shared-mem scan (default) |
| `--sort=N` | Material sorting `0/1` (default off) |
| `--rng=N` | `0` LCG (default), `1` scrambled Halton |
| `--benchmark` | Enable profiler CSV output to `profiler_output/<scene>_<timestamp>/` |
| `--warmup=N` | Profiler warmup iterations (default 3) |
| `--verbose` | Print per-bounce path counts to the console |
| `--save` / `--save-at=N1,N2,...` | Save final / checkpoint images |
| `--config=PATH` | Load a config JSON (default `config.local.json` in CWD) |
| `-h`, `--help` | Help text |

Most settings can also be toggled live in the ImGui overlay; those mutate the same `g_opts` singleton.

## Project structure

```text
src/
├── main.cpp                  # GLFW/GL window, camera, ImGui, render loop → pathtrace()
├── pathtrace.cu / .h         # GPU pipeline + runtime getters/setters
├── config/                   # AppConfig singleton, JSON + CLI merge
├── scene/                    # scene container + scene_loader (OBJ via tinyobjloader, glTF/GLB via cgltf)
├── kernels/                  # __global__ kernels: ray_generation, bvh_traversal, shading, accumulation
├── pipeline/                 # host-side orchestration: sort, compact, postprocess dispatch
├── postprocess/              # bloom, tonemap (ACES + sRGB), chromatic_aberration, vignette
├── interactions/             # scatterRay, Fresnel (Schlick/Accurate), hemisphere/Phong sampling
├── intersection/             # triangle.h (Möller–Trumbore), ray utils
├── rng/                      # RngState: LCG + scrambled Halton
├── profiler/                 # cudaEvent/chrono timers, CSV export
├── stream_compaction/        # global-scan + shared-mem hierarchical-scan compaction
├── utils/                    # utilities, logger
├── ImGui/                    # Dear ImGui source
└── json.hpp                  # nlohmann/json (header-only)
```

Other folders:

- `scenes/` — scene JSON files (cornell, cornellRefra, cornellGlossy, cornell_shapes, cornell_inv, john_marston, …)
- `scenes/models/` — mesh assets; `gen_shapes.py` regenerates the procedural shapes, plus `glTF-Sample-Assets/` for downloaded glTF models
- `docs/` — design notes, benchmarking guide, profiler output structure
- `tests/` — standalone loader/RNG/config/refraction tests (not wired into the root build)
- `scripts/` — benchmark runner + plot scripts

## Scenes & models

Scenes are JSON files with `Materials`, `Camera`, and `Objects` sections. Objects are meshes placed by `TRANS` / `ROTAT` / `SCALE`:

```json
{ "TYPE":"mesh", "MATERIAL":"glass", "FILE":"models/sphere.gltf",
  "TRANS":[2.0,0.7,-1.8], "ROTAT":[0,0,0], "SCALE":[0.7,0.7,0.7] }
```

- `scenes/models/gen_shapes.py` generates sphere / cylinder / cone / torus / capsule as both `.gltf` and `.glb`, plus `*_inv` twins (flipped normals + winding — the glTF equivalent of `sphere_inv.obj`).
- **Winding / normals matter.** The renderer trusts the model's winding and normal direction. Opaque materials orient the normal toward the ray, but refraction reads its sign to classify enter vs exit — an inward-wound glass mesh renders as inside-out glass (Fresnel reflections, no lensing/caustics). Use outward-wound meshes for solid glass.
- **glTF node transforms are applied.** The scene graph is walked and each node's accumulated TRS/matrix transform places its mesh, so multi-part models assemble exactly as the file specifies. (Caveat: some Sketchfab/asset exports bake the world-space transform into the vertices **and** carry a redundant node matrix — applying both double-transforms the model. For such files, zero out the redundant node matrices or adjust the scene `SCALE`.)

## Controls

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

The camera is a free-fly camera: `cam.position` is independent state, translated by WASD / middle-pan / scroll-dolly, while left-drag only changes the view orientation — the camera turns in place and never moves.

## Documentation

- `docs/benchmarking-guide.md` — profiler usage, experiment recipes, CSV formats
- `docs/OUTPUT_STRUCTURE.md` — profiler output layout
- `docs/bvh-design.md` — implemented BVH acceleration design
- `docs/direct-lighting-design.md` — planned next-event estimation / direct lighting
- `docs/bloom-design.md`, `docs/postprocess-effects-design.md`, `docs/rng-design.md` — implemented feature designs

## Notes for contributors

- No test suite in the main build — validation is visual inspection of the rendered output. Standalone tests live under `tests/` (`bvh_test`, `loader_test`, `config_test`, `rng_test`, `refraction_test`), each a separate CMake project.
- `-use_fast_math` is enabled; renders differ from a precise-math build in the last few bits.
- Intersection is a single world-space BVH closest-hit traversal (`bvhTraverse`) — no per-mesh loop or ray transform (see `docs/bvh-design.md`).
