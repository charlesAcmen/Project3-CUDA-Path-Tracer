# CLAUDE.md

This document describes the current source tree. Treat `src/` as the implementation authority when this file and code disagree.

## Build and run

- Windows / Visual Studio: `cmake -B build`, then `cmake --build build --config Release`.
- Linux / WSL: `make` or `make Release` (CMake output is under `build/`).
- Run: `build/bin/Release/cis565_path_tracer <scene.json>`, for example `build/bin/Release/cis565_path_tracer scenes/cornell_box.json`.
- Standalone projects under `tests/` are not wired into the root CMake build. Generate each one separately with Visual Studio 2022 x64 **and the installed CUDA toolset**, then build its Release configuration. With the current CUDA 12.8 installation, use (from the repo root):
  `cmake -S tests/bvh_test -B tests/bvh_test/build -G "Visual Studio 17 2022" -A x64 -T cuda=12.8`, then
  `cmake --build tests/bvh_test/build --config Release`, then
  `tests/bvh_test/build/Release/bvh_test.exe`.
  The CUDA installer must have installed Visual Studio Integration for VS 2022: the file
  `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations\CUDA 12.8.props`
  must exist. `nvcc --version` alone is insufficient; if CMake reports that this `.props` file is missing, repair or modify CUDA 12.8 and select Visual Studio Integration before retrying. Rendering changes are primarily validated by the user through visual inspection.

Do not run builds or render scenes merely to validate a source change. State the exact command for the user to run and the expected visual or console result instead.

## Runtime configuration

`CLI flags > explicitly selected --config file > config.local.json > code defaults`. `initAppConfig()` in `src/config/config.cpp` owns this merge and `appConfig()` is the singleton consumed by the application and renderer.

The defaults are shared-memory compaction, material sorting off, LCG RNG, and auto-save on. JSON `compactMethod` and `rngMode` accept either a case-insensitive name or their legacy integer; CLI forms are numeric.

| Flag | Current meaning |
|---|---|
| `--compact=N` | `0` off, `1` global scan, `2` Thrust, `3` shared-memory scan |
| `--sort=N` | Material sorting; nonzero enables it |
| `--rng=N` | `0` LCG, `1` scrambled Halton |
| `--benchmark`, `--warmup=N`, `--verbose` | CSV profiler, warm-up count, per-bounce logging |
| `--save`, `--save-at=N1,N2,...` | Final save and checkpoint saves |
| `--config=PATH`, `-h`, `--help` | Config path and help |

`config.local.json` also controls bloom, chromatic aberration, vignette and profiler settings. The ImGui panel changes the same live renderer settings; camera changes reset accumulation.

## Source ownership and entry flow

```
main.cpp
  -> config/config.cpp          startup configuration
  -> scene/scene_loader.cpp     JSON orchestration
       -> obj_loader.cpp / gltf_loader.cpp / texture_loader.cpp
  -> window_setup.cpp           GLFW/OpenGL/CUDA-GL setup
  -> app_loop.cpp               per-frame CUDA map, display, save/cleanup
       -> camera_controller.cpp input and free-fly camera state
       -> render_ui.cpp         ImGui controls and profiler display
       -> pathtrace.cu          GPU pipeline orchestration
```

`AppState` is application/window state only. `Scene` owns parsed CPU scene data; `pathtrace.cu` allocates and frees the GPU copies. Keep vendored `external/` headers read-only unless a task explicitly authorizes changes.

Important source groups:

- `sceneStructs.h`: GPU-shared layouts and runtime enums.
- `scene/`: JSON parser plus format-specific OBJ, glTF and texture loading.
- `bvh/`: host-side scene bake/build, GPU-readable nodes and traversal helper.
- `kernels/`: primary-ray generation, BVH traversal, shading, accumulation.
- `interactions/`: texture sampling, normal mapping, GGX/diffuse/refraction scattering and Fresnel helpers.
- `pipeline/`: optional sorting/compaction and post-processing orchestration.
- `postprocess/`: bloom, tonemap, chromatic aberration and vignette kernels.
- `profiler/`: per-phase GPU timing and CSV output.

## Scene representation and loading

All renderable geometry is triangulated. Source geometry uses parallel `TrianglePos` (positions) and `TriangleAttr` (normals, UV0, COLOR_0 and a source surface-binding id) arrays. A `Geom` references a source slice and contains its JSON transform.

`buildSceneBvh()` applies each geom transform to positions and inverse transpose to normals, then creates one world-space BVH over all meshes. It also maps each `(materialId, SurfaceBinding)` pair to one compact runtime `Surface`. Consequently traversal only loads positions and records `t`, barycentrics and triangle index in a 20-byte `HitRecord`; shading expands the winning triangle's attributes and surface afterwards.

glTF nodes are walked with accumulated matrix or `T * R * S` transforms. Supported vertex attributes are POSITION, NORMAL, TEXCOORD_0, COLOR_0 and TANGENT; higher UV/color sets, skinning, morph targets and Draco are not handled. OBJ and glTF mesh paths are relative to the scene JSON file.

## Materials, textures and scattering

The JSON material selects the BSDF: `Diffuse`, `Emitting`, `Specular`, `PBR`, or `Refractive`. `Specular` maps to reflective chrome-like GGX behavior and may use `SPECULAR_COLOR`; `PBR` uses metallic-roughness GGX; `Refractive` uses optional `IOR` (default 1.5). For a PBR material missing `RGB`, the loader deliberately uses a sentinel so a glTF material's base color can win.

Texture ownership is asset-driven: no JSON `TEXTURE` or `UV_SCALE` path is parsed. glTF contributes baseColor, normal, metallicRoughness (ORM), occlusion and emissive slots. OBJ companion MTL contributes `map_Kd`, `map_Bump`/`map_bump`, and `map_Ke`. Color maps are decoded sRGB-to-linear; normal/ORM/occlusion data maps remain linear bytes. glTF images are decoded in parallel, then appended serially to retain deterministic texture ids; external and bufferView-backed GLB images are supported.

The relevant source chains are:

- PBR base color: glTF base-color texture times its glTF factor, otherwise the JSON material color; vertex COLOR_0 multiplies the result when present.
- Roughness/metallic: ORM G/B times their glTF factors, otherwise non-default glTF factors, otherwise the PBR/Reflective type defaults.
- Emission: `(emissive texture or white) * emissiveFactor * emissiveStrength`. A JSON `Emitting` material multiplies this by `emittance` and terminates; nonzero glTF emission on any other BSDF is additive auto-glow and scattering continues.

Normal maps prefer an interpolated glTF vertex tangent; when it is absent or invalid, they use the existing per-triangle tangent generated from UV derivatives. A degenerate-UV sentinel disables perturbation. Opaque scattering orients the shading normal against the incident ray; secondary-ray origins use the winning triangle's geometric normal and extent-aware offset, while `PathSegment` carries the previous triangle id so closest traversal skips only that numerical self-hit. Direct-light visibility likewise skips only the receiver triangle by default; all other geometry remains an occluder. Refraction keeps the geometric normal and uses the front-face classification, preserving winding-dependent behavior. The renderer is double-sided at intersection time.

## Per-iteration GPU pipeline

1. `generateRayFromCamera` creates one primary path per pixel, including AA jitter and optional thin-lens depth of field.
2. Each bounce traverses the one world-space BVH, optionally sorts active paths by resolved material, shades/scatters, then optionally gathers dead paths and compacts survivors.
3. With compaction disabled, terminated paths are gathered after the bounce loop. With compaction enabled, gathering occurs inside compaction.
4. Post-processing runs bloom in linear HDR, averages/composites into the display buffer, then ACES+sRGB tonemapping, chromatic aberration and vignette. `pathtraceCopyDisplayToHost()` makes saved PNGs match the display, rather than exporting raw HDR accumulation.

RNG is per `(iteration, pixel, bounce)`. Halton uses fixed Cranley-Patterson rotations per pixel/bounce/dimension and advances its index across iterations. Dimensions 0--10 are allocated; 11--15 are reserved.

## Runtime controls

- `W/A/S/D`, `Space`, `Left Shift`: fly on camera axes.
- Left drag: rotate in place. Right drag (vertical) or wheel: dolly.
- Middle drag: pan. `R`: restore the loaded camera. `P`: save. `Esc`: save and exit.

`cam.position` is authoritative. `lookAt` is derived from the view direction and reference zoom distance, not the position controller's anchor.

## Performance and limits

- The BVH is built on the CPU with exhaustive three-axis SAH sorting, then flattened into leaf-contiguous triangle arrays. It is one-time work, but can dominate startup on high-poly assets.
- Device texture data is one concatenated RGB buffer plus image descriptors; the one-time host-to-device copy can be substantial for large texture sets.
- CUDA error checks synchronize only in Debug and RelWithDebInfo. Release is asynchronous and is the appropriate configuration for timing comparisons.
- CUDA uses `-use_fast_math`; last-bit image comparisons require matching compiler flags. `KernelConfig` obtains device properties through its cached `DeviceInfo` singleton.

## Known scope limits

- Occlusion is loaded and carried through the surface binding, but is not sampled by current shading.
- Emissive maps are used; glTF alpha, transparent traversal, skinning, morphs, Draco, motion blur and extra UV/color sets are not implemented.
- BVH maximum depth and leaf size are compile-time constants in `constants.h`.
- Root CMake does not include the standalone `tests/` projects.

## Change checklist

For a propagated renderer change, follow the data through producer, transport, consumer and validation: loader/source layout -> BVH bake/device upload -> kernel/pipeline -> scene or focused test. Preserve `HitRecord`'s 20-byte GPU layout unless every consumer and its tests are intentionally updated. Do not claim a visual pass; give the user a scene command and the expected effect.
