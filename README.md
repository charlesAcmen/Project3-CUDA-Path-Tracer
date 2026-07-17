# CUDA Path Tracer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

This repository contains a CUDA-based Monte Carlo path tracer with a refactored modular architecture. It renders globally illuminated scenes using GPU kernels, OpenGL/GLFW interop, and an ImGui-based debug overlay.

## What this project includes

- A full GPU path tracing pipeline with ray generation, intersection testing, shading, and accumulation
- Refactored source organization for scene loading, RNG, intersections, interactions, pipeline control, profiling, and post-processing
- Advanced rendering features such as:
  - stream compaction
  - material sorting
  - diffuse/specular/refraction-aware scattering
  - depth-of-field
  - tone mapping and bloom post-processing
  - benchmark-oriented CLI and profiler hooks

## Project structure

```text
src/
├── main.cpp                  # Window, input handling, render loop
├── pathtrace.cu / pathtrace.h
│                              # Core path tracing pipeline and dispatch logic
├── intersections.cu / intersections.h
│                              # Ray-primitive intersection tests
├── interactions/             # BSDF / scattering related logic
├── pipeline/                 # Rendering pipeline helpers and execution flow
├── postprocess/              # Tone mapping and bloom passes
├── profiler/                 # Timing / profiling utilities
├── rng/                      # RNG implementations and sampler state
├── scene.cpp / scene.h       # Scene loading from JSON
├── sceneStructs.h            # Shared data structures
├── stream_compaction/        # Path compaction implementations
└── utilities.*               # Math constants and shared helpers
```

Other important folders:

- `scenes/` — input scene JSON files
- `outputs/` — rendered output images and generated results
- `docs/` — design notes and benchmarking guidance
- `build/` — CMake build output

## Build and run

### Windows / Visual Studio

```powershell
cmake -B build
cmake --build build --config Debug
```

Then run the executable with a scene file:

```powershell
build\bin\Debug\cis565_path_tracer.exe scenes\cornell.json
```

### Linux / WSL

```bash
make
# or
make Release
```

Run:

```bash
./build/bin/cis565_path_tracer scenes/cornell.json
```

## Basic usage

- Launch the program with a scene JSON file.
- Use the mouse to orbit / zoom / pan the camera.
- Press `Space` to reset the camera view.
- Press `S` to save the current image.
- Press `Esc` to save and exit.

## Notes for contributors

- The main render loop is driven by the path tracer entry point in `src/pathtrace.cu`.
- Runtime behavior can be controlled with benchmark-oriented flags documented in `docs/benchmarking-guide.md`.
- The project does not ship with a formal test suite; visual inspection of rendered output is the primary validation method.

## Implemented features

- Stream compaction with multiple implementations
- Russian roulette path termination
- Material sorting for better GPU execution coherence
- Diffuse, specular, and refraction-aware scattering
- Depth-of-field camera model
- Halton-based sampling and improved RNG structure
- Tone mapping and bloom post-processing
- ImGui overlay for interactive tuning

