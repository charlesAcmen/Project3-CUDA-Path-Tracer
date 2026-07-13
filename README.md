CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.


Development notes — Implemented features
---------------------------------------

Below are the items from `INSTRUCTION.md` / project notes and which ones
are currently implemented in the codebase (for developer record). Keep the
original README content above — this is just an addition for tracking.

- Stream compaction: implemented three working variants
	- Global-memory scan version (compact method 1)
	- Thrust `copy_if` reference version (compact method 2)
	- Shared-memory multi-block scan version (compact method 3) — default
	- `0` disables compaction entirely

- Russian roulette: implemented (path termination via Russian roulette where applicable)

- Material sorting: implemented (`sort by material` path + intersection permutation before shading)

- BSDF / scattering: implemented diffuse Lambertian scattering, perfect specular reflection, and imperfect glossy/specular reflection with roughness-driven sampling for non-perfect mirrors

- Visual Improvements (Instruction.md): the refraction-oriented visual improvement is implemented, including Fresnel-based reflection/refraction selection and imperfect specular lighting for rougher specular materials

- Random sampling: the path tracer now uses one centralized seeded RNG helper so sampling stays consistent across path generation, shading, and Russian roulette decisions

Other notes:

- The runtime mappings are in `src/pathtrace.cu` (see `g_compactMethod` and
	`setCompactMethod`). The dispatch table (`g_compactCore`) points to the
	corresponding function for each method.
- Profiler and CLI flags (`--compact`, `--sort`, `--benchmark`, `--save`,
	`--verbose`, `--warmup`) control runtime behavior and are documented in
	`docs/benchmarking-guide.md`. The default compact method in code is
	the shared-memory multi-block scan (method 3).

If you think I missed anything that's implemented, tell me and I'll update
this note (keeping the existing README content intact).

