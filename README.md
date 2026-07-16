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

- The refraction-oriented visual improvement is implemented, including Fresnel-based reflection/refraction selection and imperfect specular lighting for rougher specular materials.
- Physically-based depth-of-field is implemented by jittering rays within an aperture.
- Random-number generation has been upgraded from the original LCG-based path to a unified Halton-based sampler. The implementation in `src/rng.h` provides a shared `RngState` interface for both LCG and Halton modes, with Halton mode using prime-base radical inverse sampling plus Cranley-Patterson rotation. The sequence start point is now hash-based per `(pixelIndex, bounceIndex)` instead of a simple linear packing, which removes the structured aliasing that caused stripe-like artifacts; the index still advances consecutively across iterations to preserve low-discrepancy convergence. The sampling calls in `src/pathtrace.cu` now use this RNG for AA jitter, depth-of-field lens sampling, diffuse/specular scattering, Fresnel roulette, and Russian roulette.
- Image post-processing includes selectable tone mapping modes: Hill ACES (full colour-matrix ACES fit) and Narkowicz ACES (simpler sRGB curve), plus a linear bypass mode.
- Bloom post-processing is implemented and wired into the display pipeline. The GPU pass operates on the accumulated HDR image in linear HDR space before tone mapping: bright pixels are first isolated by a threshold-extraction kernel, then blurred with a two-pass separable Gaussian filter (horizontal + vertical) implemented with shared-memory tiled CUDA kernels, and finally composited back into the HDR image before ACES tone mapping and sRGB gamma correction. The effect is controllable from the ImGui overlay via enable/threshold/intensity/radius parameters, and the implementation lives in `src/postprocess/bloom.cuh` and `src/pathtrace.cu`.
- The ImGui overlay supports dynamic adjustment of focal distance and aperture radius, and shows the focal plane in green at the intersection point for visual validation of depth-of-field correctness.
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

