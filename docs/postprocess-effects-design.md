# Chromatic Aberration & Vignette Post-Processing Effects Design

## Overview

This document describes the design and implementation of two camera-lens simulation post-processing effects:

- **Chromatic Aberration (色差)** — Simulates the wavelength-dependent refractive index of camera lenses, causing different color channels to focus at slightly different positions. The result is color fringing at high-contrast edges, most visible near the image periphery.
- **Vignette (暗角)** — Simulates the natural light falloff at the edges of a camera lens, darkening the corners of the image relative to the center.

Both effects operate on the **tone-mapped sRGB image** after ACES filmic tone mapping, in display-ready [0,1] space. They are the final post-processing steps before `sendImageToPBO` writes the result to the OpenGL pixel buffer.

This feature corresponds to the **:three: Use final rays to apply post-processing shaders** item in the project instructions (3-point optional feature).

## Pipeline Order

```
g_dev.image (HDR accumulation)
    │
    ├──[bloom]─── thresholdExtract → blurH → blurV  (HDR linear space)
    │
    ▼
prepareDisplayKernel    (avg HDR + composite bloom)  → imageDisplay
    │
    ▼
tonemapKernel           (ACES filmic + sRGB gamma)   → imageDisplay (in-place)
    │
    ▼
chromaticAberrationKernel (radial color shift)       → bloomBufB → imageDisplay
    │
    ▼
vignetteKernel          (corner darkening)           → imageDisplay (in-place)
    │
    ▼
sendImageToPBO          (float → uchar4 → OpenGL PBO)
```

**Why after tone mapping?** Both chromatic aberration and vignette are camera/display artifacts, not scene-referred effects. They belong in the display pipeline after the ACES filmic transform, matching how these effects are applied in game engines and post-production pipelines.

## Chromatic Aberration

### Algorithm

Chromatic aberration simulates a lens that refracts different wavelengths (colors) at slightly different angles:

1. For each pixel at `(x, y)`, compute its vector from the image center in pixel space.
2. Compute the radial distance `dist` from center.
3. **Red channel**: sample at a position shifted **outward** along the radial direction.
4. **Green channel**: keep at the original position (reference channel).
5. **Blue channel**: sample at a position shifted **inward** along the radial direction.

The shift magnitude is proportional to the radial distance (more shift at edges, zero at center):

```
shift = intensity * dist
```

where `intensity` is a small user-controlled value (typically 0.001–0.01 in UV space).

### Bilinear Sampling

Since sub-pixel offsets are required (the shift is typically 0.3–3 pixels), we use bilinear interpolation for smooth results:

```cpp
__device__ inline glm::vec3 sampleBilinear(
    const glm::vec3* __restrict__ src,
    int width, int height,
    float fx, float fy)
```

- Reads the four nearest texels around `(fx, fy)`.
- Interpolates using fractional weights.
- Edge clamping: coordinates are clamped to `[0, width-1]` × `[0, height-1]` to avoid out-of-bounds access.

### Kernel Signature

```cuda
__global__ void chromaticAberrationKernel(
    const glm::vec3* __restrict__ srcImage,    // imageDisplay (sRGB input)
    glm::vec3*       __restrict__ dstImage,     // bloomBufB (temporary output)
    glm::ivec2 resolution,
    float intensity);                           // shift magnitude
```

### Special Cases

- **Center region** (`dist < 0.5 pixels`): pass-through (no shift), avoiding division-by-zero in the direction normalization.
- **Edge pixels**: bilinear sampler clamps coordinates to valid range, so edge pixels naturally lose their fringe (which is physically correct — the fringe shifts beyond the sensor).

---

## Vignette

### Algorithm

Vignette darkens image corners via a radial falloff:

1. Compute normalized distance from image center: `ndist = dist / maxDist` in [0, 1].
2. Compute falloff factor:
   ```
   factor = 1.0 - intensity * pow(ndist, exponent)
   ```
3. Clamp factor to [0, 1].
4. Multiply pixel color by factor.

### Kernel Signature

```cuda
__global__ void vignetteKernel(
    const glm::vec3* __restrict__ srcImage,    // input (imageDisplay or bloomBufB)
    glm::vec3*       __restrict__ dstImage,     // output (always imageDisplay)
    glm::ivec2 resolution,
    float intensity,    // 0.0–1.0: darkness at corners
    float exponent);    // 0.5–8.0: falloff curve steepness
```

The kernel supports both in-place (`src == dst = imageDisplay`) and separate source/destination (`src = bloomBufB`, `dst = imageDisplay`) operation, enabling efficient chaining with chromatic aberration.

### Visual Effects by Parameter

| Intensity | Exponent | Visual Result |
|-----------|----------|---------------|
| 0.0 | any | No effect |
| 0.3 | 2.0 | Subtle, natural lens falloff |
| 0.5 | 1.5 | Moderate, noticeable corner darkening |
| 0.7 | 1.0 | Strong, gradual darkening from center to edges |
| 1.0 | 4.0 | Extreme corners nearly black, tight falloff |
| 0.5 | 0.5 | Wide, gradual falloff affecting large area |

---

## File Structure

```
src/postprocess/
├── tonemap.cuh                  # ACES tone mapping + sRGB (unchanged)
├── bloom.cuh                    # Bloom effect kernels (unchanged)
├── chromatic_aberration.cuh     # NEW: chromatic aberration kernel + bilinear sampler
└── vignette.cuh                 # NEW: vignette kernel
```

## Configuration

### ChromaticAberrationConfig

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `enabled` | bool | false | — | Master toggle (disabled by default) |
| `intensity` | float | 0.003 | 0.0–0.02 | Radial shift magnitude (UV fraction) |

### VignetteConfig

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `enabled` | bool | false | — | Master toggle (disabled by default) |
| `intensity` | float | 0.5 | 0.0–1.0 | Darkness at corners |
| `exponent` | float | 2.0 | 0.5–8.0 | Radial falloff power |

Both default to **disabled** to preserve backward compatibility.

## GPU Buffers

No additional device memory is needed. Both effects reuse the existing `bloomBufB` buffer as a temporary read/write surface:

| Scenario | CA action | Vignette action | Final imageDisplay |
|----------|-----------|-----------------|---------------------|
| Neither enabled | — | — | unchanged from tonemapKernel |
| CA only | imageDisplay → bloomBufB | — | cudaMemcpy bloomBufB → imageDisplay |
| Vignette only | — | imageDisplay → imageDisplay (in-place) | vignette-applied |
| Both enabled | imageDisplay → bloomBufB | bloomBufB → imageDisplay | CA + vignette applied |

### Buffer State Transitions

| State | imageDisplay | bloomBufB |
|-------|-------------|-----------|
| After tonemapKernel | sRGB [0,1] | Free (stale bloom data) |
| After CA (if enabled) | Unchanged | CA result |
| After Vignette (if both) | Vignette-applied result | CA result (stale) |
| After Vignette (if CA only) | cudaMemcpy'd from bloomBufB | CA result (stale) |
| After Vignette (if vignette only) | Vignette-applied result | Unchanged |
| After Vignette (if neither) | sRGB [0,1] | Free |

## Kernel Launch Configuration

Both kernels use the standard 2D post-process launch configuration:

```cpp
const dim3 blockSize2d(8, 8);
const dim3 blocksPerGrid2d(
    (resolution.x + blockSize2d.x - 1) / blockSize2d.x,
    (resolution.y + blockSize2d.y - 1) / blockSize2d.y);
```

No shared memory needed — these are purely per-pixel operations with no inter-thread communication.

## Runtime Controls (ImGui)

```
┌──────────────────────────────────┐
│ Chromatic Aberration (色差):      │
│ [✓] Enable Chromatic Aberration  │
│     Intensity: [===|========] 0.00300 │
├──────────────────────────────────┤
│ Vignette (暗角):                 │
│ [✓] Enable Vignette              │
│     Intensity: [====|========] 0.50 │
│     Exponent:  [====|========] 2.0  │
└──────────────────────────────────┘
```

All controls are live — changes take effect on the next frame without restart.

## Code Style

- **English comments only** — document each kernel's purpose, algorithm, and parameter meanings, matching the style of `bloom.cuh` / `tonemap.cuh`.
- **`__restrict__`** on all pointer parameters for compiler optimization hints.
- **`const` correctness** for read-only input buffers.
- **No file-scope statics** in `.cuh` files — all state lives in `PathTracerOptions`.

## Design Decisions & Trade-offs

### Chromatic Aberration: Bilinear vs Nearest-Neighbor

Bilinear interpolation adds ~3 extra texture reads per channel (4 vs 1) but eliminates aliasing from sub-pixel shifts. Since CA shifts are typically < 2 pixels, nearest-neighbor would produce visible stairstepping. The cost is negligible on modern GPUs (the kernel remains memory-bandwidth-bound rather than compute-bound).

### Chromatic Aberration: Radial vs Directional

A pure radial model (shift along the radius vector) is chosen over directional models because:
- It matches physical lens chromatic aberration (magnification varies with field angle).
- It produces zero shift at center (lens optical axis), which is physically correct.
- It produces maximum shift at corners, matching real-world lens behavior.

### Vignette: pow() vs smoothstep() or cos⁴

A simple `pow(ndist, exponent)` model is chosen over more complex formulas because:
- It matches the standard game-engine vignette implementation.
- The single `exponent` parameter provides intuitive control over falloff shape.
- It is computationally cheap (single `powf()` per pixel).
- More physically accurate models (cos⁴ law) offer no visual benefit for this use case.

### No New GPU Buffers

Reusing `bloomBufB` avoids allocating extra device memory. The bloom pipeline completes before CA/vignette run, so `bloomBufB` is always available. A single `cudaMemcpy` DeviceToDevice (optional, only when CA is enabled but vignette is not) copies ~8 MB at 1920×1080, which takes < 0.1 ms on modern GPUs.

## References

- Bevy Engine, "Chromatic Aberration" WGSL shader (radial variant with bilinear sampling)
- Playdead's "Inside" GDC 2016 presentation on post-processing
- GPU Gems 3, Chapter 28: "Post-Processing Effects on Mobile Devices"
- Real-Time Rendering, 4th Edition, §10.12 Bloom (vignette and lens effects)
- Unreal Engine 5 Post-Processing Effects documentation
