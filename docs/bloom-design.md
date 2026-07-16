# Bloom Post-Processing Effect Design

## Overview

Bloom (泛光) is a post-processing effect that simulates the scattering of bright light in a camera lens or the human eye. It produces a soft glow around high-intensity regions, enhancing the perception of brightness beyond what a display's limited dynamic range can convey.

## Pipeline Order: Why HDR Space Before Tone Mapping?

```
                    Linear HDR Space                          Display Space
                    ────────────────                          ─────────────
g_dev.image ──→ thresholdExtract ──→ blurH ──→ blurV ──→ tonemapKernel ──→ PBO
  (HDR sum)      (bright areas)    (H-temp)   (final)   (HDR+bloom→LDR)
```

**Bloom operates on linear HDR values BEFORE tone mapping.** This is critical:

1. **Physical meaning of threshold.** A threshold of `1.0` means "brighter than the display's nominal maximum." If we applied tone mapping first, the ACES S-curve would compress highlights, making the threshold arbitrary and scene-dependent.

2. **Energy is additive.** Bloom represents *actual light energy* scattering in the lens. Adding bloom energy in physical (linear) space, then tone-mapping the composite, is physically motivated. Tone-mapping the base image and bloom separately, then adding, would double-apply the transfer function.

3. **This is standard industry practice.** Unreal Engine, Unity HDRP, Blender Cycles, and virtually all film VFX pipelines do bloom in scene-referred linear space before the display transform.

## Algorithm

### Step 1: Threshold Extraction

```
bloomSource = max(HDR_average - threshold, 0)   // per-channel
```

Only pixels whose per-sample HDR average exceeds `threshold` contribute to bloom. This prevents dark/mid-tone areas from producing glow.

**Per-channel vs luminance-based thresholding:**
- We use **per-channel**: `max(pix.r - threshold, 0)`, same for G, B.
- This preserves the color of bright highlights (a red neon sign produces red bloom).
- Luminance-based thresholding (`max(luminance(pix) - threshold, 0) * color`) is slightly more physically accurate but adds a division that can produce NaNs at near-black pixels. Per-channel is the default in Unreal and Unity and gives equivalent visual results in practice.

### Step 2: Separable Gaussian Blur

A 2D Gaussian blur kernel can be factored into two 1D passes:
```
G(x, y) = G(x) * G(y) = [exp(-x²/2σ²)] * [exp(-y²/2σ²)]
```

This reduces complexity from O(N × r²) to O(2N × r) where N = pixel count and r = radius.

#### Shared Memory Tiling

Each 1D pass uses CUDA shared memory to load a tile plus `radius` halo pixels on each side:

```
Horizontal pass shared memory layout:
┌──────────┬──────────────────────────┬───────────┐
│  halo_L  │       center tile        │  halo_R   │
│ (radius) │    (BLOOM_BLOCK_SIZE)    │ (radius)  │
└──────────┴──────────────────────────┴───────────┘
```

- **Block size:** 256 threads (1D)
- **Shared memory:** `(256 + 2 × radius) × 12 bytes` (float3 per pixel)
- **Max shared memory:** ~3.8 KB at radius=30 — well within 48 KB limit
- **Launch config:**
  - Horizontal: `grid(ceil(width/256), height)`, `block(256, 1)`
  - Vertical:   `grid(ceil(height/256), width)`,  `block(256, 1)` (transposed)

#### Halo Loading Strategy

Each thread loads one center pixel. The first `radius` threads additionally load the left/top halo; the last `radius` threads load the right/bottom halo. This avoids extra kernel launches and minimizes global memory transactions.

### Step 3: Composite + Tone Map

Bloom is added to the original HDR image inside `tonemapKernel`, then the combined result passes through ACES tone mapping + sRGB gamma:

```
pix = (HDR_input + bloomIntensity * bloomBlurred) / iter
pix = ACES_tone_map(pix)        // S-curve + highlight desaturation
pix = LinearToSRGB(pix)         // Gamma encode for 8-bit display
```

Compositing inside tonemapKernel saves one full-screen kernel launch and one intermediate buffer.

## File Structure

```
src/postprocess/
├── tonemap.cuh        # ACES tone mapping + sRGB (modified: accepts bloom input)
├── bloom.cuh           # thresholdExtract + blurHorizontal + blurVertical kernels
```

## Configuration

`BloomConfig` (in `pathtrace.h`):

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `enabled` | bool | true | — | Master bloom toggle |
| `threshold` | float | 1.0 | 0.1–10.0 | HDR brightness cutoff |
| `intensity` | float | 0.5 | 0.0–2.0 | Bloom blend strength |
| `radius` | int | 10 | 1–30 | Gaussian blur radius (pixels) |
| `sigma` | float | radius/2 | — | Gaussian standard deviation (auto-computed) |

### Expected Visual Effects

| Setting | Effect |
|---------|--------|
| Low threshold (0.5) | Even moderately bright areas glow — "dreamy" look |
| High threshold (5.0) | Only very intense light sources bloom |
| Low intensity (0.2) | Subtle glow, barely noticeable |
| High intensity (1.5) | Strong halation, "overexposed" look |
| Small radius (3) | Tight, sharp glow around light sources |
| Large radius (25) | Wide, soft, atmospheric bloom |

## GPU Buffers

Two additional HDR buffers are allocated for separable blur ping-pong:

| Buffer | Size | Purpose |
|--------|------|---------|
| `g_dev.bloomBufA` | `W×H×12B` | Threshold output → final blurred bloom |
| `g_dev.bloomBufB` | `W×H×12B` | Horizontal blur intermediate |
| `g_dev.bloomWeights` | `65×4B` | 1D Gaussian kernel (max 2×32+1 entries) |

**Total overhead:** ~2.04× resolution (e.g., 800×600 → ~11.5 MB). All buffers are managed in `pathtraceInit()`/`pathtraceFree()`.

## Runtime Controls (ImGui)

```
[✓] Enable Bloom
    Threshold: [========|====] 1.00
    Intensity: [====|========] 0.50
    Radius:    [===|=========] 10
```

All controls are live — changes take effect on the next frame without restart.

## Design Decisions & Trade-offs

### No Downsampling (Yet)

A common optimization is to run bloom at 1/4 or 1/8 resolution (bilinear downsample → blur → upsample). We skip this for initial implementation because:
- At typical path-tracing resolutions (800×600 to 1920×1080), full-res blur with shared memory is already fast.
- Downsampling adds 2 more kernels (downsample, upsample) and 2 more buffers.
- The blur kernels take << 1 ms on modern GPUs.

Downsampling can be added later if bloom becomes a bottleneck.

### Sigma Auto-Derivation

Sigma is set to `radius / 2.0` automatically. This means:
- At radius=10, sigma=5.0 → ±2σ ≈ ±10 pixels covered by the kernel (good coverage).
- At radius=1, sigma=0.5 → mostly just neighbor pixels (tight blur).
- Reasonable for all practical radius values.

### Why Not a Compute Shader?

This is a CUDA path tracer — all processing happens in CUDA kernels on the same device as the accumulation buffer. Using CUDA kernels avoids GPU↔CPU round-trips and keeps all data resident in GPU memory.

## Adding Future Post-Processing Effects

The `src/postprocess/` directory is designed for extensibility. To add a new effect (e.g., vignette, chromatic aberration, film grain):

1. Create `src/postprocess/<effect>.cuh` with the kernel(s).
2. Add configuration to `PathTracerOptions` and extra buffers to `DeviceBuffers` in `pathtrace.h`.
3. Insert the kernel launch in `pathtrace()` in the appropriate pipeline position (before or after tone mapping, depending on the effect).
4. Add ImGui controls in `main.cpp`.

For longer-term maintainability, consider extracting a `runPostProcess()` function that orchestrates all post-processing kernels. This keeps `pathtrace()` focused on the core path-tracing loop.

## References

- Krzysztof Narkowicz, "ACES Filmic Tone Mapping Curve" (2016)
- Stephen Hill / MJP, "BakingLab ACES" (HLSL fitted ACES approximation)
- GPU Gems 3, Chapter 28: "Post-Processing Effects on Mobile Devices"
- Real-Time Rendering, 4th Edition, §10.12 Bloom
- NVIDIA CUDA Toolkit Documentation: Shared Memory
