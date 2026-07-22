# Direct Lighting (Next-Event Estimation)

## Overview

The current path tracer uses **pure unidirectional path tracing** — light enters the accumulation buffer only when a BSDF-sampled continuation ray randomly happens to hit an emissive surface. This is inefficient for small or distant light sources.

**Direct Lighting (Next-Event Estimation, NEE)** explicitly samples a point on an emissive object at each non-emissive surface hit, casts a shadow ray, and adds the contribution directly. The BSDF continuation ray still carries indirect illumination.

This implements the explicit light sampling technique described in PBRTv4 §13.4.

---

## Algorithm

At each non-emissive surface hit point `x` with surface normal `n_x`:

### Step-by-step

1. **Select a light source** — Uniformly pick one emissive geometry from the scene's light list.
2. **Sample a point** `y` on its surface with uniform area PDF `p_A = 1 / A_total`.
3. **Set up shadow ray** — Ray from `x + EPSILON * n_x` toward `y`. Maximum distance = `||y - x||`.
4. **Visibility test** — If any geometry occludes (`0 < t < dist - EPSILON`), contribution is zero (the light is blocked).
5. **Compute geometry term**:
   ```
   G = max(0, dot(n_x, wi)) * max(0, dot(n_y, -wi)) / ||x - y||^2
   ```
   where `wi = normalize(y - x)` is the direction from surface toward light, and `n_y` is the light surface normal.
6. **Evaluate BSDF** — For diffuse surfaces: `fr = albedo / PI` (Lambertian, energy-conserving). The BSDF is evaluated for `wo` (outgoing toward camera) and `wi` (incoming from light).
7. **Look up emission** — `Le = material_y.color * material_y.emittance`.
8. **Unbiased estimator**:
   ```
   L_direct = throughput * fr * Le * V * G / p_A
   ```
   where `p_A = 1 / totalArea` is the combined PDF of picking this light and this point, in area measure.
9. **Accumulate** — Add `L_direct` to the pixel's accumulation buffer via `atomicAdd` (component-wise).
10. **Continue** — The indirect path continues with `scatterRay` on the original throughput (not reduced by direct lighting).

### Energy Conservation

The estimator is unbiased:

```
Sampling in solid angle measure:
  L = fr * Le * V * |cosθ_receiver| / p_ω

Area-to-solid-angle conversion (Jacobian):
  p_ω = p_A * |cosθ_light| / r²

Substituting:
  L = fr * Le * V * |cosθ_receiver| / (p_A * |cosθ_light| / r²)
    = fr * Le * V * |cosθ_receiver| * r² / (p_A * |cosθ_light|)
    = fr * Le * V * G / p_A

where G = |cosθ_receiver| * |cosθ_light| / r²
      p_A = 1 / totalArea

Expected value:
  E[L] = ∫_A fr * Le * V * G dA = true direct illumination  ✓
```

The indirect path (BSDF continuation ray) remains unchanged and unbiased. It can still independently hit light sources — both contributions accumulate to the same pixel, and their sum converges to the correct result.

### Implementation Note — Only Diffuse Materials

In this initial implementation, NEE is only evaluated for `MaterialType::Diffuse` surfaces. Specular/reflective/refractive materials rely on their BSDF sampling for both direct and indirect light. This avoids the complexity of evaluating the glossy BSDF for arbitrary sampled light directions (which would have near-zero contribution for mirror-like surfaces anyway). MIS (Multiple Importance Sampling) can combine both strategies in a future extension.

---

## Multiple Importance Sampling (MIS) — Power Heuristic

### Motivation

Pure NEE (one strategy — light sampling) produces fireflies when the
light-sampled direction coincides with a low-weight region of the BSDF.
For diffuse surfaces this is less severe than for glossy ones, but even
the cosine-weighted BSDF can have arbitrarily small values near the
horizon (cosθ → 0), causing the `fr * G / p_A` ratio to explode.

MIS eliminates fireflies at the source by combining **two** sampling
strategies and down-weighting each where its PDF is small relative
to the other.

### Combined Two-Strategy Estimator

At each non-emissive diffuse surface hit, evaluate both strategies
and weight by the power heuristic (β=2):

```
L_direct = w_light * f(ω_light) / p_light(ω_light)
         + w_bsdf  * f(ω_bsdf)  / p_bsdf(ω_bsdf)

where:

  f(ω) = throughput * fr(ω) * Le(ω) * V(ω) * |cosθ_receiver|
       (the integrand — incident radiance × BSDF × cosine)

  w_s(ω) = p_s(ω)² / (p_light(ω)² + p_bsdf(ω)²)
         (power heuristic, β=2)

  w_light(ω) + w_bsdf(ω) = 1  for any ω where both PDFs > 0
```

The estimator is unbiased (proof in PBRTv4 §14.6).

### Strategy 1 — Light Sampling (existing NEE, now weighted)

The existing `evaluateDirectLighting` already computes:

```
f(ω_light) / p_light(ω_light) = throughput * fr * Le * V * G / p_A
```

Add the MIS weight to the existing contribution before atomicAdd:

```
p_light_solidAngle = p_A * |cosθ_light| / r²    (area → solid-angle PDF)
p_bsdf(ω_light)    = |cosθ_receiver| / PI       (cosine-weighted hemisphere PDF)
w_light            = p_light² / (p_light² + p_bsdf²)

directContrib *= w_light
```

When `|cosθ_receiver| → 0` (light at the horizon), `p_bsdf → 0` and
`w_light → 1` — light sampling naturally gets full weight where BSDF
sampling cannot reach.

### Strategy 2 — BSDF Sampling (new)

After `scatterRay` sets the continuation ray direction `ω_bsdf`, trace
the scattered ray against all scene geometry to find the closest hit:

- **Closest hit is emissive**: the scattered ray reaches a light source
  directly. Compute:
  ```
  Le = emittedRadiance of the hit light
  p_bsdf(ω_bsdf) = |dot(normal, ω_bsdf)| / PI
  p_light_solidAngle(ω_bsdf) = (1/(numLights * lightArea))
                                 * |dot(lightNormal, -ω_bsdf)| / dist²
  w_bsdf = p_bsdf² / (p_bsdf² + p_light_solidAngle²)
  contrib_bsdf = w_bsdf * pathSegment.color * Le
                (pathSegment.color after scatterRay carries f/p_bsdf)
  ```
  Accumulate via `atomicAdd`, then **terminate the path** (set
  `remainingBounces = 0`).  The direct portion has been fully accounted
  for; the next bounce must not double-count it via the emissive-hit
  branch.

- **Closest hit is non-emissive, or no hit**: the BSDF direction does
  not directly reach a light.  The path continues normally with indirect
  scattering — BSDF sampling contributes zero direct light this bounce.

### Geometry Traversal

The BSDF ray is traced using the same O(numGeoms) loop as the shadow ray
test, reusing `boxIntersectionTest` / `sphereIntersectionTest`.  A single
pass finds the closest intersection; if the hit geometry's index matches
a `LightInfo` entry, it is emissive.

### Implementation Notes

- **Diffuse only**: MIS is only applied for `MaterialType::Diffuse`,
  consistent with the existing NEE guard.  Specular/refractive materials
  continue to rely solely on BSDF sampling for direct light.
- **No new fields**: MIS weights are computed on-the-fly from known
  quantities (PDFs, cosines).  No changes to `PathSegment`, `ShadingConfig`,
  or `LightInfo`.
- **Path termination**: When BSDF sampling hits a light, termination
  prevents double-counting through the emissive-hit path in the next
  iteration of `shadeMaterial`.  This is correct because an emissive
  surface does not reflect light in the current material model — there
  is no indirect component to lose.
- **Power heuristic β=2**: Standard choice.  Higher β reduces variance
  further but can increase bias in the presence of PDF estimation error;
  β=2 is the PBRT default.

---

## RNG Dimensions

Added to the existing Halton sequence (all 16 dimensions now used):

| Dim | Prime | Usage | Where |
|-----|-------|-------|-------|
| 10 | 31 | `LightSelection` — pick which emissive geometry to sample | `sampleLightSource` |
| 11 | 37 | `LightSampleU` — u coordinate on light surface | `samplePointOnLight` |
| 12 | 41 | `LightSampleV` — v coordinate on light surface | `samplePointOnLight` |

---

## Data Structures

### `LightInfo` — GPU descriptor for one emissive geometry

```cpp
/**
 * LightInfo — Compact description of one emissive geometry (light source).
 *
 * Stored in a GPU device array and passed to the shading kernel for
 * direct-light sampling.  Each emissive geometry in the scene produces
 * one entry.
 *
 * The emittedRadiance field is pre-multiplied (color × emittance) at
 * init time to avoid a device-memory lookup of the Material array.
 */
struct LightInfo {
    int geomIndex;              // index into the device geoms array
    float area;                 // world-space surface area
    float inverseArea;          // 1/area (precomputed, avoids GPU division)
    glm::vec3 emittedRadiance;  // material.color * material.emittance (Le)
};
```

### `ShadeableIntersection` — new field

```cpp
// Added after existing fields:
int geomIndex;     // index of the hit geometry in the geoms array (-1 = miss)
```

This is needed by the direct lighting code to look up geometry transforms when sampling light points. It is set in the `computeIntersections` kernel.

### `ShadingConfig` — new fields

```cpp
// Added to the ShadingConfig struct:
int numLights;           // number of emissive geometries (0 = no direct lighting)
LightInfo* lightInfos;   // device array of LightInfo descriptors (nullptr if none)
Geom* geoms;             // device array of all geometry (for light-sampling transforms)
int numGeoms;            // total geometry count (for shadow ray bounds)
float totalLightArea;    // sum of all emissive surface areas (for PDF computation)
```

---

## Surface Area Computation

For proper PDF calculation, the world-space surface area of each emissive geometry must be computed. These are computed on the host during `pathtraceInit`:

| Geometry | Formula | Notes |
|----------|---------|-------|
| **Cube** | `2 * (sx*sy + sx*sz + sy*sz)` | Canonical cube `[-0.5, 0.5]³` scaled by `(sx, sy, sz)` |
| **Sphere** | `PI * s²` (uniform scale) | Sphere radius 0.5 in object space. For non-uniform scale, uses volume-equivalent radius `0.5 * cbrt(sx * sy * sz)` |

**Light selection**: Uniform among lights (equal probability each light). Area-weighted selection would be better but is left for future optimization.

**PDF**: When selecting uniformly among N lights:
```
p_light = 1 / numLights
p_area = 1 / light.area
totalPdf = p_light * p_area = 1 / (numLights * area)
```

Since we sample uniformly among all lights and then uniformly on the chosen light's surface, the combined PDF in area measure is:
```
p_A = 1 / (numLights * area_of_selected_light)
```

**Wait — correction for uniform light selection**: If we select lights uniformly (not area-weighted), the combined PDF in area measure is `1 / (numLights * lightArea_i)` where `lightArea_i` is the area of the selected light. The total PDF for the MIS weight should be computed per-sample.

For this implementation (uniform light selection, then uniform area sampling on that light):
```
lightPdf = 1.0f / numLights
areaPdf  = 1.0f / selectedLight.area
totalPdf = lightPdf * areaPdf = 1.0f / (numLights * selectedLight.area)
```

This is passed directly to the estimator as `p_A = totalPdf`.

---

## Light Surface Sampling

### Sphere

Uniform sampling on the sphere surface (radius 0.5 in object space):

```cpp
float theta = TWO_PI * u;           // azimuth
float phi = acosf(1.0f - 2.0f * v); // zenith (uniform on sphere surface)
glm::vec3 objPoint(
    0.5f * sinf(phi) * cosf(theta),
    0.5f * cosf(phi),
    0.5f * sinf(phi) * sinf(theta)
);
// Transform to world space
outPoint = multiplyMV(geom.transform, glm::vec4(objPoint, 1.0f));
```

### Cube

Uniform sampling over 6 faces:

```cpp
int face = (int)(u * 6.0f);  // 0..5, equally likely per face
// Face coordinates in [-0.5, 0.5]:
float fx = v - 0.5f;
// Need a third random value for the second face coordinate.
// For simplicity and to avoid another RNG call, use:
//   faceU = LightSampleU (already consumed for face selection)
//   faceV = LightSampleV (the 'v' param)
// But LightSampleU was consumed for face selection. 
// Solution: call rng.next() again — each call is a fresh dimension.
float fy = rng.next(HaltonDim::LightSampleV) - 0.5f;
// Actually, re-consume dim 11: we only need 2 dimensions per light sample
// (face selection reuses the same sample, and then we need 2 for the face UV).
// Better to use 3 draws: face, faceU, faceV all from the same state.
```

**Practical implementation**: Consume 3 sequential draws from the RNG state for light surface sampling. Since the RNG uses multi-dimensional Halton, each draw is a unique dimension (or advances the engine in LCG mode), producing independent values.

---

## Shadow Ray Test

```cpp
/**
 * Any-hit intersection for shadow rays.
 *
 * Unlike computeIntersections which finds the CLOSEST hit, this function
 * returns as soon as ANY intersection is found (early exit).
 *
 * @param ray        Shadow ray (origin already offset by EPSILON along normal)
 * @param maxT       Distance to the light point (use maxT - EPSILON as upper bound)
 * @param geoms      Device array of all geometry
 * @param numGeoms   Number of geometries
 * @return           true if occluded, false if the light is visible
 */
__device__ bool testShadowRay(const Ray& ray, float maxT,
                               const Geom* geoms, int numGeoms)
{
    for (int i = 0; i < numGeoms; i++)
    {
        glm::vec3 tmp_point, tmp_normal;
        bool outside;
        float t = intersectSingleGeom(geoms[i], ray, tmp_point, tmp_normal, outside);
        // EPSILON avoids self-intersection with the originating surface.
        // maxT - EPSILON avoids hitting the light surface itself (the light
        // emitter should not shadow itself — it's a surface we're aiming at).
        if (t > EPSILON && t < maxT - EPSILON)
            return true;  // occluded
    }
    return false;  // visible
}
```

---

## Integration in the Shading Kernel

### Modified `shadeMaterial` flow

```
shadeMaterial(iter, num_paths, intersections, paths, materials,
              ShadingConfig config, Geom* geoms, int numGeoms,
              LightInfo* lightInfos, int numLights, float totalArea,
              glm::vec3* imageAccum)
{
    for each active path:
        if remainingBounces <= 0: skip
        if intersection.t < 0: set color = black, terminate (unchanged)
        if material.emittance > 0: multiply by emission, terminate (unchanged)

        --- NEW: Direct Lighting ---
        if numLights > 0 && material.type == Diffuse:
            1. Sample light → point, normal, Le, pdf
            2. Compute shadow ray direction wi = normalize(lightPoint - hitPoint)
            3. If visible:
               - bsdf = albedo / PI
               - G = max(0, dot(normal, wi)) * max(0, dot(lightNormal, -wi)) / r²
               - contrib = throughput * bsdf * Le * G / pdf
               - atomicAdd to imageAccum[pixelIndex]
        --- End Direct Lighting ---

        scatterRay(pathSegment, ...)    (indirect, unchanged)
        russianRoulette(...)             (unchanged)
}
```

### Launch site (`pathtrace.cu`)

```cpp
// Before bounce loop, build ShadingConfig:
ShadingConfig shadingCfg = {
    traceDepth, hst_scene->state.rrMinBounces,
    hst_scene->state.fresnelMode, g_opts.rngMode, cam, hst_scene->state.debug,
    g_dev.numLights, g_dev.lightInfos, g_dev.geoms,
    (int)hst_scene->geoms.size(), g_dev.totalLightArea
};

// In bounce loop:
LAUNCH_KERNEL_AUTO(shadeMaterial, num_paths,
    iter, num_paths,
    g_dev.intersections, g_dev.paths, g_dev.materials,
    shadingCfg,
    g_dev.geoms, hst_scene->geoms.size(),
    g_dev.lightInfos, g_dev.numLights, g_dev.totalLightArea,
    g_dev.image);
```

---

## Pipeline Integration

```
pathtraceInit()
├── Copy geoms, materials to device (unchanged)
├── Scan for emissive geometry → build LightInfo array → copy to device  ← NEW
└── Allocate all buffers (unchanged)

pathtrace() — per iteration:
├── generateRayFromCamera (unchanged)
├── Bounce loop:
│   ├── computeIntersections (now stores geomIndex)
│   ├── sortPathsByMaterial (unchanged)
│   ├── shadeMaterial (now with direct lighting + imageAccum atomicAdd)
│   └── compactActivePaths (unchanged)
├── finalGather (unchanged — catch-all for paths that hit lights indirectly)
└── runPostProcess (unchanged)
```

---

## File Change Summary

### New Files

| File | Purpose |
|------|---------|
| `src/lighting/light_sampling.h` | Declarations: `samplePointOnLight`, `sampleLightSource`, `testShadowRay` |
| `src/lighting/light_sampling.cu` | Implementations of all three functions |

### Modified Files

| File | Changes |
|------|---------|
| `src/sceneStructs.h` | New `LightInfo` struct; add `geomIndex` to `ShadeableIntersection`; add `numLights`, `lightInfos`, `geoms`, `numGeoms`, `totalLightArea` to `ShadingConfig` |
| `src/rng/rng.h` | New HaltonDim: `LightSelection` (10), `LightSampleU` (11), `LightSampleV` (12); update comments |
| `src/pathtrace.h` | Add `LightInfo* lightInfos`, `int numLights`, `float totalLightArea` to `DeviceBuffers` |
| `src/pathtrace.cu` | `pathtraceInit`: scan for emissive geoms, build/copy LightInfo array; `pathtraceFree`: free LightInfo buffer; `pathtrace`: pass new args to `shadeMaterial` |
| `src/kernels/intersection.cuh` | Store `hit_geom_index` in `ShadeableIntersection.geomIndex` |
| `src/kernels/shading.cuh` | Add `imageAccum` parameter; insert NEE code before `scatterRay`; add `Geom*`, `LightInfo*`, light count params |
| `CMakeLists.txt` | Add `src/lighting/light_sampling.cu` to sources; add `src/lighting/light_sampling.h` to headers |

---

## Verification

1. **Build**: `make` compiles with no errors. New `.cu` files are in CMakeLists.txt.
2. **Baseline correctness**: Render `cornell.json` (800×800, 1000 iterations) — image should be visually correct, no black pixels or fireflies from the NEE code.
3. **Faster convergence**: At low iteration counts (50-200), NEE should produce noticeably less noise on diffuse walls and the ceiling.
4. **Furnace test**: `FurnaceDiffuse.json` — the scene should be uniformly lit; NEE should not introduce any energy gain or loss compared to the non-NEE version.
5. **Energy conservation**: No color channels should blow up or go negative. The shadow ray test should not produce artifacts around shadow boundaries that differ from the reference.
