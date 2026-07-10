# Refraction Design — Fresnel Dielectric BSDF

> **Target:** INSTRUCTION.md Part 2 — Visual Improvements — :two: Refraction
>
> **Status:** design document (not implemented)
>
> **Requirements:**
> - Refraction (glass/water) with Fresnel effects
> - Schlick's approximation **and** accurate Fresnel — toggleable at runtime
> - Use `glm::refract` for Snell's law

---

## 1. Overview of Changes

| File | Change |
|------|--------|
| `sceneStructs.h` | Add `indexOfRefraction` to `Material` (already present), add `FresnelMode` enum |
| `scene.cpp` | Parse `"TYPE": "Refractive"` and `"IOR"` from JSON |
| `interactions.h` | Declare Fresnel helpers: `fresnelSchlick`, `fresnelAccurate` |
| `interactions.cu` | Implement both Fresnel variants + extend `scatterRay` for refraction/reflection branching |
| `pathtrace.h` | Declare `setFresnelMode(int)` / `getFresnelMode()` |
| `pathtrace.cu` | Add `g_fresnelMode` global + setter/getter |
| `main.cpp` | Parse `--fresnel=0/1` CLI flag |
| `scenes/` | Example scene with refractive sphere/cube |

---

## 2. Data Structures

### 2.1 `sceneStructs.h` — `Material` (existing + additions)

```cpp
struct Material
{
    glm::vec3 color;             // surface albedo / transmittance tint
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;         // non-zero = reflective
    float hasRefractive;         // non-zero = refractive (already exists!)
    float indexOfRefraction;     // IOR, e.g. 1.5 for glass, 1.33 for water (already exists!)
    float emittance;             // non-zero = emissive
};
```

**No changes needed to `Material`** — `hasRefractive` and `indexOfRefraction` already exist.

### 2.2 `sceneStructs.h` — Fresnel mode enum (new)

```cpp
// Fresnel evaluation mode — runtime toggle via --fresnel=N
enum class FresnelMode : int {
    Schlick  = 0,  // Schlick's approximation (fast, ~95% accurate for dielectrics)
    Accurate = 1   // Full unpolarized Fresnel equations (reference)
};
```

### 2.3 `sceneStructs.h` — surface hit side indicator (new)

```cpp
// Whether a ray hit the outside or inside of a surface.
// Needed for correct IOR ratio in Snell's law.
enum class HitSide : int {
    Outside = 0,  // ray struck the outside face (normal points toward ray origin)
    Inside  = 1   // ray struck the inside face (ray originated inside the medium)
};
```

---

## 3. Runtime Configuration

### 3.1 `pathtrace.h` — API

```cpp
// In addition to existing setters:
void setFresnelMode(int mode);   // 0 = Schlick, 1 = Accurate
int  getFresnelMode();
```

### 3.2 `pathtrace.cu` — global + getter/setter

```cpp
// Fresnel mode — Schlick by default (fast, good enough for most scenes)
static int g_fresnelMode = 0;

void setFresnelMode(int mode) { g_fresnelMode = mode; }
int  getFresnelMode()         { return g_fresnelMode; }
```

### 3.3 `main.cpp` — CLI parsing

```cpp
// In the for (int i = 2; i < argc; ++i) loop:
} else if (arg.rfind("--fresnel=", 0) == 0) {
    setFresnelMode(std::stoi(arg.substr(10)));
}
```

Usage:
```
# Schlick's approximation (default)
cis565_path_tracer.exe ../scenes/cornell.json

# Accurate Fresnel
cis565_path_tracer.exe ../scenes/cornell.json --fresnel=1
```

---

## 4. Fresnel Implementations

### 4.1 Physics Background

When light hits a dielectric interface (e.g. air→glass), part reflects and part transmits.
The **Fresnel reflectance** `R` is the fraction of power reflected. It depends on:

- `n₁` — IOR of the medium the ray is coming FROM
- `n₂` — IOR of the medium the ray is going INTO
- `cosθᵢ` — cosine of the incident angle (dot product of ray direction and surface normal)

The **transmittance** is `T = 1 - R` (conservation of energy).

### 4.2 `interactions.h` — declarations

```cpp
/**
 * Compute the Fresnel reflectance for a dielectric interface using
 * Schlick's approximation.
 *
 * @param cosThetaI   Absolute cosine of the incident angle (ray.dir dot normal).
 *                    Must be in [0, 1]. The caller handles sign (inside/outside).
 * @param n1          Index of refraction of the medium the ray is coming FROM.
 * @param n2          Index of refraction of the medium the ray is going INTO.
 * @return            Reflectance R in [0, 1].
 */
__host__ __device__ float fresnelSchlick(float cosThetaI, float n1, float n2);

/**
 * Compute the Fresnel reflectance using the full unpolarized dielectric
 * Fresnel equations.  Averages s-polarized and p-polarized reflectance.
 *
 * @param cosThetaI   Absolute cosine of the incident angle.
 * @param n1          IOR of incoming medium.
 * @param n2          IOR of outgoing medium.
 * @return            Reflectance R in [0, 1].
 */
__host__ __device__ float fresnelAccurate(float cosThetaI, float n1, float n2);

/**
 * Determine whether a ray is entering or exiting a medium.
 *
 * @param rayDir        Normalized ray direction (pointing TOWARD the surface).
 * @param surfaceNormal Normalized geometric normal (always points outward).
 * @param ior           The material's index of refraction.
 * @param outN1         [out] IOR of the medium the ray is coming FROM.
 * @param outN2         [out] IOR of the medium the ray is going INTO.
 * @param outCosThetaI  [out] Absolute cosine of the incident angle.
 * @return              HitSide::Outside or HitSide::Inside.
 */
__host__ __device__ HitSide classifyRefraction(
    glm::vec3 rayDir,
    glm::vec3 surfaceNormal,
    float     ior,
    float&    outN1,
    float&    outN2,
    float&    outCosThetaI);
```

### 4.3 `interactions.cu` — Schlick's approximation

```cpp
__host__ __device__ float fresnelSchlick(float cosThetaI, float n1, float n2)
{
    // Reflectance at normal incidence (cosθ = 1, i.e. ray perpendicular to surface)
    // R0 = ((n1 - n2) / (n1 + n2))^2
    float r0 = (n1 - n2) / (n1 + n2);
    r0 = r0 * r0;

    // Schlick's polynomial: R(θ) = R0 + (1 - R0) * (1 - cosθ)^5
    // This is an empirical approximation — extremely fast, one multiply-add chain.
    float oneMinusCos = 1.0f - cosThetaI;
    float oneMinusCos2 = oneMinusCos * oneMinusCos;          // (1-cosθ)^2
    float oneMinusCos5 = oneMinusCos2 * oneMinusCos2 * oneMinusCos;  // (1-cosθ)^5

    return r0 + (1.0f - r0) * oneMinusCos5;
}
```

### 4.4 `interactions.cu` — Accurate Fresnel

```cpp
__host__ __device__ float fresnelAccurate(float cosThetaI, float n1, float n2)
{
    // Snell's law: n1 * sinθi = n2 * sinθt
    // Compute cosθt (cosine of transmitted angle)
    float sinThetaI = sqrtf(fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI));
    float sinThetaT = (n1 / n2) * sinThetaI;

    // Total internal reflection — no transmission possible
    // Occurs when sinθt >= 1 (physically: light cannot exit to a lower-IOR medium
    // at grazing angles, e.g. looking up from underwater at a shallow angle)
    if (sinThetaT >= 1.0f)
    {
        return 1.0f;  // 100% reflection
    }

    float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sinThetaT * sinThetaT));

    // s-polarized (perpendicular) reflectance
    // Rs = ((n1*cosθi - n2*cosθt) / (n1*cosθi + n2*cosθt))^2
    float rParallel = (n2 * cosThetaI - n1 * cosThetaT) /
                      (n2 * cosThetaI + n1 * cosThetaT);

    // p-polarized (parallel) reflectance
    // Rp = ((n1*cosθt - n2*cosθi) / (n1*cosθt + n2*cosθi))^2
    float rPerpendicular = (n1 * cosThetaT - n2 * cosThetaI) /
                           (n1 * cosThetaT + n2 * cosThetaI);

    // Unpolarized light: average of both polarizations
    // R = (Rs^2 + Rp^2) / 2
    return (rParallel * rParallel + rPerpendicular * rPerpendicular) * 0.5f;
}
```

### 4.5 `interactions.cu` — media classification helper

```cpp
__host__ __device__ HitSide classifyRefraction(
    glm::vec3 rayDir,
    glm::vec3 surfaceNormal,
    float     ior,
    float&    outN1,
    float&    outN2,
    float&    outCosThetaI)
{
    // cosθ between ray direction and surface normal.
    // Negative = ray enters surface (hitting from outside).
    // Positive = ray exits surface (hitting from inside).
    float cosTheta = glm::dot(rayDir, surfaceNormal);

    if (cosTheta < 0.0f)
    {
        // Ray enters the medium: air (or whatever is outside) → material
        outN1        = 1.0f;         // assume air/vacuum outside
        outN2        = ior;          // material IOR
        outCosThetaI = -cosTheta;    // make positive (abs of dot product)
        return HitSide::Outside;
    }
    else
    {
        // Ray exits the medium: material → air
        outN1        = ior;          // material IOR
        outN2        = 1.0f;         // assume air/vacuum outside
        outCosThetaI = cosTheta;     // already positive
        return HitSide::Inside;
    }
}
```

---

## 5. `scatterRay` — Extended for Refraction

### 5.1 Pseudocode (design only — NOT implementation)

```
scatterRay(pathSegment, intersect, normal, material, rng):

    // --- Common ---
    offset ray origin along normal by EPSILON to prevent self-intersection
    pathSegment.remainingBounces--

    // --- Diffuse (existing, unchanged) ---
    if material is diffuse:
        newDir = cosine-weighted hemisphere sample around normal
        offset = intersect + normal * EPSILON
        color *= material.color
        return

    // --- Refractive (NEW) ---
    if material.hasRefractive:

        // 1. Determine entering vs exiting the medium
        (n1, n2, cosThetaI) = classifyRefraction(rayDir, normal, material.IOR)

        // 2. Compute Fresnel reflectance
        if g_fresnelMode == 0:
            R = fresnelSchlick(cosThetaI, n1, n2)
        else:
            R = fresnelAccurate(cosThetaI, n1, n2)

        // 3. Russian-roulette: probabilistically reflect OR refract
        //    Unbiased: divide by the chosen branch's probability
        if random01 < R:
            // --- REFLECT ---
            // Perfect specular reflection around the normal
            reflectDir = glm::reflect(rayDir, sign-adjusted normal)
            offset = intersect + normal * EPSILON
            color *= material.color      // tint the reflection
            color /= R                    // unbiased: compensate for branch probability
        else:
            // --- REFRACT ---
            // Use the geometric normal (always outward) for glm::refract.
            // When exiting, we need the INWARD normal — flip the normal.
            if exiting:
                refractNormal = -normal
            else:
                refractNormal = normal

            bool hasRefraction = glm::refract(rayDir, refractNormal, n1/n2, refractDir)

            if hasRefraction:
                // Ray successfully refracted — offset slightly into the medium
                // When entering: go just INSIDE (+normal * EPSILON)
                // When exiting:  go just OUTSIDE (-normal * EPSILON)
                offset = intersect + refractNormal * EPSILON
                color *= material.color
                color /= (1.0f - R)       // unbiased: compensate for branch probability
            else:
                // Total internal reflection — force reflection
                reflectDir = glm::reflect(rayDir, sign-adjusted normal)
                offset = intersect + normal * EPSILON
                color *= material.color
                // No probability division needed — this branch was deterministic
                // (R was already 1.0 from the Fresnel computation)
```

### 5.2 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Russian roulette split** (reflect OR refract, not both) | Both would double the number of active paths per bounce (exponential explosion). Splitting with probability `R` and compensating via `color /= R` or `color /= (1-R)` keeps the result unbiased. |
| **`classifyRefraction` helper** | Separates the entering/exiting logic from the scattering code. Easier to test, easier to read. The `cosTheta` sign determines the direction: negative → entering (ray hits from outside), positive → exiting (ray hits from inside). |
| **Always use geometric (outward-facing) normal** | The surface normal from intersection always points outward. `glm::refract` needs the normal pointing toward the incoming ray. The helper flips it when the ray is exiting. |
| **Total internal reflection fallback** | `glm::refract` returns `false` when `sinθt > 1` (TIR). In this case we fall back to specular reflection. The Fresnel equations already return `R=1.0` for TIR, so the probabilistic branch never selects refraction in this case — the fallback is a safety net. |
| **Reflection = specular (not diffuse)** | For a smooth dielectric surface, reflected rays follow the law of reflection exactly (`glm::reflect`). Rough/imperfect specular (glossy) is a separate feature. |
| **`color *= material.color` at every interaction** | Dielectric absorption (colored glass) is handled by tinting the throughput. A "pure" glass would have `RGB = [1,1,1]` — no absorption. |

---

## 6. Scene File Format

### 6.1 New material type: `"Refractive"`

```json
{
    "Materials": {
        "glass": {
            "TYPE": "Refractive",
            "RGB":  [0.98, 0.98, 0.98],
            "IOR":   1.5
        },
        "water": {
            "TYPE": "Refractive",
            "RGB":  [0.95, 0.97, 1.0],
            "IOR":   1.33
        }
    }
}
```

### 6.2 `scene.cpp` — material loading (addition)

```cpp
// New branch in the material TYPE parsing loop:
else if (p["TYPE"] == "Refractive")
{
    const auto& col = p["RGB"];
    newMaterial.color             = glm::vec3(col[0], col[1], col[2]);
    newMaterial.hasRefractive     = 1.0f;           // flag as refractive
    newMaterial.indexOfRefraction = p.value("IOR", 1.5f);  // default to glass
}
```

### 6.3 Example scene: glass sphere in Cornell box

```json
{
    "Materials": {
        "light":          { "TYPE":"Emitting",  "RGB":[1,1,1],   "EMITTANCE":5.0 },
        "diffuse_white":  { "TYPE":"Diffuse",   "RGB":[0.98,0.98,0.98] },
        "diffuse_red":    { "TYPE":"Diffuse",   "RGB":[0.85,0.35,0.35] },
        "diffuse_green":  { "TYPE":"Diffuse",   "RGB":[0.35,0.85,0.35] },
        "glass":          { "TYPE":"Refractive","RGB":[0.98,0.98,0.98], "IOR":1.5 }
    },
    "Camera": {
        "RES":[800,800], "FOVY":45.0, "ITERATIONS":500,
        "DEPTH":12, "RR_DEPTH":3, "FILE":"refraction_test",
        "EYE":[0,5,10.5], "LOOKAT":[0,5,0], "UP":[0,1,0]
    },
    "Objects": [
        { "TYPE":"cube",   "MATERIAL":"light",         "TRANS":[0,10,0],  "ROTAT":[0,0,0],   "SCALE":[3,0.3,3] },
        { "TYPE":"cube",   "MATERIAL":"diffuse_white", "TRANS":[0,0,0],   "ROTAT":[0,0,0],   "SCALE":[10,0.01,10] },
        { "TYPE":"cube",   "MATERIAL":"diffuse_white", "TRANS":[0,10,0],  "ROTAT":[0,0,90],  "SCALE":[0.01,10,10] },
        { "TYPE":"cube",   "MATERIAL":"diffuse_white", "TRANS":[0,5,-5],  "ROTAT":[0,90,0],  "SCALE":[0.01,10,10] },
        { "TYPE":"cube",   "MATERIAL":"diffuse_red",   "TRANS":[-5,5,0],  "ROTAT":[0,0,0],   "SCALE":[0.01,10,10] },
        { "TYPE":"cube",   "MATERIAL":"diffuse_green", "TRANS":[5,5,0],   "ROTAT":[0,0,0],   "SCALE":[0.01,10,10] },
        { "TYPE":"sphere", "MATERIAL":"glass",         "TRANS":[0,3.5,1], "ROTAT":[0,0,0],   "SCALE":[2,2,2] }
    ]
}
```

Note: `"DEPTH":12` — refractive paths need more bounces (enter → travel inside → exit → continue).

---

## 7. Toggle Summary

| Flag | Values | Default | Meaning |
|------|--------|---------|---------|
| `--fresnel=0` | `0` | ✓ | Schlick's approximation — fast, ~95% accurate for common dielectrics |
| `--fresnel=1` | `1` | | Full unpolarized Fresnel equations — physically correct, includes TIR handling |

Both modes share the same `scatterRay` code path; only the Fresnel reflectance computation differs. This makes it easy to measure the performance difference in isolation.

Schlick's is ~5 floating-point operations. Accurate Fresnel is ~20 operations plus a `sqrtf`. The runtime difference is negligible compared to the ray-scene intersection test, so the toggle is primarily for correctness validation, not performance.

---

## 8. Implementation Order (suggested)

1. **Add `FresnelMode` enum and `HitSide` enum** to `sceneStructs.h`
2. **Add `g_fresnelMode` global + setter/getter** to `pathtrace.cu` / `pathtrace.h`
3. **Implement `fresnelSchlick`, `fresnelAccurate`, `classifyRefraction`** in `interactions.cu`
4. **Extend `scatterRay`** with the refractive branch (reflection + refraction via Russian roulette)
5. **Parse `"TYPE": "Refractive"` and `"IOR"`** in `scene.cpp`
6. **Parse `--fresnel=0/1`** in `main.cpp`
7. **Create test scene** (`scenes/refraction_test.json`) with a glass sphere
8. **Increase `traceDepth`** in test scenes (refractive paths need ~12+ bounces for multi-bounce caustics)

---

## 9. Potential Pitfalls (for the implementor)

| Pitfall | Prevention |
|---------|-----------|
| **Total internal reflection not handled** | `fresnelAccurate` checks `sinThetaT >= 1.0f`. Also check `glm::refract` return value. |
| **EPSILON offset in wrong direction** | When entering: offset along `+normal`. When exiting: offset along `-normal` (the refracted ray normal was flipped). |
| **Division by zero in probability compensation** | Ensure `R` is clamped away from 0 and 1 before dividing, or handle the deterministic case separately. |
| **Infinite recursion inside a refractive object** | The `remainingBounces` decrement + Russian roulette should prevent this. Set `traceDepth` high enough (≥12). |
| **`color` blow-up from probability division** | The `color /= R` or `color /= (1-R)` compensates for Russian roulette. If many paths survive (low absorption), this is fine. |
| **Self-intersection after refraction** | The EPSILON offset along the refracted direction should prevent this. If shadow acne appears, increase EPSILON slightly. |
| **Refractive object embedded in another refractive object** | The current design assumes outside medium IOR = 1.0 (air). Nested dielectrics require tracking a media stack — out of scope for this feature. |

---

## 10. Future Extensions

- **Imperfect specular / glossy refraction** — jitter the refracted direction for frosted glass (requires roughness parameter on refractive materials)
- **Beer-Lambert absorption** — exponential color absorption based on distance traveled inside the medium (colored glass with thickness-dependent tint)
- **Nested dielectrics** — track a stack of IORs for objects inside other refractive objects (e.g. air bubble in water in glass)
- **Caustic photon mapping** — pure path tracing converges slowly for caustics; a photon map pass would accelerate convergence
