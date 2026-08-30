# Direct Lighting: Triangle NEE + MIS

> **Status: implemented on `codex/direct-lighting-nee`; pending user build and
> render validation.** It is written against the current `src/` tree, not the
> original primitive-based proposal. The intended end state is an efficient,
> unbiased direct-light estimator with explicit protections against PDF-related
> fireflies.

## 1. Current renderer facts and resulting constraints

The current renderer is pure unidirectional path tracing. A light contributes
only when a BSDF continuation ray happens to hit it. This converges slowly for
Sponza-like spaces, where a diffuse point sees a small solid angle of the
ceiling emitter and many indirect bounces are needed.

The implementation must preserve these source-level facts:

- `buildSceneBvh()` bakes every `Geom` into **one world-space triangle BVH**.
  There is no device `Geom` array, no per-mesh traversal, and no primitive
  sphere/cube intersection path.
- The traversal hot path reads `TrianglePos` only and produces a 20-byte
  `HitRecord` (`t`, barycentrics, `triangleIndex`). `shadeMaterial`
  expands `TriangleAttr`, `Surface`, and `SurfaceBinding` after the
  closest triangle has been selected. NEE must not enlarge `HitRecord`.
- `traverseBvhClosest(ray, nodes, triangles, maxT)` already accepts a far
  plane. A shadow query must reuse this world-space BVH, rather than restore
  an O(number of geoms) loop.
- JSON `Emitting` surfaces terminate on a hit and use
  `resolveEmissive(binding, textures, uv, material) * material.emittance`.
  glTF/OBJ auto-glow uses the same `resolveEmissive` result additively and
  continues scattering. Both are radiance sources and must be eligible for
  NEE when their emitted radiance can be non-zero.
- `PathSegment::accumulatedRadiance` is gathered once on termination. NEE
  must add to that per-path value; it must **not** write to `image` with
  `atomicAdd`.
- Current diffuse sampling is cosine-weighted. Current PBR/GGX sampling is a
  Fresnel-weighted lobe mixture, but exposes neither a BSDF evaluator nor a
  solid-angle PDF. Correct MIS therefore requires a small BSDF-interface
  refactor before glossy NEE is enabled.
- Halton dimensions 0--10 are occupied; 11--15 are reserved. One direct-light
  sample needs three independent dimensions.

`pathtrace()` remains one primary path per pixel per iteration. Compaction may
move path records, but `pixelIndex` remains the stable accumulation address.

## 2. Target estimator

At a non-delta surface point `x`, sample an emissive triangle and a point
`y` on it. Let:

```text
wi       = normalize(y - x)                 direction from receiver to light
r2       = dot(y - x, y - x)
nx       = receiver shading normal, oriented to the current ray side
nl       = normalized cross(light.v1-light.v0, light.v2-light.v0)
cosX     = max(0, dot(nx, wi))
cosL     = emissionCosine(nl, -wi)          material-directed emission
f(x,wo,wi)                                  receiver BSDF value
Le(y,-wi)                                   emitted radiance at sampled UV
```

Triangle intersection remains double-sided, but emission direction is now an
explicit material property. `EMISSION_SIDEDNESS: "OneSided"` uses
`max(0, dot(nl, -wi))`; the backwards-compatible default `TwoSided` uses the
absolute value. The same helper gates both NEE and BSDF-hit emission, so the
two MIS strategies always describe the same light source.

If triangle `i` is selected with discrete PMF `pSelect(i)` and its world
area is `Ai`, then its area-measure PDF is:

```text
pA = pSelect(i) / Ai
```

The direct-light contribution is evaluated in area measure:

```text
C_light = beta * Le * f * cosX * cosL / (r2 * pA) * V * w_light
```

where `beta` is the incoming path throughput and `V` is binary visibility.
This form is preferred for the contribution because it has no division by
`cosL`; a grazing light sample naturally tends to zero instead of producing a
large intermediate value.

MIS compares PDFs in **the same measure**, so it uses solid angle:

```text
pLightOmega = pA * r2 / cosL
pBsdfOmega  = BSDF mixture PDF for wi
w_light     = pLightOmega^2 / (pLightOmega^2 + pBsdfOmega^2)
```

The conversion above is the critical one:

```text
d omega = cosL / r2 * dA
p(omega) = p(A) * dA/d omega = pA * r2 / cosL
```

It is **not** `pA * cosL / r2`. Reversing this Jacobian was a flaw in the old
proposal and gives wrong MIS weights, especially for grazing or distant
emitters.

For a BSDF-sampled continuation that directly hits emissive triangle `i`, use:

```text
w_bsdf = pBsdfOmega^2 / (pBsdfOmega^2 + pLightOmega^2)
C_bsdf = beta_after_scatter * Le * w_bsdf
```

Here `pLightOmega` is evaluated for the actual hit point on `i`, using that
triangle's stored `pSelect(i)` and area. For a delta event (perfect mirror or
specular refraction), or a camera ray, set `w_bsdf = 1`: light sampling cannot
sample a Dirac direction, so there is no competing continuous strategy.

Use the power heuristic (exponent two) with a scale-safe implementation:

```text
powerHeuristic(a, b):
    if a == 0: return 0
    m = max(a, b)
    a /= m; b /= m
    return a*a / (a*a + b*b)
```

Rescaling before squaring avoids an avoidable float overflow when a light is
nearly edge-on. It does not alter the mathematical weight.

## 3. Efficient light representation

Build the light sampler **after** `buildSceneBvh()` has baked and flattened
the world-space triangle arrays. This is the only point where the final GPU
triangle index, world area, runtime `Surface`, and source binding are all
available together.

The host creates one entry per emissive triangle:

```cpp
struct LightTriangle {
    int   triangleIndex;  // index into the reordered world-space arrays
    float area;           // 0.5 * length(cross(v1-v0, v2-v0))
    float selectPmf;      // exact discrete pSelect(triangleIndex)
};

struct AliasEntry {
    float q;              // Walker/Vose alias threshold in [0, 1]
    int   alias;
};
```

An additional `int lightIndexByTriangle[numTriangles]`, initialized to `-1`,
maps an emissive hit triangle to its `LightTriangle` in O(1). It is read only
on an emission hit, not by the traversal hot loop.

### Selection distribution

Use a Walker/Vose alias table, sampled in O(1), rather than one uniform pick
per mesh or a linear scan. The selection weight is:

```text
weight_i = area_i * estimatedMeanLuminance_i
pSelect(i) = weight_i / sum(weight_j)
```

`estimatedMeanLuminance_i` is a host-side importance estimate only:

- flat JSON emission: luminance of `material.color * emittance`;
- factor-only asset emission: luminance of
  `emissiveFactor * emissiveStrength`, multiplied by `emittance` when present;
- an emissive texture: mean linear luminance of the decoded texture, multiplied
  by the corresponding factor/strength/emittance.

The device must still evaluate directional `Le(y,-wi)` at the sampled triangle
UV through one shared emitted-radiance function. A texture average is not the
final radiance; it merely concentrates samples. Triangles below the small
mean-linear-luminance cutoff are excluded from explicit light sampling to
avoid JPEG near-black residue turning an entire mesh into NEE lights. This is
unbiased: the omitted emission remains visible to BSDF paths, whose light PDF
is zero and therefore receives MIS weight one.

The light list includes both terminating JSON emitters and non-terminating
auto-glow surfaces. Eligibility and `Le` evaluation must come from a single
helper shared by the existing emission-hit branch and NEE. Duplicating the
current `resolveEmissive` rules is an eventual brightness mismatch.

### Sampling a point on a triangle

With two uniform values `u1`, `u2`:

```text
s  = sqrt(u1)
b0 = 1 - s
b1 = s * (1 - u2)
b2 = s * u2
y  = b0*v0 + b1*v1 + b2*v2
uv = b0*uv0 + b1*uv1 + b2*uv2
```

This is uniform over area. It must not use three independently normalized
random numbers, which does not produce the required distribution.

Reserve these RNG dimensions in `rng.h`:

| Dimension | Prime | Use |
|---:|---:|---|
| 11 | 37 | alias-table light selection |
| 12 | 41 | triangle barycentric `u1` |
| 13 | 43 | triangle barycentric `u2` |

They are drawn from the same bounce-local `RngState`, consistent with existing
Halton usage. No random value may be reused for both an alias decision and a
triangle coordinate.

## 4. Visibility: a bounded BVH any-hit query

NEE adds one shadow query only for a valid non-delta light sample. It should
not call the current closest-hit traversal and discard its detailed result, and
must never restore the old per-geometry loop.

Add `traverseBvhAnyHit(ray, nodes, triangles, maxT)` beside
`traverseBvhClosest` in `src/bvh/bvh.h`:

- same world-space nodes, positions, slab tests, near-first ordering, and
  `RAY_EPSILON` lower bound;
- return immediately on the first triangle intersection with `t < maxT`;
- no barycentrics, attributes, normals, or materials are loaded;
- caller gives the upper bound strictly before the sampled point on the light.

The shadow origin must use the current scale-aware `offsetRayOrigin`, not the
former fixed `EPSILON`. Offset along the **unperturbed geometric side** that
contains `wi`; use the shading/normal-mapped normal only for the BSDF cosine.
The scale passed to the offset includes the triangle extent, so a large
triangle centered near the origin does not under-offset because its hit point
has a small coordinate magnitude.
The query also receives the flattened receiver-triangle index and skips that
one primitive as a numerical self-hit guard; every other triangle remains a
valid occluder. A sampled light triangle equal to the receiver is discarded,
because a zero-thickness triangle has no nonzero solid angle to itself.
This prevents normal-map terminator leaks while retaining the current robust
large-coordinate behavior.

After offsetting, derive the limit from the actual origin:

```text
tLight = dot(y - shadowOrigin, wi)
maxT   = nextafterf(tLight, 0)
```

The one-ULP reduction prevents the explicitly sampled endpoint from
self-occluding due to rounding. It is not a hand-tuned world-unit epsilon.
Any invalid/non-finite value, `tLight <= 0`, `area <= 0`, `r2 <= 0`,
`cosX <= 0`, or `cosL <= 0` contributes exactly zero and launches no invalid
traversal.

## 5. BSDF interface required for correct MIS

Do not bolt NEE onto `scatterRay()` as it exists today. Its output is a
sampled direction and a compensated throughput, not the `f`, continuous PDF,
or delta classification required by MIS. Recomputing an approximate PDF in the
shading kernel is precisely the type of mismatch that produces rare, enormous
samples.

Before enabling glossy NEE, factor the current scattering semantics into a
single shared interface conceptually equivalent to:

```cpp
struct BsdfSample {
    glm::vec3 wi;
    glm::vec3 weight;      // f * abs(N dot wi) / pdf for this sampled event
    float pdfOmega;        // continuous mixture PDF; 0 for a delta event
    bool isDelta;
};

glm::vec3 evaluateBsdf(..., glm::vec3 wo, glm::vec3 wi);
float evaluateBsdfPdf(..., glm::vec3 wo, glm::vec3 wi);
BsdfSample sampleBsdf(..., RngState& rng);
```

The concrete names may differ, but all three operations must share material
parameter resolution, normal-map orientation, the Fresnel lobe probability,
and texture sampling. `scatterRay()` then becomes the caller that applies
`BsdfSample` to the path.

Required material behavior:

- **Diffuse:** `f = albedo / PI`, `pdf = cosX / PI`. This is the first NEE
  implementation target and covers Sponza's primary transport.
- **Rough PBR/GGX:** direct evaluation uses the same GGX `D`, Smith `G`,
  Schlick Fresnel, and diffuse/specular mixture as sampling. The continuous
  PDF is the weighted sum of every continuous lobe that can produce `wi`, not
  merely the selected lobe's PDF.
- **Smooth mirror / refractive events:** delta. Do not launch an NEE shadow
  ray, and use MIS weight one when the sampled direction reaches an emitter.
- **Smooth PBR with both a delta specular lobe and diffuse lobe:** NEE
  evaluates the continuous diffuse part only. A diffuse sampled continuation
  stores its continuous mixture PDF; a specular sampled continuation is delta.

This refactor is required for correctness, not abstraction for its own sake.
It also makes the existing lobe probability compensation testable directly.

Store one `float previousBsdfPdfOmega` in `PathSegment`; initialize it to
zero for camera rays and write zero for delta samples. On a direct emitter hit,
zero means `w_bsdf = 1`; otherwise it is the competing BSDF PDF. This avoids
changing `HitRecord` and adds only the state MIS actually needs.

## 6. Shading and accumulation flow

The intended bounce order in `shadeMaterial` is:

```text
expand closest HitRecord -> SurfaceBinding / normal / UV
|
|-- hit emission:
|     Le = shared evaluateEmittedRadiance(...)
|     accumulatedRadiance += throughput * Le * MIS_for_previous_event
|     if JSON Emitting: terminate
|     otherwise continue to BSDF sampling
|
|-- non-delta BSDF-capable surface:
|     sample one LightTriangle + one uniform point
|     evaluate f, pBsdfOmega, Le, geometry, and visibility
|     accumulatedRadiance += C_light
|
|-- sample BSDF continuation; record previousBsdfPdfOmega / delta state
|-- apply Russian roulette to continuation only
```

The terminating-emitter assignment in the current shader must become `+=`,
not `=`. Before NEE this distinction is usually invisible because a path only
needed one terminal contribution; after NEE, assignment would erase all direct
lighting banked at earlier vertices. The same rule preserves earlier auto-glow
contributions.

`gatherTerminatedPaths` and compaction already collect
`accumulatedRadiance`; no atomics and no additional image buffer are needed.
Direct-light energy is banked before Russian roulette, because roulette only
estimates the continuation path. Applying its survival compensation to an
already evaluated direct sample would be wrong.

## 7. Firefly prevention: invariants before tuning

MIS reduces variance; it does **not** excuse invalid PDFs, nor does it make all
fireflies impossible. The implementation must satisfy these invariants before
considering a contribution clamp or a visual tweak.

| Risk | Required invariant / response |
|---|---|
| Mixed area and solid-angle PDFs | Area PDF is used only in the area estimator; both MIS terms use `pA * r2 / cosL`. |
| Wrong discrete light probability | Every sampled triangle stores the exact `selectPmf` used by the alias table; do not substitute `1/lightCount` or total area. |
| Bright textured emitter sampled with the wrong radiance | Selection weight is only an estimate; evaluate actual `Le(y)` at the sampled UV for every contribution. |
| Back side of an emitting panel contributes | Gate NEE and BSDF-hit emission with the same `EmissionSidedness` rule and geometric normal. |
| Target light self-occludes | Offset the receiver with `offsetRayOrigin`; set `maxT` from the offset origin and one ULP before `y`. |
| Double counting | Light samples get `w_light`; BSDF light hits get the complementary `w_bsdf`. Delta/camera hits get one. |
| BSDF/PDF disagreement | Sampling, `evaluateBsdf`, and `evaluateBsdfPdf` share one material implementation and are tested as a unit. |
| Normal-map leaks | BSDF cosine uses the shading normal; ray offset side and visibility robustness use the geometric normal. |
| Overflow in power heuristic | Normalize PDF pair by their maximum before squaring. |
| Bad geometry or arithmetic | Reject non-finite/zero-area/zero-distance/non-positive-cosine samples. Count them in a focused diagnostic test; never replace them with arbitrary brightness. |
| “Fixing” spikes by clamping radiance | Do not clamp individual samples. It hides a bug and biases the renderer. Diagnose the PDF/visibility/event state first. |

The diagnostic order when a bright isolated sample appears is:

1. Verify `triangleIndex -> lightIndex`, triangle area, alias PMF, and `pA`.
2. Verify `r2`, `cosX`, `cosL`, then recompute `pLightOmega` from the same
   sample with the formula above.
3. Verify the BSDF evaluator, sampler, and stored previous PDF agree on lobe
   type and measure.
4. Verify the visibility query is bounded before the sampled light point.
5. Verify the direct term was added once and the eventual BSDF light hit used
   the complementary MIS weight.
6. Only after these checks inspect normal maps, emission texture variance, or
   inherently difficult caustic paths.

A temporary **test-only structured counter buffer** may record invalid samples
and the largest finite contribution by cause. Do not add unconstrained device
`printf` calls to the render loop, and do not use a production clamp as a
debugging substitute.

## 8. Implementation sequence

1. Add host light-list construction after the world BVH bake, plus device
   uploads/frees in `DeviceBuffers`. Unit-test areas, eligibility, PMF sum,
   and alias sampling before touching shading.
2. Add header-level world-BVH any-hit traversal and test it against
   `traverseBvhClosest(..., maxT)` for random rays and blockers.
3. Introduce the shared emitted-radiance helper and change terminal emission
   accumulation to `+=`. Confirm the no-light path remains behaviorally
   equivalent apart from the intended accumulation correction.
4. Refactor diffuse BSDF sampling/evaluation/PDF into the shared interface;
   implement diffuse triangle NEE plus both MIS weights. This is the first
   functional Sponza milestone.
5. Refactor rough PBR/GGX into the same interface and enable glossy NEE; leave
   delta events BSDF-only.
6. Add texture-luminance-weighted selection. It changes sampling efficiency,
   not the estimator or its PDFs.
7. Profile Sponza after correctness is established. The expected cost is at
   most one early-out BVH shadow traversal per eligible surface vertex; the
   intended gain is sharply reduced variance and fewer useful indirect
   bounces, not a claim that `computeIntersections` alone becomes cheaper.

## 9. Verification plan

Root CMake does not build standalone `tests/`; create/generate their Visual
Studio projects separately when these tests are implemented.

### Deterministic tests

- **Light table:** all PMFs are finite and positive, sum to one, every alias
  result is in range, zero-area and near-black triangles are excluded, and
  empirical alias frequencies match the stored PMFs.
- **Emission sidedness:** a `OneSided` panel emits from its geometric-normal
  front side only in both NEE and BSDF-hit paths; the default `TwoSided`
  material preserves legacy output.
- **Triangle sampling:** sampled barycentrics sum to one and reconstructed
  points lie in the selected triangle. For a known triangle, estimated area
  integral converges to its analytic area.
- **Any-hit:** for random rays/max distances, `anyHit` is true exactly when
  closest traversal finds `t < maxT`; include an occluder before the light and
  the sampled light endpoint itself.
- **PDF conversion:** assert
  `pLightOmega == selectPmf / area * r2 / cosL` for fixed geometry. Include
  grazing and distant values, where the reversed Jacobian would fail clearly.
- **MIS complement:** for finite positive PDFs,
  `w_light + w_bsdf == 1` within float tolerance; delta/camera events always
  produce BSDF-hit weight one.
- **BSDF consistency:** numerically verify sampled diffuse/GGX events use the
  same `f`, PDF, lobe probability, and `f*cos/pdf` weight returned by the
  evaluator. This is the highest-value firefly regression test.

### Scene validation performed by the user

After compiling, render the same Cornell and Sponza cameras with NEE disabled
and enabled. At equal low sample counts, diffuse walls and the dark Sponza
interior should converge materially faster with NEE. At high sample counts, the
mean brightness must agree with the disabled reference within Monte Carlo
noise; NEE may change noise distribution, not scene energy. Inspect bright
small lights, grazing receiver edges, textured emitters, normal-mapped
surfaces, mirrors, and glass separately. Any isolated hot pixel is a reason to
run the invariant checks above, not a reason to raise an image clamp.

## 10. Explicit non-goals

- No environment/IBL importance sampling: the current renderer has a black
  miss background and no environment-light representation.
- No change to the 20-byte `HitRecord` or to the existing single world-space
  BVH ownership model.
- No bidirectional path tracing, photon mapping, ReSTIR, or temporal reuse.
  They are separate algorithms, not prerequisites for robust NEE.
- No arbitrary per-sample radiance clamp, PDF floor, or fixed shadow epsilon.
  Those conceal errors or introduce bias; the scale-aware ray offset and the
  explicit PDF contract are the intended robustness mechanisms.
