#pragma once

// ====================================================================
// Shading Kernel
//
// BSDF evaluation and ray scattering for each active path.
// Material sorting should be applied BEFORE this kernel to group
// same-material paths together, reducing warp divergence.
// ====================================================================

#include "sceneStructs.h"
#include "scene/scene.h"          // ShadingConfig
#include "intersection/intersections.h"   // getExactPointOnRay
#include "interactions/interactions.h"   // scatterRay
#include "rng/rng.h"
#include "constants.h"

/**
 * Russian roulette — probabilistically terminate low-throughput paths
 * without introducing bias.
 *
 * Survival probability p = max(R,G,B) clamped to [RR_P_MIN, RR_P_MAX].
 *   - max component gives conservative survival (fewer fireflies).
 *   - RR_P_MIN prevents extreme compensation (max 1/0.2 = 5x).
 *   - RR_P_MAX = 1.0 means full-throughput paths always survive.
 *
 * Unbiased: survivors have color /= p (compensation factor).
 * Terminated paths keep their (zero) color — gatherTerminatedPaths
 * collects it during the next compaction pass.
 *
 * @return  true if the path should be terminated.
 */
__device__ bool russianRouletteTerminate(
    glm::vec3& color,
    int remainingBounces,
    int traceDepth,
    int rrMinBounces,
    RngState& rng);

/**
 * BSDF evaluation and path-scattering kernel.
 *
 * For each active path:
 *   - Light source hit  → accumulate emission, terminate path.
 *   - Surface hit       → scatter the ray according to material BSDF
 *                         (diffuse, glossy, specular, refractive), then
 *                         apply Russian roulette for early termination.
 *   - Miss              → terminate with background colour (black).
 *
 * PERFORMANCE NOTES
 *   This kernel suffers from severe warp divergence when adjacent threads
 *   hit different materials (different if/else branches serialise within
 *   each warp).  Sorting paths by materialId before launching this kernel
 *   groups same-material paths together, dramatically reducing divergence.
 *
 *   Register pressure is high — ShadeableIntersection + PathSegment +
 *   Material + RNG state per thread.  High register count lowers SM
 *   occupancy.  This is driven by the live values (path state, material
 *   and RNG must coexist through the branchy scatter logic) and is
 *   layout-independent — SoA changes where fields sit in memory, not how
 *   many scalars are live per thread, so it would NOT reduce register
 *   pressure.  The real levers are live-range trimming and the
 *   scatterRay call boundary (a cross-TU device call, not inlined, so
 *   the ABI save/restore can spill registers).
 *
 *   Material array access is uncoalesced when materialId varies across
 *   threads in a warp — material sorting also mitigates this.
 */
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    TextureTable textures,        // scene texture assets (pixels + slice table)
    ShadingConfig config);

