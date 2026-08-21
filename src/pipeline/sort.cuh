#pragma once

// ====================================================================
// Material Sorting Pipeline
//
// Sorts path segments by the materialId resolved from each hit's Surface so that
// same-material paths become contiguous before shadeMaterial runs.
//
// Why this helps:
//   1. Warp divergence:  threads in a warp hitting different materials
//      take different branches, serialising execution.  Clustering by
//      materialId means most warps execute a single branch path.
//   2. Coalesced memory: adjacent threads load adjacent Material structs
//      instead of scattering across the array.
// ====================================================================

#include "sceneStructs.h"
#include "kernels/kernel_config.h"
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

// Resolve the sort key only when material sorting is enabled.  Traversal
// otherwise remains position-only and HitRecord stays independent of material
// storage.  Misses retain -1, matching the old sorting behavior.
__global__ void buildMaterialSortKeys(
    int numPaths,
    const HitRecord* __restrict__ intersections,
    const TriangleAttr* __restrict__ triangleAttrs,
    const Surface* __restrict__ surfaces,
    int* __restrict__ sortKeys)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPaths) return;

    const HitRecord& hit = intersections[idx];
    sortKeys[idx] = (hit.triangleIndex >= 0 && triangleAttrs != nullptr && surfaces != nullptr)
        ? surfaces[triangleAttrs[hit.triangleIndex].surfaceId].materialId
        : -1;
}

/**
 * Permute g_dev.paths and g_dev.intersections so that paths with the
 * same materialId become contiguous.
 *
 * Algorithm (Thrust-based, ping-pong via g_dev.pathsCompacted):
 *   1. buildMaterialSortKeys — resolve hit surface → materialId key
 *   2. thrust::sequence    — sortIndices = [0, 1, 2, ..., n-1]
 *   3. thrust::sort_by_key — sortIndices maps sorted_pos → original_pos
 *   4. thrust::gather      — reorder paths       into pathsCompacted
 *   5. thrust::gather      — reorder intersections into intersectionsSorted
 *   6. std::swap           — the sorted buffers become the "live" ones
 *
 * No-op when g_opts.sortByMaterial is false (runtime toggle, no rebuild
 * needed).
 */
static void sortPathsByMaterial(int num_paths)
{
    if (!g_opts.sortByMaterial) return;
    if (num_paths <= 1) return;

    // 1. Resolve material keys through TriangleAttr.surfaceId → Surface.
    LAUNCH_KERNEL_AUTO(buildMaterialSortKeys, num_paths,
        num_paths, g_dev.intersections, g_dev.deviceTriangleAttrs,
        g_dev.deviceSurfaces, g_dev.sortKeys);

    // 2. Initialise permutation: [0, 1, 2, ..., n-1]
    thrust::sequence(thrust::device,
        g_dev.sortIndices, g_dev.sortIndices + num_paths);

    // 3. Sort indices by material ID
    thrust::sort_by_key(thrust::device,
        g_dev.sortKeys, g_dev.sortKeys + num_paths,
        g_dev.sortIndices);

    // 4. Gather path segments into sorted order (reuse pathsCompacted)
    thrust::gather(thrust::device,
        g_dev.sortIndices, g_dev.sortIndices + num_paths,
        g_dev.paths,
        g_dev.pathsCompacted);

    // 5. Gather intersections into sorted order
    thrust::gather(thrust::device,
        g_dev.sortIndices, g_dev.sortIndices + num_paths,
        g_dev.intersections,
        g_dev.intersectionsSorted);

    // 6. Swap pointers — sorted arrays become the live ones
    std::swap(g_dev.paths, g_dev.pathsCompacted);
    std::swap(g_dev.intersections, g_dev.intersectionsSorted);
}
