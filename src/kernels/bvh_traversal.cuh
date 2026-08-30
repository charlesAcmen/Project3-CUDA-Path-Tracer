#pragma once

// ====================================================================
// BVH Traversal Kernel
//
// GPU-side closest-hit traversal over the single scene-wide world-space
// BVH built on the host (src/bvh/bvh.cu).  One thread per active path, ONE
// traverseBvhClosest call per ray — no per-geom loop, no ray transform.
//
// The ray is already world-space (camera/scatter output); the triangles
// were baked to world space by buildSceneBvh, so a single traversal over
// the whole tree finds the closest hit of any mesh.  Root is node 0.
//
// AABB pruning only skips subtrees that cannot contain a hit closer than
// the current far plane, so traversal never misses a closer hit.
// ====================================================================

#include "sceneStructs.h"
#include "bvh/bvh.h"                       // BvhNode, BvhHit, traverseBvhClosest

/**
 * Compute the nearest ray–scene intersection for every active path using
 * the single world-space BVH.
 *
 * \param num_paths         Number of active paths
 * \param pathSegments      Active-path buffer (including the previous
 *                             primitive self-hit guard)
 * \param intersections     [out] Compact closest-hit record per path
 * \param deviceTrianglePositions   World-space position array (REORDERED by the BVH build)
 * \param deviceBvhNodes    Node array (device, built on host)
 */
__global__ void bvhTraverse(
    int num_paths,
    PathSegment* __restrict__ pathSegments,
    HitRecord* __restrict__ intersections,
    const TrianglePos* __restrict__ deviceTrianglePositions,
    BvhNode* __restrict__ deviceBvhNodes);
