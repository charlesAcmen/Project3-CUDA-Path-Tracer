#pragma once

// ====================================================================
// BVH Traversal Kernel
//
// GPU-side closest-hit traversal over the per-mesh BVHs built on the host
// (src/bvh/bvh.cu).  One thread per active path, outer loop over all
// geoms, writing the ShadeableIntersection layout the shading kernels read.
//
// Each geom's subtree is located through deviceBvhMeta (rootNodeIndex;
// -1 = empty mesh, skipped).  The ray is transformed to object space with
// the exact same transformRayToObjectSpace helper, and the subtree is
// traversed with traverseBvhClosest (src/bvh/bvh.h) — the identical
// iterative closest-hit algorithm the host test validates.
//
// AABB pruning only skips subtrees that cannot contain a hit closer than
// the current far plane, so traversal never misses a closer hit.
// ====================================================================

#include "sceneStructs.h"
#include "intersection/intersections.h"   // transformRayToObjectSpace, recordWorldNormal
#include "bvh/bvh.h"                       // BvhNode, BvhMeta, traverseBvhClosest

/**
 * Compute the nearest ray–mesh intersection for every active path using
 * the per-mesh BVHs.
 *
 * \param num_paths         Number of active paths
 * \param pathSegments      Active-path buffer
 * \param intersections     [out] Closest-hit result per path
 * \param deviceTriangles   Flat triangle array (REORDERED by the BVH build)
 * \param deviceBvhNodes    Node array (device, built on host)
 * \param deviceBvhMeta     Per-geom BVH metadata (rootNodeIndex = -1 → skip)
 */
__global__ void bvhTraverse(
    int num_paths,
    PathSegment* pathSegments,
    ShadeableIntersection* intersections,
    Triangle* deviceTriangles,
    BvhNode* deviceBvhNodes,
    BvhMeta* deviceBvhMeta);
