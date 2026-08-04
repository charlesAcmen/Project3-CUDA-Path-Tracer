#pragma once

// ====================================================================
// Intersection Testing Kernel
//
// Naive O(N_geoms × N_paths) linear scan over all geometries for every
// active path.  Each thread processes one path and records the closest
// hit into the ShadeableIntersection buffer.
//
// All geometries are expected to be MESH (triangulated).  Every ray is
// transformed to object space before the triangle test via the geom's
// inverseTransform; non-mesh types silently miss (-1).
//
// Triangle intersection logic (Möller-Trumbore via GLM) lives in
// intersection/triangle.h.  No primitive-type dispatcher exists — mesh
// is the only geometry primitive supported.
//
// TODO: replace linear scan with BVH traversal for O(log N) scaling.
// ====================================================================

#include "sceneStructs.h"
#include "intersection/intersections.h"   // multiplyMV
#include "intersection/triangle.h"  // triangleIntersectionTest

/**
 * Compute the nearest ray–mesh intersection for every active path.
 *
 * Naive linear scan: each thread iterates over all scene geometries,
 * transforms the ray to object space, and tests against the mesh's
 * triangle slice.  The closest hit (smallest t > 0) is recorded.
 *
 * \param depth             Current bounce depth (unused — reserved)
 * \param num_paths         Number of active paths
 * \param pathSegments      Active-path buffer
 * \param geoms             Host-side geom array (copied to device)
 * \param geoms_size        Number of geoms
 * \param intersections     [out] Closest-hit result per path
 * \param deviceTriangles   Flat triangle array (all meshes)

 * Compute the nearest ray-geometry intersection for every active path.
 *
 * Naive O(N_geoms × N_paths) linear scan.  Each thread processes one path
 * and tests against every scene geometry, recording the closest hit into
 * the ShadeableIntersection buffer (t < 0 = miss, otherwise t = distance +
 * materialId + surfaceNormal).
 *
 * TODO: replace with BVH traversal for O(log N) asymptotic scaling.
 */
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
    Triangle* deviceTriangles);

