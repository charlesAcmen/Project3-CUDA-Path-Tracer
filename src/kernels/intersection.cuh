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
    Triangle* deviceTriangles)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index >= num_paths) return;

    PathSegment pathSegment = pathSegments[path_index];

    float t_min = FLT_MAX;
    int   hit_geom_index = -1;
    glm::vec3 hit_normal;

    for (int i = 0; i < geoms_size; i++)
    {
        Geom& geom = geoms[i];

        // // ---- Skip non-mesh geoms ----
        // if (geom.type != MESH) continue;

        // ---- Transform ray to object space via inverseTransform ----
        // Triangles are stored in object space; bring the ray into the
        // mesh's local frame so that the intersection test is performed
        // in the same coordinate system as the vertices.
        Ray objRay;
        objRay.origin    = multiplyMV(geom.inverseTransform,
                            glm::vec4(pathSegment.ray.origin, 1.0f));//1.0f:point
        objRay.direction = multiplyMV(geom.inverseTransform,
                            glm::vec4(pathSegment.ray.direction, 0.0f));//0.0f:vector

        // ---- Linear scan over this mesh's triangle slice ----
        if (deviceTriangles == nullptr || geom.meshTriangleCount <= 0)
            continue;

        float closestT = 1e30f;
        bool  hit = false;
        glm::vec3 objNormal;

        for (int j = 0; j < geom.meshTriangleCount; j++)
        {
            float t;
            glm::vec3 triNormal;
            const Triangle& tri = deviceTriangles[geom.meshTriangleOffset + j];

            if (triangleIntersectionTest(objRay, tri, t, triNormal))
            {
                if (t < closestT)
                {
                    closestT  = t;
                    objNormal = triNormal;
                    hit = true;
                }
            }
        }

        if (!hit) continue;

        // ---- Record closest hit (world space) ----
        // Surface point:   transform object-space hit via geom.transform
        // Surface normal:  transform via invTranspose (preserves
        //                  orthogonality under non-uniform scaling).
        t_min = closestT;
        hit_geom_index = i;
        hit_normal = glm::normalize(multiplyMV(
            geom.invTranspose, glm::vec4(objNormal, 0.0f)));
    }

    if (hit_geom_index == -1)
    {
        intersections[path_index].t = -1.0f;
    }
    else
    {
        intersections[path_index].t           = t_min;
        intersections[path_index].materialId  = geoms[hit_geom_index].materialid;
        intersections[path_index].surfaceNormal = hit_normal;
    }
}
