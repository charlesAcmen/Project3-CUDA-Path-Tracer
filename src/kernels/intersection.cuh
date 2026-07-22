#pragma once

// ====================================================================
// Intersection Testing Kernel
//
// Contains the device helper and kernel that perform ray-geometry
// intersection tests.  Currently a naive O(N) linear scan; the intent
// is that a BVH traversal replaces this function when acceleration
// structures are added (see intersectSingleGeom — add new primitives
// like TRIANGLE by extending the if/else chain).
// ====================================================================

#include "sceneStructs.h"
#include "intersections.h"   // boxIntersectionTest, sphereIntersectionTest

/**
 * Dispatch a single geometry intersection test based on type.
 *
 * Extract so that adding a new primitive (triangle, metaball, CSG …)
 * only requires extending this function — the loop in
 * computeIntersections stays unchanged.
 *
 * Returns parametric distance t along the ray, or -1 on miss.
 */
__device__ float intersectSingleGeom(
    const Geom& geom,
    const Ray& ray,
    glm::vec3& outPoint,
    glm::vec3& outNormal,
    bool& outOutside)
{
    if (geom.type == CUBE)
    {
        return boxIntersectionTest(geom, ray, outPoint, outNormal, outOutside);
    }
    else if (geom.type == SPHERE)
    {
        return sphereIntersectionTest(geom, ray, outPoint, outNormal, outOutside);
    }
    // TODO: add more intersection tests here... triangle? metaball? CSG?
    return -1.0f;
}

/**
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
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            t = intersectSingleGeom(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);

            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
            intersections[path_index].geomIndex = -1;
        }
        else
        {
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].geomIndex = hit_geom_index;
        }
    }
}
