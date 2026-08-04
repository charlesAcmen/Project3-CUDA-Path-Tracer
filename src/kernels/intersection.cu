#include "intersection.cuh"

// ====================================================================
// Intersection Testing Kernel Implementation
// ====================================================================

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

    const PathSegment& pathSegment = pathSegments[path_index];

    float t_min = LARGE_T;
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

        float closestT = LARGE_T;
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

        // ---- Record closest hit (world-space) ----
        // Only overwrite if this mesh's closest triangle is nearer than
        // the best hit found so far across all geometries.
        if (closestT >= t_min) continue;

        t_min = closestT;
        hit_geom_index = i;
        glm::vec3 worldNormal = multiplyMV(
            geom.invTranspose, glm::vec4(objNormal, 0.0f));
        float wLen2 = glm::dot(worldNormal, worldNormal);
        hit_normal = (isnan(wLen2) || wLen2 < RAY_EPSILON) ? glm::vec3(0.0f, 1.0f, 0.0f) : worldNormal * glm::inversesqrt(wLen2);
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
