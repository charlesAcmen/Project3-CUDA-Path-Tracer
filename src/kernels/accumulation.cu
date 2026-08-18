#include "accumulation.cuh"

// ====================================================================
// Accumulation & Display Kernels Implementation
// ====================================================================

__global__ void sendImageToPBO(uchar4* __restrict__ pbo, glm::ivec2 resolution, glm::vec3* __restrict__ image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__global__ void gatherTerminatedPaths(int nPaths, glm::vec3* __restrict__ image, PathSegment* __restrict__ paths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        const PathSegment& path = paths[index];
        if (path.remainingBounces <= 0)
        {
            image[path.pixelIndex] += path.accumulatedRadiance;
        }
    }
}
