#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

// Note: checkCUDAError and checkCUDAErrorFn are now defined in utilities.h/cu

// ====================================================================
// Stream Compaction: compile-time method selection
// ====================================================================
//   0 = disabled (paths are never compacted; terminated paths are
//       guarded by the remainingBounces check in shadeMaterial)
//   1 = custom work-efficient scan-based compaction (Project 2)
//   2 = Thrust copy_if (reference / baseline; requires extended lambdas)
//
// The three modes are mutually exclusive -- exactly one is active.
// ====================================================================
#define COMPACT_METHOD 2

// Predicate functor for Thrust copy_if (used when COMPACT_METHOD == 2)
struct IsPathActive {
    __device__ bool operator()(const PathSegment& p) const {
        return p.remainingBounces > 0;
    }
};

// Functor for extracting materialId from ShadeableIntersection (used in material sorting)
struct ExtractMaterialId {
    __device__ int operator()(const ShadeableIntersection& isect) const {
        return isect.materialId;
    }
};

#if COMPACT_METHOD == 1
    #include "../stream_compaction/efficient.h"
#elif COMPACT_METHOD == 2
    #include <thrust/copy.h>
#elif COMPACT_METHOD != 0
    #error "COMPACT_METHOD must be 0 (disabled), 1 (custom), or 2 (Thrust)"
#endif

// Material sorting toggle -- groups path segments by material ID before
// shading to reduce warp divergence and improve memory coalescing.
//
// When enabled, dev_paths and dev_intersections are permuted so that paths
// hitting the same material become contiguous.  Threads in the same warp
// then follow identical emissive/diffuse/specular branches instead of
// diverging across all three.
//
// Note: SORT_BY_MATERIAL is now defined via CMake compilation flags.
// Do NOT manually define it here. Use the appropriate build target instead:
//   - cis565_path_tracer_sorted   (SORT_BY_MATERIAL=1)
//   - cis565_path_tracer_unsorted (SORT_BY_MATERIAL=0)
#ifndef SORT_BY_MATERIAL
    #define SORT_BY_MATERIAL 1  // Default fallback if not defined by build system
#endif

#if SORT_BY_MATERIAL
    #include <thrust/sort.h>
    #include <thrust/gather.h>
    #include <thrust/sequence.h>
#endif

//index:spatial correlation,ensuring that the different pixels will have different random seeds
//depth:depth correlation ,ensuring that generated random number in different bounces is independent for a ray
//iter:temporal correlation,ensuring that the generated random number in different iterations is independent for a pixel
//iter ensures ray trace is different in every iterations.
//Note that engine is created whenever determining new ray direction,do NOT fill the engine in the PathSegment struct
//for optimizing gpu memory bandwidth by utilizing GPU high calculation performance
__host__ __device__ thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// Temporary buffer for stream compaction
static PathSegment* dev_paths_compacted = NULL;
// Buffers for material-based sorting
// store the materialId of each intersection for sorting
static int* dev_sortKeys = NULL;
// store the original index of each intersection for sorting
static int* dev_sortIndices = NULL;
static ShadeableIntersection* dev_intersections_sorted = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_paths_compacted, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
#if SORT_BY_MATERIAL
    cudaMalloc(&dev_sortKeys, pixelcount * sizeof(int));
    cudaMalloc(&dev_sortIndices, pixelcount * sizeof(int));
    cudaMalloc(&dev_intersections_sorted, pixelcount * sizeof(ShadeableIntersection));
#endif

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_paths_compacted);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
#if SORT_BY_MATERIAL
    cudaFree(dev_sortKeys);
    cudaFree(dev_sortIndices);
    cudaFree(dev_intersections_sorted);
#endif
    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        // segment.ray.direction = glm::normalize(cam.view
        //     - cam.right * cam.pixelLength.x * ((float)x  - (float)cam.resolution.x * 0.5f)
        //     - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        // );
        // // Antialiasing: Add random jitter to ray direction for stochastic sampling
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float jitterX = u01(rng) - 0.5f;  // Random offset in [-0.5, 0.5]
        float jitterY = u01(rng) - 0.5f;
        
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
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

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
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
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

/**
 * Shader kernel that performs BSDF evaluation and generates new rays.
 * This kernel handles path termination and ray scattering based on material properties.
 * 
 * For each path segment:
 * - If ray hits a light source: accumulate emitted light and terminate path
 * - If ray hits a surface: scatter ray according to material BSDF
 * - If ray misses all geometry: terminate path (background)
 * this code causes severe wrap divergence: unpredictable branching (different ray path leads to different results)
 * TODO: sort pathSegments by materialId (same material in a group) to reduce divergence and uncoalesced global memory access
 */
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    int traceDepth,
    int rrMinBounces)
{
    //Memory-Bound:Occupancy is limited by the number of registers used per thread
    //so designing smallest and aligned data structure 
    //and replacing AoS(Array of Structures) with SoA(Structure of Arrays) is important
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment& pathSegment = pathSegments[idx];

        // Skip paths that have already terminated (hit light or missed geometry).
        // Without this guard, a path that hits an emissive surface will be
        // stuck on the same intersection for every remaining bounce - its ray
        // was never moved - and get multiplied by emittance repeatedly,
        // causing the image to blow out to white as iterations accumulate.
        if (pathSegment.remainingBounces <= 0)
        {
            return;
        }

        //Register Heavy:ShadeableIntersection+PathSegment+Material+engine
        //the register count of every stream multiprocessor(SM) is limited
        //potentially bringing down the occupancy of SMs by reducing active warps per SM drastically
        ShadeableIntersection intersection = shadeableIntersections[idx];

        // Check if ray intersected with scene geometry
        if (intersection.t > 0.0f)
        {
            // Setup random number generator for this path
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);

            // Get material properties at intersection point
            // material is not sorted,hence leading to uncoalesced global memory access
            Material material = materials[intersection.materialId];

            // Compute intersection point on the ray
            glm::vec3 intersectionPoint = getExactPointOnRay(pathSegment.ray, intersection.t);
            
            // Check if we hit a light source (emissive material)   
            if (material.emittance > 0.0f)
            {
                // Accumulate light contribution and terminate path
                pathSegment.color *= (material.color * material.emittance);
                pathSegment.remainingBounces = 0;
            }
            else
            {
                // Non-emissive surface: scatter ray according to BSDF
                // This updates the ray direction and attenuates color based on material
                scatterRay(pathSegment, intersectionPoint, intersection.surfaceNormal, material, rng);

                // Russian roulette: probabilistically terminate paths whose
                // throughput has dropped below a useful level.  Only applies
                // after rrMinBounces guaranteed bounces.
                //
                // Survival probability p = max(R,G,B) clamped to [RR_P_MIN, RR_P_MAX].
                //   - Using max component is conservative (highest survival chance
                //     among the three channels -> fewest fireflies).
                //   - RR_P_MIN prevents extreme compensation factors (max 1/0.2 = 5x).
                //   - RR_P_MAX = 1.0 means paths with full throughput always survive.
                //
                // Unbiased: survivors have throughput /= p.
                // Terminated paths keep their color intact -- gatherTerminatedPaths
                // collects it during the next compaction pass.
                if (pathSegment.remainingBounces > 0 &&
                    pathSegment.remainingBounces <= traceDepth - rrMinBounces)
                {
                    float p = fmaxf(fmaxf(pathSegment.color.r,
                                           pathSegment.color.g),
                                           pathSegment.color.b);
                    p = fminf(fmaxf(p, RR_P_MIN), RR_P_MAX);

                    thrust::uniform_real_distribution<float> u01(0, 1);
                    if (u01(rng) < p)
                    {
                        pathSegment.color /= p;    // unbiased compensation
                    }
                    else
                    {
                        pathSegment.remainingBounces = 0;  // terminate
                    }
                }
            }
        }
        else
        {
            // Ray didn't hit anything - terminate path with background color
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Gathers the accumulated color of TERMINATED paths (remainingBounces <= 0)
 * into the accumulation image BEFORE stream compaction discards them.
 *
 * This is critical for correctness with stream compaction:
 * - When a path hits a light, its color *= emittance is the final result for that
 *   sample, and it must be recorded in the accumulation buffer.
 * - When a path exhausts all bounces without hitting a light, its attenuated color
 *   is also a valid sample and must be recorded.
 * - Without this pre-compaction gather, these terminated paths are removed by
 *   compaction before finalGather can collect their contributions, resulting in
 *   a black image.
 *
 * Active paths (remainingBounces > 0) are NOT gathered here -- they will continue
 * bouncing and their final color will be gathered when they eventually terminate.
 */
__global__ void gatherTerminatedPaths(int nPaths, glm::vec3* image, PathSegment* paths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment path = paths[index];
        // Only collect contributions from paths that have finished their journey.
        // Active paths are skipped -- they will contribute when they terminate.
        if (path.remainingBounces <= 0)
        {
            image[path.pixelIndex] += path.color;
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

#if SORT_BY_MATERIAL
/**
 * Permutes dev_paths and dev_intersections so that paths hitting the same
 * material become contiguous in memory.
 *
 * Why this helps:
 *   shadeMaterial branches on material.emittance > 0 (light vs. surface) and
 *   material type (diffuse vs. specular vs. refractive).  Without sorting,
 *   threads within a warp hit different materials and take divergent paths,
 *   serialising execution.  After sorting, same-material paths cluster
 *   together, so most warps execute a single branch path.
 *
 *   Additionally, materials[id] reads become coalesced: adjacent threads
 *   load adjacent Material structs instead of scattering across the array.
 *
 * Algorithm (Thrust-based, in-place via ping-pong):
 *   1. Extract materialId from dev_intersections -> dev_sortKeys
 *   2. Fill dev_sortIndices = [0, 1, 2, ...]
 *   3. sort_by_key(keys, indices)  -- indices now map sorted_pos -> original_pos
 *   4. gather paths    via indices -> dev_paths_compacted  (reuses compaction buffer)
 *   5. gather intersections via indices -> dev_intersections_sorted
 *   6. Swap pointers so dev_paths / dev_intersections point to sorted data
 */
static void sortPathsByMaterial(int num_paths)
{
    if (num_paths <= 1) return;

    // 1. Extract sort keys (materialId from each intersection)
    thrust::transform(thrust::device,//transform in GPU
        dev_intersections, dev_intersections + num_paths,
        dev_sortKeys,
        ExtractMaterialId());

    // 2. Initialise permutation indices: [0, 1, 2, ..., n-1]
    thrust::sequence(thrust::device,
        dev_sortIndices, dev_sortIndices + num_paths);

    // 3. Sort indices by material ID (stable radix sort)
    thrust::sort_by_key(thrust::device,
        dev_sortKeys, dev_sortKeys + num_paths,
        dev_sortIndices);
    // dev_sortIndices[sorted_pos] now gives the original position

    // 4. Gather path segments into sorted order
    //    Reuses dev_paths_compacted as the gather destination; its previous
    //    contents (from stream compaction) are stale and safe to overwrite.
    thrust::gather(thrust::device,
        dev_sortIndices, dev_sortIndices + num_paths,
        dev_paths,               // input  (unsorted)
        dev_paths_compacted);    // output (sorted)
    //for (i = 0 to n-1):
    //output[i] = input[indices[i]]


    // 5. Gather intersections into sorted order (separate temp buffer)
    thrust::gather(thrust::device,
        dev_sortIndices, dev_sortIndices + num_paths,
        dev_intersections,        // input  (unsorted)
        dev_intersections_sorted);// output (sorted)

    // 6. Swap pointers -- the sorted arrays are now the "live" ones
    { PathSegment* tmp = dev_paths;
      dev_paths = dev_paths_compacted;
      dev_paths_compacted = tmp; }

    { ShadeableIntersection* tmp = dev_intersections;
      dev_intersections = dev_intersections_sorted;
      dev_intersections_sorted = tmp; }
}
#endif // SORT_BY_MATERIAL

/**
 * Gathers terminated path colors into dev_image, then stream-compacts the path
 * array to remove entries with remainingBounces <= 0.
 *
 * WHY before compaction:
 *   After shadeMaterial, paths that hit a light carry their final radiance in
 *   pathSegment.color.  If we compact first, those contributions are discarded
 *   and the accumulation buffer stays empty --> black image.
 *
 *   Active paths (remainingBounces > 0) are deliberately skipped here -- they
 *   continue bouncing and contribute when they eventually terminate.
 *
 * Uses ping-pong buffers (dev_paths <-> dev_paths_compacted) to avoid a
 * separate allocation per bounce.
 *
 * @param num_paths    [in/out]  Active path count; set to survivors after compaction.
 * @param blockSize1d            1D block size for kernel launches.
 * @return                       true if every path terminated (caller should
 *                               exit the bounce loop immediately).
 */
static bool compactActivePaths(int& num_paths, int blockSize1d)
{
    // Nothing to do when compaction is disabled -- terminated paths are
    // already guarded by the remainingBounces check in shadeMaterial, and
    // finalGather collects every path at the end of the bounce loop.
#if COMPACT_METHOD == 0
    (void)num_paths; (void)blockSize1d;
    return false;

    // ====================================================================
    // Custom work-efficient compaction (Project 2)
    // ====================================================================
#elif COMPACT_METHOD == 1
    dim3 numBlocks((num_paths + blockSize1d - 1) / blockSize1d);

    // 1. Bank terminated-path colors before they disappear.
    gatherTerminatedPaths<<<numBlocks, blockSize1d>>>(
        num_paths, dev_image, dev_paths);
    checkCUDAError("gatherTerminatedPaths");

    // 2. Compact: keep only paths with remainingBounces > 0.
    int survivors = StreamCompaction::Efficient::compactPathSegments(
        num_paths,
        dev_paths_compacted,   // output
        dev_paths);            // input

    // 3. Ping-pong the buffer pointers.
    PathSegment* tmp = dev_paths;
    dev_paths = dev_paths_compacted;
    dev_paths_compacted = tmp;

    num_paths = survivors;
    return (num_paths == 0);

    // ====================================================================
    // Thrust copy_if (reference / baseline)
    // ====================================================================
#elif COMPACT_METHOD == 2
    dim3 numBlocks((num_paths + blockSize1d - 1) / blockSize1d);

    // 1. Bank terminated-path colors before they disappear.
    gatherTerminatedPaths<<<numBlocks, blockSize1d>>>(
        num_paths, dev_image, dev_paths);
    checkCUDAError("gatherTerminatedPaths");

    // 2. Compact: keep only paths with remainingBounces > 0.
    PathSegment* end = thrust::copy_if(
        thrust::device,
        dev_paths, dev_paths + num_paths,
        dev_paths_compacted,
        IsPathActive());

    // 3. Ping-pong the buffer pointers.
    PathSegment* tmp = dev_paths;
    dev_paths = dev_paths_compacted;
    dev_paths_compacted = tmp;

    num_paths = static_cast<int>(end - dev_paths);
    return (num_paths == 0);

#endif
}

// ---------------------------------------------------------------------------
// Main path-tracing entry point (called once per frame / iteration)
// ---------------------------------------------------------------------------
//
// Pipeline overview (one iteration = one sample per pixel):
//
//   generateRayFromCamera          primary rays -> PathSegment buffer
//   for each bounce:
//       computeIntersections        ray <-> scene test
//       [sortPathsByMaterial]       group by materialId for coalesced shading  (optional)
//       shadeMaterial               BSDF eval, color attenuation / emission
//       compactActivePaths          gather dead paths -> compact -> ping-pong  (optional)
//   finalGather                     remaining paths -> accumulation buffer
//   sendImageToPBO                  tone-map -> OpenGL display
//
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam    = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    // ---- 1. Primary rays ------------------------------------------------
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(
        cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int  depth     = 0;
    int  num_paths = pixelcount;
    bool done      = false;

    // ---- 2. Bounce loop -------------------------------------------------
    while (!done)
    {
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        dim3 numBlocks((num_paths + blockSize1d - 1) / blockSize1d);
        computeIntersections<<<numBlocks, blockSize1d>>>(
            depth, num_paths, dev_paths,
            dev_geoms, hst_scene->geoms.size(), dev_intersections);
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

#if SORT_BY_MATERIAL
        sortPathsByMaterial(num_paths);
#endif

        shadeMaterial<<<numBlocks, blockSize1d>>>(
            iter, num_paths,
            dev_intersections, dev_paths, dev_materials,
            traceDepth, hst_scene->state.rrMinBounces);

        bool allDead = compactActivePaths(num_paths, blockSize1d);
        done = allDead || (depth >= traceDepth);

        if (guiData != NULL)
            guiData->TracedDepth = depth;
    }

    // ---- 3. Accumulation ------------------------------------------------
    // Survivors (paths that reached traceDepth without terminating) also
    // carry valid attenuated colors and must contribute.
    {
        dim3 numBlocks((pixelcount + blockSize1d - 1) / blockSize1d);
        finalGather<<<numBlocks, blockSize1d>>>(
            num_paths, dev_image, dev_paths);
    }

    // ---- 4. Display -----------------------------------------------------
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(
        pbo, cam.resolution, iter, dev_image);

    cudaMemcpy(hst_scene->state.image.data(), dev_image,
               pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
