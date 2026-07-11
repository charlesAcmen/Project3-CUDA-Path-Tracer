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
#include "profiler.h"

// Note: checkCUDAError and checkCUDAErrorFn are now defined in utilities.h/cu

// ====================================================================
// Stream Compaction & Material Sorting: runtime-configurable toggles
// ====================================================================
//   COMPACT_METHOD:  0 = disabled, 1 = custom scan, 2 = Thrust copy_if
//   SORT_BY_MATERIAL:  true = group paths by materialId before shading
//
// Defaults set here (not from CMake).  Override at runtime via:
//   --compact=N  --sort=0/1  (requires --benchmark)
// ====================================================================

// Always include all dependencies -- runtime branching replaces the old
// #if guards so a single executable supports every combination.
#include "../stream_compaction/efficient.h"
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>

// Predicate functor for Thrust copy_if (used when g_compactMethod == 2)
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

// Runtime configuration -- consolidated into PathTracerOptions (defined in pathtrace.h).
// Can be overridden at runtime via --compact=N --sort=0/1 --fresnel=0/1 command-line flags
static PathTracerOptions g_opts;

// Forward declarations for the dispatch table (defined below).
using CompactCoreFunc = int (*)(int n, PathSegment* dst, const PathSegment* src);
static int compactCoreThrust(int n, PathSegment* dst, const PathSegment* src);
static int compactCoreGlobalMem(int n, PathSegment* dst, const PathSegment* src);
static int compactCoreSharedMem(int n, PathSegment* dst, const PathSegment* src);
static CompactCoreFunc g_compactCore = nullptr;

void setCompactMethod(int method) {
    g_opts.compactMethod = method;
    switch (method) {
        case 0:  g_compactCore = nullptr;                  break;
        case 1:  g_compactCore = compactCoreGlobalMem;     break;
        case 2:  g_compactCore = compactCoreThrust;        break;
        case 3:  g_compactCore = compactCoreSharedMem;     break;
        default: g_compactCore = compactCoreSharedMem;     break;
    }
}
void setSortByMaterial(bool enable) { g_opts.sortByMaterial = enable; }
int  getCompactMethod()             { return g_opts.compactMethod; }
bool getSortByMaterial()            { return g_opts.sortByMaterial; }
void setFresnelMode(int mode)       { g_opts.fresnelMode = (mode == 1 ? 1 : 0); }
int  getFresnelMode()              { return g_opts.fresnelMode; }

// ====================================================================
// Compaction dispatch implementations (forward-declared above).
// ====================================================================
static int compactCoreThrust(int n, PathSegment* dst, const PathSegment* src) {
    PathSegment* end = thrust::copy_if(thrust::device, src, src + n, dst, IsPathActive());
    return static_cast<int>(end - dst);
}

static int compactCoreGlobalMem(int n, PathSegment* dst, const PathSegment* src) {
    return StreamCompaction::Efficient::compactPathSegments(n, dst, src);
}

static int compactCoreSharedMem(int n, PathSegment* dst, const PathSegment* src) {
    return StreamCompaction::Efficient::compactPathSegmentsSharedMemory(n, dst, src);
}

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

// All GPU device buffers consolidated into a single struct (defined in pathtrace.h).
// Previously 9 separate static dev_* pointers.
static DeviceBuffers g_dev;

// TODO: static variables for device memory, any extra info you need, etc
// ...

// Tracks whether pathtraceInit has been called at least once.
// Avoids calling pathtraceFree() on uninitialised pointers.
static bool s_initialized = false;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;
    setCompactMethod(g_opts.compactMethod);

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    const int maxPaddedPathCount = 1 << ilog2ceil(pixelcount);

    cudaMalloc(&g_dev.image, pixelcount * sizeof(glm::vec3));
    cudaMemset(g_dev.image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&g_dev.paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&g_dev.pathsCompacted, pixelcount * sizeof(PathSegment));

    cudaMalloc(&g_dev.geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(g_dev.geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&g_dev.materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(g_dev.materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&g_dev.intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(g_dev.intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    StreamCompaction::Efficient::initCompactionWorkspace(maxPaddedPathCount);

    // Sort buffers -- always allocated (overhead is negligible); the sorting
    // function early-returns when g_opts.sortByMaterial is false at runtime.
    cudaMalloc(&g_dev.sortKeys, pixelcount * sizeof(int));
    cudaMalloc(&g_dev.sortIndices, pixelcount * sizeof(int));
    cudaMalloc(&g_dev.intersectionsSorted, pixelcount * sizeof(ShadeableIntersection));

    s_initialized = true;

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    if (!s_initialized)
        return;

    s_initialized = false;

    cudaFree(g_dev.image);  // no-op if g_dev.image is null
    cudaFree(g_dev.paths);
    cudaFree(g_dev.pathsCompacted);
    cudaFree(g_dev.geoms);
    cudaFree(g_dev.materials);
    cudaFree(g_dev.intersections);
    cudaFree(g_dev.sortKeys);
    cudaFree(g_dev.sortIndices);
    cudaFree(g_dev.intersectionsSorted);
    StreamCompaction::Efficient::freeCompactionWorkspace();
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

// Dispatch a single geometry intersection test based on geometry type.
// Extracted from computeIntersections so that adding a new primitive type
// (triangle, metaball, CSG, etc.) only requires modifying this one function.
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

            t = intersectSingleGeom(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);

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

// Russian roulette: probabilistically terminate paths whose throughput has
// dropped below a useful level.  Extracted from shadeMaterial for clarity.
// Returns true if the path should be terminated.
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
__device__ bool russianRouletteTerminate(
    glm::vec3& color,
    int remainingBounces,
    int traceDepth,
    int rrMinBounces,
    thrust::default_random_engine& rng)
{
    // Only applies after rrMinBounces guaranteed bounces.
    if (remainingBounces <= 0 ||
        remainingBounces >= traceDepth - rrMinBounces)
    {
        return false; // still within guaranteed bounces
    }

    float p = fmaxf(fmaxf(color.r, color.g), color.b);
    p = fminf(fmaxf(p, RR_P_MIN), RR_P_MAX);

    thrust::uniform_real_distribution<float> u01(0, 1);
    if (u01(rng) < p)
    {
        color /= p;    // unbiased compensation
        return false;  // survived
    }
    return true; // terminated
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
    int rrMinBounces,
    int fresnelMode)
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
                scatterRay(pathSegment, intersectionPoint, intersection.surfaceNormal, material, rng, fresnelMode);

                // Russian roulette: probabilistically terminate paths whose
                // throughput has dropped below a useful level.  Only applies
                // after rrMinBounces guaranteed bounces.
                //
                // Note: scatterRay() has already decremented remainingBounces, so we check
                // the UPDATED value. With traceDepth=8 and rrMinBounces=3:
                //   - Bounce 0: remainingBounces = 7 after scatter, 7 < 5? No -> no RR
                //   - Bounce 1: remainingBounces = 6 after scatter, 6 < 5? No -> no RR
                //   - Bounce 2: remainingBounces = 5 after scatter, 5 < 5? No -> no RR
                //   - Bounce 3: remainingBounces = 4 after scatter, 4 < 5? Yes -> RR starts
                // This ensures the first rrMinBounces (3) bounces are guaranteed.
                if (russianRouletteTerminate(pathSegment.color,
                    pathSegment.remainingBounces, traceDepth, rrMinBounces, rng))
                {
                    pathSegment.remainingBounces = 0;  // terminate
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

/**
 * Add the current iteration's output to the overall image.
 *
 * NOTE: The (1/eta^2) radiance scaling was removed because it caused energy
 * loss in the glass furnace test.  pathEta is only updated when refraction
 * occurs; it is NOT updated on Fresnel reflection or TIR.  Therefore, when a
 * path exhausts its bounce budget while still inside glass (e.g. after TIR),
 * pathEta retains the glass IOR (1.5) and the former 1/eta^2 factor (~0.44)
 * incorrectly darkened those contributions, making glass balls appear gray
 * instead of invisible in a uniform-white furnace environment.
 *
 * For paths that do exit the glass and then hit the furnace walls the energy
 * is already correctly accounted for by the Fresnel Russian-roulette weights
 * inside scatterRay; no additional correction is needed here.
 */
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
        // See finalGather for why the (1/pathEta^2) correction was removed.
        if (path.remainingBounces <= 0)
        {
            image[path.pixelIndex] += path.color;
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

/**
 * Permutes g_dev.paths and g_dev.intersections so that paths hitting the same
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
 *   1. Extract materialId from g_dev.intersections -> g_dev.sortKeys
 *   2. Fill g_dev.sortIndices = [0, 1, 2, ...]
 *   3. sort_by_key(keys, indices)  -- indices now map sorted_pos -> original_pos
 *   4. gather paths    via indices -> g_dev.pathsCompacted  (reuses compaction buffer)
 *   5. gather intersections via indices -> g_dev.intersectionsSorted
 *   6. Swap pointers so g_dev.paths / g_dev.intersections point to sorted data
 *
 * When g_opts.sortByMaterial is false this function returns immediately (runtime
 * toggle -- no rebuild needed for performance comparisons).
 */
static void sortPathsByMaterial(int num_paths)
{
    if (!g_opts.sortByMaterial) return;   // runtime toggle
    if (num_paths <= 1) return;

    // 1. Extract sort keys (materialId from each intersection)
    thrust::transform(thrust::device,//transform in GPU
        g_dev.intersections, g_dev.intersections + num_paths,
        g_dev.sortKeys,
        ExtractMaterialId());

    // 2. Initialise permutation indices: [0, 1, 2, ..., n-1]
    thrust::sequence(thrust::device,
        g_dev.sortIndices, g_dev.sortIndices + num_paths);

    // 3. Sort indices by material ID (stable radix sort)
    thrust::sort_by_key(thrust::device,
        g_dev.sortKeys, g_dev.sortKeys + num_paths,
        g_dev.sortIndices);
    // g_dev.sortIndices[sorted_pos] now gives the original position

    // 4. Gather path segments into sorted order
    //    Reuses g_dev.pathsCompacted as the gather destination; its previous
    //    contents (from stream compaction) are stale and safe to overwrite.
    thrust::gather(thrust::device,
        g_dev.sortIndices, g_dev.sortIndices + num_paths,
        g_dev.paths,               // input  (unsorted)
        g_dev.pathsCompacted);    // output (sorted)
    //for (i = 0 to n-1):
    //output[i] = input[indices[i]]


    // 5. Gather intersections into sorted order (separate temp buffer)
    thrust::gather(thrust::device,
        g_dev.sortIndices, g_dev.sortIndices + num_paths,
        g_dev.intersections,        // input  (unsorted)
        g_dev.intersectionsSorted);// output (sorted)

    // 6. Swap pointers -- the sorted arrays are now the "live" ones
    std::swap(g_dev.paths, g_dev.pathsCompacted);
    std::swap(g_dev.intersections, g_dev.intersectionsSorted);
}

/**
 * Gathers terminated path colors into g_dev.image, then stream-compacts the path
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
 * Uses ping-pong buffers (g_dev.paths <-> g_dev.pathsCompacted) to avoid a
 * separate allocation per bounce.
 *
 * Implicit Host-Device Synchronization:
 *   This function returns the survivor count (num_paths) to the CPU, which is
 *   used to determine loop termination: done = (num_paths == 0 || depth >= max).
 *   The cudaMemcpy operations inside the stream compaction implementations
 *   (to retrieve the final count) implicitly synchronize the device, ensuring
 *   all prior GPU work completes before the CPU reads the result.
 *
 * @param num_paths    [in/out]  Active path count; set to survivors after compaction.
 * @param blockSize1d            1D block size for kernel launches.
 * @return                       true if every path terminated (caller should
 *                               exit the bounce loop immediately).
 */
static bool compactActivePaths(int& num_paths, int blockSize1d)
{
    Profiler& prof = g_profiler();

    // Compaction disabled at runtime -- terminated paths are guarded by the
    // remainingBounces check in shadeMaterial; finalGather collects everything.
    if (g_compactCore == nullptr) {
        return false;
    }

    dim3 numBlocks((num_paths + blockSize1d - 1) / blockSize1d);

    // 1. Bank terminated-path colors before they disappear.
    prof.gpuStart(ProfilerOp::GatherTerminatedPaths);
    gatherTerminatedPaths<<<numBlocks, blockSize1d>>>(
        num_paths, g_dev.image, g_dev.paths);
    prof.gpuStop(ProfilerOp::GatherTerminatedPaths);
    checkCUDAError("gatherTerminatedPaths");

    // 2. Compact: dispatch through function pointer (set once at startup).
    prof.cpuStart(ProfilerOp::CompactPaths);
    int survivors = g_compactCore(num_paths, g_dev.pathsCompacted, g_dev.paths);
    prof.cpuStop(ProfilerOp::CompactPaths);

    // 3. Swap buffers: compacted array becomes the active one.
    std::swap(g_dev.paths, g_dev.pathsCompacted);

    num_paths = survivors;
    return (num_paths == 0);
}

// ---------------------------------------------------------------------------
// Host-side helpers extracted from pathtrace() for readability.
// ---------------------------------------------------------------------------

// Print per-bounce path survival counts.  Enabled only when the profiler's
// --verbose flag is set.
static void debugPrintBounce(int iter, int depth, int num_paths) {
    if (g_profiler().verbose()) {
        printf("  iter=%d depth=%d paths=%d\n", iter, depth, num_paths);
    }
}

// Update the ImGui trace-depth display and per-kernel timing after each frame.
static void updateGuiAfterFrame(Profiler& prof, GuiDataContainer* gui) {
    if (prof.enabled() && gui != NULL) {
        prof.updateGuiData(gui);
    }
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

    Profiler& prof = g_profiler();
    prof.beginIteration(iter);

    // ---- 1. Primary rays ------------------------------------------------
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(
        cam, iter, traceDepth, g_dev.paths);
    checkCUDAError("generate camera ray");

    int  depth     = 0;
    int  num_paths = pixelcount;
    bool done      = false;

    // ---- 2. Bounce loop -------------------------------------------------
    while (!done)
    {
        // Note: No need to zero out g_dev.intersections here.
        // computeIntersections will overwrite all active path entries completely.
        // cudaMemset(g_dev.intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        prof.recordBounce(depth, num_paths);

        dim3 numBlocks((num_paths + blockSize1d - 1) / blockSize1d);

        prof.gpuStart(ProfilerOp::ComputeIntersections);
        computeIntersections<<<numBlocks, blockSize1d>>>(
            depth, num_paths, g_dev.paths,
            g_dev.geoms, hst_scene->geoms.size(), g_dev.intersections);
        prof.gpuStop(ProfilerOp::ComputeIntersections);
        checkCUDAError("trace one bounce");
        depth++;

        // GPU timer: sortPathsByMaterial consists of asynchronous Thrust calls
        // (transform, sequence, sort_by_key, gatherx2).  None of these implicitly
        // synchronise -- they launch GPU kernels and return immediately.  A CPU
        // timer would only capture launch overhead (~us), missing the actual GPU
        // work.  Using cudaEvent captures true GPU execution time and the
        // cudaEventSynchronize inside gpuStop provides the sync point.
        //
        // Contrast with compactPaths (Thrust copy_if): that returns a host-visible
        // iterator, so Thrust internally syncs to count survivors.  CPU timer is
        // correct there because it naturally captures the full blocking cost.
        prof.gpuStart(ProfilerOp::SortByMaterial);
        sortPathsByMaterial(num_paths);  // no-op when g_opts.sortByMaterial==false
        prof.gpuStop(ProfilerOp::SortByMaterial);

        prof.gpuStart(ProfilerOp::ShadeMaterial);
        shadeMaterial<<<numBlocks, blockSize1d>>>(
            iter, num_paths,
            g_dev.intersections, g_dev.paths, g_dev.materials,
            traceDepth, hst_scene->state.rrMinBounces,
            g_opts.fresnelMode);
        prof.gpuStop(ProfilerOp::ShadeMaterial);

        bool allDead = compactActivePaths(num_paths, blockSize1d);
        done = allDead || (depth >= traceDepth);

        debugPrintBounce(iter, depth, num_paths);

        if (guiData != NULL)
            guiData->TracedDepth = depth;
    }

    // ---- 3. Accumulation ------------------------------------------------
    // Survivors (paths that reached traceDepth without terminating) also
    // carry valid attenuated colors and must contribute.
    {
        dim3 numBlocks((pixelcount + blockSize1d - 1) / blockSize1d);
        finalGather<<<numBlocks, blockSize1d>>>(
            num_paths, g_dev.image, g_dev.paths);
    }

    // ---- 4. Display -----------------------------------------------------
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(
        pbo, cam.resolution, iter, g_dev.image);

    cudaMemcpy(hst_scene->state.image.data(), g_dev.image,
               pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");

    prof.endIteration();
    updateGuiAfterFrame(prof, guiData);

    // CSV output is flushed by atexit(profiler.shutdown) in main.cpp.
    // Doing it here would race with endFrame() in runCuda(), which fires
    // after pathtrace() returns.
}
