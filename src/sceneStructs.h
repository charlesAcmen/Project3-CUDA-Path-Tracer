#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

// Per-triangle texture binding: which image in the texture table serves each
// role.  A value >= 0 indexes the concatenated texture array; -1 = no
// texture for that role (use the material's own color / fall back).
//
// The roles mirror glTF material slots (base / normal / metallic-roughness
// ORM / occlusion / emissive).  Only baseColor is sampled by the current
// Lambert/Phong shading; the rest are stored so future features (normal
// mapping, PBR, emissive maps) just sample the slot that is already bound.
struct TextureBinding
{
    int baseColor        = -1;
    int normal           = -1;
    int metallicRoughness = -1;   // ORM: metallic (B) + roughness (G) packed
    int occlusion        = -1;
    int emissive         = -1;
};

// Triangle backed by an OBJ mesh.
//
// Triangles are stored in object space in the Scene's hostTriangles, then
// baked to WORLD space by buildSceneBvh (vertices via the geom transform,
// normals via the inverse-transpose).  The device array the traversal reads
// is therefore world-space; materialId tags which material the triangle
// belongs to so the shading kernel can resolve it from the hit triangle.
struct Triangle {
    glm::vec3 v0, v1, v2;  // three vertex positions
    glm::vec3 n0, n1, n2;  // vertex normals (smooth shading interpolation)
    glm::vec2 uv0{ 0.0f }, uv1{ 0.0f }, uv2{ 0.0f };  // per-vertex texture coordinates (UVs)
    int materialId = -1;   // material index; set during the world-space bake
    TextureBinding tex;    // per-triangle glTF texture slots (all -1 unless glTF-assigned)
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    // Mesh geometry: slice into the device-wide flat triangle array.
    // Set to (-1, 0) for non-mesh primitives.
    int meshTriangleOffset;   // first triangle index in the device array
    int meshTriangleCount;    // number of triangles belonging to this mesh
};

enum class MaterialType
{
    Diffuse,
    Reflective,
    Refractive,
    Emissive
};

struct Material
{
    glm::vec3 color;              // Base albedo or surface tint for diffuse/refraction throughput
    //基础反射率或漫反射/折射的表面色调
    struct
    {
        float exponent;           // Phong exponent or glossiness for specular highlight falloff
        //Phong指数或高光的光泽度，用于高光衰减
        float invExponentPlusOne; // Precomputed 1/(exponent+1) to avoid GPU division in Phong sampling
        glm::vec3 color;          // Specular color tint for mirror-like reflections
        //镜面反射的高光颜色色调
        // Raw scene ROUGHNESS scalar, kept so the shader can distinguish
        // "author explicitly wrote ROUGHNESS" (∈ [0,1]) from "unspecified"
        // (-1).  The explicit value wins over a glTF roughnessFactor fallback;
        // exponent/invExponentPlusOne above are its precomputed conversion.
        float roughness = -1.0f;
    } specular;
    MaterialType type;            // Explicit material classification used by scattering logic
    float indexOfRefraction;      // IOR of the refractive material, e.g. 1.5 for glass
    float invIndexOfRefraction;   // Precomputed inverse IOR to avoid GPU division
    float emittance;              // Emission strength for light sources (nonzero = emissive)

    // Texture mapping.  textureId selects the image to sample in the
    // shading step; -1 = no texture (use `color`), -2 = procedural
    // checkerboard, >= 0 = index into the scene's texture array.
    int   textureId = -1;
    float uvScale   = 1.0f;       // UV repeat scale (1 = one tile over [0,1])
};

// One image in the scene's texture array, referenced by Material::textureId
// (>= 0).  On the GPU all images' texels live concatenated in one flat
// buffer; TextureInfo describes this image's slice of it.
struct TextureInfo
{
    int pixelOffset = 0;   // texel offset into the concatenated pixel buffer
    int width       = 0;   // texel width
    int height      = 0;   // texel height
};

// The scene's texture assets, uploaded once at init and read-only afterward.
// `pixels` holds every image's texels concatenated into one flat LINEAR-RGB
// buffer; `infos[count]` describes each image's slice of it.  Shading code
// indexes it through a TextureBinding slot (>= 0), never directly.
//
// Bundled as one struct rather than passing pixels+infos as separate kernel
// parameters so the sampler signatures stay stable as texture roles are
// added — a future feature (normal map, emissive map) reads its slot through
// the same table without touching any function signature.
struct TextureTable
{
    // Owned device pointers (cudaMalloc/cudaFree) — non-const here; samplers
    // take them as `const` parameters.
    glm::vec3*   pixels = nullptr;   // concatenated texel buffer (linear RGB)
    TextureInfo* infos  = nullptr;   // per-image slice descriptors
    int          count  = 0;         // number of images in the table
};

// Whether a ray is entering or exiting a refractive medium.
enum class HitSide : int
{
    Outside = 0,
    Inside = 1
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float lensRadius;       // lens aperture radius; 0 = pinhole camera (backward compatible)
    float focalDistance;    // distance from camera to plane of perfect focus
};

// Centralized debug configuration.  Passed by value to GPU kernels where
// every thread reads the same flag → uniform branch → zero warp divergence.
// ImGui / JSON / command-line can all set these; runtime toggling avoids
// recompilation.
struct DebugConfig
{
    bool showDOFOverlay = false;   // overlay focal-plane pixels in green
    float focalTolerance = 0.5f;   // distance threshold (world units) for
                                   // "at focal plane"
};

// POD projection of RenderState fields for GPU kernel parameters.
// Does NOT own data — RenderState is the single source of truth.
// Assembled locally at kernel launch time from hst_scene->state.
// ---- Typed enums for runtime configuration ---------------------------
// Numeric values match the legacy --compact=N / --rng=N
// CLI flags so existing usage is backwards-compatible.

enum class CompactMethod : int {
    Off        = 0,
    GlobalScan = 1,
    Thrust     = 2,
    SharedMem  = 3
};

inline const char* toString(CompactMethod m) {
    switch (m) {
        case CompactMethod::Off:        return "Off";
        case CompactMethod::GlobalScan: return "GlobalScan";
        case CompactMethod::Thrust:     return "Thrust";
        case CompactMethod::SharedMem:  return "SharedMem";
    }
    return "?";
}

enum class RngMode : int {
    LCG    = 0,
    HALTON = 1
};

inline const char* toString(RngMode m) {
    switch (m) {
        case RngMode::LCG:    return "LCG";
        case RngMode::HALTON: return "Halton";
    }
    return "?";
}

struct ShadingConfig
{
    int           traceDepth;
    int           rrMinBounces;    // guaranteed bounces before Russian roulette
    RngMode       rngMode;         // LCG / Halton
    Camera cam;
    DebugConfig debug;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    int rrMinBounces;  // guaranteed bounces before Russian roulette (default 3)
    std::vector<glm::vec3> image;
    std::string imageName;
    DebugConfig debug;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;// parametric distance along the ray
  //t > 0.0f: intersection with an object
  //t < 0.0f: no intersection with an object(initial value)
  glm::vec3 surfaceNormal;
  int materialId;
  glm::vec2 uv;   // interpolated texture coordinate at the hit point
  TextureBinding tex;   // per-triangle texture slots (copied from the hit triangle)
};
