#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

// Object-space triangle backed by an OBJ mesh.
// The intersection test transforms rays to object space via
// the geometry's inverseTransform, keeping triangles untransformed.
struct Triangle {
    glm::vec3 v0, v1, v2;  // three vertex positions in object space
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
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
        glm::vec3 color;          // Specular color tint for mirror-like reflections
        //镜面反射的高光颜色色调
    } specular;
    MaterialType type;            // Explicit material classification used by scattering logic
    float indexOfRefraction;      // IOR of the refractive material, e.g. 1.5 for glass
    float invIndexOfRefraction;   // Precomputed inverse IOR to avoid GPU division
    float emittance;              // Emission strength for light sources (nonzero = emissive)
};

// Fresnel evaluation mode used by refractive materials.
enum class FresnelMode : int
{
    Schlick = 0,
    Accurate = 1
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

enum class RngMode : int {
    LCG    = 0,
    HALTON = 1
};

struct ShadingConfig
{
    int           traceDepth;
    int           rrMinBounces;    // guaranteed bounces before Russian roulette
    FresnelMode   fresnelMode;     // Schlick / Accurate
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
    FresnelMode fresnelMode = FresnelMode::Schlick;
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
};
