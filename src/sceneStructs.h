#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE
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

/**
 * LightInfo — Compact description of one emissive geometry (light source).
 *
 * Stored in a GPU device array for next-event estimation in the shading
 * kernel.  Each emissive geometry in the scene produces one entry.
 *
 * The emittedRadiance field is pre-multiplied at init time (color × emittance)
 * so the shading kernel does not need to look up the Material array for the
 * emission value — one fewer device-memory load per direct-light evaluation.
 */
struct LightInfo {
    int geomIndex;              // index into the device geoms array
    float area;                 // world-space surface area (for PDF computation)
    float inverseArea;          // 1/area (precomputed to avoid GPU division)
    glm::vec3 emittedRadiance;  // material.color * material.emittance = Le
};

// POD projection of RenderState fields for GPU kernel parameters.
// Does NOT own data — RenderState is the single source of truth.
// Assembled locally at kernel launch time from hst_scene->state.
struct ShadingConfig
{
    int traceDepth;
    int rrMinBounces;    // guaranteed bounces before Russian roulette
    int fresnelMode;     // 0=Schlick, 1=Accurate
    int rngMode;         // 0=LCG, 1=scrambled Halton
    Camera cam;
    DebugConfig debug;

    // --- Direct lighting (next-event estimation) ---
    int      numLights;      // number of emissive geometries (0 = skip NEE)
    LightInfo* lightInfos;   // device array of LightInfo (nullptr if none)
    Geom*      geoms;        // device array of all geoms (for light-sampling transforms)
    int        numGeoms;     // total geometry count (for shadow ray bounds)
    float      totalLightArea; // sum of emissive surface areas (for PDF denominator)
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    int rrMinBounces;  // guaranteed bounces before Russian roulette (default 3)
    int fresnelMode;   // 0=Schlick (default), 1=Accurate
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
  int geomIndex;        // index of the hit geometry in the device geoms array (-1 = miss).
                        // Needed for direct lighting (light sampling transforms)
                        // and future SSS (geometry containment tests).
};
