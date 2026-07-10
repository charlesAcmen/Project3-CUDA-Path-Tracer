#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

/**
 * Generates a random direction vector in a hemisphere oriented around a given surface normal,
 * with cosine-weighted distribution (importance sampling for diffuse surfaces).
 * 
 * This function implements Monte Carlo importance sampling for physically-based rendering.
 * Cosine weighting means rays closer to the normal are more likely to be sampled than
 * rays near the horizon, which matches the cosine term in Lambert's law and reduces variance.
 * 
 * The algorithm consists of three main steps:
 * 1. Sample spherical coordinates using inverse transform sampling
 * 2. Construct an orthonormal basis (ONB) aligned with the surface normal
 * 3. Transform the local sample to world space coordinates
 * 
 * @param normal The surface normal vector (assumed to be normalized)
 * @param rng Random number generator for Monte Carlo sampling
 * @return A unit direction vector in the hemisphere, cosine-weighted around the normal
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    // STEP 1: Polar Coordinate Sampling (Inverse Transform Sampling)
    // ----------------------------------------------------------------
    // To achieve cosine-weighted distribution, we use the probability density function:
    // p(theta) = cos(theta) * sin(theta), where theta is the polar angle from the normal
    // 
    // The cumulative distribution function (CDF) is: CDF(theta) = sin^2(theta)
    // Inverting the CDF gives: theta = arcsin(sqrt(xi_1)), where xi_1 is a uniform random variable
    // 
    // This simplifies to:
    //   cos(theta) = sqrt(xi_1)         (vertical component, weighted toward normal)
    //   sin(theta) = sqrt(1 - xi_1)     (horizontal component, from Pythagorean identity)
    
    float up = sqrt(u01(rng)); // cos(theta) - probability of sampling decreases with angle from normal
    float over = sqrt(1 - up * up); // sin(theta) - derived from trigonometric identity sin^2(theta) + cos^2(theta) = 1
    float around = u01(rng) * TWO_PI; // phi - azimuthal angle, uniformly distributed in [0, 2*PI]

    // At this point, we have spherical coordinates (theta, phi) that represent a direction
    // in a LOCAL coordinate system where the normal is aligned with the Z-axis:
    //   Local coordinates: (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))

    // STEP 2: Construct Orthonormal Basis (ONB)
    // ------------------------------------------
    // To transform our local sample into world space, we need to build a coordinate frame
    // where the normal becomes the "up" direction (local Z-axis), and we need two
    // perpendicular vectors to complete the basis (local X and Y axes).
    // 
    // Peter Kutz's Trick: Since the normal is a unit vector (x^2 + y^2 + z^2 = 1),
    // it's mathematically impossible for all three components to have absolute values
    // greater than sqrt(1/3) ~= 0.577. At least one component must be smaller.
    // 
    // By selecting the axis corresponding to the smallest component, we guarantee
    // that the cross product won't degenerate to zero (which would happen if the
    // normal and our chosen axis were parallel or nearly parallel).

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0); // X-axis is safe to use
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0); // Y-axis is safe to use
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1); // Z-axis is safe to use
    }

    // Generate two perpendicular direction vectors using cross products
    // These form the tangent (U) and bitangent (V) vectors of our local coordinate system
    // 
    // First perpendicular direction (U-axis): orthogonal to both normal and our helper vector
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    
    // Second perpendicular direction (V-axis): orthogonal to both normal and first perpendicular
    // This completes the right-handed orthonormal basis {U, V, N}
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    // STEP 3: Transform from Local to World Space
    // --------------------------------------------
    // Now we have an orthonormal basis:
    //   - normal (N): the "up" direction (local Z-axis)
    //   - perpendicularDirection1 (U): tangent direction (local X-axis)
    //   - perpendicularDirection2 (V): bitangent direction (local Y-axis)
    // 
    // Our sampled direction in local coordinates is:
    //   local_dir = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
    //             = (over * cos(around), over * sin(around), up)
    // 
    // To transform to world space, we compute the linear combination:
    //   world_dir = cos(theta) * N + sin(theta)cos(phi) * U + sin(theta)sin(phi) * V
    //             = up * normal + cos(around) * over * U + sin(around) * over * V
    // 
    // This gives us our final cosine-weighted random direction in world space.

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ float fresnelSchlick(float cosThetaI, float n1, float n2)
{
    float r0 = (n1 - n2) / (n1 + n2);
    r0 = r0 * r0;
    float oneMinusCos = 1.0f - cosThetaI;
    float oneMinusCos2 = oneMinusCos * oneMinusCos;
    float oneMinusCos5 = oneMinusCos2 * oneMinusCos2 * oneMinusCos;
    return r0 + (1.0f - r0) * oneMinusCos5;
}
//returns the fraction of non-polarized light reflected at the interface between two materials with indices of refraction n1 and n2, 
//given the cosine of the incident angle cosThetaI.
__host__ __device__ float fresnelAccurate(float cosThetaI, float n1, float n2)
{
    float sinThetaI = sqrtf(fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI));
    //SNELL'S LAW: n1 * sin(thetaI) = n2 * sin(thetaT)
    float sinThetaT = (n1 / n2) * sinThetaI;
    if (sinThetaT >= 1.0f)
    {// Total internal reflection occurs when the angle of incidence exceeds the critical angle, resulting in no refraction.
        return 1.0f;
    }

    float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sinThetaT * sinThetaT));
    float rParallel = (n2 * cosThetaI - n1 * cosThetaT) /
                      (n2 * cosThetaI + n1 * cosThetaT);
    // Correct perpendicular (s-polarized) Fresnel term:
    // r_perp = (n1 * cosThetaI - n2 * cosThetaT) / (n1 * cosThetaI + n2 * cosThetaT)
    float rPerpendicular = (n1 * cosThetaI - n2 * cosThetaT) /
                           (n1 * cosThetaI + n2 * cosThetaT);
    return (rParallel * rParallel + rPerpendicular * rPerpendicular) * 0.5f;
}

__host__ __device__ HitSide classifyRefraction(
    glm::vec3 rayDir,//assumed to be normalized
    glm::vec3 surfaceNormal,//assumed to be normalized
    float ior,//index of refraction of the material
    float& outN1,//from IOR
    float& outN2,//to IOR
    float& outCosThetaI//a positive value of costheta incident angle
)
{
    //>=0:exit the object
    //<0:enter the object
    float cosTheta = glm::dot(rayDir, surfaceNormal);

    if (cosTheta < 0.0f)
    {
        outN1 = 1.0f;
        outN2 = ior;
        outCosThetaI = -cosTheta;//invert the sign to make it positive
        return HitSide::Outside;
    }
    else
    {
        outN1 = ior;
        outN2 = 1.0f;
        outCosThetaI = cosTheta;
        return HitSide::Inside;
    }
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng,
    int fresnelMode)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    // Generate new random direction for diffuse reflection (cosine-weighted hemisphere sampling)

    // CRITICAL: Offset ray origin along the NORMAL direction to prevent self-intersection
    // Using EPSILON (1e-5) provides sufficient clearance for the Cornell Box scale (units: ~10)
    // 
    // Common mistake: offsetting along newDirection instead of normal
    // - When newDirection is nearly parallel to the surface (grazing angle),
    //   offset along newDirection has almost zero normal component
    // - This causes the ray to start below the surface -> self-intersection -> shadow acne
    // 
    // Why 1e-5 works for this scene:
    // - Scene scale: [-5, 10] -> typical values around 1-10 units
    // - Float precision: approx 1e-7 relative error
    // - Accumulated error after transforms: approx 1e-6 to 1e-5
    // - Safety margin: 1e-5 is approx 10x the expected numerical error
    // 
    // When to adjust epsilon:
    // - Increase to 1e-4 if seeing shadow acne (black speckles on surfaces)
    // - Decrease to 1e-6 for scenes with very thin geometry (< 0.01 units)
    // - For large-scale scenes (>1000 units), scale proportionally
    
    //Note that since only Refraction always requires opposite direction offset
    //Let's leave sign judging to future release.i.e.TODO: judge sign by dot product of normal and newDirection
    thrust::uniform_real_distribution<float> u01(0, 1);

    if (m.hasRefractive > 0.5f)
    {
        float n1, n2, cosThetaI;
        HitSide side = classifyRefraction(pathSegment.ray.direction, normal, m.indexOfRefraction, n1, n2, cosThetaI);
        float etaRatio = n1 / n2;

        float reflectance = (fresnelMode == 1)
            ? fresnelAccurate(cosThetaI, n1, n2)
            : fresnelSchlick(cosThetaI, n1, n2);

        bool totalInternalReflection = false;
        float sinThetaI = sqrtf(fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI));
        float sinThetaT = etaRatio * sinThetaI;
        if (sinThetaT >= 1.0f)
        {
            totalInternalReflection = true;
        }

        bool reflect = totalInternalReflection || (u01(rng) < reflectance);

        if (reflect)
        {
            glm::vec3 reflectedDir = glm::reflect(pathSegment.ray.direction, normal);
            float offsetSign = glm::dot(reflectedDir, normal) > 0.0f ? 1.0f : -1.0f;
            pathSegment.ray.origin = intersect + normal * (EPSILON * offsetSign);
            pathSegment.ray.direction = glm::normalize(reflectedDir);
            pathSegment.color *= m.color / fmaxf(reflectance, 1e-6f);
        }
        else
        {
            glm::vec3 refractedDir = glm::refract(pathSegment.ray.direction, normal, etaRatio);
            float offsetSign = glm::dot(refractedDir, normal) > 0.0f ? 1.0f : -1.0f;
            pathSegment.ray.origin = intersect + normal * (EPSILON * offsetSign);
            pathSegment.ray.direction = glm::normalize(refractedDir);
            float transmitProb = 1.0f - reflectance;
            pathSegment.color *= m.color / fmaxf(transmitProb, 1e-6f);
        }
    }
    else if (m.hasReflective > 0.5f)
    {
        glm::vec3 reflectedDir = glm::reflect(pathSegment.ray.direction, normal);
        float offsetSign = glm::dot(reflectedDir, normal) > 0.0f ? 1.0f : -1.0f;
        pathSegment.ray.origin = intersect + normal * (EPSILON * offsetSign);
        pathSegment.ray.direction = glm::normalize(reflectedDir);
        pathSegment.color *= m.color;
    }
    else
    {
        glm::vec3 newDirection = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect + normal * EPSILON;
        // Apply diffuse material color (energy attenuation)
        // multiplier = fr * cos theta/pdf(omega)
        // where pdf(omega) = cos theta / PI
        // BSDF of diffuse reflection: fr = R / PI
        pathSegment.ray.direction = glm::normalize(newDirection);
        pathSegment.color *= m.color;
    }

    // Decrement remaining bounces
    pathSegment.remainingBounces--;
}

