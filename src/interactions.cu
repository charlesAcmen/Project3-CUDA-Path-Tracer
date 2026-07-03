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
    // p(θ) = cos(θ) * sin(θ), where θ is the polar angle from the normal
    // 
    // The cumulative distribution function (CDF) is: CDF(θ) = sin²(θ)
    // Inverting the CDF gives: θ = arcsin(√ξ₁), where ξ₁ is a uniform random variable
    // 
    // This simplifies to:
    //   cos(θ) = √ξ₁         (vertical component, weighted toward normal)
    //   sin(θ) = √(1 - ξ₁)   (horizontal component, from Pythagorean identity)
    
    float up = sqrt(u01(rng)); // cos(theta) - probability of sampling decreases with angle from normal
    float over = sqrt(1 - up * up); // sin(theta) - derived from trigonometric identity sin²θ + cos²θ = 1
    float around = u01(rng) * TWO_PI; // phi - azimuthal angle, uniformly distributed in [0, 2π]

    // At this point, we have spherical coordinates (θ, φ) that represent a direction
    // in a LOCAL coordinate system where the normal is aligned with the Z-axis:
    //   Local coordinates: (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ))

    // STEP 2: Construct Orthonormal Basis (ONB)
    // ------------------------------------------
    // To transform our local sample into world space, we need to build a coordinate frame
    // where the normal becomes the "up" direction (local Z-axis), and we need two
    // perpendicular vectors to complete the basis (local X and Y axes).
    // 
    // Peter Kutz's Trick: Since the normal is a unit vector (x² + y² + z² = 1),
    // it's mathematically impossible for all three components to have absolute values
    // greater than √(1/3) ≈ 0.577. At least one component must be smaller.
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
    //   local_dir = (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ))
    //             = (over * cos(around), over * sin(around), up)
    // 
    // To transform to world space, we compute the linear combination:
    //   world_dir = cos(θ) * N + sin(θ)cos(φ) * U + sin(θ)sin(φ) * V
    //             = up * normal + cos(around) * over * U + sin(around) * over * V
    // 
    // This gives us our final cosine-weighted random direction in world space.
    
    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    
    // Generate new random direction for diffuse reflection (cosine-weighted hemisphere sampling)
    glm::vec3 newDirection = calculateRandomDirectionInHemisphere(normal, rng);
    
    // CRITICAL: Offset ray origin along the NORMAL direction to prevent self-intersection
    // Using EPSILON (1e-5) provides sufficient clearance for the Cornell Box scale (units: ~10)
    // 
    // Common mistake: offsetting along newDirection instead of normal
    // - When newDirection is nearly parallel to the surface (grazing angle),
    //   offset along newDirection has almost zero normal component
    // - This causes the ray to start below the surface → self-intersection → shadow acne
    // 
    // Why 1e-5 works for this scene:
    // - Scene scale: [-5, 10] → typical values around 1-10 units
    // - Float precision: ~1e-7 relative error
    // - Accumulated error after transforms: ~1e-6 to 1e-5
    // - Safety margin: 1e-5 is ~10x the expected numerical error
    // 
    // When to adjust epsilon:
    // - Increase to 1e-4 if seeing shadow acne (black speckles on surfaces)
    // - Decrease to 1e-6 for scenes with very thin geometry (< 0.01 units)
    // - For large-scale scenes (>1000 units), scale proportionally
    
    //Note that since only Refraction always requires opposite direction offset
    //Let's leave sign judging to future release.i.e.TODO:judge sign by dot product of normal and newDirection
    pathSegment.ray.origin = intersect + normal * EPSILON;
    pathSegment.ray.direction = newDirection;
    
    // Apply diffuse material color (energy attenuation)
    //multiplier = fr * cos theta/pdf(omega)
    //where pdf(omega) = coos theta / π
    //BSDF of diffuse reflection：fr = R / π
    pathSegment.color *= m.color;
    
    // Decrement remaining bounces
    pathSegment.remainingBounces--;
}
