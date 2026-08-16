#include "ray_generation.cuh"

// ====================================================================
// Primary Ray Generation Kernel Implementation
// ====================================================================

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, RngMode rngMode)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // RNG state shared across all primary-ray sampling.
        // Halton mode:  next(dim) selects a distinct prime base per dimension,
        //               so sequential calls with dim 0..3 produce independent
        //               values from the same Halton point — identical to using
        //               four separate RngState objects with the same index.
        // LCG mode:     sequential draws advance the engine, producing different
        //               values per call (unlike separate engines with the same
        //               seed, which would produce identical first draws).
        //   dim 0 (prime 2)  = AA jitter x
        //   dim 1 (prime 3)  = AA jitter y
        //   dim 2 (prime 5)  = lens aperture u  (only when DoF is active)
        //   dim 3 (prime 7)  = lens aperture v  (only when DoF is active)
        // depth = -1: a sentinel distinct from every bounce's
        // bounceNum * MAX_DRAWS_PER_BOUNCE encoding.  Bounce-0 shading uses
        // depth 0, so a 0 here would give the LCG branch the EXACT same seed
        // as the first-bounce scatter RNG → AA jitter and bounce-0 BSDF draws
        // become 100% correlated (a regression from the old
        // descending-remainingBounces schedule).  -1 keeps the primary-ray
        // stream independent in LCG mode; Halton is unaffected either way
        // (different dims already produce different Owen seeds).
        RngState rng = makeRngState(iter, index, -1, rngMode);

        // Anti-aliasing: stochastic sub-pixel jitter
        float jitterX = rng.next(HaltonDim::AaJitterX) - 0.5f;
        float jitterY = rng.next(HaltonDim::AaJitterY) - 0.5f;

        // Pinhole ray direction (centre-of-lens ray, undeflected)
        glm::vec3 pinholeDir = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up    * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f));

        if (cam.lensRadius > 0.0f) {
            // ---- Thin-lens depth of field ----
            // 1. Intersect pinhole ray with the focal plane.
            // cosTheta is the ray's direction cosine to the view axis; the focal
            // plane is hit iff cosTheta > 0 (ray points forward).  It is always
            // ≥ cos(fov/2) for on-frustum rays, so the fallback is just a guard.
            float cosTheta = glm::dot(pinholeDir, cam.view);
            float ft = (cosTheta > 0.0f) ? (cam.focalDistance / cosTheta)
                                         : cam.focalDistance;
            glm::vec3 pFocus = cam.position + ft * pinholeDir;

            // 2. Sample a point on the lens aperture via concentric disk mapping
            float lensU = rng.next(HaltonDim::LensApertureU);  // dim 2 (prime 5): aperture u
            float lensV = rng.next(HaltonDim::LensApertureV);  // dim 3 (prime 7): aperture v
            float dx, dy;
            concentricSampleDisk(lensU, lensV, dx, dy);

            // 3. Offset ray origin within the aperture
            glm::vec3 lensOffset = cam.lensRadius * (dx * cam.right + dy * cam.up);
            segment.ray.origin = cam.position + lensOffset;

            // 4. Aim ray at the focal-plane point — all rays for this pixel converge there
            segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
        } else {
            // Pinhole camera (default, lensRadius == 0)
            segment.ray.origin = cam.position;
            segment.ray.direction = pinholeDir;
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}
