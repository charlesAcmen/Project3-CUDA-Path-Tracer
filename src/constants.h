#pragma once

// ====================================================================
// Mathematical constants used throughout the renderer.
//
// This header has zero dependencies -- it can be included from any
// compilation unit (host or device) without pulling in I/O libraries.
// ====================================================================

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define DEG_TO_RAD        (PI / 180.0f)   // degrees → radians (scene rotations, FOV)
#define RAD_TO_DEG        (180.0f / PI)   // radians → degrees (FOV back-calc)
#define EPSILON           0.00001f    // ray origin offset for self-intersection prevention
#define RAY_EPSILON       1e-10f      // minimum ray–scene distance; rejects near-parallel / self-hit
// Minimum squared length a refract() result must exceed to be a valid
// direction.  Unit |d|² = 1.0; a NaN (TIR: sqrt of negative) compares
// false against ANY threshold; a zero vector is 0.0.  0.5 sits safely
// between all three.
#define REFRACT_VALID_SQ_LEN_MIN 0.5f
#define ROUGHNESS_THRESHOLD 0.001f
// A specular exponent ≈ 0 means maximum roughness (ROUGHNESS = 1 → exponent
// = 2/1² − 2 = 0), where the Phong lobe is uniform and the powf(x, 1/(n+1))
// in samplePhongSpecularDir reduces to x.  Used only to skip that powf —
#define SPECULAR_EXPONENT_ZERO_EPSILON 0.00001f
#define RR_P_MIN          0.2f
#define RR_P_MAX          1.0f
#define LARGE_T           1e30f       // sentinel > any valid ray–scene intersection

// Bloom post-processing limits — single source of truth shared by the device
// kernels (postprocess/bloom.cuh) and the host config (src/config.h), so the
// legal radius range can't drift between the two.
#define MAX_BLOOM_RADIUS 32           // max bloom Gaussian radius (device weights buffer size)
#define BLOOM_BLOCK_SIZE 256          // threads per 1D block for the bloom blur kernels
