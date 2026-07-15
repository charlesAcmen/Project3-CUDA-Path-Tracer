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
#define EPSILON           0.00001f
#define ROUGHNESS_THRESHOLD 0.001f
#define RR_P_MIN          0.2f
#define RR_P_MAX          1.0f
