#pragma once

// ====================================================================
// trace.h — shared logging facility for the TRACED copies of the BVH
// production code (aabb_traced.h / bvh_traced.h / bvh_traced.cu).
//
// The traced copies are functional duplicates of src/bvh/* with printf
// instrumentation added, so the production code stays free of any
// test-only logic.  A single verbosity knob controls how much of the
// intermediate state of the key structs (AABB bounds, BvhNode fields,
// order permutation, pref/suff arrays, stack walk) is printed:
//
//     level 0 — silent (validation runs use this)
//     level 1 — process narrative (node decisions, stack pops, hits)
//     level 2 — full per-step detail (every expand, every slab, every
//               per-triangle test inside a leaf)
//
// The __CUDA_ARCH__ guard makes the TRACE macro a no-op on the device
// pass of nvcc, so __host__ __device__ functions in the traced headers
// still compile for device even though printf is host-only here.
// ====================================================================

#include <cstdio>

#include "glm/glm.hpp"

namespace trace {
    inline int level = 0;   // 0=silent, 1=narrative, 2=full per-step detail

    // Print 2*depth spaces so recursive/stack output aligns by depth.
    void indent(int depth);

    // Print a labeled vec3 as "(x, y, z)".
    void vec(const char* label, const glm::vec3& v);
}

#ifdef __CUDA_ARCH__
#define TRACE(lvl, ...) ((void)0)
#else
#define TRACE(lvl, ...)                                   \
    do { if (::trace::level >= (lvl)) std::printf(__VA_ARGS__); } while (0)
#endif
