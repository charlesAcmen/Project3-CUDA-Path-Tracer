#pragma once

#include "common.h"

// Forward declaration for PathSegment (defined in sceneStructs.h)
struct PathSegment;

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);

        // Specialized compact for PathSegment arrays (device memory version)
        // Removes terminated paths (remainingBounces <= 0) from the array
        // Input and output are both device pointers
        int compactPathSegments(int n, PathSegment *dev_odata, const PathSegment *dev_idata);

        // Shared-memory variants (GPU Gems 3, Chapter 39) — separate from global-memory APIs above
        void scanSharedMemory(int n, int *odata, const int *idata);
        int compactSharedMemory(int n, int *odata, const int *idata);
        int compactPathSegmentsSharedMemory(int n, PathSegment *dev_odata, const PathSegment *dev_idata);

        // Export kernels for use by other modules (e.g., radix sort)
        __global__ void kernUpSweep(int n, int d, int *data);
        __global__ void kernDownSweep(int n, int d, int *data);
    }
}
