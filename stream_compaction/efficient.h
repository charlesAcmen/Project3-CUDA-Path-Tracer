#pragma once

#include "common.h"

// Forward declaration for PathSegment (defined in sceneStructs.h)
struct PathSegment;

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        // Specialized compact for PathSegment arrays (device memory version)
        // Removes terminated paths (remainingBounces <= 0) from the array
        // Input and output are both device pointers
        int compactPathSegments(int n, PathSegment *dev_odata, const PathSegment *dev_idata);
        int compactPathSegmentsSharedMemory(int n, PathSegment *dev_odata, const PathSegment *dev_idata);
    }
}
