#pragma once

#include "common.h"

// Forward declaration for PathSegment (defined in sceneStructs.h)
struct PathSegment;

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        // ========================================================================
        // PathSegment Stream Compaction API
        // ========================================================================
        
        /**
         * Compacts PathSegment arrays using global memory scan.
         * Removes terminated paths (remainingBounces <= 0) from the array.
         * 
         * Uses traditional Blelloch scan algorithm with global memory operations.
         * Suitable for general use cases.
         * 
         * @param n           Number of PathSegments in dev_idata
         * @param dev_odata   Device pointer to output array (must be pre-allocated)
         * @param dev_idata   Device pointer to input array
         * @returns           Number of active paths remaining after compaction
         */
        int compactPathSegments(int n, PathSegment *dev_odata, const PathSegment *dev_idata);
        
        /**
         * Compacts PathSegment arrays using shared memory scan across multiple blocks.
         * Removes terminated paths (remainingBounces <= 0) from the array.
         * 
         * Implements work-efficient stream compaction using shared memory
         * as described in GPU Gems 3, Chapter 39. Performs per-block exclusive
         * scans in shared memory with hierarchical block-offset propagation.
         * Generally offers better performance than global memory version for large arrays.
         * 
         * @param n           Number of PathSegments in dev_idata
         * @param dev_odata   Device pointer to output array (must be pre-allocated)
         * @param dev_idata   Device pointer to input array
         * @returns           Number of active paths remaining after compaction
         */
        int compactPathSegmentsSharedMemory(int n, PathSegment *dev_odata, const PathSegment *dev_idata);
    }
}

