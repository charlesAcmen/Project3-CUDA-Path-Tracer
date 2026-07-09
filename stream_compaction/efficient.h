#pragma once

#include "common.h"

// Forward declaration for PathSegment (defined in sceneStructs.h)
struct PathSegment;

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        /**
         * Device-side workspace shared by all stream-compaction paths.
         * Allocated once at scene-load time and freed at shutdown.
         *
         * scanBuffer   - int buffer for scan results: reused by both the global-
         *                 memory and shared-memory compaction paths (avoids
         *                 per-frame cudaMalloc / cudaFree).
         * scanScratch  - hierarchical block-sum scratch space; used only by the
         *                 shared-memory path.
         * flagBuffer   - uint8_t boolean flags (shared-memory uint8 fast path).
         *                 Separated from scanBuffer so that scan loads are 4x
         *                 smaller while scan stores stay int-sized.
         * scanBlockSize / scanBlockElements - auto-detected optimal block
         *                 dimensions for the templated shared-memory scan kernels.
         */
        struct CompactionWorkspace {
            int*     scanBuffer        = nullptr;
            int*     scanScratch       = nullptr;
            unsigned char* flagBuffer  = nullptr;   // uint8_t
            size_t   scanBufferInts    = 0;
            size_t   scanScratchInts   = 0;
            int      maxElements       = 0;
            int      scanBlockSize     = 256;       // threads per block
            int      scanBlockElements = 512;       // elements per block (= 2 * scanBlockSize)
        };

        /**
         * Initializes the fixed scratch workspace used by stream compaction.
         * The workspace is sized for the maximum number of path segments the
         * renderer can produce for the current scene resolution.
         *
         * Must be called once before any compaction and freed in pathtraceFree().
         */
        void initCompactionWorkspace(int maxElements);

        /**
         * Releases the compaction workspace allocated by initCompactionWorkspace().
         */
        void freeCompactionWorkspace();

        // ========================================================================
        // PathSegment Stream Compaction API
        // ========================================================================

        /**
         * Compacts PathSegment arrays using global memory scan.
         * Removes terminated paths (remainingBounces <= 0) from the array.
         *
         * Uses traditional Blelloch scan algorithm with global memory operations.
         * Prefers the pre-allocated workspace scanBuffer when available.
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
         * Uses uint8_t boolean flags to reduce scan-phase memory bandwidth,
         * and auto-selects optimal block size (256 or 512) based on device.
         *
         * @param n           Number of PathSegments in dev_idata
         * @param dev_odata   Device pointer to output array (must be pre-allocated)
         * @param dev_idata   Device pointer to input array
         * @returns           Number of active paths remaining after compaction
         */
        int compactPathSegmentsSharedMemory(int n, PathSegment *dev_odata, const PathSegment *dev_idata);
    }
}
