#pragma once

// ====================================================================
// Stream Compaction Pipeline
//
// Removes terminated paths (remainingBounces <= 0) from the active
// set so that subsequent bounces only process surviving paths.
//
// Must include kernels/accumulation.cuh BEFORE this file so the
// gatherTerminatedPaths kernel is available for pre-compaction gathering.
// ====================================================================

#include "sceneStructs.h"
#include "profiler/profiler.h"
#include "kernel_config.h"  // LAUNCH_KERNEL_AUTO
#include "efficient.h"      // StreamCompaction::Efficient
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <algorithm>        // std::swap
#include "kernels/accumulation.cuh"  // gatherTerminatedPaths

// ---- Predicate for Thrust copy_if ----

struct IsPathActive {
    __device__ bool operator()(const PathSegment& p) const {
        return p.remainingBounces > 0;
    }
};

// ---- Compaction dispatch implementations ----

static int compactCoreThrust(int n, PathSegment* dst, const PathSegment* src) {
    PathSegment* end = thrust::copy_if(thrust::device, src, src + n, dst, IsPathActive());
    return static_cast<int>(end - dst);
}

static int compactCoreGlobalMem(int n, PathSegment* dst, const PathSegment* src) {
    return StreamCompaction::Efficient::compactPathSegments(n, dst, src);
}

static int compactCoreSharedMem(int n, PathSegment* dst, const PathSegment* src) {
    return StreamCompaction::Efficient::compactPathSegmentsSharedMemory(n, dst, src);
}

/**
 * Gather terminated path colors into the accumulation buffer, then
 * stream-compact the PathSegment array to remove dead entries.
 *
 * Compaction is only applied when g_opts.compactMethod != 0.
 * Uses ping-pong buffers (g_dev.paths <-> g_dev.pathsCompacted)
 * to avoid a separate allocation per bounce.
 *
 * @param num_paths   [in/out]  Active path count; set to survivors after.
 * @return            true if EVERY path terminated (caller may exit
 *                    bounce loop immediately).
 */
static bool compactActivePaths(int& num_paths)
{
    Profiler& prof = g_profiler();

    // Compaction disabled → nothing to do
    if (g_opts.compactMethod == 0) {
        return false;
    }

    // 1. Bank terminated-path colors before compaction discards them.
    //    Without this gather, paths that hit a light would have their
    //    radiance lost, producing a black image.
    prof.gpuStart(ProfilerOp::GatherTerminatedPaths);
    LAUNCH_KERNEL_AUTO(gatherTerminatedPaths, num_paths,
        num_paths, g_dev.image, g_dev.paths);
    prof.gpuStop(ProfilerOp::GatherTerminatedPaths);
    checkCUDAError("gatherTerminatedPaths");

    // 2. Compact via the runtime-selected method.
    //    CPU timer is correct here because each method implicitly syncs
    //    (Thrust copy_if returns a host iterator; custom scans do a
    //    cudaMemcpy for the survivor count).
    prof.cpuStart(ProfilerOp::CompactPaths);
    int survivors = 0;
    if (g_opts.compactMethod == 1) {
        survivors = compactCoreGlobalMem(num_paths, g_dev.pathsCompacted, g_dev.paths);
    } else if (g_opts.compactMethod == 2) {
        survivors = compactCoreThrust(num_paths, g_dev.pathsCompacted, g_dev.paths);
    } else {
        survivors = compactCoreSharedMem(num_paths, g_dev.pathsCompacted, g_dev.paths);
    }
    prof.cpuStop(ProfilerOp::CompactPaths);

    // 3. Swap buffers — compacted array becomes the active one
    std::swap(g_dev.paths, g_dev.pathsCompacted);

    num_paths = survivors;
    return (num_paths == 0);
}
