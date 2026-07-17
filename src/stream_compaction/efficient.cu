#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "kernel_config.h"

// Include PathSegment definition (need full definition, not just forward declaration)
#include "../src/sceneStructs.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // ========================================================================
        // Compaction workspace (single instance, shared by all paths)
        // ========================================================================

        static CompactionWorkspace s_compactionWorkspace;

        static size_t computeScanScratchInts(int n, int blockElements)
        {
            size_t totalInts = 0;
            int current = (n + blockElements - 1) / blockElements;

            while (current > 0)
            {
                totalInts += static_cast<size_t>(current);

                if (current == 1)
                {
                    break;
                }

                current = (current + blockElements - 1) / blockElements;
            }

            return totalInts;
        }

        void initCompactionWorkspace(int maxElements)
        {
            freeCompactionWorkspace();

            if (maxElements <= 0)
            {
                return;
            }

            // Auto-detect optimal scan block size from device capabilities.
            // Decision lives in DeviceInfo so device queries stay centralized.
            const int blockSize     = DeviceInfo::getInstance().getOptimalScanBlockSize();
            const int blockElements = 2 * blockSize;

            s_compactionWorkspace.maxElements       = maxElements;
            s_compactionWorkspace.scanBlockSize     = blockSize;
            s_compactionWorkspace.scanBlockElements = blockElements;
            s_compactionWorkspace.scanBufferInts    = static_cast<size_t>(maxElements);
            s_compactionWorkspace.scanScratchInts   =
                computeScanScratchInts(maxElements, blockElements);

            // --- Allocations ---

            // int scan buffer - reused by global-mem and shared-mem paths
            cudaMalloc(
                reinterpret_cast<void**>(&s_compactionWorkspace.scanBuffer),
                s_compactionWorkspace.scanBufferInts * sizeof(int));
            checkCUDAError("cudaMalloc scanBuffer failed");

            // uint8_t flag buffer - shared-mem uint8 fast path
            cudaMalloc(
                reinterpret_cast<void**>(&s_compactionWorkspace.flagBuffer),
                static_cast<size_t>(maxElements) * sizeof(unsigned char));
            checkCUDAError("cudaMalloc flagBuffer failed");

            // Hierarchical block-sum scratch - shared-mem path
            if (s_compactionWorkspace.scanScratchInts > 0)
            {
                cudaMalloc(
                    reinterpret_cast<void**>(&s_compactionWorkspace.scanScratch),
                    s_compactionWorkspace.scanScratchInts * sizeof(int));
                checkCUDAError("cudaMalloc scanScratch failed");
            }
        }

        void freeCompactionWorkspace()
        {
            cudaFree(s_compactionWorkspace.scanBuffer);
            s_compactionWorkspace.scanBuffer = nullptr;

            cudaFree(s_compactionWorkspace.flagBuffer);
            s_compactionWorkspace.flagBuffer = nullptr;

            cudaFree(s_compactionWorkspace.scanScratch);
            s_compactionWorkspace.scanScratch = nullptr;

            s_compactionWorkspace.scanBufferInts  = 0;
            s_compactionWorkspace.scanScratchInts = 0;
            s_compactionWorkspace.maxElements     = 0;
        }

        // ========================================================================
        // Forward Declaration (Template Host Function)
        // ========================================================================
        //
        // scanExclusiveSharedMemoryDevice<BLOCK_SIZE, FlagType>
        //   FlagType = unsigned char : top level  -- reads uint8 flags from DRAM
        //   FlagType = int           : recursive  -- reads int block-sums (in-place)
        //
        // Forward-declared so the recursive call from the uint8_t instantiation
        // to the int instantiation is visible.

        template <int BLOCK_SIZE, typename FlagType>
        static void scanExclusiveSharedMemoryDevice(
            int n, int* dev_odata, const FlagType* dev_idata,
            int* dev_scratch, size_t scratchInts);

        // ========================================================================
        // Non-Template Kernels (global-memory scan + map/scatter)
        // ========================================================================

        __global__ void kernUpSweep(int n, int d, int *data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int stride = 1 << (d + 1);
            if (index * stride >= n) return;
            int i = (index + 1) * stride - 1;
            int offset = 1 << d;
            data[i] += data[i - offset];
        }

        __global__ void kernDownSweep(int n, int d, int *data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int stride = 1 << (d + 1);
            if (index * stride >= n) return;
            int i = (index + 1) * stride - 1;
            int offset = 1 << d;
            int temp = data[i - offset];
            data[i - offset] = data[i];
            data[i] += temp;
        }

        __global__ void kernMapPathSegmentToBoolean(int n, int *flags, const PathSegment *paths) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;
            flags[index] = (paths[index].remainingBounces > 0) ? 1 : 0;
        }

        __global__ void kernMapPathSegmentToBooleanU8(int n, unsigned char *flags, const PathSegment *paths) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;
            flags[index] = (paths[index].remainingBounces > 0) ? 1 : 0;
        }

        __global__ void kernScatterPathSegment(int n, PathSegment *odata,
                const PathSegment *idata, const int *indices) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;
            if (idata[index].remainingBounces > 0) {
                odata[indices[index]] = idata[index];
            }
        }

        // ========================================================================
        // Template Block-Scan Kernel (single kernel, two input types)
        // ========================================================================
        //
        // FlagType = unsigned char : loads 1-byte boolean flags from DRAM,
        //     converts to int in shared memory.  Used at the top level.
        // FlagType = int           : loads int block-sums for in-place
        //     recursive scans.
        //
        // Shared memory is always int[] because partial sums can exceed 255.
        // Output is always int[] because prefix sums range 0..N-1.

        template <int BLOCK_SIZE, typename FlagType>
        __global__ void kernBlockExclusiveScan(
            int n, int *odata, const FlagType *idata, int *blockSums)
        {
            constexpr int ELEMENTS = 2 * BLOCK_SIZE;
            __shared__ int temp[ELEMENTS];

            int thid = threadIdx.x;
            int bid  = blockIdx.x;
            int offset = 1;

            int index0 = bid * ELEMENTS + 2 * thid;
            int index1 = index0 + 1;

            // One load line --- the only difference between the two instantiations.
            // static_cast<int> is a no-op when FlagType == int.
            temp[2 * thid]     = (index0 < n) ? static_cast<int>(idata[index0]) : 0;
            temp[2 * thid + 1] = (index1 < n) ? static_cast<int>(idata[index1]) : 0;

            // Up-sweep
            for (int d = BLOCK_SIZE; d > 0; d >>= 1)
            {
                __syncthreads();
                if (thid < d)
                {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    temp[bi] += temp[ai];
                }
                offset *= 2;
            }

            if (thid == 0)
            {
                blockSums[bid] = temp[ELEMENTS - 1];
                temp[ELEMENTS - 1] = 0;
            }

            // Down-sweep
            for (int d = 1; d < ELEMENTS; d <<= 1)
            {
                offset >>= 1;
                __syncthreads();
                if (thid < d)
                {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }
            __syncthreads();

            if (index0 < n) odata[index0] = temp[2 * thid];
            if (index1 < n) odata[index1] = temp[2 * thid + 1];
        }

        // ========================================================================
        // Template Block-Offset Kernel
        // ========================================================================

        template <int BLOCK_SIZE>
        __global__ void kernAddBlockOffsets(int n, int *data, const int *blockOffsets)
        {
            constexpr int ELEMENTS = 2 * BLOCK_SIZE;
            int bid  = blockIdx.x;
            int thid = threadIdx.x;

            int index0 = bid * ELEMENTS + 2 * thid;
            int index1 = index0 + 1;
            int off = blockOffsets[bid];

            if (index0 < n) data[index0] += off;
            if (index1 < n) data[index1] += off;
        }

        // ========================================================================
        // Template Hierarchical Scan Function (single function, two input types)
        // ========================================================================
        //
        // Top-level call:    FlagType = unsigned char
        //   Reads uint8 flags from dev_iflags, writes int indices to dev_odata.
        //   Recurses with FlagType = int for block-sum scanning.
        //
        // Recursive call:    FlagType = int
        //   In-place int-to-int scan of block sums (dev_odata == dev_idata).

        template <int BLOCK_SIZE, typename FlagType>
        static void scanExclusiveSharedMemoryDevice(
            int n, int *dev_odata, const FlagType *dev_idata,
            int *dev_scratch, size_t scratchInts)
        {
            constexpr int ELEMENTS = 2 * BLOCK_SIZE;

            if (n <= 0) return;

            int numBlocks = (n + ELEMENTS - 1) / ELEMENTS;

            // Internal invariant: scratch was sized by computeScanScratchInts
            // during init.  If this fires, the sizing calculation is buggy.
            assert(dev_scratch != nullptr);
            assert(scratchInts >= static_cast<size_t>(numBlocks));

            int* dev_blockSums      = dev_scratch;
            int* childScratch       = dev_scratch + numBlocks;
            size_t childScratchInts = scratchInts - static_cast<size_t>(numBlocks);

            // Block-level exclusive scan
            kernBlockExclusiveScan<BLOCK_SIZE, FlagType><<<numBlocks, BLOCK_SIZE>>>(
                n, dev_odata, dev_idata, dev_blockSums);
            checkCUDAError("kernBlockExclusiveScan failed");

            if (numBlocks > 1)
            {
                // Recursive scan of block sums --- always int (in-place)
                scanExclusiveSharedMemoryDevice<BLOCK_SIZE, int>(
                    numBlocks,
                    dev_blockSums,    // odata
                    dev_blockSums,    // idata (same buffer = in-place)
                    childScratch,
                    childScratchInts);

                // Add per-block offsets
                kernAddBlockOffsets<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE>>>(
                    n, dev_odata, dev_blockSums);
                checkCUDAError("kernAddBlockOffsets failed");
            }
        }

        // ========================================================================
        // Global-Memory Stream Compaction
        // ========================================================================

        int compactPathSegments(int n, PathSegment *dev_odata, const PathSegment *dev_idata) {
            if (n <= 0) {
                return 0;
            }

            int paddedN = 1 << ilog2ceil(n);

            // Reuse pre-allocated workspace buffer when available; fall back to
            // cudaMalloc when the workspace hasn't been initialized.
            CompactionWorkspace& ws = s_compactionWorkspace;
            int* dev_indices;
            bool using_workspace = (ws.scanBuffer != nullptr &&
                                    ws.scanBufferInts >= static_cast<size_t>(paddedN));

            if (using_workspace) {
                dev_indices = ws.scanBuffer;
            } else {
                cudaMalloc((void**)&dev_indices, paddedN * sizeof(int));
                checkCUDAError("cudaMalloc failed in compactPathSegments");
            }

            // Step 1: Map PathSegments to boolean flags
            LAUNCH_KERNEL_AUTO(kernMapPathSegmentToBoolean, n, n, dev_indices, dev_idata);
            checkCUDAError("kernMapPathSegmentToBoolean failed");

            // Zero-pad the scan tail so the tree is a full power of two
            if (paddedN > n) {
                cudaMemset(dev_indices + n, 0, (paddedN - n) * sizeof(int));
            }

            // Step 2: Exclusive scan --- up-sweep
            for (int d = 0; d < ilog2ceil(paddedN); d++) {
                int numThreads = paddedN / (1 << (d + 1));
                LAUNCH_KERNEL_AUTO(kernUpSweep, numThreads, paddedN, d, dev_indices);
                checkCUDAError("kernUpSweep failed in compactPathSegments");
            }

            // Set root to zero
            cudaMemset(dev_indices + paddedN - 1, 0, sizeof(int));

            // Down-sweep
            for (int d = ilog2ceil(paddedN) - 1; d >= 0; d--) {
                int numThreads = paddedN / (1 << (d + 1));
                LAUNCH_KERNEL_AUTO(kernDownSweep, numThreads, paddedN, d, dev_indices);
                checkCUDAError("kernDownSweep failed in compactPathSegments");
            }

            // Step 3: Scatter
            LAUNCH_KERNEL_AUTO(kernScatterPathSegment, n, n, dev_odata, dev_idata, dev_indices);
            checkCUDAError("kernScatterPathSegment failed");

            // Count active paths
            PathSegment lastPath;
            int lastIndex;
            cudaMemcpy(&lastPath, dev_idata + (n - 1), sizeof(PathSegment), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + (lastPath.remainingBounces > 0 ? 1 : 0);

            if (!using_workspace) {
                cudaFree(dev_indices);
            }

            return count;
        }

        // ========================================================================
        // Shared-Memory Stream Compaction
        // ========================================================================

        int compactPathSegmentsSharedMemory(
            int n, PathSegment *dev_odata, const PathSegment *dev_idata)
        {
            if (n <= 0)
            {
                return 0;
            }

            int paddedN = 1 << ilog2ceil(n);

            CompactionWorkspace& ws = s_compactionWorkspace;
            if (ws.flagBuffer == nullptr ||
                ws.scanBuffer  == nullptr ||
                ws.scanScratch == nullptr ||
                ws.maxElements < paddedN)
            {
                return 0;  // workspace not initialized or too small
            }

            unsigned char* dev_flags   = ws.flagBuffer;
            int*           dev_indices = ws.scanBuffer;

            // Step 1: Map PathSegments to uint8_t boolean flags
            LAUNCH_KERNEL_AUTO(kernMapPathSegmentToBooleanU8, n, n, dev_flags, dev_idata);
            checkCUDAError("kernMapPathSegmentToBooleanU8 failed");

            // Zero-pad the flag tail for the power-of-two scan
            if (paddedN > n) {
                cudaMemset(dev_flags + n, 0, (paddedN - n) * sizeof(unsigned char));
            }

            // Step 2: Exclusive scan from uint8 flags to int indices.
            // Dispatch to the template instance matching the auto-detected block size.
            switch (ws.scanBlockSize) {
                case 512:
                    scanExclusiveSharedMemoryDevice<512, unsigned char>(
                        paddedN, dev_indices, dev_flags,
                        ws.scanScratch, ws.scanScratchInts);
                    break;
                case 256:
                default:
                    scanExclusiveSharedMemoryDevice<256, unsigned char>(
                        paddedN, dev_indices, dev_flags,
                        ws.scanScratch, ws.scanScratchInts);
                    break;
            }

            // Step 3: Scatter PathSegments to compacted output
            LAUNCH_KERNEL_AUTO(kernScatterPathSegment, n, n, dev_odata, dev_idata, dev_indices);
            checkCUDAError("kernScatterPathSegment failed");

            // Count survivors
            PathSegment lastPath;
            int lastIndex;
            cudaMemcpy(&lastPath, dev_idata + (n - 1), sizeof(PathSegment), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + (lastPath.remainingBounces > 0 ? 1 : 0);

            return count;
        }

        // ========================================================================
        // Explicit Template Instantiations
        // ========================================================================
        //
        // Emit device code for every (BLOCK_SIZE, FlagType) combination used.
        // ========================================================================

        // Block scan kernel --- four combinations (2 sizes x 2 types)
        template __global__ void kernBlockExclusiveScan<256, int>(
            int n, int *odata, const int *idata, int *blockSums);
        template __global__ void kernBlockExclusiveScan<256, unsigned char>(
            int n, int *odata, const unsigned char *idata, int *blockSums);
        template __global__ void kernBlockExclusiveScan<512, int>(
            int n, int *odata, const int *idata, int *blockSums);
        template __global__ void kernBlockExclusiveScan<512, unsigned char>(
            int n, int *odata, const unsigned char *idata, int *blockSums);

        // Block-offset kernel --- two sizes
        template __global__ void kernAddBlockOffsets<256>(
            int n, int *data, const int *blockOffsets);
        template __global__ void kernAddBlockOffsets<512>(
            int n, int *data, const int *blockOffsets);

    }  // namespace Efficient
}  // namespace StreamCompaction
