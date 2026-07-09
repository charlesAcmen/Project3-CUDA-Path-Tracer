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

        struct SharedMemoryCompactionWorkspace
        {
            int* scanData = nullptr;
            int* scanScratch = nullptr;
            size_t scanDataInts = 0;
            size_t scanScratchInts = 0;
            int maxElements = 0;
        };

        static SharedMemoryCompactionWorkspace s_sharedMemoryCompactionWorkspace;

        static size_t computeSharedScanScratchInts(int n)
        {
            size_t totalInts = 0;
            int current = (n + 511) / 512;

            while (current > 0)
            {
                totalInts += static_cast<size_t>(current);

                if (current == 1)
                {
                    break;
                }

                current = (current + 511) / 512;
            }

            return totalInts;
        }

        void freeSharedMemoryCompactionWorkspace();

        void initSharedMemoryCompactionWorkspace(int maxElements)
        {
            freeSharedMemoryCompactionWorkspace();

            if (maxElements <= 0)
            {
                return;
            }

            s_sharedMemoryCompactionWorkspace.maxElements = maxElements;
            s_sharedMemoryCompactionWorkspace.scanDataInts = static_cast<size_t>(maxElements);
            s_sharedMemoryCompactionWorkspace.scanScratchInts = computeSharedScanScratchInts(maxElements);

            cudaMalloc(
                reinterpret_cast<void**>(&s_sharedMemoryCompactionWorkspace.scanData),
                s_sharedMemoryCompactionWorkspace.scanDataInts * sizeof(int));
            checkCUDAError("cudaMalloc scanData failed");

            if (s_sharedMemoryCompactionWorkspace.scanScratchInts > 0)
            {
                cudaMalloc(
                    reinterpret_cast<void**>(&s_sharedMemoryCompactionWorkspace.scanScratch),
                    s_sharedMemoryCompactionWorkspace.scanScratchInts * sizeof(int));
                checkCUDAError("cudaMalloc scanScratch failed");
            }
        }

        void freeSharedMemoryCompactionWorkspace()
        {
            cudaFree(s_sharedMemoryCompactionWorkspace.scanData);
            s_sharedMemoryCompactionWorkspace.scanData = nullptr;

            cudaFree(s_sharedMemoryCompactionWorkspace.scanScratch);
            s_sharedMemoryCompactionWorkspace.scanScratch = nullptr;

            s_sharedMemoryCompactionWorkspace.scanDataInts = 0;
            s_sharedMemoryCompactionWorkspace.scanScratchInts = 0;
            s_sharedMemoryCompactionWorkspace.maxElements = 0;
        }

        // ========================================================================
        // Internal Implementation Details
        // ========================================================================
        // Note: Only functions/kernels that require forward declaration (e.g.,
        // recursive functions) are declared here. Other functions are defined
        // in order of dependency to avoid unnecessary forward declarations.
        // ========================================================================

        // Cache device info at startup (internal utility)
        static DeviceInfo& getDeviceInfo() {
            static DeviceInfo& info = DeviceInfo::getInstance();
            return info;
        }

        // ========================================================================
        // Forward Declaration (Required for Recursion)
        // ========================================================================
        
        // This recursive helper function needs forward declaration
        static void scanExclusiveSharedMemoryDevice(
            int n, int* dev_data, int* dev_scratch, size_t scratchInts);

        // ========================================================================
        // Global Memory Scan Kernels (Original Implementation)
        // ========================================================================

        /**
         * Up-sweep (reduce) phase of work-efficient scan
         * Builds a balanced binary tree on the input data
         * 
         * @param n      Number of elements (must be power of 2)
         * @param d      Current depth (iteration) of the tree
         * @param data   Array to operate on (in-place)
         */
        __global__ void kernUpSweep(int n, int d, int *data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            
            int stride = 1 << (d + 1); // 2^(d+1)
            
            if (index * stride >= n) {
                return;
            }
            
            int i = (index + 1) * stride - 1;
            int offset = 1 << d; // 2^d
            
            data[i] += data[i - offset];
        }

        /**
         * Down-sweep phase of work-efficient scan
         * Traverses down the tree to build the scan from the partial sums
         * 
         * @param n      Number of elements (must be power of 2)
         * @param d      Current depth (iteration) of the tree
         * @param data   Array to operate on (in-place)
         */
        __global__ void kernDownSweep(int n, int d, int *data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            
            int stride = 1 << (d + 1); // 2^(d+1)
            
            if (index * stride >= n) {
                return;
            }
            
            int i = (index + 1) * stride - 1;
            int offset = 1 << d; // 2^d
            
            int temp = data[i - offset];
            data[i - offset] = data[i];
            data[i] += temp;
        }

        // ========================================================================
        // PathSegment-specific stream compaction kernels
        // ========================================================================

        /**
         * Maps PathSegment array to boolean array.
         * Paths with remainingBounces > 0 map to 1, others map to 0.
         */
        __global__ void kernMapPathSegmentToBoolean(int n, int *flags, const PathSegment *paths) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            
            if (index >= n) {
                return;
            }
            
            // Keep paths that still have bounces remaining
            flags[index] = (paths[index].remainingBounces > 0) ? 1 : 0;
        }

        /**
         * Scatters PathSegment array based on boolean and indices arrays.
         * Only paths where bools[idx] == 1 are copied to output.
         */
        __global__ void kernScatterPathSegment(int n, PathSegment *odata,
                const PathSegment *idata, const int *indices) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            
            if (index >= n) {
                return;
            }
            
            if (idata[index].remainingBounces > 0) {
                odata[indices[index]] = idata[index];
            }
        }

        /**
         * Performs stream compaction on PathSegment arrays (device memory version).
         * Removes paths with remainingBounces <= 0.
         * 
         * This function operates entirely on GPU memory - no host-device copies.
         * Both input and output must be device pointers.
         *
         * @param n           The number of PathSegments in dev_idata.
         * @param dev_odata   Device pointer to output array (must be pre-allocated).
         * @param dev_idata   Device pointer to input array.
         * @returns           The number of active paths remaining after compaction.
         */
        int compactPathSegments(int n, PathSegment *dev_odata, const PathSegment *dev_idata) {
            if (n <= 0) {
                return 0;
            }

            // Round up to next power of 2 for scan.
            // The map + scan happen in a single reusable buffer so this path
            // only needs one temporary allocation.
            int paddedN = 1 << ilog2ceil(n);

            int* dev_indices;
            cudaMalloc((void**)&dev_indices, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc failed in compactPathSegments");

            // Step 1: Map PathSegments to a 0/1 flag buffer in-place.
            LAUNCH_KERNEL_AUTO(kernMapPathSegmentToBoolean, n, n, dev_indices, dev_idata);
            checkCUDAError("kernMapPathSegmentToBoolean failed");

            // Zero-pad the scan tail so the tree is a full power of two.
            if (paddedN > n) {
                cudaMemset(dev_indices + n, 0, (paddedN - n) * sizeof(int));
            }
            
            // Step 2: Exclusive scan (prefix sum) on indices
            // Up-sweep phase
            for (int d = 0; d < ilog2ceil(paddedN); d++) {
                int numThreads = paddedN / (1 << (d + 1));
                LAUNCH_KERNEL_AUTO(kernUpSweep, numThreads, paddedN, d, dev_indices);
                checkCUDAError("kernUpSweep failed in compactPathSegments");
            }
            
            // Set root to zero
            cudaMemset(dev_indices + paddedN - 1, 0, sizeof(int));
            
            // Down-sweep phase
            for (int d = ilog2ceil(paddedN) - 1; d >= 0; d--) {
                int numThreads = paddedN / (1 << (d + 1));
                LAUNCH_KERNEL_AUTO(kernDownSweep, numThreads, paddedN, d, dev_indices);
                checkCUDAError("kernDownSweep failed in compactPathSegments");
            }
            
            // Step 3: Scatter PathSegments to output array
            LAUNCH_KERNEL_AUTO(kernScatterPathSegment, n, n, dev_odata, dev_idata, dev_indices);
            checkCUDAError("kernScatterPathSegment failed");

            // Calculate the count of active paths
            PathSegment lastPath;
            int lastIndex;
            cudaMemcpy(&lastPath, dev_idata + (n - 1), sizeof(PathSegment), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + (lastPath.remainingBounces > 0 ? 1 : 0);

            // Free intermediate device memory
            cudaFree(dev_indices);

            return count;
        }

        /**
         * Device-side PathSegment stream compaction using shared-memory scan.
         * Parallel API to compactPathSegments() (global memory).
         */
        int compactPathSegmentsSharedMemory(
            int n, PathSegment *dev_odata, const PathSegment *dev_idata)
        {
            if (n <= 0)
            {
                return 0;
            }

            // Round up to next power of 2 so the shared-mem scan operates on
            // a padded array whose size is a multiple of SCAN_BLOCK_ELEMENTS.
            // Without padding, the last block processes a partial batch whose
            // out-of-bounds reads can corrupt the scan result.
            int paddedN = 1 << ilog2ceil(n);

            if (s_sharedMemoryCompactionWorkspace.scanScratch == nullptr ||
                s_sharedMemoryCompactionWorkspace.maxElements < paddedN)
            {
                fprintf(stderr,
                    "ERROR: shared-memory compaction workspace is not initialized for %d elements.\n",
                    paddedN);
                return 0;
            }

            int* dev_indices = s_sharedMemoryCompactionWorkspace.scanData;

            LAUNCH_KERNEL_AUTO(kernMapPathSegmentToBoolean, n, n, dev_indices, dev_idata);
            checkCUDAError("kernMapPathSegmentToBoolean failed");

            // Zero-pad the tail so the scan runs on a power-of-two length.
            if (paddedN > n) {
                cudaMemset(dev_indices + n, 0, (paddedN - n) * sizeof(int));
            }

            scanExclusiveSharedMemoryDevice(
                paddedN,
                dev_indices,
                s_sharedMemoryCompactionWorkspace.scanScratch,
                s_sharedMemoryCompactionWorkspace.scanScratchInts);

            LAUNCH_KERNEL_AUTO(kernScatterPathSegment, n, n, dev_odata, dev_idata, dev_indices);
            checkCUDAError("kernScatterPathSegment failed");

            // Count survivors: exclusive-scan result at last active element + its bool
            PathSegment lastPath;
            int lastIndex;
            cudaMemcpy(&lastPath, dev_idata + (n - 1), sizeof(PathSegment), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + (lastPath.remainingBounces > 0 ? 1 : 0);

            return count;
        }


        // ========================================================================
        // Shared-memory stream compaction (GPU Gems 3, Chapter 39)
        // Separate from the global-memory implementation above. Each CUDA block
        // performs a work-efficient exclusive scan in shared memory; block totals
        // are scanned recursively and propagated as cross-block offsets.
        // ========================================================================

        /**
         * Configuration constants for shared memory scan.
         * SCAN_BLOCK_SIZE: Number of threads per block (256 chosen for good occupancy)
         * SCAN_BLOCK_ELEMENTS: Each thread processes 2 elements to reduce kernel launches
         */
        static const int SCAN_BLOCK_SIZE = 256;
        static const int SCAN_BLOCK_ELEMENTS = SCAN_BLOCK_SIZE * 2;

        /**
         * Per-block work-efficient exclusive scan using shared memory.
         * Each thread loads two elements (bank-conflict-free layout) so one CUDA
         * block processes SCAN_BLOCK_ELEMENTS items.
         * 
         * Note: This kernel uses a FIXED block size (SCAN_BLOCK_SIZE) because:
         * 1. Shared memory array size must be known at compile time
         * 2. The algorithm logic depends on the specific block size
         * 3. Cannot use dynamic occupancy optimization here
         */
        __global__ void kernBlockExclusiveScan(
            int n, int *odata, const int *idata, int *blockSums)
        {
            __shared__ int temp[2 * SCAN_BLOCK_SIZE];

            int thid = threadIdx.x;
            int bid = blockIdx.x;
            int offset = 1;

            int index0 = bid * SCAN_BLOCK_ELEMENTS + 2 * thid;
            int index1 = index0 + 1;

            temp[2 * thid]     = (index0 < n) ? idata[index0] : 0;
            temp[2 * thid + 1] = (index1 < n) ? idata[index1] : 0;

            // Up-sweep: log2(2*SCAN_BLOCK_SIZE) levels.
            // The tree has 2*SCAN_BLOCK_SIZE leaves (= SCAN_BLOCK_ELEMENTS),
            // so the first level pairs elements (0,1), (2,3), ..., requiring
            // SCAN_BLOCK_SIZE threads (all threads in the block).
            for (int d = SCAN_BLOCK_SIZE; d > 0; d >>= 1)
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
                blockSums[bid] = temp[2 * SCAN_BLOCK_SIZE - 1];
                temp[2 * SCAN_BLOCK_SIZE - 1] = 0;
            }

            // Down-sweep: same number of levels as up-sweep.
            for (int d = 1; d < SCAN_BLOCK_ELEMENTS; d <<= 1)
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

        /**
         * Adds the exclusive-scan offset of each block to that block's elements.
         */
        __global__ void kernAddBlockOffsets(int n, int *data, const int *blockOffsets)
        {
            int bid = blockIdx.x;
            int thid = threadIdx.x;

            int index0 = bid * SCAN_BLOCK_ELEMENTS + 2 * thid;
            int index1 = index0 + 1;
            int offset = blockOffsets[bid];

            if (index0 < n) data[index0] += offset;
            if (index1 < n) data[index1] += offset;
        }

        /**
         * Device-side exclusive prefix sum using shared memory across multiple
         * blocks (internal helper for the shared-memory API below).
         * 
         * Algorithm:
         * 1. Each block performs local exclusive scan on SCAN_BLOCK_ELEMENTS items
         * 2. Collect the sum from each block into blockSums array
         * 3. Recursively scan blockSums to get per-block offsets
         * 4. Add the offsets back to each block's results
         * 
         * Complexity: O(n) work, O(log n) depth
         * Recursion depth: ~log₅₁₂(n), safe for any practical input size
         */
        static void scanExclusiveSharedMemoryDevice(
            int n, int *dev_data, int *dev_scratch, size_t scratchInts)
        {
            if (n <= 0)
            {
                return;
            }

            int numBlocks = (n + SCAN_BLOCK_ELEMENTS - 1) / SCAN_BLOCK_ELEMENTS;

            if (dev_scratch == nullptr || scratchInts < static_cast<size_t>(numBlocks))
            {
                fprintf(stderr, "ERROR: insufficient shared-memory compaction scratch space.\n");
                return;
            }

            int* dev_blockSums = dev_scratch;
            int* childScratch = dev_scratch + numBlocks;
            size_t childScratchInts = scratchInts - static_cast<size_t>(numBlocks);

            // Step 1: Per-block exclusive scan, save each block's total sum
            kernBlockExclusiveScan<<<numBlocks, SCAN_BLOCK_SIZE>>>(
                n, dev_data, dev_data, dev_blockSums);
            checkCUDAError("kernBlockExclusiveScan failed");

            // Step 2: If multiple blocks, compute cross-block offsets
            if (numBlocks > 1)
            {
                // Recursively scan block sums to get per-block offsets.
                scanExclusiveSharedMemoryDevice(numBlocks, dev_blockSums, childScratch, childScratchInts);

                // Step 3: Add block offsets to each block's local scan results
                kernAddBlockOffsets<<<numBlocks, SCAN_BLOCK_SIZE>>>(
                    n, dev_data, dev_blockSums);
                checkCUDAError("kernAddBlockOffsets failed");
            }
            // else: single block, no cross-block offsets needed
        }
    }
}
