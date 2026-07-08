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

        // Cache device info at startup
        static DeviceInfo& getDeviceInfo() {
            static DeviceInfo& info = DeviceInfo::getInstance();
            return info;
        }

        // ========================================================================
        // Forward Declarations of Internal Kernels
        // ========================================================================
        
        // Global memory scan kernels
        __global__ void kernUpSweep(int n, int d, int *data);
        __global__ void kernDownSweep(int n, int d, int *data);
        
        // PathSegment-specific kernels
        __global__ void kernMapPathSegmentToBoolean(int n, int *bools, const PathSegment *paths);
        __global__ void kernScatterPathSegment(int n, PathSegment *odata,
                const PathSegment *idata, const int *bools, const int *indices);
        
        // Shared memory scan kernels
        __global__ void kernBlockExclusiveScan(int n, int *odata, const int *idata, int *blockSums);
        __global__ void kernAddBlockOffsets(int n, int *data, const int *blockOffsets);
        
        // Helper function
        static void scanExclusiveSharedMemoryDevice(int n, int *dev_odata, const int *dev_idata);

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
        __global__ void kernMapPathSegmentToBoolean(int n, int *bools, const PathSegment *paths) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            
            if (index >= n) {
                return;
            }
            
            // Keep paths that still have bounces remaining
            bools[index] = (paths[index].remainingBounces > 0) ? 1 : 0;
        }

        /**
         * Scatters PathSegment array based on boolean and indices arrays.
         * Only paths where bools[idx] == 1 are copied to output.
         */
        __global__ void kernScatterPathSegment(int n, PathSegment *odata,
                const PathSegment *idata, const int *bools, const int *indices) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            
            if (index >= n) {
                return;
            }
            
            if (bools[index] == 1) {
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
            // Round up to next power of 2 for scan
            int paddedN = 1 << ilog2ceil(n);
            
            // Allocate device memory for intermediate arrays
            int *dev_bools, *dev_indices;
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc failed in compactPathSegments");

            // Step 1: Map PathSegments to boolean array
            LAUNCH_KERNEL_AUTO(kernMapPathSegmentToBoolean, n, n, dev_bools, dev_idata);
            checkCUDAError("kernMapPathSegmentToBoolean failed");
            
            // Copy bools to indices array and pad with zeros
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
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
            LAUNCH_KERNEL_AUTO(kernScatterPathSegment, n, n, dev_odata, dev_idata, dev_bools, dev_indices);
            checkCUDAError("kernScatterPathSegment failed");

            // Calculate the count of active paths
            int lastBool, lastIndex;
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + lastBool;

            // Free intermediate device memory
            cudaFree(dev_bools);
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
            int *dev_bools, *dev_indices;
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, n * sizeof(int));
            checkCUDAError("cudaMalloc failed in compactPathSegmentsSharedMemory");

            LAUNCH_KERNEL_AUTO(kernMapPathSegmentToBoolean, n, n, dev_bools, dev_idata);
            checkCUDAError("kernMapPathSegmentToBoolean failed");

            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            scanExclusiveSharedMemoryDevice(n, dev_indices, dev_indices);

            LAUNCH_KERNEL_AUTO(kernScatterPathSegment, n, n, dev_odata, dev_idata, dev_bools, dev_indices);
            checkCUDAError("kernScatterPathSegment failed");

            int lastBool, lastIndex;
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + lastBool;

            cudaFree(dev_bools);
            cudaFree(dev_indices);

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

            for (int d = SCAN_BLOCK_SIZE >> 1; d > 0; d >>= 1)
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

            for (int d = 1; d < SCAN_BLOCK_SIZE; d <<= 1)
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
         */
        static void scanExclusiveSharedMemoryDevice(
            int n, int *dev_odata, const int *dev_idata)
        {
            if (n <= 0)
            {
                return;
            }

            int numBlocks = (n + SCAN_BLOCK_ELEMENTS - 1) / SCAN_BLOCK_ELEMENTS;

            int *dev_blockSums;
            cudaMalloc((void**)&dev_blockSums, numBlocks * sizeof(int));
            checkCUDAError("cudaMalloc dev_blockSums failed");

            kernBlockExclusiveScan<<<numBlocks, SCAN_BLOCK_SIZE>>>(
                n, dev_odata, dev_idata, dev_blockSums);
            checkCUDAError("kernBlockExclusiveScan failed");

            if (numBlocks > 1)
            {
                int *dev_blockOffsets;
                cudaMalloc((void**)&dev_blockOffsets, numBlocks * sizeof(int));
                checkCUDAError("cudaMalloc dev_blockOffsets failed");

                scanExclusiveSharedMemoryDevice(numBlocks, dev_blockOffsets, dev_blockSums);

                kernAddBlockOffsets<<<numBlocks, SCAN_BLOCK_SIZE>>>(
                    n, dev_odata, dev_blockOffsets);
                checkCUDAError("kernAddBlockOffsets failed");

                cudaFree(dev_blockOffsets);
            }

            cudaFree(dev_blockSums);
        }
    }
}
