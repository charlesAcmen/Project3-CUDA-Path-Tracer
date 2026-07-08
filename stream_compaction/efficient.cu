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

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Round up to next power of 2
            int paddedN = 1 << ilog2ceil(n);
            
            // Allocate device memory
            int *dev_data;
            cudaMalloc((void**)&dev_data, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed");
            
            // Copy input data and pad with zeros
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (paddedN > n) {
                cudaMemset(dev_data + n, 0, (paddedN - n) * sizeof(int));
            }
            checkCUDAError("cudaMemcpy to device failed");

            timer().startGpuTimer();

            // Up-sweep phase
            for (int d = 0; d < ilog2ceil(paddedN); d++) {
                // Thread Compaction: compute number of active threads
                int numThreads = paddedN / (1 << (d + 1));
                
                // Use dynamic kernel configuration
                KernelConfig config(numThreads);
                
                kernUpSweep<<<config.gridSize, config.blockSize>>>(paddedN, d, dev_data);
                checkCUDAError("kernUpSweep failed");
            }
            
            // Set root to zero
            cudaMemset(dev_data + paddedN - 1, 0, sizeof(int));
            
            // Down-sweep phase
            for (int d = ilog2ceil(paddedN) - 1; d >= 0; d--) {
                // Thread Compaction: compute number of active threads
                int numThreads = paddedN / (1 << (d + 1));
                
                // Use dynamic kernel configuration
                KernelConfig config(numThreads);
                
                kernDownSweep<<<config.gridSize, config.blockSize>>>(paddedN, d, dev_data);
                checkCUDAError("kernDownSweep failed");
            }

            timer().endGpuTimer();

            // Copy result back to host
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to host failed");

            // Free device memory
            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            // Round up to next power of 2 for scan
            int paddedN = 1 << ilog2ceil(n);
            
            // Allocate device memory
            int *dev_idata, *dev_bools, *dev_indices, *dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, paddedN * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc failed");
            
            // Copy input data to device
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed");

            timer().startGpuTimer();

            // Step 1: Map to boolean
            KernelConfig configMap(n);
            StreamCompaction::Common::kernMapToBoolean<<<configMap.gridSize, configMap.blockSize>>>(n, dev_bools, dev_idata);
            checkCUDAError("kernMapToBoolean failed");
            
            // Copy bools to indices array and pad with zeros
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            if (paddedN > n) {
                cudaMemset(dev_indices + n, 0, (paddedN - n) * sizeof(int));
            }
            
            // Step 2: Scan (exclusive prefix sum) - inline implementation
            // Up-sweep phase
            for (int d = 0; d < ilog2ceil(paddedN); d++) {
                int numThreads = paddedN / (1 << (d + 1));
                KernelConfig config(numThreads);
                kernUpSweep<<<config.gridSize, config.blockSize>>>(paddedN, d, dev_indices);
                checkCUDAError("kernUpSweep failed");
            }
            
            // Set root to zero
            cudaMemset(dev_indices + paddedN - 1, 0, sizeof(int));
            
            // Down-sweep phase
            for (int d = ilog2ceil(paddedN) - 1; d >= 0; d--) {
                int numThreads = paddedN / (1 << (d + 1));
                KernelConfig config(numThreads);
                kernDownSweep<<<config.gridSize, config.blockSize>>>(paddedN, d, dev_indices);
                checkCUDAError("kernDownSweep failed");
            }
            
            // Step 3: Scatter
            KernelConfig configScatter(n);
            StreamCompaction::Common::kernScatter<<<configScatter.gridSize, configScatter.blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            checkCUDAError("kernScatter failed");

            timer().endGpuTimer();

            // Get the count of non-zero elements
            int lastBool, lastIndex;
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + lastBool;
            
            // Copy result back to host
            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to host failed");

            // Free device memory
            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);

            return count;
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
            KernelConfig configMap(n);
            kernMapPathSegmentToBoolean<<<configMap.gridSize, configMap.blockSize>>>(n, dev_bools, dev_idata);
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
                KernelConfig config(numThreads);
                kernUpSweep<<<config.gridSize, config.blockSize>>>(paddedN, d, dev_indices);
                checkCUDAError("kernUpSweep failed in compactPathSegments");
            }
            
            // Set root to zero
            cudaMemset(dev_indices + paddedN - 1, 0, sizeof(int));
            
            // Down-sweep phase
            for (int d = ilog2ceil(paddedN) - 1; d >= 0; d--) {
                int numThreads = paddedN / (1 << (d + 1));
                KernelConfig config(numThreads);
                kernDownSweep<<<config.gridSize, config.blockSize>>>(paddedN, d, dev_indices);
                checkCUDAError("kernDownSweep failed in compactPathSegments");
            }
            
            // Step 3: Scatter PathSegments to output array
            KernelConfig configScatter(n);
            kernScatterPathSegment<<<configScatter.gridSize, configScatter.blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
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
    }
}
