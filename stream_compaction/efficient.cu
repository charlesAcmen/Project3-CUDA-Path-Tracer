#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        const int blockSize = 128;

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
                //Thread Compaction
                int numThreads = paddedN / (1 << (d + 1));
                dim3 fullBlocksPerGrid((numThreads + blockSize - 1) / blockSize);
                
                kernUpSweep<<<fullBlocksPerGrid, blockSize>>>(paddedN, d, dev_data);
                checkCUDAError("kernUpSweep failed");
            }
            
            // Set root to zero
            cudaMemset(dev_data + paddedN - 1, 0, sizeof(int));
            
            // Down-sweep phase
            for (int d = ilog2ceil(paddedN) - 1; d >= 0; d--) {
                //Thread Compaction
                int numThreads = paddedN / (1 << (d + 1));
                dim3 fullBlocksPerGrid((numThreads + blockSize - 1) / blockSize);
                
                kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(paddedN, d, dev_data);
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
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            StreamCompaction::Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(n, dev_bools, dev_idata);
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
                dim3 blocks((numThreads + blockSize - 1) / blockSize);
                kernUpSweep<<<blocks, blockSize>>>(paddedN, d, dev_indices);
                checkCUDAError("kernUpSweep failed");
            }
            
            // Set root to zero
            cudaMemset(dev_indices + paddedN - 1, 0, sizeof(int));
            
            // Down-sweep phase
            for (int d = ilog2ceil(paddedN) - 1; d >= 0; d--) {
                int numThreads = paddedN / (1 << (d + 1));
                dim3 blocks((numThreads + blockSize - 1) / blockSize);
                kernDownSweep<<<blocks, blockSize>>>(paddedN, d, dev_indices);
                checkCUDAError("kernDownSweep failed");
            }
            
            // Step 3: Scatter
            StreamCompaction::Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
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
    }
}
