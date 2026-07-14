#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

/**
 * KernelConfig - Dynamic kernel launch configuration management
 * 
 * This class provides automatic grid/block size configuration based on:
 * 1. Device SM capabilities (max threads per block, SM count, etc.)
 * 2. Problem size (number of elements to process)
 * 3. Kernel-specific requirements
 * 
 * Usage:
 *   KernelConfig config(numElements);
 *   myKernel<<<config.gridSize, config.blockSize>>>(args...);
 * 
 * When to use:
 * - General-purpose kernels without specific resource constraints
 * - Simple memory-bound operations
 * - When you want consistent block sizes across similar operations
 */
class KernelConfig {
public:
    dim3 gridSize;
    dim3 blockSize;
    int effectiveThreads;  // Actual number of threads that will be launched
    
    /**
     * Constructor: Automatically configures grid and block dimensions
     * 
     * @param numElements    Number of elements to process
     * @param deviceId       CUDA device ID (default: 0)
     * @param preferredBlockSize  Preferred block size hint (0 = auto-detect)
     */
    KernelConfig(int numElements, int deviceId = 0, int preferredBlockSize = 0)
        : effectiveThreads(numElements) {
        
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceId);
        
        // Determine optimal block size
        if (preferredBlockSize > 0) {
            // Use user-specified block size (clamped to device limits)
            blockSize.x = std::min(preferredBlockSize, deviceProp.maxThreadsPerBlock);
        } else {
            // Auto-detect optimal block size based on device architecture
            blockSize.x = computeOptimalBlockSize(deviceProp);
        }
        //one dimension
        blockSize.y = 1;
        blockSize.z = 1;
        
        // Calculate grid size to cover all elements
        gridSize.x = (numElements + blockSize.x - 1) / blockSize.x;
        gridSize.y = 1;
        gridSize.z = 1;
        
        // Clamp grid size to device limits
        gridSize.x = std::min(gridSize.x, (unsigned int)deviceProp.maxGridSize[0]);
    }
    
    /**
     * Get device properties for current device
     */
    static cudaDeviceProp getDeviceProperties(int deviceId = 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);
        return prop;
    }
    
    /**
     * Print configuration details (useful for debugging)
     */
    void print() const {
        printf("KernelConfig: grid(%d, %d, %d) block(%d, %d, %d) -> %d threads\n",
               gridSize.x, gridSize.y, gridSize.z,
               blockSize.x, blockSize.y, blockSize.z,
               effectiveThreads);
    }
    
private:
    /**
     * Compute optimal block size based on device architecture
     * 
     * Strategy:
     * - For modern GPUs (compute capability >= 3.0): prefer 256 threads
     * - For older GPUs: prefer 128 threads
     * - Consider warp size (always 32) and max threads per block
     */
    static int computeOptimalBlockSize(const cudaDeviceProp& prop) {
        // Extract compute capability
        int computeCapability = prop.major * 10 + prop.minor;
        
        int optimalSize;
        if (computeCapability >= 30) {  // Kepler and newer
            // Modern GPUs benefit from larger blocks (more parallelism)
            optimalSize = 256;
        } else {
            // Older GPUs: use smaller blocks
            optimalSize = 128;
        }
        
        // Ensure block size is multiple of warp size (32)
        optimalSize = (optimalSize / prop.warpSize) * prop.warpSize;
        
        // Clamp to device limits
        optimalSize = std::min(optimalSize, prop.maxThreadsPerBlock);
        
        // Ensure at least one warp
        optimalSize = std::max(optimalSize, prop.warpSize);
        
        return optimalSize;
    }
};

/**
 * OccupancyConfig - Advanced configuration using CUDA Occupancy Calculator
 * 
 * Uses cudaOccupancyMaxPotentialBlockSize to find optimal configuration
 * that maximizes GPU occupancy for a specific kernel.
 * 
 * Usage:
 *   OccupancyConfig config = OccupancyConfig::forKernel(myKernel, numElements);
 *   myKernel<<<config.gridSize, config.blockSize>>>(args...);
 * 
 * When to use:
 * - Compute-intensive kernels where occupancy matters most
 * - Kernels with significant shared memory usage
 * - Kernels with complex register usage patterns
 * - Performance-critical hotspots identified through profiling
 * 
 * When NOT to use:
 * - Simple memory-bound operations (bandwidth is the bottleneck, not occupancy)
 * - Kernels that need consistent block sizes for algorithmic reasons
 * - Early development phase (use simpler KernelConfig first, optimize later)
 */
class OccupancyConfig {
public:
    int gridSize;
    int blockSize;
    int effectiveThreads;
    
    /**
     * Configure for maximum occupancy of a specific kernel
     * 
     * @param func           Kernel function pointer
     * @param numElements    Number of elements to process
     * @param dynamicSMemSize Dynamic shared memory per block (default: 0)
     * @param blockSizeLimit  Maximum block size (0 = no limit)
     */
    template<typename KernelFunc>
    static OccupancyConfig forKernel(
        KernelFunc func,
        int numElements,
        size_t dynamicSMemSize = 0,
        int blockSizeLimit = 0
    ) {
        OccupancyConfig config;
        
        int minGridSize;
        cudaOccupancyMaxPotentialBlockSize(
            &minGridSize,
            &config.blockSize,
            func,
            dynamicSMemSize,
            blockSizeLimit
        );
        
        // Calculate actual grid size for the problem
        config.gridSize = (numElements + config.blockSize - 1) / config.blockSize;
        config.effectiveThreads = config.gridSize * config.blockSize;
        
        return config;
    }
    
    void print() const {
        printf("OccupancyConfig: grid(%d) block(%d) -> %d threads\n",
               gridSize, blockSize, effectiveThreads);
    }
};

/**
 * DeviceInfo - Singleton for caching device properties
 * Avoids repeated cudaGetDeviceProperties calls
 */
class DeviceInfo {
public:
    static DeviceInfo& getInstance(int deviceId = 0) {
        static DeviceInfo instance(deviceId);
        return instance;
    }
    
    const cudaDeviceProp& getProperties() const { return prop; }
    
    int getMaxThreadsPerBlock() const { return prop.maxThreadsPerBlock; }
    int getMultiProcessorCount() const { return prop.multiProcessorCount; }
    int getWarpSize() const { return prop.warpSize; }
    int getComputeCapability() const { return prop.major * 10 + prop.minor; }

    /**
     * Returns the recommended scan block size (threads per block) for the
     * hierarchical shared-memory scan kernels.
     *
     * 512 is preferred on devices that support it (CC 3.0+ with >=512
     * threads/block) because it halves the number of recursion levels
     * compared to 256.  Shared memory pressure is minimal (4 KB vs 2 KB
     * per block) and does not hurt occupancy on any GPU since Kepler.
     */
    int getOptimalScanBlockSize() const {
        return (prop.maxThreadsPerBlock >= 512) ? 512 : 256;
    }

    void printDeviceInfo() const {
        printf("=== CUDA Device Info ===\n");
        printf("Device: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Warp Size: %d\n", prop.warpSize);
        printf("Max Grid Size: (%d, %d, %d)\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("========================\n");
    }
    
    // Delete copy/move constructors
    DeviceInfo(const DeviceInfo&) = delete;
    DeviceInfo& operator=(const DeviceInfo&) = delete;
    
private:
    cudaDeviceProp prop;
    
    explicit DeviceInfo(int deviceId) {
        cudaGetDeviceProperties(&prop, deviceId);
    }
};

/**
 * Helper macro for simple kernel launches with auto-configuration
 * 
 * Usage:
 *   LAUNCH_KERNEL_AUTO(myKernel, numElements, arg1, arg2, arg3);
 * 
 * Expands to:
 *   KernelConfig config(numElements);
 *   myKernel<<<config.gridSize, config.blockSize>>>(arg1, arg2, arg3);
 */
#define LAUNCH_KERNEL_AUTO(kernel, numElements, ...) \
    do { \
        KernelConfig _cfg(numElements); \
        kernel<<<_cfg.gridSize, _cfg.blockSize>>>(__VA_ARGS__); \
    } while(0)

/**
 * Helper macro for kernel launches with occupancy optimization
 * 
 * Usage:
 *   LAUNCH_KERNEL_OCCUPANCY(myKernel, numElements, arg1, arg2, arg3);
 */
#define LAUNCH_KERNEL_OCCUPANCY(kernel, numElements, ...) \
    do { \
        auto _cfg = OccupancyConfig::forKernel(kernel, numElements); \
        kernel<<<_cfg.gridSize, _cfg.blockSize>>>(__VA_ARGS__); \
    } while(0)
