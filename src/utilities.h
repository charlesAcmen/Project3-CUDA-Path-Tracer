#pragma once

#include "constants.h"
#include "glm/glm.hpp"

#include <string>

namespace utilityCore
{
    extern std::string convertIntToString(int number);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
}

// CUDA error checking utilities
// Available only when compiling with NVCC
//
// checkCUDAError is a development-time safety net: checkCUDAErrorFn performs
// a full cudaDeviceSynchronize() before querying the error, which serialises
// the entire GPU pipeline on every call.  It is therefore compiled in ONLY
// for Debug / RelWithDebInfo builds — CMakeLists.txt defines the
// CIS565_ENABLE_CUDA_ERRORCHECK macro for exactly those configurations.  In
// Release the macro expands to a no-op so kernels run fully asynchronously.
#ifdef __CUDACC__
    #ifdef CIS565_ENABLE_CUDA_ERRORCHECK
        #define ERRORCHECK 1
        #define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
        #define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
        void checkCUDAErrorFn(const char* msg, const char* file = nullptr, int line = -1);
    #else
        #define ERRORCHECK 0
        #define checkCUDAError(msg) ((void)0)
    #endif
#endif
