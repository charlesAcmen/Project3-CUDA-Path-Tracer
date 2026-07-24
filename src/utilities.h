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
#ifdef __CUDACC__
    #define ERRORCHECK 1
    #define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
    #define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
    void checkCUDAErrorFn(const char* msg, const char* file = nullptr, int line = -1);
#endif
