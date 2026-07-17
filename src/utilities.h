#pragma once

#include "constants.h"
#include "glm/glm.hpp"

#include <algorithm>
#include <istream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;

    // Per-frame timing summary (populated by profiler, displayed by ImGui).
    // Sized to match ProfilerOp::COUNT in profiler.h.
    float perKernelMs[7] = {};
    int   lastBounceCount = 0;
};

namespace utilityCore
{
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}

// CUDA error checking utilities
// Available only when compiling with NVCC
#ifdef __CUDACC__
    #define ERRORCHECK 1
    #define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
    #define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
    void checkCUDAErrorFn(const char* msg, const char* file = nullptr, int line = -1);
#endif
