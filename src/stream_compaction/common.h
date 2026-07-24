#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

// Use unified CUDA error checking from utilities module
#include "utilities.h"

inline int ilog2(int x) {
    int lg = 0;
    //right shift x by 1 bit until x is 0
    while (x >>= 1) {
        ++lg;
    }
    //returns the number of bits in x
    return lg;
}
inline int ilog2ceil(int x) {
    return x == 1 ? 0 : ilog2(x - 1) + 1;
}

namespace StreamCompaction {
    namespace Common {
        //map the input array to an array of 0s and 1s,FIRST STEP OF COMPACTWITHSCAN
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata);

        //scatter the input array to an array of 0s and 1s,SECOND STEP OF COMPACTWITHSCAN
        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices);

        // NOTE: Timing functionality is provided by the Profiler singleton
        // (src/profiler/profiler.h).  The legacy PerformanceTimer class that
        // lived here has been removed — Profiler covers both cudaEvent GPU
        // timing and std::chrono CPU timing with CSV output and GUI integration.
    }
}
