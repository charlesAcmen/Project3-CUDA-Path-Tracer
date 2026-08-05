#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

// Use unified CUDA error checking from utilities module
#include "utils/utilities.h"

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
