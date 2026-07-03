#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);

        // Export kernels for use by other modules (e.g., radix sort)
        __global__ void kernUpSweep(int n, int d, int *data);
        __global__ void kernDownSweep(int n, int d, int *data);
    }
}
