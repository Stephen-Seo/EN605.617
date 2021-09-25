#include "mathexpressions.h"

__global__
void non_branching_mathexpressions(int *x, int *y, int *out) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int sidx = thread_idx % 4;
    // i=0  (i & 1) >> (3 - i) = 0
    // i=1  (i & 1) >> (3 - i) = 0
    // i=2  (i & 1) >> (3 - i) = 0
    // i=3  (i & 1) >> (3 - i) = 1
    //
    // i=0  ((3-i) & 1) >> i = 1
    // i=1  ((3-i) & 1) >> i = 0
    // i=2  ((3-i) & 1) >> i = 0
    // i=3  ((3-i) & 1) >> i = 0
    //
    // i=0  (3 >> i) * i = 0
    // i=1  (3 >> i) * i = 1
    // i=2  (3 >> i) * i = 0
    // i=3  (3 >> i) * i = 0
    //
    // i=0  (3 >> (3-i)) * (3-i) = 0
    // i=1  (3 >> (3-i)) * (3-i) = 0
    // i=2  (3 >> (3-i)) * (3-i) = 1
    // i=3  (3 >> (3-i)) * (3-i) = 0
    out[thread_idx] =
        (((3 - sidx) & 1) >> sidx)         * (x[thread_idx] + y[thread_idx])
      + ((3 >> sidx) * sidx)               * (x[thread_idx] - y[thread_idx])
      + ((3 >> (3-sidx)) * (3-sidx))       * (x[thread_idx] * y[thread_idx])
      + ((sidx & 1) >> (3 - sidx))         * (x[thread_idx] % y[thread_idx]);
}

// vim: cindent: ts=4: sw=4: et
