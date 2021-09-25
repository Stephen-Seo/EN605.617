#include "mathexpressions.h"

__global__
void branching_mathexpressions(int *x, int *y, int *out) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int subidx = thread_idx % 4;
    if (subidx == 0) {
        out[thread_idx] = x[thread_idx] + y[thread_idx];
    } else if (subidx == 1) {
        out[thread_idx] = x[thread_idx] - y[thread_idx];
    } else if (subidx == 2) {
        out[thread_idx] = x[thread_idx] * y[thread_idx];
    } else {
        out[thread_idx] = x[thread_idx] % y[thread_idx];
    }
}

// vim: cindent: ts=4: sw=4: et
