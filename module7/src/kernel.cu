#include "kernel.h"

__global__
void mathexpressions_events_and_streams(int *a, int *b, int *out) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int subidx = thread_idx % 4;
    if (subidx == 0) {
        out[thread_idx] = a[thread_idx] + b[thread_idx];
    } else if (subidx == 1) {
        out[thread_idx] = a[thread_idx] - b[thread_idx];
    } else if (subidx == 2) {
        out[thread_idx] = a[thread_idx] * b[thread_idx];
    } else {
        out[thread_idx] = a[thread_idx] % b[thread_idx];
    }
}
