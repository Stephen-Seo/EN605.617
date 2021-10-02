#include "mathexpressions.h"

#include "constants.h"

__device__
void copy_to_shared(const int *x,
                    const int *y,
                    int *x_shared,
                    int *y_shared,
                    const unsigned int thread_idx) {
    x_shared[threadIdx.x] = x[thread_idx];
    y_shared[threadIdx.x] = y[thread_idx];
}

__device__
void copy_from_shared_out(int *out,
                          const int *out_shared,
                          const unsigned int thread_idx) {
    out[thread_idx] = out_shared[threadIdx.x];
}

__global__
void mathexpressions_shared(const int *x, const int *y, int *out) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    __shared__ int x_shared[NUM_THREADS];
    __shared__ int y_shared[NUM_THREADS];
    __shared__ int out_shared[NUM_THREADS];

    copy_to_shared(x, y, x_shared, y_shared, thread_idx);

    // sync here to ensure shared data is fully populated
    __syncthreads();

    const unsigned int subidx = thread_idx % 4;
    if (subidx == 0) {
        out_shared[threadIdx.x] = x_shared[threadIdx.x] + y_shared[threadIdx.x];
    } else if (subidx == 1) {
        out_shared[threadIdx.x] = x_shared[threadIdx.x] - y_shared[threadIdx.x];
    } else if (subidx == 2) {
        out_shared[threadIdx.x] = x_shared[threadIdx.x] * y_shared[threadIdx.x];
    } else {
        out_shared[threadIdx.x] = x_shared[threadIdx.x] % y_shared[threadIdx.x];
    }

    // sync here to ensure out_shared is fully populated
    __syncthreads();

    copy_from_shared_out(out, out_shared, thread_idx);
}

// vim: cindent: ts=4: sw=4: et
