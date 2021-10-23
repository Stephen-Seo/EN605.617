#include "cuda_rand_kernel.h"

#include "constants.h"

__global__
void GenerateRandom(curandState_t *state, unsigned int seed, unsigned int *out) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    curand_init(seed,
                thread_idx,
                0,
                &state[thread_idx]);

    out[thread_idx] = curand(&state[thread_idx]) % kMaxRand;
}
