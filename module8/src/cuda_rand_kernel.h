#ifndef IGPUP_MODULE_8_CUDA_RAND_KERNEL_H
#define IGPUP_MODULE_8_CUDA_RAND_KERNEL_H

#include <curand_kernel.h>

__global__
void GenerateRandom(curandState_t *state, unsigned int seed, unsigned int *out);

#endif
