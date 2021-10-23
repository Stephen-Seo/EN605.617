#include "helpers.h"

void helpers::InitRandStates(curandState_t **states,
                               unsigned int num_blocks,
                               unsigned int num_threads) {
    cudaMalloc(states, num_blocks * num_threads * sizeof(curandState_t));
}

void helpers::FreeRandStates(curandState_t **states) {
    if(states && *states) {
        cudaFree(*states);
        *states = nullptr;
    }
}
