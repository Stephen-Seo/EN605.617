#ifndef IGPUP_MODULE_8_HELPERS_H
#define IGPUP_MODULE_8_HELPERS_H

#include <curand_kernel.h>

namespace helpers {
    void InitRandStates(curandState_t **states,
                          unsigned int num_blocks,
                          unsigned int num_threads);

    void FreeRandStates(curandState_t **states);

    template <typename T>
    void InitDeviceMemory(T **device_mem,
                          unsigned int num_blocks,
                          unsigned int num_threads) {
        cudaMalloc(device_mem, num_blocks * num_threads * sizeof(T));
    }

    template <typename T>
    void FreeDeviceMemory(T **device_mem) {
        if(device_mem && *device_mem) {
            cudaFree(*device_mem);
            *device_mem = nullptr;
        }
    }
} // namespace helpers

#endif
