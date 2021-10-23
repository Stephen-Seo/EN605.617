#ifndef IGPUP_MODULE_8_HELPERS_H
#define IGPUP_MODULE_8_HELPERS_H

#define HELPERS_IDX(row, col, height) ((((col))*height)+((row)))

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

    void InitMatrices(float **first_matrix_host,
                      float **second_matrix_host,
                      float **result_matrix_host,
                      float **first_matrix_device,
                      float **second_matrix_device,
                      float **result_matrix_device,
                      unsigned int final_width,
                      unsigned int final_height,
                      unsigned int in_between);

    void FreeMatrices(float **first_matrix_host,
                      float **second_matrix_host,
                      float **result_matrix_host,
                      float **first_matrix_device,
                      float **second_matrix_device,
                      float **result_matrix_device);

    void PrintMatrix(float *matrix, unsigned int cols, unsigned int rows);
} // namespace helpers

#endif
