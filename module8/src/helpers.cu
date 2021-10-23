#include "helpers.h"

#include <cstdio>

#include <cublas.h>

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

void helpers::InitMatrices(float **first_matrix_host,
                           float **second_matrix_host,
                           float **result_matrix_host,
                           float **first_matrix_device,
                           float **second_matrix_device,
                           float **result_matrix_device,
                           unsigned int final_width,
                           unsigned int final_height,
                           unsigned int in_between) {
    // create host matrices
    *first_matrix_host = (float*)malloc(
                            sizeof(float)
                            * final_height      // height of first matrix
                            * in_between);      // width of first matrix
    *second_matrix_host = (float*)malloc(
                            sizeof(float)
                            * in_between        // height of second matrix
                            * final_width);     // width of second matrix
    *result_matrix_host = (float*)malloc(
            sizeof(float) * final_height * final_width);

    // assign to host matrices
    for (unsigned int i = 0; i < final_height * in_between; ++i) {
        (*first_matrix_host)[i] = static_cast<float>(i + 1);
    }

    for (unsigned int i = 0; i < in_between * final_width; ++i) {
        (*second_matrix_host)[i] = static_cast<float>(i + 1);
    }

    // create device matrices
    cublasAlloc(final_height * in_between, sizeof(float),
            (void**)first_matrix_device);
    cublasAlloc(in_between * final_width, sizeof(float),
            (void**)second_matrix_device);
    cublasAlloc(final_height * final_width, sizeof(float),
            (void**)result_matrix_device);

    // assign to device matrices
    cublasSetMatrix(final_height, in_between, sizeof(float),
                    *first_matrix_host, final_height,
                    *first_matrix_device, final_height);
    cublasSetMatrix(in_between, final_width, sizeof(float),
                    *second_matrix_host, in_between,
                    *second_matrix_device, in_between);
}

void helpers::FreeMatrices(float **first_matrix_host,
                           float **second_matrix_host,
                           float **result_matrix_host,
                           float **first_matrix_device,
                           float **second_matrix_device,
                           float **result_matrix_device) {
    if (result_matrix_device && *result_matrix_device) {
        cublasFree(*result_matrix_device);
        *result_matrix_device = nullptr;
    }
    if (second_matrix_device && *second_matrix_device) {
        cublasFree(*second_matrix_device);
        *second_matrix_device = nullptr;
    }
    if (first_matrix_device && *first_matrix_device) {
        cublasFree(*first_matrix_device);
        *first_matrix_device = nullptr;
    }

    if (result_matrix_host && *result_matrix_host) {
        free(*result_matrix_host);
        *result_matrix_host = nullptr;
    }
    if (second_matrix_host && *second_matrix_host) {
        free(*second_matrix_host);
        *second_matrix_host = nullptr;
    }
    if (first_matrix_host && *first_matrix_host) {
        free(*first_matrix_host);
        *first_matrix_host = nullptr;
    }
}

void helpers::PrintMatrix(float *matrix, unsigned int cols, unsigned int rows) {
    for (unsigned int j = 0; j < rows; ++j) {
        for (unsigned int i = 0; i < cols; ++i) {
            printf("%8.1f ", matrix[HELPERS_IDX(j, i, rows)]);
        }
        puts("");
    }
}
