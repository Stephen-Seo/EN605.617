#include "helpers.h"

#include <cstdlib>
#include <ctime>

#include "constants.h"

__host__
void Helpers::allocateAndPrepareHostMemory(int **hostX, int **hostY, int **hostOut) {
    (*hostX) = (int*)malloc(sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    (*hostY) = (int*)malloc(sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    (*hostOut) = (int*)malloc(sizeof(int) * NUM_BLOCKS * NUM_THREADS);

    std::srand(std::time(nullptr));
    for (unsigned int i = 0; i < NUM_BLOCKS * NUM_THREADS; ++i) {
        (*hostX)[i] = i;
        (*hostY)[i] = std::rand() % 4;
        (*hostOut)[i] = 0;
    }
}

__host__
void Helpers::freeHostMemory(int **hostX, int **hostY, int **hostOut) {
    if(hostX && *hostX) {
        free(*hostX);
        *hostX = nullptr;
    }
    if(hostY && *hostY) {
        free(*hostY);
        *hostY = nullptr;
    }
    if(hostOut && *hostOut) {
        free(*hostOut);
        *hostOut = nullptr;
    }
}

__host__
void Helpers::allocateDeviceMemory(int **deviceX,
                                  int **deviceY,
                                  int **deviceOut) {
    cudaMalloc(deviceX, sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    CHECK_ERROR();
    cudaMalloc(deviceY, sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    CHECK_ERROR();
    cudaMalloc(deviceOut, sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    CHECK_ERROR();
}

__host__
void Helpers::freeDeviceMemory(int **deviceX,
                               int **deviceY,
                               int **deviceOut) {
    if(deviceX && *deviceX) {
        cudaFree(*deviceX);
        CHECK_ERROR();
        *deviceX = nullptr;
    }
    if(deviceY && *deviceY) {
        cudaFree(*deviceY);
        CHECK_ERROR();
        *deviceY = nullptr;
    }
    if(deviceOut && *deviceOut) {
        cudaFree(*deviceOut);
        CHECK_ERROR();
        *deviceOut = nullptr;
    }
}

__host__
void Helpers::hostToDeviceXY(int *hostX, int *hostY,
                             int *deviceX, int *deviceY) {
    cudaMemcpy(deviceX, hostX, sizeof(int) * NUM_BLOCKS * NUM_THREADS,
            cudaMemcpyHostToDevice);
    CHECK_ERROR();
    cudaMemcpy(deviceY, hostY, sizeof(int) * NUM_BLOCKS * NUM_THREADS,
            cudaMemcpyHostToDevice);
    CHECK_ERROR();
}

__host__
void Helpers::deviceToHostOut(int *hostOut, int *deviceOut) {
    cudaMemcpy(hostOut, deviceOut, sizeof(int) * NUM_BLOCKS * NUM_THREADS,
            cudaMemcpyDeviceToHost);
    CHECK_ERROR();
}
