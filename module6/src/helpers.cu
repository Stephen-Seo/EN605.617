#include "helpers.h"

#include <cstdlib>
#include <ctime>

#include "constants.h"

__host__
void Helpers::allocateAndPrepareHostMemory(int **hostX,
                                           int **hostY,
                                           int **hostOut,
                                           unsigned int size) {
    (*hostX) = (int*)malloc(sizeof(int) * size);
    (*hostY) = (int*)malloc(sizeof(int) * size);
    (*hostOut) = (int*)malloc(sizeof(int) * size);

    std::srand(std::time(nullptr));
    for (unsigned int i = 0; i < size; ++i) {
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
                                   int **deviceOut,
                                   unsigned int size) {
    cudaMalloc(deviceX, sizeof(int) * size);
    CHECK_ERROR();
    cudaMalloc(deviceY, sizeof(int) * size);
    CHECK_ERROR();
    cudaMalloc(deviceOut, sizeof(int) * size);
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
                             int *deviceX, int *deviceY,
                             unsigned int size) {
    cudaMemcpy(deviceX, hostX, sizeof(int) * size,
            cudaMemcpyHostToDevice);
    CHECK_ERROR();
    cudaMemcpy(deviceY, hostY, sizeof(int) * size,
            cudaMemcpyHostToDevice);
    CHECK_ERROR();
}

__host__
void Helpers::deviceToHostOut(int *hostOut, int *deviceOut, unsigned int size) {
    cudaMemcpy(hostOut, deviceOut, sizeof(int) * size,
            cudaMemcpyDeviceToHost);
    CHECK_ERROR();
}
