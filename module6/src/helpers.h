#ifndef IGPUP_MODULE_6_HELPERS_H
#define IGPUP_MODULE_6_HELPERS_H

#include <iostream>

#define CHECK_ERROR() \
    do { \
        cudaError_t errorValue = cudaGetLastError(); \
        if (errorValue != cudaSuccess) { \
            std::cout \
                << __FILE__ << ':' << __LINE__ \
                << ' ' << cudaGetErrorName(errorValue) << ": " \
                << cudaGetErrorString(errorValue) \
                << std::endl; \
        } \
    } while(false);

namespace Helpers {
    __host__
    void allocateAndPrepareHostMemory(int **hostX,
                                      int **hostY,
                                      int **hostOut,
                                      unsigned int size);
    __host__
    void freeHostMemory(int **hostX, int **hostY, int **hostOut);

    __host__
    void allocateDeviceMemory(int **deviceX,
                              int **deviceY,
                              int **deviceOut,
                              unsigned int size);
    __host__
    void freeDeviceMemory(int **deviceX, int **deviceY, int **deviceOut);

    __host__
    void hostToDeviceXY(int *hostX,
                        int *hostY,
                        int *deviceX,
                        int *deviceY,
                        unsigned int size);
    __host__
    void deviceToHostOut(int *hostOut, int *deviceOut, unsigned int size);
} // namespace Helper

#endif
