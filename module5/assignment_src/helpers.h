#ifndef IGPUP_MODULE_5_HELPERS_H
#define IGPUP_MODULE_5_HELPERS_H

#include <iostream>

#define CHECK_ERROR() \
    do { \
        cudaError_t errorValue = cudaGetLastError(); \
        if (errorValue != cudaSuccess) { \
            std::cout \
                << __FILE__ << ':' << __LINE__ \
                << " CUDA ERROR: " \
                << cudaGetErrorString(errorValue) \
                << std::endl; \
        } \
    } while(false);

#endif
