#include "helpers.h"

#include <stdio.h>

#include "constants.h"

bool checkError(cudaError_t cudaError) {
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaError));
        return true;
    }
    return false;
}

void allocAndSetupHostMemory(int **hostX, int **hostY, int **hostOut) {
    if (!hostX || !hostY || !hostOut) {
        return;
    }

    *hostX = (int*)malloc(sizeof(int) * totalThreads);
    *hostY = (int*)malloc(sizeof(int) * totalThreads);
    *hostOut = (int*)malloc(sizeof(int) * totalThreads);

    srand(time(NULL));
    for(unsigned int i = 0; i < totalThreads; ++i) {
        (*hostX)[i] = i;
        (*hostY)[i] = rand() % 4;
        (*hostOut)[i] = 0;
    }
}

void freeHostMemory(int *hostX, int *hostY, int *hostOut) {
    free(hostX);
    free(hostY);
    free(hostOut);
}

void allocDeviceMemory(int **x, int **y, int **out) {
    cudaMalloc((void**)x, sizeof(int) * totalThreads);
    checkError(cudaPeekAtLastError());
    cudaMalloc((void**)y, sizeof(int) * totalThreads);
    checkError(cudaPeekAtLastError());
    cudaMalloc((void**)out, sizeof(int) * totalThreads);
    checkError(cudaPeekAtLastError());
}

void freeDeviceMemory(int *x, int *y, int *out) {
    cudaFree(x);
    checkError(cudaPeekAtLastError());
    cudaFree(y);
    checkError(cudaPeekAtLastError());
    cudaFree(out);
    checkError(cudaPeekAtLastError());
}

void allocAndSetupPinnedMemory(int **x, int **y, int **out) {
    cudaHostAlloc((void**)x, sizeof(int) * totalThreads, 0);
    checkError(cudaPeekAtLastError());
    cudaHostAlloc((void**)y, sizeof(int) * totalThreads, 0);
    checkError(cudaPeekAtLastError());
    cudaHostAlloc((void**)out, sizeof(int) * totalThreads, 0);
    checkError(cudaPeekAtLastError());

    srand(time(NULL));
    for(unsigned int i = 0; i < totalThreads; ++i) {
        (*x)[i] = i;
        (*y)[i] = rand() % 4;
        (*out)[i] = 0;
    }
}

void freePinnedMemory(int *x, int *y, int *out) {
    cudaFreeHost(x);
    checkError(cudaPeekAtLastError());
    cudaFreeHost(y);
    checkError(cudaPeekAtLastError());
    cudaFreeHost(out);
    checkError(cudaPeekAtLastError());
}

void hostToDeviceXY(int *hostX, int *hostY, int *deviceX, int *deviceY) {
    cudaMemcpy(deviceX, hostX, sizeof(int) * totalThreads,
            cudaMemcpyHostToDevice);
    checkError(cudaPeekAtLastError());
    cudaMemcpy(deviceY, hostY, sizeof(int) * totalThreads,
            cudaMemcpyHostToDevice);
    checkError(cudaPeekAtLastError());
}

void deviceToHostOut(int *hostOut, int *deviceOut) {
    cudaMemcpy(hostOut, deviceOut, sizeof(int) * totalThreads,
            cudaMemcpyDeviceToHost);
}

void printHostOut(int *hostOut) {
    for(unsigned int j = 0; j <= totalThreads / 4; ++j) {
        if (j * 4 < totalThreads) {
            printf("%4u: %4d\t", j * 4, hostOut[j * 4]);
            if (1 + j * 4 < totalThreads) {
                printf("%4u: %4d\t", 1 + j * 4, hostOut[1 + j * 4]);
                if (2 + j * 4 < totalThreads) {
                    printf("%4u: %4d\t", 2 + j * 4, hostOut[2 + j * 4]);
                    if (3 + j * 4 < totalThreads) {
                        printf("%4u: %4d\n", 3 + j * 4, hostOut[3 + j * 4]);
                    } else {
                        printf("\n");
                        break;
                    }
                } else {
                    printf("\n");
                    break;
                }
            } else {
                printf("\n");
                break;
            }
        } else {
            printf("\n");
            break;
        }
    }
}
