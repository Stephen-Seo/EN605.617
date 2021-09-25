#include "helpers.h"

#include <stdio.h>
#include <string.h>

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

void freeHostMemory(int **hostX, int **hostY, int **hostOut) {
    if (hostX && *hostX) {
        free(*hostX);
        *hostX = NULL;
    }
    if (hostY && *hostY) {
        free(*hostY);
        *hostY = NULL;
    }
    if (hostOut && *hostOut) {
        free(*hostOut);
        *hostOut = NULL;
    }
}

void allocDeviceMemory(int **x, int **y, int **out) {
    if (!x || !y || !out) {
        return;
    }
    cudaMalloc((void**)x, sizeof(int) * totalThreads);
    checkError(cudaPeekAtLastError());
    cudaMalloc((void**)y, sizeof(int) * totalThreads);
    checkError(cudaPeekAtLastError());
    cudaMalloc((void**)out, sizeof(int) * totalThreads);
    checkError(cudaPeekAtLastError());
}

void freeDeviceMemory(int **x, int **y, int **out) {
    if (x && *x) {
        cudaFree(*x);
        checkError(cudaPeekAtLastError());
        *x = NULL;
    }
    if (y && *y) {
        cudaFree(*y);
        checkError(cudaPeekAtLastError());
        *y = NULL;
    }
    if (out && *out) {
        cudaFree(*out);
        checkError(cudaPeekAtLastError());
        *out = NULL;
    }
}

void allocAndSetupPinnedMemory(int **x, int **y, int **out) {
    if (!x || !y || !out) {
        return;
    }

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

void freePinnedMemory(int **x, int **y, int **out) {
    if (x && *x) {
        cudaFreeHost(*x);
        checkError(cudaPeekAtLastError());
        *x = NULL;
    }
    if (y && *y) {
        cudaFreeHost(*y);
        checkError(cudaPeekAtLastError());
        *y = NULL;
    }
    if (out && *out) {
        cudaFreeHost(*out);
        checkError(cudaPeekAtLastError());
        *out = NULL;
    }
}

void hostToDeviceXY(int *hostX, int *hostY, int *deviceX, int *deviceY) {
    if (!hostX || !hostY || !deviceX || !deviceY) {
        return;
    }
    cudaMemcpy(deviceX, hostX, sizeof(int) * totalThreads,
            cudaMemcpyHostToDevice);
    checkError(cudaPeekAtLastError());
    cudaMemcpy(deviceY, hostY, sizeof(int) * totalThreads,
            cudaMemcpyHostToDevice);
    checkError(cudaPeekAtLastError());
}

void deviceToHostOut(int *hostOut, int *deviceOut) {
    if (!hostOut || !deviceOut) {
        return;
    }
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

void cypher_allocAndSetupHostMemory(char **host) {
    if (!host) {
        return;
    }

    *host = (char*)malloc(sizeof(char) * totalThreads);

    unsigned int i = 0;
    for (; i < totalThreads; i += cypherPhraseSize) {
        memcpy(*host + i, cypherPhrase, sizeof(char) * 26);
    }
    if (i > totalThreads) {
        i -= cypherPhraseSize;
        memcpy(*host + i, cypherPhrase, sizeof(char) * (totalThreads - i));
    }
}

void cypher_freeHostMemory(char **host) {
    if (host && *host) {
        free(*host);
        *host = NULL;
    }
}

void cypher_allocDeviceMemory(char **device) {
    if(!device) {
        return;
    }
    cudaMalloc((void**)device, sizeof(char) * totalThreads);
    checkError(cudaPeekAtLastError());
}

void cypher_freeDeviceMemory(char **device) {
    if(device && *device) {
        cudaFree(*device);
        checkError(cudaPeekAtLastError());
        *device = NULL;
    }
}

void cypher_hostToDevice(char *host, char *device) {
    if(!host || !device) {
        return;
    }
    cudaMemcpy(device, host, sizeof(char) * totalThreads,
            cudaMemcpyHostToDevice);
    checkError(cudaPeekAtLastError());
}

void cypher_deviceToHost(char *host, char *device) {
    if(!host || !device) {
        return;
    }
    cudaMemcpy(host, device, sizeof(char) * totalThreads,
            cudaMemcpyDeviceToHost);
    checkError(cudaPeekAtLastError());
}

void cypher_printChars(char *host) {
    unsigned int j = 0;
    for (; j * 64 < totalThreads; ++j) {
        for (unsigned int i = 0; i < 64; ++i) {
            printf("%c", host[i + j * 64]);
        }
        printf("\n");
    }
    if (j * 64 < totalThreads) {
        for (unsigned int i = j * 64; i < totalThreads; ++i) {
            printf("%c", host[i]);
        }
        printf("\n");
    }
}

// vim: cindent: ts=4: sw=4: et
