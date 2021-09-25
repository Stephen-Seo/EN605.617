#ifndef M4_HELPERS_H
#define M4_HELPERS_H

bool checkError(cudaError_t cudaError);

void allocAndSetupHostMemory(int **hostX, int **hostY, int **hostOut);
void freeHostMemory(int **hostX, int **hostY, int **hostOut);
void allocDeviceMemory(int **x, int **y, int **out);
void freeDeviceMemory(int **x, int **y, int **out);
void allocAndSetupPinnedMemory(int **x, int **y, int **out);
void freePinnedMemory(int **x, int **y, int **out);
void hostToDeviceXY(int *hostX, int *hostY, int *deviceX, int *deviceY);
void deviceToHostOut(int *hostOut, int *deviceOut);
void printHostOut(int *hostOut);

void cipher_allocAndSetupHostMemory(char **host);
void cipher_freeHostMemory(char **host);
void cipher_allocDeviceMemory(char **device);
void cipher_freeDeviceMemory(char **device);
void cipher_hostToDevice(char *host, char *device);
void cipher_deviceToHost(char *host, char *device);
void cipher_printChars(char *host);

#endif
