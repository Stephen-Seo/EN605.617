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

void cypher_allocAndSetupHostMemory(char **host);
void cypher_freeHostMemory(char **host);
void cypher_allocDeviceMemory(char **device);
void cypher_freeDeviceMemory(char **device);
void cypher_hostToDevice(char *host, char *device);
void cypher_deviceToHost(char *host, char *device);
void cypher_printChars(char *host);

#endif
