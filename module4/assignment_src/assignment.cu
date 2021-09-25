#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <chrono>

#include "constants.h"
#include "helpers.h"
#include "mathexpressions.h"
#include "cypher.h"

void printHelp() {
    printf("Usage:\n");
    printf("\t-h | --help\tprint this usage text\n");
    printf("\t--use-paged\trun algorithm with paged memory\n");
    printf("\t--use-pinned\trun algorithm with pinned memory\n");
    printf("\t--use-cypher [offset]\trun Caesar Cypher kernel (default offset "
            "is 3)\n");
    printf("\t--print-results\toutput results of kernel execution\n");
}

void runPaged(bool printResults) {
    int *hostX;
    int *hostY;
    int *hostOut;

    int *deviceX;
    int *deviceY;
    int *deviceOut;

    allocAndSetupHostMemory(&hostX, &hostY, &hostOut);
    allocDeviceMemory(&deviceX, &deviceY, &deviceOut);

    hostToDeviceXY(hostX, hostY, deviceX, deviceY);

    branching_mathexpressions<<<blockSize, totalThreads>>>(
            deviceX,
            deviceY,
            deviceOut);

    deviceToHostOut(hostOut, deviceOut);
    freeDeviceMemory(&deviceX, &deviceY, &deviceOut);

    if (printResults) {
        printHostOut(hostOut);
    }

    freeHostMemory(&hostX, &hostY, &hostOut);
}

void runPinned(bool printResults) {
    int *hostX;
    int *hostY;
    int *hostOut;

    int *deviceX;
    int *deviceY;
    int *deviceOut;

    allocAndSetupPinnedMemory(&hostX, &hostY, &hostOut);
    allocDeviceMemory(&deviceX, &deviceY, &deviceOut);

    hostToDeviceXY(hostX, hostY, deviceX, deviceY);

    branching_mathexpressions<<<blockSize, totalThreads>>>(
            deviceX,
            deviceY,
            deviceOut);

    deviceToHostOut(hostOut, deviceOut);
    freeDeviceMemory(&deviceX, &deviceY, &deviceOut);

    if (printResults) {
        printHostOut(hostOut);
    }

    freePinnedMemory(&hostX, &hostY, &hostOut);
}

void validateOffset(int *offset) {
    while (*offset < 0) {
        *offset += 26;
    }
    *offset = *offset % 26;
}

void runCypher(bool printResults, int offset) {
    char *host_orig;
    char *host_edit;

    char *device_first;
    char *device_second;

    cypher_allocAndSetupHostMemory(&host_orig);
    cypher_allocAndSetupHostMemory(&host_edit);

    if (printResults) {
        printf("Original text:\n");
        cypher_printChars(host_orig);
    }

    cypher_allocDeviceMemory(&device_first);
    cypher_allocDeviceMemory(&device_second);

    cypher_hostToDevice(host_orig, device_first);

    // first shift
    caesar_cypher<<<blockSize, totalThreads>>>(
            device_first,
            device_second,
            offset);

    cypher_deviceToHost(host_edit, device_second);

    if (printResults) {
        printf("Shifted text:\n");
        cypher_printChars(host_edit);
    }

    // second shift, reverse of previous shift
    caesar_cypher<<<blockSize, totalThreads>>>(
            device_second,
            device_first,
            -offset);

    cypher_deviceToHost(host_edit, device_first);

    if (printResults) {
        printf("Reverse shifted text:\n");
        cypher_printChars(host_edit);
        printf("Reverse shifted result is %s to original\n",
                (strncmp(host_orig, host_edit, totalThreads) == 0
                    ? "same" : "different"));
    }

    cypher_freeDeviceMemory(&device_second);
    cypher_freeDeviceMemory(&device_first);

    cypher_freeHostMemory(&host_edit);
    cypher_freeHostMemory(&host_orig);
}

int main(int argc, char **argv) {
    bool usePaged = false;
    bool usePinned = false;
    bool printResults = false;
    bool useCaesarCypher = false;
    int cypherOffset = 3;

    --argc; ++argv;
    for (; argc > 0; --argc, ++argv) {
        if (strcmp(argv[0], "--use-paged") == 0) {
            usePaged = true;
        } else if (strcmp(argv[0], "--use-pinned") == 0) {
            usePinned = true;
        } else if (strcmp(argv[0], "-h") == 0
                || strcmp(argv[0], "--help") == 0) {
            printHelp();
            return 0;
        } else if (strcmp(argv[0], "--print-results") == 0) {
            printResults = true;
        } else if (strcmp(argv[0], "--use-cypher") == 0) {
            useCaesarCypher = true;
            if (argc > 1) {
                int offset = atoi(argv[1]);
                if (offset != 0) {
                    validateOffset(&offset);
                    if (offset != 0) {
                        cypherOffset = offset;
                        --argc; ++argv;
                    }
                }
            }
        }
    }

    if (!usePaged && !usePinned && !useCaesarCypher) {
        printHelp();
        return 0;
    }

    if (usePaged) {
        printf("Running \"paged\" algorithm\n");
        runPaged(printResults);
    }
    if (usePinned) {
        printf("Running \"pinned\" algorithm\n");
        runPinned(printResults);
    }
    if (useCaesarCypher) {
        printf("Running \"Caesar Cypher\" algorithm\n");
        runCypher(printResults, cypherOffset);
    }

    return 0;
}

// vim: cindent: ts=4: sw=4: et
