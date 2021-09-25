#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <chrono>

#include "helpers.h"
#include "mathexpressions.h"
#include "constants.h"

void printHelp() {
    printf("Usage:\n");
    printf("\t-h | --help\tprint this usage text\n");
    printf("\t--use-paged\trun algorithm with paged memory\n");
    printf("\t--use-pinned\trun algorithm with pinned memory\n");
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

    non_branching_mathexpressions<<<blockSize, totalThreads>>>(
            deviceX,
            deviceY,
            deviceOut);

    deviceToHostOut(hostOut, deviceOut);
    freeDeviceMemory(deviceX, deviceY, deviceOut);

    if (printResults) {
        printHostOut(hostOut);
    }

    freeHostMemory(hostX, hostY, hostOut);
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

    non_branching_mathexpressions<<<blockSize, totalThreads>>>(
            deviceX,
            deviceY,
            deviceOut);

    deviceToHostOut(hostOut, deviceOut);
    freeDeviceMemory(deviceX, deviceY, deviceOut);

    if (printResults) {
        printHostOut(hostOut);
    }

    freePinnedMemory(hostX, hostY, hostOut);
}

int main(int argc, char **argv) {
    bool usePaged = false;
    bool usePinned = false;
    bool printResults = false;

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
        }
    }

    if (!usePaged && !usePinned) {
        printHelp();
        return 0;
    }

    if (usePaged) {
        runPaged(printResults);
    }
    if (usePinned) {
        runPinned(printResults);
    }

    return 0;
}

// vim: cindent: ts=4: sw=4: et
