#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <chrono>

#include "constants.h"
#include "helpers.h"
#include "mathexpressions.h"
#include "cipher.h"

void printHelp() {
    printf("Usage:\n");
    printf("\t-h | --help\tprint this usage text\n");
    printf("\t--use-paged\trun algorithm with paged memory\n");
    printf("\t--use-pinned\trun algorithm with pinned memory\n");
    printf("\t--use-cipher\trun Caesar Cipher kernel\n");
    printf("\t--use-cipher-offset <offset>\tuse the specified offset when "
            "running the Caesar Cipher algorithm\n");
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

    branching_mathexpressions<<<BLOCK_SIZE, TOTAL_THREADS>>>(
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

    branching_mathexpressions<<<BLOCK_SIZE, TOTAL_THREADS>>>(
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

void runCipher(bool printResults, int offset) {
    char *host_orig;
    char *host_edit;

    char *device_first;
    char *device_second;

    cipher_allocAndSetupHostMemory(&host_orig);
    cipher_allocAndSetupHostMemory(&host_edit);

    if (printResults) {
        printf("Original text:\n");
        cipher_printChars(host_orig);
    }

    cipher_allocDeviceMemory(&device_first);
    cipher_allocDeviceMemory(&device_second);

    cipher_hostToDevice(host_orig, device_first);

    // first shift
    caesar_cipher<<<BLOCK_SIZE, TOTAL_THREADS>>>(
            device_first,
            device_second,
            offset);

    cipher_deviceToHost(host_edit, device_second);

    if (printResults) {
        printf("Shifted text:\n");
        cipher_printChars(host_edit);
    }

    // second shift, reverse of previous shift
    caesar_cipher<<<BLOCK_SIZE, TOTAL_THREADS>>>(
            device_second,
            device_first,
            -offset);

    cipher_deviceToHost(host_edit, device_first);

    if (printResults) {
        printf("Reverse shifted text:\n");
        cipher_printChars(host_edit);
        printf("Reverse shifted result is %s to original\n",
                (strncmp(host_orig, host_edit, TOTAL_THREADS) == 0
                    ? "same" : "different"));
    }

    cipher_freeDeviceMemory(&device_second);
    cipher_freeDeviceMemory(&device_first);

    cipher_freeHostMemory(&host_edit);
    cipher_freeHostMemory(&host_orig);
}

int main(int argc, char **argv) {
    bool usePaged = false;
    bool usePinned = false;
    bool printResults = false;
    bool useCaesarCipher = false;
    int cipherOffset = 3;

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
        } else if (strcmp(argv[0], "--use-cipher") == 0) {
            useCaesarCipher = true;
        } else if (strcmp(argv[0], "--use-cipher-offset") == 0) {
            if (argc > 1) {
                int offset = atoi(argv[1]);
                if (offset != 0) {
                    validateOffset(&offset);
                    if (offset != 0) {
                        cipherOffset = offset;
                        --argc; ++argv;
                    } else {
                        printf("ERROR: offset resolves to \"0\", use a "
                                "different number\n");
                        return 1;
                    }
                } else {
                    printf("ERROR: offset is invalid or resolves to \"0\"\n");
                    return 1;
                }
            } else {
                printf("ERROR: --use-cihper-offset specified without offset\n");
                return 1;
            }
        }
    }

    if (!usePaged && !usePinned && !useCaesarCipher) {
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
    if (useCaesarCipher) {
        printf("Running \"Caesar Cipher\" algorithm\n");
        runCipher(printResults, cipherOffset);
    }

    return 0;
}

// vim: cindent: ts=4: sw=4: et
