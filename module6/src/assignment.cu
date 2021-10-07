#include <iostream>
#include <cstdio>

#include "arg_parse.h"
#include "constants.h"
#include "helpers.h"
#include "mathexpressions.h"

void runKernel(bool usingRegisterAlgorithm, bool printOutput, bool doTimings) {
    int *hostX;
    int *hostY;
    int *hostOut;

    int *deviceX;
    int *deviceY;
    int *deviceOut;

    Helpers::allocateAndPrepareHostMemory(&hostX, &hostY, &hostOut);
    Helpers::allocateDeviceMemory(&deviceX, &deviceY, &deviceOut);
    Helpers::hostToDeviceXY(hostX, hostY, deviceX, deviceY);

    if (usingRegisterAlgorithm) {
        mathexpressions_registers<<<NUM_BLOCKS, NUM_THREADS>>>(deviceX,
                                                               deviceY,
                                                               deviceOut);
    } else {
        mathexpressions_shared<<<NUM_BLOCKS, NUM_THREADS>>>(deviceX,
                                                            deviceY,
                                                            deviceOut);
    }
    CHECK_ERROR();
    cudaDeviceSynchronize();
    CHECK_ERROR();

    Helpers::deviceToHostOut(hostOut, deviceOut);
    CHECK_ERROR();

    Helpers::freeDeviceMemory(&deviceX, &deviceY, &deviceOut);
    CHECK_ERROR();

    if (printOutput) {
        for(unsigned int i = 0; i < NUM_BLOCKS * NUM_THREADS; ++i) {
            if (i % 4 < 3 && i + 1 < NUM_BLOCKS * NUM_THREADS) {
                std::printf("%7d ", hostOut[i]);
            } else {
                std::printf("%7d\n", hostOut[i]);
            }
        }
    }

    Helpers::freeHostMemory(&hostX, &hostY, &hostOut);
}

int main(int argc, char **argv) {
    Args args;
    if (args.parseArgs(argc, argv)) {
        // help was printed, so stop program
        return 0;
    }

    if (args.runRegisterBasedMemory) {
        std::cout << "Running Register Memory based algorithm...\n";
        runKernel(true, args.enablePrintOutput, args.enableTimings);
    } else if (args.runSharedBasedMemory) {
        std::cout << "Running Shared Memory based algorithm...\n";
        runKernel(false, args.enablePrintOutput, args.enableTimings);
    } else {
        std::cout << "ERROR: Neither Register or Shared memory algorithms "
            "specified.\n";
        args.displayHelp();
        return 1;
    }

    return 0;
}
