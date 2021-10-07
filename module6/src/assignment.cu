#include <iostream>
#include <cstdio>
#include <chrono>

#include "arg_parse.h"
#include "constants.h"
#include "helpers.h"
#include "mathexpressions.h"

void runTimings(bool usingRegisterAlgorithm,
                int *deviceX, int *deviceY, int *deviceOut) {
    unsigned long long count = 0;
    std::chrono::nanoseconds duration;

    for (unsigned int i = 0; i < 25; ++i) {
        auto startClock = std::chrono::high_resolution_clock::now();
        if (usingRegisterAlgorithm) {
            mathexpressions_registers<<<NUM_BLOCKS, NUM_THREADS>>>(deviceX,
                                                                   deviceY,
                                                                   deviceOut);
        } else {
            mathexpressions_shared<<<NUM_BLOCKS, NUM_THREADS>>>(deviceX,
                                                                deviceY,
                                                                deviceOut);
        }
        cudaDeviceSynchronize();
        auto endClock = std::chrono::high_resolution_clock::now();
        if (i > 4) {
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>
                    (endClock - startClock);
            std::printf("Register iteration %2u took %7llu ns\n",
                    i - 5, duration.count());
            count += duration.count();
        }
    }

    std::cout << "Average of 20 runs == " << (count / 20) << " ns"
        << std::endl;
}

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
        if (doTimings) {
            // time the register-using algorithm
            runTimings(true, deviceX, deviceY, deviceOut);
        } else {
            mathexpressions_registers<<<NUM_BLOCKS, NUM_THREADS>>>(deviceX,
                                                                   deviceY,
                                                                   deviceOut);
        }
    } else {
        if (doTimings) {
            // time the shared-memory-using algorithm
            runTimings(false, deviceX, deviceY, deviceOut);
        } else {
            mathexpressions_shared<<<NUM_BLOCKS, NUM_THREADS>>>(deviceX,
                                                                deviceY,
                                                                deviceOut);
        }
    }
    CHECK_ERROR();
    cudaDeviceSynchronize();
    CHECK_ERROR();

    Helpers::deviceToHostOut(hostOut, deviceOut);
    CHECK_ERROR();

    Helpers::freeDeviceMemory(&deviceX, &deviceY, &deviceOut);
    CHECK_ERROR();

    if (printOutput && !doTimings) {
        for (unsigned int i = 0; i < NUM_BLOCKS * NUM_THREADS; ++i) {
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
