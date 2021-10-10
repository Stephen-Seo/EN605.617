#include <iostream>
#include <cstdio>
#include <chrono>

#include "arg_parse.h"
#include "constants.h"
#include "helpers.h"
#include "mathexpressions.h"

void runTimings(bool usingRegisterAlgorithm,
                int *deviceX, int *deviceY, int *deviceOut,
                unsigned int num_blocks) {
    unsigned long long count = 0;
    std::chrono::nanoseconds duration;

    for (unsigned int i = 0; i < 25; ++i) {
        auto startClock = std::chrono::high_resolution_clock::now();
        if (usingRegisterAlgorithm) {
            mathexpressions_registers<<<num_blocks, NUM_THREADS>>>(deviceX,
                                                                   deviceY,
                                                                   deviceOut);
        } else {
            mathexpressions_shared<<<num_blocks, NUM_THREADS>>>(deviceX,
                                                                deviceY,
                                                                deviceOut);
        }
        cudaDeviceSynchronize();
        auto endClock = std::chrono::high_resolution_clock::now();
        if (i > 4) {
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>
                    (endClock - startClock);
            std::printf("%s iteration %2u took %7llu ns\n",
                    (usingRegisterAlgorithm ? "Register" : "Shared"),
                    i - 5, duration.count());
            count += duration.count();
        }
    }

    std::cout << "Average of 20 runs == " << (count / 20) << " ns"
        << std::endl;
}

void runKernel(bool usingRegisterAlgorithm, bool printOutput, bool doTimings,
               unsigned int num_blocks) {
    int *hostX;
    int *hostY;
    int *hostOut;

    int *deviceX;
    int *deviceY;
    int *deviceOut;

    unsigned int size = num_blocks * NUM_THREADS;

    Helpers::allocateAndPrepareHostMemory(&hostX, &hostY, &hostOut, size);
    Helpers::allocateDeviceMemory(&deviceX, &deviceY, &deviceOut, size);
    Helpers::hostToDeviceXY(hostX, hostY, deviceX, deviceY, size);

    if (usingRegisterAlgorithm) {
        if (doTimings) {
            // time the register-using algorithm
            runTimings(true,
                       deviceX, deviceY, deviceOut,
                       num_blocks);
        } else {
            mathexpressions_registers<<<num_blocks, NUM_THREADS>>>(deviceX,
                                                                   deviceY,
                                                                   deviceOut);
        }
    } else {
        if (doTimings) {
            // time the shared-memory-using algorithm
            runTimings(false,
                       deviceX, deviceY, deviceOut,
                       num_blocks);
        } else {
            mathexpressions_shared<<<num_blocks, NUM_THREADS>>>(deviceX,
                                                                deviceY,
                                                                deviceOut);
        }
    }
    CHECK_ERROR();
    cudaDeviceSynchronize();
    CHECK_ERROR();

    Helpers::deviceToHostOut(hostOut, deviceOut, size);
    CHECK_ERROR();

    Helpers::freeDeviceMemory(&deviceX, &deviceY, &deviceOut);
    CHECK_ERROR();

    if (printOutput && !doTimings) {
        for (unsigned int i = 0; i < size; ++i) {
            if (i % 4 < 3 && i + 1 < size) {
                std::printf("%7d ", hostOut[i]);
            } else {
                std::printf("%7d\n", hostOut[i]);
            }
        }
    }

    Helpers::freeHostMemory(&hostX, &hostY, &hostOut);
}

int main(int argc, char **argv) {
    unsigned int num_blocks = DEFAULT_NUM_BLOCKS;

    Args args;
    if (args.parseArgs(argc, argv)) {
        // help was printed, so stop program
        return 0;
    }

    if (args.num_blocks > 0) {
        num_blocks = args.num_blocks;
        std::cout << "Setting num_blocks to " << num_blocks << std::endl;
    } else {
        std::cout << "Defaulting num_blocks to " << num_blocks << std::endl;
    }

    if (args.runRegisterBasedMemory) {
        std::cout << "Running Register Memory based algorithm...\n";
        runKernel(true,
                  args.enablePrintOutput,
                  args.enableTimings,
                  num_blocks);
    } else if (args.runSharedBasedMemory) {
        std::cout << "Running Shared Memory based algorithm...\n";
        runKernel(false,
                  args.enablePrintOutput,
                  args.enableTimings,
                  num_blocks);
    } else {
        std::cout << "ERROR: Neither Register or Shared memory algorithms "
            "specified.\n";
        args.displayHelp();
        return 1;
    }

    return 0;
}
