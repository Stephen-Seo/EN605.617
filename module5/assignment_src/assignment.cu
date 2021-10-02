#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <chrono>

#include "arg_parse.h"
#include "mathexpressions.h"
#include "constants.h"
#include "helpers.h"

// Tried to declare these in "constants.h", but couldn't figure out having a
// separate declaration in header and definition in source file.
static int host_x[NUM_BLOCKS * NUM_THREADS];
static int host_y[NUM_BLOCKS * NUM_THREADS];
__constant__ int const_x[NUM_BLOCKS * NUM_THREADS];
__constant__ int const_y[NUM_BLOCKS * NUM_THREADS];

// Kernel needs to be in same "scope" as __constant__ variables. This could be
// placed in a separate file only if I could figure out having a static
// __constant__ array declared in a header and defined in a source separately.
__global__
void mathexpressions_constant(int *out) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int subidx = thread_idx % 4;

    if (subidx == 0) {
        out[thread_idx] = const_x[thread_idx] + const_y[thread_idx];
    } else if (subidx == 1) {
        out[thread_idx] = const_x[thread_idx] - const_y[thread_idx];
    } else if (subidx == 2) {
        out[thread_idx] = const_x[thread_idx] * const_y[thread_idx];
    } else {
        out[thread_idx] = const_x[thread_idx] % const_y[thread_idx];
    }
}

__host__ void prepare_xyOut(int *x, int *y, int *out) {
    srand(time(nullptr));
    for (unsigned int i = 0; i < NUM_BLOCKS * NUM_THREADS; ++i) {
        x[i] = i;
        y[i] = rand() % 4;
        out[i] = 0;
    }
}

__host__ void prepare_device_xyOut(int **deviceX, int **deviceY, int **deviceOut) {
    cudaMalloc(deviceX, sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    CHECK_ERROR();
    cudaMalloc(deviceY, sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    CHECK_ERROR();
    cudaMalloc(deviceOut, sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    CHECK_ERROR();
}

__host__ void host_to_device_xy(int *x, int *y, int *deviceX, int *deviceY) {
    cudaMemcpy(deviceX, x, sizeof(int) * NUM_BLOCKS * NUM_THREADS,
            cudaMemcpyHostToDevice);
    CHECK_ERROR();
    cudaMemcpy(deviceY, y, sizeof(int) * NUM_BLOCKS * NUM_THREADS,
            cudaMemcpyHostToDevice);
    CHECK_ERROR();
}

__host__ void device_to_host_out(int *out, int *deviceOut) {
    cudaMemcpy(out, deviceOut, sizeof(int) * NUM_BLOCKS * NUM_THREADS,
            cudaMemcpyDeviceToHost);
    CHECK_ERROR();
}

__host__ void print_results(int *out) {
    for (unsigned int i = 0; i < NUM_BLOCKS * NUM_THREADS; ++i) {
        if (i % 4 < 3 && i + 1 < NUM_BLOCKS * NUM_THREADS) {
            std::printf("%7d ", out[i]);
        } else {
            std::printf("%7d\n", out[i]);
        }
    }
}

__host__ void prepare_data(int **x, int **y, int **out,
                  int **deviceX, int **deviceY, int **deviceOut) {
    // prepare local data
    *x = (int*)std::malloc(sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    *y = (int*)std::malloc(sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    *out = (int*)std::malloc(sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    prepare_xyOut(*x, *y, *out);

    // prepare device data
    prepare_device_xyOut(deviceX, deviceY, deviceOut);

    host_to_device_xy(*x, *y, *deviceX, *deviceY);
}

__host__ void prepare_constant_data(int **out, int **deviceOut) {
    // prepare local data
    *out = (int*)std::malloc(sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    prepare_xyOut(host_x, host_y, *out);

    // prepare device data
    cudaMalloc(deviceOut, sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    CHECK_ERROR();

    // prepare constant data
    cudaMemcpyToSymbol(const_x,
                       host_x,
                       sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    CHECK_ERROR();
    cudaMemcpyToSymbol(const_y,
                       host_y,
                       sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    CHECK_ERROR();
}

__host__ void run_shared(bool printOutputs, bool useTimings) {
    int *x;
    int *y;
    int *out;

    int *deviceX;
    int *deviceY;
    int *deviceOut;

    prepare_data(&x, &y, &out, &deviceX, &deviceY, &deviceOut);

    if (useTimings) {
        unsigned long long count = 0;
        std::chrono::nanoseconds duration;

        for (unsigned int i = 0; i < 25; ++i) {
            auto startClock = std::chrono::high_resolution_clock::now();
            // run kernel
            mathexpressions_shared<<<NUM_BLOCKS, NUM_THREADS>>>(deviceX,
                                                                deviceY,
                                                                deviceOut);
            cudaDeviceSynchronize();
            auto endClock = std::chrono::high_resolution_clock::now();

            // warmup for first 5 iterations, time the remaining 20
            if (i > 4) {
                duration = std::chrono::duration_cast<std::chrono::nanoseconds>
                    (endClock - startClock);
                std::printf("Shared iteration %3u took %9llu ns\n",
                        i - 5, duration.count());
                count += duration.count();
            }
        }

        std::cout << "Average of 20 runs == " << (count / 20) << " ns"
            << std::endl;
    } else {
        // run kernel
        mathexpressions_shared<<<NUM_BLOCKS, NUM_THREADS>>>(deviceX,
                                                            deviceY,
                                                            deviceOut);
        CHECK_ERROR();

        device_to_host_out(out, deviceOut);

        if (printOutputs) {
            print_results(out);
        }
    }

    // free data
    cudaFree(deviceOut);
    cudaFree(deviceY);
    cudaFree(deviceX);
    free(out);
    free(y);
    free(x);
}

__host__ void run_constant(bool printOutputs, bool useTimings) {
    int *out;

    int *deviceOut;

    prepare_constant_data(&out, &deviceOut);

    if (useTimings) {
        unsigned long long count = 0;
        std::chrono::nanoseconds duration;

        for (unsigned int i = 0; i < 25; ++i) {
            auto startClock = std::chrono::high_resolution_clock::now();
            // run kernel
            mathexpressions_constant<<<NUM_BLOCKS, NUM_THREADS>>>(deviceOut);
            cudaDeviceSynchronize();
            auto endClock = std::chrono::high_resolution_clock::now();

            // warmup for first 5 iterations, time the remaining 20
            if (i > 4) {
                duration = std::chrono::duration_cast<std::chrono::nanoseconds>
                    (endClock - startClock);
                std::printf("Constant iteration %3u took %9llu ns\n",
                        i - 5, duration.count());
                count += duration.count();
            }
        }

        std::cout << "Average of 20 runs == " << (count / 20) << " ns"
            << std::endl;
    } else {
        // run kernel
        mathexpressions_constant<<<NUM_BLOCKS, NUM_THREADS>>>(deviceOut);
        CHECK_ERROR();

        device_to_host_out(out, deviceOut);

        if (printOutputs) {
            print_results(out);
        }
    }

    // free data
    cudaFree(deviceOut);
    free(out);
}

int main(int argc, char **argv) {
    Args args;
    if (args.parseArgs(argc, argv)) {
        return 0;
    } else if (args.runShared) {
        std::cout << "Running shared-memory algorithm" << std::endl;
        run_shared(args.enablePrintOutput, args.enableTimings);
    } else if (args.runConstant) {
        std::cout << "Running constant-memory algorithm" << std::endl;
        run_constant(args.enablePrintOutput, args.enableTimings);
    } else {
        std::cout << "shared or constant algorithm not specified.\n";
        Args::displayHelp();
    }

    return 0;
}

// vim: cindent
