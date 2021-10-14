#include <iostream>
#include <cstdio>

#include "arg_parse.h"
#include "constants.h"
#include "kernel.h"
#include "helpers.h"

int main(int argc, char **argv) {
    Args args{};
    if(args.parseArgs(argc, argv)) {
        // help printed, just stop with success
        return 0;
    }

    unsigned int block_size = DEFAULT_BLOCK_SIZE;
    unsigned int thread_size = DEFAULT_THREAD_SIZE;

    if (args.num_blocks > 0) {
        block_size = args.num_blocks;
        std::cout << "Setting block_size to " << block_size << std::endl;
    } else {
        std::cout << "Defaulting block_size to " << block_size << std::endl;
    }

    if (args.num_threads > 0) {
        thread_size = args.num_threads;
        std::cout << "Setting thread_size to " << thread_size << std::endl;
    } else {
        std::cout << "Defaulting thread_size to " << thread_size << std::endl;
    }

    // parameters set, now setting up kernel function invocation
    int *host_a;
    int *host_b;
    int *host_out;

    int *device_a;
    int *device_b;
    int *device_out;

    cudaStream_t stream;
    cudaEvent_t event_start;
    cudaEvent_t event_end;

    Helpers::setUpHostMemory(&host_a, &host_b, &host_out,
                             block_size, thread_size);
    Helpers::setUpDeviceMemory(&device_a, &device_b, &device_out,
                               block_size, thread_size);
    Helpers::setUpStreamAndEvents(&stream, &event_start, &event_end);

    if (args.enableTimings) {
        // enableTimings is set, time the latter 20 of 25 runs
        float sumMilliseconds = 0.0F;
        float elapsedMilliseconds;
        for (unsigned int i = 0; i < 25; ++i) {
            Helpers::asyncMemcpyToDevice(host_a, host_b, device_a, device_b,
                                         stream, event_start,
                                         block_size, thread_size);
            Helpers::invokeKernel(device_a, device_b, device_out,
                                  block_size, thread_size, stream);
            Helpers::asyncMemcpyToHost(host_out, device_out, stream, event_end,
                                       block_size, thread_size);
            if (i > 4) {
                Helpers::getEventElapsedTime(event_start, event_end,
                                             &elapsedMilliseconds);
                std::cout << "Run " << i - 4 << " took " << elapsedMilliseconds
                    << " milliseconds\n";
                sumMilliseconds += elapsedMilliseconds;
            }
        }
        std::cout << "Average of 20 runs == " << sumMilliseconds / 20.0F
            << " milliseconds" << std::endl;
    } else {
        // enableTimings is not set, just run normally
        Helpers::asyncMemcpyToDevice(host_a, host_b, device_a, device_b,
                                     stream, event_start,
                                     block_size, thread_size);
        Helpers::invokeKernel(device_a, device_b, device_out,
                              block_size, thread_size, stream);
        Helpers::asyncMemcpyToHost(host_out, device_out, stream, event_end,
                                   block_size, thread_size);
        if (args.enablePrintOutput) {
            // enablePrintOutput is set, print the output
            unsigned int size = block_size * thread_size;
            for(unsigned int i = 0; i < size; ++i) {
                if (i % 4 != 3 && i + 1 != size) {
                    printf("%7d ", host_out[i]);
                } else {
                    printf("%7d\n", host_out[i]);
                }
            }
        } else {
            // neither enableTimings nor enablePrintOutput was set
            std::cout << "print-output or timings were not enabled, please "
                "specify \"-p\" or \"-t\" to get outputs" << std::endl;
            Args::displayHelp();
        }
    }

    Helpers::cleanupStreamAndEvents(stream, event_start, event_end);
    Helpers::cleanupDeviceMemory(&device_a, &device_b, &device_out);
    Helpers::cleanupHostMemory(&host_a, &host_b, &host_out);

    return 0;
}
