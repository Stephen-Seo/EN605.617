#include <ctime>
#include <cstdio>

#include "arg_parse.h"
#include "constants.h"
#include "helpers.h"
#include "cuda_rand_kernel.h"

void runCudaRandKernel(unsigned int num_blocks,
                       unsigned int num_threads,
                       bool print_output,
                       bool do_timings) {
    curandState_t *states;
    unsigned int *kernel_output;

    helpers::InitRandStates(&states, num_blocks, num_threads);
    helpers::InitDeviceMemory<unsigned int>(&kernel_output,
                                            num_blocks,
                                            num_threads);

    GenerateRandom<<<num_blocks, num_threads>>>(states,
                                                std::time(nullptr),
                                                kernel_output);

    helpers::FreeRandStates(&states);

    if (print_output) {
        unsigned int *host_output =
                (unsigned int*)malloc(
                    num_blocks * num_threads * sizeof(unsigned int));
        cudaMemcpy(host_output,
                   kernel_output,
                   num_blocks * num_threads * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
        for (unsigned int i = 0; i < num_blocks * num_threads; ++i) {
            if (i + 1 < num_blocks * num_threads && i % 12 != 11) {
                std::printf("%4u ", host_output[i]);
            } else {
                std::printf("%4u\n", host_output[i]);
            }
        }
    } else {
        std::puts("WARNING: GenerateRandom kernel was run, but print output "
                "was not enabled");
    }

    helpers::FreeDeviceMemory(&kernel_output);
}

int main(int argc, char **argv) {
    unsigned int num_blocks = kDefaultBlocks;
    unsigned int num_threads = kDefaultThreads;

    Args args{};
    if (args.ParseArgs(argc, argv)) {
        // help was printed, just exit
        return 0;
    } else if(!args.run_cuda_rand) {
        std::puts("ERROR: Kernel to run not specificed");
        Args::DisplayHelp();
        return 1;
    }

    if (args.num_blocks > 0) {
        num_blocks = args.num_blocks;
    }
    if (args.num_threads > 0) {
        num_threads = args.num_threads;
    }

    if (args.run_cuda_rand) {
        runCudaRandKernel(num_blocks,
                          num_threads,
                          args.enable_print_output,
                          args.enable_timings);
    }

    return 0;
}
