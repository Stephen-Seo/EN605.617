#include <iostream>

#include "arg_parse.h"
#include "constants.h"

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

    return 0;
}
