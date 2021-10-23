#include "arg_parse.h"

#include <cstring>
#include <iostream>

Args::Args() :
    enable_print_output(false),
    enable_timings(false),
    run_cuda_rand(false),
    num_blocks(0),
    num_threads(0)
{}

Args::~Args() {}

bool Args::ParseArgs(int argc, char **argv) {
    --argc; ++argv;
    while (argc > 0) {
        if (std::strcmp(argv[0], "-h") == 0
                || std::strcmp(argv[0], "--help") == 0) {
            Args::DisplayHelp();
            return true;
        } else if (std::strcmp(argv[0], "-p") == 0
                || std::strcmp(argv[0], "--print-output") == 0) {
            enable_print_output = true;
        } else if (std::strcmp(argv[0], "-t") == 0
                || std::strcmp(argv[0], "--timings") == 0) {
            enable_timings = true;
        } else if (std::strcmp(argv[0], "-r") == 0
                || std::strcmp(argv[0], "--rand") == 0) {
            run_cuda_rand = true;
        } else if (std::strcmp(argv[0], "--num-blocks") == 0
                && argc > 1) {
            num_blocks = std::strtoul(argv[1], nullptr, 10);
            if (num_blocks == 0) {
                std::cout << "WARNING: --num-blocks specified but got 0, "
                    "ignoring..." << std::endl;
            }
            --argc; ++argv;
        } else if (std::strcmp(argv[0], "--num-threads") == 0 && argc > 1) {
            num_threads = std::strtoul(argv[1], nullptr, 10);
            if (num_threads == 0) {
                std::cout << "WARNING: --num-threads specified but got 0, "
                    "ignoring..." << std::endl;
            }
            --argc; ++argv;
        } else {
            std::cout << "Ignoring invalid argument \"" << argv[0] << "\"\n";
        }
        --argc; ++argv;
    }

    return false;
}

void Args::DisplayHelp() {
    std::cout << "Usage:\n"
        "  -h | --help\t\t\tprint this help text\n"
        "  -p | --print-output\t\tprint result outputs when running algorithms\n"
        "  -t | --timings\t\tprint timings\n"
        "  -r | --rand\t\t\tRun Cuda Rand Kernel\n"
        "  --num-blocks <blocks>\t\tset the number of blocks to run\n"
        "  --num-threads <threads>\tset the number of threads to run\n"
        "    Note that -p and -t are mutually exclusive.\n"
        "    If both are specified, -t takes precedence.\n";
}
