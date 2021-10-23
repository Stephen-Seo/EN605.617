#include "arg_parse.h"

#include <cstring>
#include <iostream>

Args::Args() :
    enable_print_output(false),
    enable_timings(false),
    run_cuda_rand(false),
    run_cuda_blas(false),
    num_blocks(0),
    num_threads(0),
    num_blas_w(0),
    num_blas_h(0),
    num_blas_i(0)
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
        } else if (std::strcmp(argv[0], "-b") == 0
                || std::strcmp(argv[0], "--blas") == 0) {
            run_cuda_blas = true;
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
        } else if (std::strcmp(argv[0], "-m") == 0 && argc > 1) {
            num_blas_w = std::strtoul(argv[1], nullptr, 10);
            if (num_blas_w == 0) {
                std::cout << "WARNING: -m specified but got 0, ignoring..."
                    << std::endl;
            }
            --argc; ++argv;
        } else if (std::strcmp(argv[0], "-n") == 0 && argc > 1) {
            num_blas_h = std::strtoul(argv[1], nullptr, 10);
            if (num_blas_h == 0) {
                std::cout << "WARNING: -n specified but got 0, ignoring..."
                    << std::endl;
            }
            --argc; ++argv;
        } else if (std::strcmp(argv[0], "-i") == 0 && argc > 1) {
            num_blas_i = std::strtoul(argv[1], nullptr, 10);
            if (num_blas_i == 0) {
                std::cout << "WARNING: -i specified but got 0, ignoring..."
                    << std::endl;
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
        "  -p | --print-output\t\tprint result outputs when running algorithms"
        "\n"
        "  -t | --timings\t\tprint timings\n"
        "  -r | --rand\t\t\tRun Cuda Rand Kernel\n"
        "  -b | --blas\t\t\tRun Cuda BLAS Kernel\n"
        "  --num-blocks <blocks>\t\tset the number of blocks to run\n"
        "  --num-threads <threads>\tset the number of threads to run\n"
        "  -m <width>\t\t\tset width of result matrix, height of first matrix\n"
        "  -n <height>\t\t\tset height of result matrix, width of second matrix"
        "\n"
        "  -i <value>\t\t\tset width of first matrix, height of second matrix\n"
        "    Note that -p and -t are mutually exclusive.\n"
        "    If both are specified, -t takes precedence.\n"
        "    Note that -r and -b are mutually exclusive.\n"
        "    If both are specified, -r takes precedence.\n";
}
