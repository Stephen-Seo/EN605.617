#ifndef IGPUP_MODULE_8_ARG_PARSE_H
#define IGPUP_MODULE_8_ARG_PARSE_H

struct Args {
    Args();
    ~Args();

    // enable copy
    Args(const Args &other) = default;
    Args& operator=(const Args &other) = default;

    // enable move
    Args(Args &&other) = default;
    Args& operator=(Args &&other) = default;

    /// Returns true if help was printed
    bool ParseArgs(int argc, char **argv);
    static void DisplayHelp();

    bool enable_print_output;
    bool enable_timings;
    bool run_cuda_rand;
    bool run_cuda_blas;
    unsigned int num_blocks;
    unsigned int num_threads;
    unsigned int num_blas_w;
    unsigned int num_blas_h;
    unsigned int num_blas_i;
};

#endif
