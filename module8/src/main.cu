#include <ctime>
#include <cstdio>

#include <cublas.h>

#include "arg_parse.h"
#include "constants.h"
#include "helpers.h"
#include "cuda_rand_kernel.h"

void runCudaRandKernel(unsigned int num_blocks,
                       unsigned int num_threads,
                       bool print_output,
                       bool do_timings) {
    if (!print_output && !do_timings) {
        puts("ERROR: print-output or timings not specified, not running Rand");
        return;
    }

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

void runCudaBlasKernel(unsigned int final_width,
                       unsigned int final_height,
                       unsigned int in_between,
                       bool print_output,
                       bool do_timings) {
    if (!print_output && !do_timings) {
        puts("ERROR: print-output or timings not specified, not running BLAS");
        return;
    }

    float *first_matrix_host;
    float *second_matrix_host;
    float *result_matrix_host;
    float *first_matrix_device;
    float *second_matrix_device;
    float *result_matrix_device;

    helpers::InitMatrices(&first_matrix_host,
                         &second_matrix_host,
                         &result_matrix_host,
                         &first_matrix_device,
                         &second_matrix_device,
                         &result_matrix_device,
                         final_width,
                         final_height,
                         in_between);

    if (print_output) {
        printf("Matrix A:\n");
        helpers::PrintMatrix(first_matrix_host, in_between, final_height);
        printf("Matrix B:\n");
        helpers::PrintMatrix(second_matrix_host, final_width, in_between);
    }

    cublasSgemm('n', 'n',               // non-transpose
                final_height,           // rows of A and C
                final_width,            // cols of B and C
                in_between,             // cols of A and rows of B
                1,                      // scalar type used for multiplication
                first_matrix_device,    // matrix A
                final_height,           // leading dimension of 2d array (A)
                second_matrix_device,   // matrix B
                in_between,             // leading dimension of 2d array (B)
                0,                      // scalar type used for multiplication
                result_matrix_device,   // matrix C
                final_height);          // leading dimension of 2d array (C)

    cublasGetMatrix(final_height,           // rows
                    final_width,            // cols
                    sizeof(float),          // size of elem
                    result_matrix_device,   // device matrix
                    final_height,           // leading dimension of 2d array (C)
                    result_matrix_host,     // host matrix
                    final_height);          // leading dimesnion of 2d array (C)

    if (print_output) {
        printf("Matrix C:\n");
        helpers::PrintMatrix(result_matrix_host, final_width, final_height);
    }

    helpers::FreeMatrices(&first_matrix_host,
                         &second_matrix_host,
                         &result_matrix_host,
                         &first_matrix_device,
                         &second_matrix_device,
                         &result_matrix_device);

}

int main(int argc, char **argv) {
    Args args{};
    if (args.ParseArgs(argc, argv)) {
        // help was printed, just exit
        return 0;
    }

    if (args.run_cuda_rand) {
        unsigned int num_blocks = kDefaultBlocks;
        unsigned int num_threads = kDefaultThreads;
        if (args.num_blocks > 0) {
            num_blocks = args.num_blocks;
        }
        if (args.num_threads > 0) {
            num_threads = args.num_threads;
        }
        runCudaRandKernel(num_blocks,
                          num_threads,
                          args.enable_print_output,
                          args.enable_timings);
    } else if (args.run_cuda_blas) {
        unsigned int num_blas_w = kDefaultNumBlasW;
        unsigned int num_blas_h = kDefaultNumBlasH;
        unsigned int num_blas_i = kDefaultNumBlasI;
        if (args.num_blas_w > 0) {
            num_blas_w = args.num_blas_w;
        }
        if (args.num_blas_h > 0) {
            num_blas_h = args.num_blas_h;
        }
        if (args.num_blas_i > 0) {
            num_blas_i = args.num_blas_i;
        }
        runCudaBlasKernel(num_blas_w,
                          num_blas_h,
                          num_blas_i,
                          args.enable_print_output,
                          args.enable_timings);
    } else {
        std::puts("ERROR: Kernel to run not specificed");
        Args::DisplayHelp();
        return 1;
    }

    return 0;
}
