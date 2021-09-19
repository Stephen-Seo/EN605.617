//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <chrono>

__global__
void add(int *x, int *y, int *out) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = x[thread_idx] + y[thread_idx];
}

__global__
void subtract(int *x, int *y, int *out) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = x[thread_idx] - y[thread_idx];
}

__global__
void mult(int *x, int *y, int *out) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = x[thread_idx] * y[thread_idx];
}

__global__
void mod(int *x, int *y, int *out) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = x[thread_idx] % y[thread_idx];
}

//__global__
//void add_offset(int *x, int *y, int *out) {
//    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
//    out[thread_idx * 4] = x[thread_idx * 4] + y[thread_idx * 4];
//}
//
//__global__
//void subtract_offset(int *x, int *y, int *out) {
//    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
//    out[thread_idx * 4 + 1] = x[thread_idx * 4 + 1] - y[thread_idx * 4 + 1];
//}
//
//__global__
//void mult_offset(int *x, int *y, int *out) {
//    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
//    out[thread_idx * 4 + 2] = x[thread_idx * 4 + 2] * y[thread_idx * 4 + 2];
//}
//
//__global__
//void mod_offset(int *x, int *y, int *out) {
//    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
//    out[thread_idx * 4 + 3] = x[thread_idx * 4 + 3] % y[thread_idx + 4 + 3];
//}

__global__
void branching_mathexpressions(int *x, int *y, int *out) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int subidx = thread_idx % 4;
    if (subidx == 0) {
        out[thread_idx] = x[thread_idx] + y[thread_idx];
    } else if (subidx == 1) {
        out[thread_idx] = x[thread_idx] - y[thread_idx];
    } else if (subidx == 2) {
        out[thread_idx] = x[thread_idx] * y[thread_idx];
    } else {
        out[thread_idx] = x[thread_idx] % y[thread_idx];
    }
}

__global__
void non_branching_mathexpressions(int *x, int *y, int *out) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int sidx = thread_idx % 4;
    // i=0  (3 ^ i) & 1 >> i = 1
    // i=1  (3 ^ i) & 1 >> i = 0
    // i=2  (3 ^ i) & 1 >> i = 0
    // i=3  (3 ^ i) & 1 >> i = 0
    //
    // i=0  ((3 >> i) & 1) * i = 0
    // i=1  ((3 >> i) & 1) * i = 1
    // i=2  ((3 >> i) & 1) * i = 0
    // i=3  ((3 >> i) & 1) * i = 0
    //
    // i=0  (3 ^ (3-i)) & 1 >> (3-i) = 0
    // i=1  (3 ^ (3-i)) & 1 >> (3-i) = 0
    // i=2  (3 ^ (3-i)) & 1 >> (3-i) = 0
    // i=3  (3 ^ (3-i)) & 1 >> (3-i) = 1
    //
    // i=0  ((3 >> (3-i)) & 1) * (3-i) = 0
    // i=1  ((3 >> (3-i)) & 1) * (3-i) = 0
    // i=2  ((3 >> (3-i)) & 1) * (3-i) = 1
    // i=3  ((3 >> (3-i)) & 1) * (3-i) = 0
    out[thread_idx] =
        (((3 ^ sidx) & 1) >> sidx)         * (x[thread_idx] + y[thread_idx])
      + (((3 >> sidx) & 1) * sidx)         * (x[thread_idx] - y[thread_idx])
      + (((3 >> (3-sidx)) & 1) * (3-sidx)) * (x[thread_idx] * y[thread_idx])
      + (((3 ^ (3-sidx)) & 1) >> (3-sidx)) * (x[thread_idx] % y[thread_idx]);
}

//__global__
//void branching_mulmod(int *x, int *y, int *out) {
//    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
//    if (thread_idx % 2 == 0) {
//        out[thread_idx] = x[thread_idx] * y[thread_idx];
//    } else {
//        out[thread_idx] = x[thread_idx] % y[thread_idx];
//    }
//}
//
//__global__
//void non_branching_mulmod(int *x, int *y, int *out) {
//    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
//    const unsigned int mod_result = thread_idx % 2;
//    out[thread_idx] = x[thread_idx] % y[thread_idx] * mod_result
//        + x[thread_idx] * y[thread_idx] * (1 - mod_result);
//}

enum MathFnToUse {
    MFN_ADD,
    MFN_SUB,
    MFN_MUL,
    MFN_MOD,
    MFN_BRANCHING,
    MFN_NON_BRANCHING
};

bool checkError(cudaError_t cudaError) {
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaError));
        return true;
    }
    return false;
}

int main(int argc, char** argv) {
	// read command line arguments
	int totalThreads = 512;
	int blockSize = 256;
    MathFnToUse fn = MFN_ADD;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
        fprintf(stderr, "Got input %3u for total threads\n", totalThreads);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
        fprintf(stderr, "Got input %3u for blockSize\n", blockSize);
	}
    if (argc >= 4) {
        if (argv[3][0] == 'a') {
            fn = MFN_ADD;
            fprintf(stderr, "Using \"add\" fn\n");
        } else if (argv[3][0] == 's') {
            fn = MFN_SUB;
            fprintf(stderr, "Using \"subtract\" fn\n");
        } else if (argv[3][0] == 'm') {
            fn = MFN_MUL;
            fprintf(stderr, "Using \"multiply\" fn\n");
        } else if (argv[3][0] == 'o') {
            fn = MFN_MOD;
            fprintf(stderr, "Using \"modulus\" fn\n");
        } else if (argv[3][0] == 'b') {
            fn = MFN_BRANCHING;
            fprintf(stderr, "Using \"branching\" fn\n");
        } else if (argv[3][0] == 'n') {
            fn = MFN_NON_BRANCHING;
            fprintf(stderr, "Using \"non-branching\" fn\n");
        } else {
            fprintf(stderr,
                    "Invalid third argument, using \"add\" fn by default\n");
        }
    }

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
    if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		fprintf(stderr, "Warning: Total thread count is not evenly divisible by"
               " the block size\n");
		fprintf(stderr,
                "The total number of threads will be rounded up to %d\n",
                totalThreads);
	}
    fprintf(stderr, "totalThreads == %3u, numBlocks == %3u, blockSize == %3u\n",
        totalThreads, numBlocks, blockSize);

    int *hostX =
        (int*)malloc(sizeof(int) * totalThreads);
    int *hostY =
        (int*)malloc(sizeof(int) * totalThreads);
    int *hostOut =
        (int*)malloc(sizeof(int) * totalThreads);

    fprintf(stderr, "Setting host values...\n");
    srand(time(NULL));
    for(unsigned int i = 0; i < totalThreads; ++i) {
        hostX[i] = i;
        hostY[i] = rand() % 4;
        hostOut[i] = 0;
    }

    int *x;
    int *y;
    int *out;

    fprintf(stderr, "cudaMalloc...\n");
    cudaMalloc((void**)&x, totalThreads * sizeof(int));
    checkError(cudaPeekAtLastError());
    cudaMalloc((void**)&y, totalThreads * sizeof(int));
    checkError(cudaPeekAtLastError());
    cudaMalloc((void**)&out, totalThreads * sizeof(int));
    checkError(cudaPeekAtLastError());

    fprintf(stderr, "cudaMemcpy...\n");
    cudaMemcpy(x, hostX, totalThreads * sizeof(int),
            cudaMemcpyHostToDevice);
    checkError(cudaPeekAtLastError());
    cudaMemcpy(y, hostY, totalThreads * sizeof(int),
            cudaMemcpyHostToDevice);
    checkError(cudaPeekAtLastError());
    cudaMemcpy(out, hostOut, totalThreads * sizeof(int),
            cudaMemcpyHostToDevice);
    checkError(cudaPeekAtLastError());

    switch (fn) {
    case MFN_ADD: {
        fprintf(stderr, "Executing \"add\"...\n");
        auto start_clock = std::chrono::high_resolution_clock::now();
        add<<<numBlocks, totalThreads>>>(x, y, out);
        cudaError_t cudaError = cudaDeviceSynchronize();
        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end_clock - start_clock);
        fprintf(stderr, "Duration of \"add\" nanos: %lld\n", duration.count());
        checkError(cudaError);
      } break;
    case MFN_SUB: {
        fprintf(stderr, "Executing \"sub\"...\n");
        auto start_clock = std::chrono::high_resolution_clock::now();
        subtract<<<numBlocks, totalThreads>>>(x, y, out);
        cudaError_t cudaError = cudaDeviceSynchronize();
        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end_clock - start_clock);
        fprintf(stderr, "Duration of \"sub\" nanos: %lld\n", duration.count());
        checkError(cudaError);
      } break;
    case MFN_MUL: {
        fprintf(stderr, "Executing \"mul\"...\n");
        auto start_clock = std::chrono::high_resolution_clock::now();
        mult<<<numBlocks, totalThreads>>>(x, y, out);
        cudaError_t cudaError = cudaDeviceSynchronize();
        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end_clock - start_clock);
        fprintf(stderr, "Duration of \"mul\" nanos: %lld\n", duration.count());
        checkError(cudaError);
      } break;
    case MFN_MOD: {
        fprintf(stderr, "Executing \"mod\"...\n");
        auto start_clock = std::chrono::high_resolution_clock::now();
        mod<<<numBlocks, totalThreads>>>(x, y, out);
        cudaError_t cudaError = cudaDeviceSynchronize();
        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end_clock - start_clock);
        fprintf(stderr, "Duration of \"mod\" nanos: %lld\n", duration.count());
        checkError(cudaError);
      } break;
    case MFN_BRANCHING: {
        fprintf(stderr, "Executing \"branching\" (%u blocks, %u threads)...\n",
                numBlocks, totalThreads);
        unsigned long long count = 0;
        for (unsigned int i = 0; i < 25; ++i) {
            auto start_clock = std::chrono::high_resolution_clock::now();
            branching_mathexpressions<<<numBlocks, totalThreads>>>(x, y, out);
            cudaError_t cudaError = cudaDeviceSynchronize();
            auto end_clock = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>
                    (end_clock - start_clock);
            if (i > 4) {
                fprintf(stderr, "Duration of \"branching\" nanos: %lld\n",
                        duration.count());
                count += duration.count();
            }
            checkError(cudaError);
        } // for
        fprintf(stderr, "Average of duration nanos == %llu\n", count / 20);
        } break;
    case MFN_NON_BRANCHING: {
        fprintf(stderr,
                "Executing \"non-branching\" (%u blocks, %u threads)...\n",
                numBlocks, totalThreads);
        unsigned long long count = 0;
        for (unsigned int i = 0; i < 25; ++i) {
            auto start_clock = std::chrono::high_resolution_clock::now();
            non_branching_mathexpressions<<<numBlocks, totalThreads>>>(x,
                                                                       y,
                                                                       out);
            cudaError_t cudaError = cudaDeviceSynchronize();
            auto end_clock = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>
                    (end_clock - start_clock);
            if (i > 4) {
                fprintf(stderr, "Duration of \"non-branching\" nanos: %lld\n",
                        duration.count());
                count += duration.count();
            }
            checkError(cudaError);
        } // for
        fprintf(stderr, "Average of duration nanos == %llu\n", count / 20);
        } break;
    default:
        fprintf(stderr, "ERROR: Invalid state\n");
        cudaFree(x);
        cudaFree(y);
        cudaFree(out);
        free(hostX);
        free(hostY);
        free(hostOut);
        return 1;
    }

    checkError(cudaPeekAtLastError());

    fprintf(stderr, "Copying result to host...\n");
    cudaMemcpy(hostOut, out, totalThreads * sizeof(int),
            cudaMemcpyDeviceToHost);

    fprintf(stderr, "Freeing device memory...\n");
    cudaFree(x);
    cudaFree(y);
    cudaFree(out);

    // print results
    for(unsigned int j = 0; j <= totalThreads / 4; ++j) {
        if (j * 4 < totalThreads) {
            printf("%4u: %4d\t", j * 4, hostOut[j * 4]);
            if (1 + j * 4 < totalThreads) {
                printf("%4u: %4d\t", 1 + j * 4, hostOut[1 + j * 4]);
                if (2 + j * 4 < totalThreads) {
                    printf("%4u: %4d\t", 2 + j * 4, hostOut[2 + j * 4]);
                    if (3 + j * 4 < totalThreads) {
                        printf("%4u: %4d\n", 3 + j * 4, hostOut[3 + j * 4]);
                    } else {
                        printf("\n");
                        break;
                    }
                } else {
                    printf("\n");
                    break;
                }
            } else {
                printf("\n");
                break;
            }
        } else {
            printf("\n");
            break;
        }
    }

    fprintf(stderr, "Freeing host memory...\n");
    free(hostX);
    free(hostY);
    free(hostOut);

    return EXIT_SUCCESS;
}

// vim: cindent: ts=4: sw=4: et
