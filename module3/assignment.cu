//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

enum MathFnToUse {
    MFN_ADD,
    MFN_SUB,
    MFN_MUL,
    MFN_MOD
};

int main(int argc, char** argv) {
	// read command line arguments
	int totalThreads = 512;
	int blockSize = 256;
    MathFnToUse fn = MFN_ADD;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
        printf("Got input %3u for total threads\n", totalThreads);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
        printf("Got input %3u for blockSize\n", blockSize);
	}
    if (argc >= 4) {
        if (argv[3][0] == 'a') {
            fn = MFN_ADD;
        } else if (argv[3][0] == 's') {
            fn = MFN_SUB;
        } else if (argv[3][0] == 'm') {
            fn = MFN_MUL;
        } else if (argv[3][0] == 'o') {
            fn = MFN_MOD;
        }
    }

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the "
               "block size\n");
		printf("The total number of threads will be rounded up to %d\n",
                totalThreads);
	}
    printf("totalThreads == %3u, numBlocks == %3u, blockSize == %3u\n",
        totalThreads, numBlocks, blockSize);

    int *hostX =
        (int*)malloc(sizeof(int) * totalThreads);
    int *hostY =
        (int*)malloc(sizeof(int) * totalThreads);
    int *hostOut =
        (int*)malloc(sizeof(int) * totalThreads);

    printf("Setting host values...\n");
    srand(time(NULL));
    for(unsigned int i = 0; i < totalThreads; ++i) {
        hostX[i] = i;
        hostY[i] = rand() % 4;
        hostOut[i] = 0;
    }

    int *x;
    int *y;
    int *out;

    printf("cudaMalloc...\n");
    cudaMalloc((void**)&x, totalThreads * sizeof(int));
    cudaMalloc((void**)&y, totalThreads * sizeof(int));
    cudaMalloc((void**)&out, totalThreads * sizeof(int));

    printf("cudaMemcpy...\n");
    cudaMemcpy(x, hostX, totalThreads * sizeof(int),
            cudaMemcpyHostToDevice);
    cudaMemcpy(y, hostY, totalThreads * sizeof(int),
            cudaMemcpyHostToDevice);

    switch (fn) {
    case MFN_ADD:
        printf("Executing \"add\"...\n");
        add<<<numBlocks, totalThreads>>>(x, y, out);
        break;
    case MFN_SUB:
        printf("Executing \"sub\"...\n");
        subtract<<<numBlocks, totalThreads>>>(x, y, out);
        break;
    case MFN_MUL:
        printf("Executing \"mul\"...\n");
        mult<<<numBlocks, totalThreads>>>(x, y, out);
        break;
    case MFN_MOD:
        printf("Executing \"mod\"...\n");
        mod<<<numBlocks, totalThreads>>>(x, y, out);
        break;
    default:
        printf("ERROR: Invalid state\n");
        cudaFree(x);
        cudaFree(y);
        cudaFree(out);
        free(hostX);
        free(hostY);
        free(hostOut);
        return 1;
    }

    printf("Copying result to host...\n");
    cudaMemcpy(hostOut, out, totalThreads * sizeof(int),
            cudaMemcpyDeviceToHost);

    printf("Freeing device memory...\n");
    cudaFree(x);
    cudaFree(y);
    cudaFree(out);

    for(unsigned int j = 0; j < totalThreads / 4; ++j) {
        printf("%3u: %4d\t%3u: %4d\t%3u: %4d\t %3u: %4d\n",
            j*4, hostOut[j*4],
            1+j*4,hostOut[1+j*4],
            2+j*4,hostOut[2+j*4],
            3+j*4,hostOut[3+j*4]);
    }

    printf("Freeing host memory...\n");
    free(hostX);
    free(hostY);
    free(hostOut);

    return EXIT_SUCCESS;
}

// vim: cindent: ts=4: sw=4: et
