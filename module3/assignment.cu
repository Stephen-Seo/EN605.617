//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__
void add(unsigned int *x, unsigned int *y, unsigned int *out) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = x[thread_idx] + y[thread_idx];
}

__global__
void subtract(unsigned int *x, unsigned int *y, unsigned int *out) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = x[thread_idx] - y[thread_idx];
}

__global__
void mult(unsigned int *x, unsigned int *y, unsigned int *out) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = x[thread_idx] * y[thread_idx];
}

__global__
void mod(unsigned int *x, unsigned int *y, unsigned int *out) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = x[thread_idx] % y[thread_idx];
}

int main(int argc, char** argv) {
	// read command line arguments
	int totalThreads = 512;
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
        printf("Got input %3u for total threads\n", totalThreads);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
        printf("Got input %3u for blockSize\n", blockSize);
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

    unsigned int *hostX =
        (unsigned int*)malloc(sizeof(unsigned int) * totalThreads);
    unsigned int *hostY =
        (unsigned int*)malloc(sizeof(unsigned int) * totalThreads);
    unsigned int *hostOut =
        (unsigned int*)malloc(sizeof(unsigned int) * totalThreads);

    printf("Setting host values...\n");
    srand(time(NULL));
    for(unsigned int i = 0; i < totalThreads; ++i) {
        hostX[i] = i;
        hostY[i] = rand() % 4;
        hostOut[i] = 0;
    }

    unsigned int *x;
    unsigned int *y;
    unsigned int *out;

    printf("cudaMalloc...\n");
    cudaMalloc((void**)&x, totalThreads * sizeof(unsigned int));
    cudaMalloc((void**)&y, totalThreads * sizeof(unsigned int));
    cudaMalloc((void**)&out, totalThreads * sizeof(unsigned int));

    printf("cudaMemcpy...\n");
    cudaMemcpy(x, hostX, totalThreads * sizeof(unsigned int),
            cudaMemcpyHostToDevice);
    cudaMemcpy(y, hostY, totalThreads * sizeof(unsigned int),
            cudaMemcpyHostToDevice);

    printf("Executing \"add\"...\n");
    add<<<numBlocks, totalThreads>>>(x, y, out);

    printf("Copying result to host...\n");
    cudaMemcpy(hostOut, out, totalThreads * sizeof(unsigned int),
            cudaMemcpyDeviceToHost);

    printf("Freeing device memory...\n");
    cudaFree(x);
    cudaFree(y);
    cudaFree(out);

    for(unsigned int j = 0; j < totalThreads / 4; ++j) {
        printf("%3u: %4u\t%3u: %4u\t%3u: %4u\t %3u: %4u\n",
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
