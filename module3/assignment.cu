//Based on the work of Andrew Krepps
#include <stdio.h>

#define ARRAY_SIZE 256
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * ARRAY_SIZE)

__global__
void do_work(unsigned int *x, unsigned int *y, unsigned int *out) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = x[thread_idx] + y[thread_idx];
}

unsigned int hostX[ARRAY_SIZE];
unsigned int hostY[ARRAY_SIZE];
unsigned int hostOut[ARRAY_SIZE];

int main(int argc, char** argv) {
	// read command line arguments
	int totalThreads = ARRAY_SIZE;
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
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
    printf("totalThreads == %3u, numBlocks == %3u, blockSize == %3u\n",
        totalThreads, numBlocks, blockSize);

    printf("Setting host values...\n");
    for(unsigned int i = 0; i < ARRAY_SIZE; ++i) {
        hostX[i] = i;
        hostY[i] = i + i;
        hostOut[i] = 0;
    }

    unsigned int *x;
    unsigned int *y;
    unsigned int *out;

    printf("cudaMalloc...\n");
    cudaMalloc((void**)&x, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&y, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&out, ARRAY_SIZE_IN_BYTES);

    printf("cudaMemcpy...\n");
    cudaMemcpy(x, hostX, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(y, hostY, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);

    printf("Doing work...\n");
    do_work<<<numBlocks, totalThreads>>>(x, y, out);

    printf("Copying result to host...\n");
    cudaMemcpy(hostOut, out, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

    printf("Freeing device memory...\n");
    cudaFree(x);
    cudaFree(y);
    cudaFree(out);

    for(unsigned int j = 0; j < ARRAY_SIZE / 4; ++j) {
        printf("%3u: %4u\t%3u: %4u\t%3u: %4u\t %3u: %4u\n",
            j*4, hostOut[j*4],
            1+j*4,hostOut[1+j*4],
            2+j*4,hostOut[2+j*4],
            3+j*4,hostOut[3+j*4]);
    }

    return EXIT_SUCCESS;
}
