#include <cstdlib>
#include <cstdio>

#include "mathexpressions.h"
#include "constants.h"
#include "helpers.h"

void prepare_xyOut(int *x, int *y, int *out) {
    srand(time(nullptr));
    for(unsigned int i = 0; i < NUM_BLOCKS * NUM_THREADS; ++i) {
        x[i] = i;
        y[i] = rand() % 4;
        out[i] = 0;
    }
}

void prepare_device_xyOut(int **deviceX, int **deviceY, int **deviceOut) {
    cudaMalloc(deviceX, sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    CHECK_ERROR();
    cudaMalloc(deviceY, sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    CHECK_ERROR();
    cudaMalloc(deviceOut, sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    CHECK_ERROR();
}

void host_to_device(int *x, int *y, int *deviceX, int *deviceY) {
    cudaMemcpy(deviceX, x, sizeof(int) * NUM_BLOCKS * NUM_THREADS,
            cudaMemcpyHostToDevice);
    CHECK_ERROR();
    cudaMemcpy(deviceY, y, sizeof(int) * NUM_BLOCKS * NUM_THREADS,
            cudaMemcpyHostToDevice);
    CHECK_ERROR();
}

void device_to_host(int *out, int *deviceOut) {
    cudaMemcpy(out, deviceOut, sizeof(int) * NUM_BLOCKS * NUM_THREADS,
            cudaMemcpyDeviceToHost);
    CHECK_ERROR();
}

void print_results(int *out) {
    for(unsigned int i = 0; i < NUM_BLOCKS * NUM_THREADS; ++i) {
        if (i % 4 < 3 && i + 1 < NUM_BLOCKS * NUM_THREADS) {
            std::printf("%7d ", out[i]);
        } else {
            std::printf("%7d\n", out[i]);
        }
    }
}

void run_shared() {
    int *x = (int*)std::malloc(sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    int *y = (int*)std::malloc(sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    int *out = (int*)std::malloc(sizeof(int) * NUM_BLOCKS * NUM_THREADS);
    prepare_xyOut(x, y, out);

    int *deviceX;
    int *deviceY;
    int *deviceOut;
    prepare_device_xyOut(&deviceX, &deviceY, &deviceOut);

    host_to_device(x, y, deviceX, deviceY);

    mathexpressions_shared<<<NUM_BLOCKS, NUM_THREADS>>>(deviceX,
                                                        deviceY,
                                                        deviceOut);
    CHECK_ERROR();

    device_to_host(out, deviceOut);

    print_results(out);

    cudaFree(deviceOut);
    cudaFree(deviceY);
    cudaFree(deviceX);
    free(out);
    free(y);
    free(x);
}

int main(int argc, char **argv) {
    run_shared();

    return 0;
}

// vim: cindent
