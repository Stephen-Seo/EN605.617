#include "helpers.h"
#include "kernel.h"

#include <cstdlib>
#include <ctime>

void Helpers::setUpHostMemory(int **host_a, int **host_b, int **host_out,
                              unsigned int block_size,
                              unsigned int thread_size) {
    cudaHostAlloc((void**)host_a, sizeof(int) * block_size * thread_size,
                  cudaHostAllocDefault);
    cudaHostAlloc((void**)host_b, sizeof(int) * block_size * thread_size,
                  cudaHostAllocDefault);
    cudaHostAlloc((void**)host_out, sizeof(int) * block_size * thread_size,
                  cudaHostAllocDefault);

    srand(time(nullptr));
    for(unsigned int i = 0; i < block_size * thread_size; ++i) {
        (*host_a)[i] = i;
        (*host_b)[i] = rand() % 4;
    }
}

void Helpers::setUpDeviceMemory(int **device_a, int **device_b,
                                int **device_out,
                                unsigned int block_size,
                                unsigned int thread_size) {
    cudaMalloc((void**)device_a, sizeof(int) * block_size * thread_size);
    cudaMalloc((void**)device_b, sizeof(int) * block_size * thread_size);
    cudaMalloc((void**)device_out, sizeof(int) * block_size * thread_size);
}

void Helpers::cleanupHostMemory(int **host_a, int **host_b, int **host_out) {
    if (host_a && *host_a) {
        cudaFreeHost(*host_a);
        *host_a = nullptr;
    }
    if (host_b && *host_b) {
        cudaFreeHost(*host_b);
        *host_b = nullptr;
    }
    if (host_out && *host_out) {
        cudaFreeHost(*host_out);
        *host_out = nullptr;
    }
}

void Helpers::cleanupDeviceMemory(int **device_a, int **device_b, int **device_out) {
    if (device_a && *device_a) {
        cudaFree(*device_a);
        *device_a = nullptr;
    }
    if (device_b && *device_b) {
        cudaFree(*device_b);
        *device_b = nullptr;
    }
    if (device_out && *device_out) {
        cudaFree(*device_out);
        *device_out = nullptr;
    }
}

void Helpers::setUpStreamAndEvents(cudaStream_t *stream,
                                   cudaEvent_t *event_start,
                                   cudaEvent_t *event_end) {
    cudaStreamCreate(stream);
    cudaEventCreate(event_start);
    cudaEventCreate(event_end);
}

void Helpers::cleanupStreamAndEvents(cudaStream_t stream,
                                     cudaEvent_t event_start,
                                     cudaEvent_t event_end) {
    cudaStreamDestroy(stream);
    cudaEventDestroy(event_start);
    cudaEventDestroy(event_end);
}

void Helpers::asyncMemcpyToDevice(int *host_a, int *host_b,
                                  int *device_a, int *device_b,
                                  cudaStream_t stream,
                                  cudaEvent_t event_start,
                                  unsigned int block_size,
                                  unsigned int thread_size) {
    cudaEventRecord(event_start);
    cudaMemcpyAsync(device_a, host_a, sizeof(int) * block_size * thread_size,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_b, host_b, sizeof(int) * block_size * thread_size,
                    cudaMemcpyHostToDevice, stream);
}

void Helpers::invokeKernel(int *device_a, int *device_b, int *device_out,
                           unsigned int block_size, unsigned int thread_size,
                           cudaStream_t stream) {
    mathexpressions_events_and_streams<<<block_size, thread_size, 0, stream>>>
            (device_a, device_b, device_out);
}

void Helpers::asyncMemcpyToHost(int *host_out, int *device_out,
                                cudaStream_t stream, cudaEvent_t event_end,
                                unsigned int block_size,
                                unsigned int thread_size) {
    cudaMemcpyAsync(host_out, device_out,
                    sizeof(int) * block_size * thread_size,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaEventRecord(event_end);
    cudaEventSynchronize(event_end);
}

void Helpers::getEventElapsedTime(cudaEvent_t event_start,
                                  cudaEvent_t event_end,
                                  float *time_out) {
    cudaEventElapsedTime(time_out, event_start, event_end);
}
