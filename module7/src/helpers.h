#ifndef IGPUP_MODULE_7_HELPERS_H
#define IGPUP_MODULE_7_HELPERS_H

namespace Helpers {
    void setUpHostMemory(int **host_a, int **host_b, int **host_out,
                         unsigned int block_size, unsigned int thread_size);
    void setUpDeviceMemory(int **device_a, int **device_b, int **device_out,
                           unsigned int block_size, unsigned int thread_size);

    void cleanupHostMemory(int **host_a, int **host_b, int **host_out);
    void cleanupDeviceMemory(int **device_a, int **device_b, int **device_out);

    void setUpStreamAndEvents(cudaStream_t *stream,
                              cudaEvent_t *event_start,
                              cudaEvent_t *event_end);
    void cleanupStreamAndEvents(cudaStream_t stream,
                                cudaEvent_t event_start,
                                cudaEvent_t event_end);

    void asyncMemcpyToDevice(int *host_a, int *host_b,
                             int *device_a, int *device_b,
                             cudaStream_t stream, cudaEvent_t event_start,
                             unsigned int block_size, unsigned int thread_size);

    void invokeKernel(int *device_a, int *device_b, int *device_out,
                      unsigned int block_size, unsigned int thread_size,
                      cudaStream_t stream);

    void asyncMemcpyToHost(int *host_out, int *device_out,
                           cudaStream_t stream, cudaEvent_t event_end,
                           unsigned int block_size, unsigned int thread_size);

    void getEventElapsedTime(cudaEvent_t event_start, cudaEvent_t event_end,
                             float *time_out);

} // namespace Helpers

#endif
