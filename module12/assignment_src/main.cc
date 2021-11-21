#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "arg_parse.h"

constexpr unsigned int kBufferSize = 16;
constexpr unsigned int kBufferWidth = 4;
constexpr unsigned int kSubBufferSize = 4;
constexpr unsigned int kActualBufferSize = kBufferSize + kSubBufferSize;

template <typename Iterable>
void PrintIterable(Iterable iter, unsigned int row_length,
                   unsigned int total_size = 0) {
  if (total_size == 0) {
    // Just print the entire Iterable
    for (unsigned int i = 0; i < iter.size(); ++i) {
      std::cout << std::setw(4) << iter.at(i) << ' ';
      if ((i + 1) % row_length == 0) {
        std::cout << '\n';
      }
    }
  } else {
    // Print up to "total_size" entries
    for (unsigned int i = 0; i < total_size; ++i) {
      std::cout << std::setw(4) << iter.at(i) << ' ';
      if ((i + 1) % row_length == 0) {
        std::cout << '\n';
      }
    }
  }
  std::cout << std::endl;
}

cl_int GetPlatformID(cl_platform_id *platform_id) {
  cl_int err_num = clGetPlatformIDs(1, platform_id, nullptr);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to get OpenCL platform" << std::endl;
  }
  return err_num;
}

cl_int GetDeviceID(const cl_platform_id *platform_id, cl_device_id *device_id) {
  cl_int err_num =
      clGetDeviceIDs(*platform_id, CL_DEVICE_TYPE_GPU, 1, device_id, nullptr);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to get OpenCL device" << std::endl;
  }

  return err_num;
}

cl_int GetContext(cl_platform_id *platform_id, cl_device_id *device_id,
                  cl_context *context) {
  cl_context_properties context_properties[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)*platform_id, 0};
  cl_int err_num = CL_SUCCESS;
  *context = clCreateContext(context_properties, 1, device_id, nullptr, nullptr,
                             &err_num);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to get OpenCL context" << std::endl;
  }

  return err_num;
}

cl_int GetProgram(cl_context *context, cl_device_id *device_id,
                  cl_program *program, const char *filename) {
  cl_int err_num = CL_SUCCESS;

  // get program source
  std::string program_source;
  {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
      std::cout << "ERROR: Failed to open \"" << filename << "\"" << std::endl;
      return ~CL_SUCCESS;
    }
    program_source = std::string(std::istreambuf_iterator<char>(ifs),
                                 std::istreambuf_iterator<char>{});
  }
  if (program_source.empty()) {
    std::cout << "ERROR: Failed to read \"" << filename << "\"" << std::endl;
    return ~CL_SUCCESS;
  }

  // Create program object
  {
    const char *program_source_cstr = program_source.c_str();
    std::size_t program_source_length = program_source.size();
    *program = clCreateProgramWithSource(*context, 1, &program_source_cstr,
                                         &program_source_length, &err_num);
    if (err_num != CL_SUCCESS) {
      std::cout << "ERROR: Failed to create OpenCL program" << std::endl;
      return err_num;
    }
  }

  // Compile program in program object
  err_num = clBuildProgram(*program, 1, device_id, nullptr, nullptr, nullptr);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to build OpenCL program" << std::endl;
    std::vector<char> build_log;
    build_log.resize(16384);
    build_log.at(16383) = 0;
    clGetProgramBuildInfo(*program, *device_id, CL_PROGRAM_BUILD_LOG,
                          build_log.size(), build_log.data(), nullptr);
    std::cout << build_log.data();
    clReleaseProgram(*program);
    return err_num;
  }

  return CL_SUCCESS;
}

void CleanupPrograms(std::vector<cl_program> *programs) {
  for (cl_program program : *programs) {
    clReleaseProgram(program);
  }
  programs->clear();
}

cl_int GetKernel(cl_program *program, cl_kernel *kernel, const char *name) {
  cl_int err_num = CL_SUCCESS;
  *kernel = clCreateKernel(*program, name, &err_num);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to create OpenCL command queue" << std::endl;
  }

  return err_num;
}

void CleanupKernels(std::vector<cl_kernel> *kernels) {
  for (cl_kernel kernel : *kernels) {
    clReleaseKernel(kernel);
  }
  kernels->clear();
}

cl_int GetQueue(cl_context *context, cl_device_id *device_id,
                cl_command_queue *queue) {
  cl_int err_num = CL_SUCCESS;
  *queue = clCreateCommandQueue(*context, *device_id, 0, &err_num);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to create OpenCL command queue" << std::endl;
  }

  return err_num;
}

void InitInputData(std::vector<int> *host_buffer) {
  host_buffer->clear();
  host_buffer->reserve(kActualBufferSize);
  unsigned int i = 0;
  // Populate first 16 entries
  for (; i < kBufferSize; ++i) {
    host_buffer->push_back(i);
  }
  // Populate repeat of first entries up to kActualBufferSize size
  // This is so that the sub-buffer doesn't go "out of range"
  for (; i < kActualBufferSize; ++i) {
    host_buffer->push_back(host_buffer->at(i - kBufferSize));
  }
}

cl_int SetUpReadBuffer(cl_context *context, std::vector<int> *host_buffer,
                       cl_mem *read_buffer) {
  cl_int err_num = CL_SUCCESS;
  *read_buffer = clCreateBuffer(
      *context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(int) * kActualBufferSize, host_buffer->data(), &err_num);

  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to create OpenCL read_bufer" << std::endl;
  }

  return err_num;
}

cl_int SetUpSubBuffers(cl_context *context, cl_command_queue *queue,
                       cl_mem *read_buffer, std::vector<cl_mem> *sub_buffers) {
  cl_int err_num = CL_SUCCESS;
  // The following code snippet fails on my system due to alignment error
  // (CL_MISALIGNED_SUB_BUFFER_OFFSET) when trying to create the sub-buffer with
  // a non-zero origin value. My system requires an alignment of 128 bytes (1024
  // bits), so this fails when requesting a sub-buffer with an offset of 4
  // bytes. The alternative is to create regular buffers and copy the
  // sub-regions to them.
  //
  // cl_buffer_region region = {
  //  0,
  //  sizeof(int) * kSubBufferSize
  //};
  // for (unsigned int i = 0; i < kBufferSize; ++i) {
  //  region.origin = sizeof(int) * i;
  //  cl_mem sub_buffer = clCreateSubBuffer(*read_buffer, CL_MEM_READ_ONLY,
  //  CL_BUFFER_CREATE_TYPE_REGION, &region, &err_num); if (err_num !=
  //  CL_SUCCESS) {
  //    std::cout << "ERROR: Failed to create OpenCL sub-buffer (iteration "
  //              << i << ", err_num " << err_num << ")"
  //              << std::endl;
  //    break;
  //  }
  //  sub_buffers->push_back(sub_buffer);
  //}

  std::vector<cl_event> copy_events;

  for (unsigned int i = 0; i < kBufferSize; ++i) {
    // Create "sub-buffer" (just a regular buffer actually)
    cl_mem buffer =
        clCreateBuffer(*context, CL_MEM_READ_ONLY, sizeof(int) * kSubBufferSize,
                       nullptr, &err_num);
    if (err_num != CL_SUCCESS) {
      std::cout << "ERROR: Failed to create OpenCL \"sub-buffer\"" << std::endl;
      break;
    }

    // copy data with offset to "sub-buffer"
    cl_event event;
    err_num =
        clEnqueueCopyBuffer(*queue, *read_buffer, buffer, i * sizeof(int), 0,
                            sizeof(int) * kSubBufferSize, 0, nullptr, &event);
    if (err_num != CL_SUCCESS) {
      std::cout << "ERROR: Failed to copy from read_buffer to \"sub-buffer\""
                   " (err_num is "
                << err_num << ")" << std::endl;
      break;
    }
    copy_events.push_back(event);
    sub_buffers->push_back(buffer);
  }

  // wait for buffer-to-buffer copies to finish
  clWaitForEvents(copy_events.size(), copy_events.data());
  for (cl_event event : copy_events) {
    clReleaseEvent(event);
  }

  return err_num;
}

void CleanUpSubBuffers(std::vector<cl_mem> *sub_buffers) {
  for (cl_mem sub_buffer : *sub_buffers) {
    clReleaseMemObject(sub_buffer);
  }
  sub_buffers->clear();
}

cl_int SetUpWriteBuffer(cl_context *context, cl_mem *write_buffer) {
  cl_int err_num = CL_SUCCESS;
  *write_buffer =
      clCreateBuffer(*context, CL_MEM_WRITE_ONLY, sizeof(float) * kBufferSize,
                     nullptr, &err_num);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to create OpenCL write_bufer" << std::endl;
  }

  return err_num;
}

cl_int SetKernelArgs(cl_kernel *kernel, cl_mem *sub_buffer,
                     cl_mem *write_buffer, int idx) {
  cl_int err_num = CL_SUCCESS;

  err_num = clSetKernelArg(*kernel, 0, sizeof(cl_mem), sub_buffer);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to set OpenCL kernel arg 0" << std::endl;
    return err_num;
  }

  err_num = clSetKernelArg(*kernel, 1, sizeof(cl_mem), write_buffer);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to set OpenCL kernel arg 1" << std::endl;
    return err_num;
  }

  err_num = clSetKernelArg(*kernel, 2, sizeof(int), &kSubBufferSize);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to set OpenCL kernel arg 2" << std::endl;
    return err_num;
  }

  err_num = clSetKernelArg(*kernel, 3, sizeof(int), &idx);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to set OpenCL kernel arg 3" << std::endl;
    return err_num;
  }

  return CL_SUCCESS;
}

cl_int SetKernelArgsAlt(cl_kernel *kernel, cl_mem *read_buffer,
                        cl_mem *write_buffer) {
  cl_int err_num = CL_SUCCESS;

  err_num = clSetKernelArg(*kernel, 0, sizeof(cl_mem), read_buffer);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to set OpenCL alt kernel arg 0" << std::endl;
    return err_num;
  }

  err_num = clSetKernelArg(*kernel, 1, sizeof(cl_mem), write_buffer);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to set OpenCL alt kernel arg 1" << std::endl;
    return err_num;
  }

  err_num = clSetKernelArg(*kernel, 2, sizeof(int), &kSubBufferSize);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to set OpenCL alt kernel arg 2" << std::endl;
    return err_num;
  }

  return err_num;
}

cl_int ExecuteKernel(cl_command_queue *queue, cl_kernel *kernel,
                     cl_event *event, std::size_t size = 1) {
  cl_int err_num = clEnqueueNDRangeKernel(*queue, *kernel, 1, nullptr, &size,
                                          &size, 0, nullptr, event);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to execute OpenCL kernel" << std::endl;
  }

  return err_num;
}

void CleanupEvents(std::vector<cl_event> *events) {
  for (cl_event event : *events) {
    clReleaseEvent(event);
  }
  events->clear();
}

cl_int WaitForKernelsToFinish(std::vector<cl_event> *events) {
  cl_int err_num =
      clWaitForEvents(events->size(), events->data()) != CL_SUCCESS;
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to wait on kernel exeuctions" << std::endl;
  }

  return err_num;
}

cl_int ReadResultData(cl_command_queue *queue, cl_mem *write_buffer,
                      std::vector<float> *host_out_buffer) {
  host_out_buffer->resize(kBufferSize);
  cl_int err_num = clEnqueueReadBuffer(
      *queue, *write_buffer, CL_TRUE, 0, sizeof(float) * kBufferSize,
      host_out_buffer->data(), 0, nullptr, nullptr);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to read result OpenCL buffer" << std::endl;
  }

  return err_num;
}

cl_int DoCellAvgWithSubBuffers(cl_context *context, cl_device_id *device_id,
                               cl_command_queue *queue,
                               std::vector<cl_program> *programs,
                               std::vector<cl_kernel> *kernels,
                               cl_mem *read_buffer, cl_mem *write_buffer,
                               std::vector<cl_mem> *sub_buffers,
                               std::vector<cl_event> *events, bool do_timings) {
  cl_int err_num = CL_SUCCESS;
  // Get programs
  for (unsigned int i = 0; i < kBufferSize; ++i) {
    cl_program program;
    err_num = GetProgram(context, device_id, &program,
                         "assignment_src/cell_avg_once.cl");
    if (err_num != CL_SUCCESS) {
      return err_num;
    }
    programs->push_back(program);
  }

  // Get kernel objects from program objects
  for (unsigned int i = 0; i < kBufferSize; ++i) {
    cl_kernel kernel;
    err_num = GetKernel(&programs->at(i), &kernel, "cell_avg_once");
    if (err_num != CL_SUCCESS) {
      return err_num;
    }
    kernels->push_back(kernel);
  }

  // Set up read-only OpenCL sub-buffers with existing read_buffer
  err_num = SetUpSubBuffers(context, queue, read_buffer, sub_buffers);
  if (err_num != CL_SUCCESS) {
    return err_num;
  }

  // Set up kernel args
  for (unsigned int i = 0; i < kBufferSize; ++i) {
    err_num =
        SetKernelArgs(&kernels->at(i), &sub_buffers->at(i), write_buffer, i);
    if (err_num != CL_SUCCESS) {
      return err_num;
    }
  }

  if (do_timings) {
    // Time kernels
    unsigned long long total_count = 0;
    for (unsigned int i = 0; i < 25; ++i) {
      CleanupEvents(events);
      auto start_time = std::chrono::high_resolution_clock::now();
      for (unsigned int j = 0; j < kBufferSize; ++j) {
        cl_event event;
        err_num = ExecuteKernel(queue, &kernels->at(j), &event, 1);
        if (err_num != CL_SUCCESS) {
          return err_num;
        }
        events->push_back(event);
      }

      // Wait for kernel executions to finish
      err_num = WaitForKernelsToFinish(events);
      if (err_num != CL_SUCCESS) {
        return err_num;
      }
      auto end_time = std::chrono::high_resolution_clock::now();

      if (i > 4) {
        unsigned long long count =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_time -
                                                                 start_time)
                .count();
        std::cout << "Iteration " << i - 4 << " took " << count
                  << " nanoseconds" << std::endl;
        total_count += count;
      }
    }
    std::cout << "Average of 20 runs == " << std::fixed << std::setprecision(2)
              << (double)total_count / 20.0 << std::endl;
  } else {
    // Execute kernels
    for (unsigned int i = 0; i < kBufferSize; ++i) {
      cl_event event;
      err_num = ExecuteKernel(queue, &kernels->at(i), &event, 1);
      if (err_num != CL_SUCCESS) {
        return err_num;
      }
      events->push_back(event);
    }

    // Wait for kernel executions to finish
    err_num = WaitForKernelsToFinish(events);
    if (err_num != CL_SUCCESS) {
      return err_num;
    }
  }

  return CL_SUCCESS;
}

cl_int DoCellAvgWithAlt(cl_context *context, cl_device_id *device_id,
                        cl_command_queue *queue,
                        std::vector<cl_program> *programs,
                        std::vector<cl_kernel> *kernels, cl_mem *read_buffer,
                        cl_mem *write_buffer, std::vector<cl_event> *events,
                        bool do_timings) {
  cl_int err_num = CL_SUCCESS;

  // Get program
  {
    cl_program program;
    err_num =
        GetProgram(context, device_id, &program, "assignment_src/cell_avg.cl");
    if (err_num != CL_SUCCESS) {
      return err_num;
    }
    programs->push_back(program);
  }

  // Get kernel object from program object
  {
    cl_kernel kernel;
    err_num = GetKernel(&programs->at(0), &kernel, "cell_avg");
    if (err_num != CL_SUCCESS) {
      return err_num;
    }
    kernels->push_back(kernel);
  }

  // Set up kernel args
  err_num = SetKernelArgsAlt(&kernels->at(0), read_buffer, write_buffer);
  if (err_num != CL_SUCCESS) {
    return err_num;
  }

  if (do_timings) {
    // Time kernel
    unsigned long long total_count = 0;
    for (unsigned int i = 0; i < 25; ++i) {
      auto start_time = std::chrono::high_resolution_clock::now();
      {
        cl_event event;
        err_num = ExecuteKernel(queue, &kernels->at(0), &event, kBufferSize);
        if (err_num != CL_SUCCESS) {
          return err_num;
        }
        events->push_back(event);
      }

      // Wait for kernel execution to finish
      err_num = WaitForKernelsToFinish(events);
      if (err_num != CL_SUCCESS) {
        return err_num;
      }
      auto end_time = std::chrono::high_resolution_clock::now();

      if (i > 4) {
        unsigned long long count =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_time -
                                                                 start_time)
                .count();
        std::cout << "Iteration " << i - 4 << " took " << count
                  << " nanoseconds" << std::endl;
        total_count += count;
      }
    }
    std::cout << "Average of 20 runs (alt) == " << std::fixed
              << std::setprecision(2) << (double)total_count / 20.0
              << std::endl;
  } else {
    // Execute kernel
    {
      cl_event event;
      err_num = ExecuteKernel(queue, &kernels->at(0), &event, kBufferSize);
      if (err_num != CL_SUCCESS) {
        return err_num;
      }
      events->push_back(event);
    }

    // Wait for kernel execution to finish
    err_num = WaitForKernelsToFinish(events);
    if (err_num != CL_SUCCESS) {
      return err_num;
    }
  }

  return CL_SUCCESS;
}

int main(int argc, char **argv) {
  Args args{};
  if (args.ParseArgs(argc, argv)) {
    return 0;
  }

  std::cout << "Running with "
            << (args.use_alt_kernel_ ? "alt kernel" : "\"sub-buffers\" kernel")
            << (args.do_timings_ ? " with timings enabled"
                                 : " with timings disabled")
            << std::endl;

  cl_int err_num = CL_SUCCESS;
  cl_platform_id platform_id;
  cl_device_id device_id;
  cl_context context;
  std::vector<cl_program> programs;
  std::vector<cl_kernel> kernels;
  cl_command_queue queue;
  cl_mem read_buffer;
  std::vector<cl_mem> sub_buffers;
  cl_mem write_buffer;
  std::vector<cl_event> events;
  std::vector<int> host_buffer;
  std::vector<float> host_out_buffer;

  // get platform_id
  if (GetPlatformID(&platform_id) != CL_SUCCESS) {
    return 1;
  }

  // Get device_id
  if (GetDeviceID(&platform_id, &device_id) != CL_SUCCESS) {
    return 1;
  }

  // Get context
  if (GetContext(&platform_id, &device_id, &context) != CL_SUCCESS) {
    return 1;
  }

  // Get command_queue "queue"
  if (GetQueue(&context, &device_id, &queue) != CL_SUCCESS) {
    CleanupKernels(&kernels);
    CleanupPrograms(&programs);
    clReleaseContext(context);
    return 1;
  }

  // Set up input data
  InitInputData(&host_buffer);

  if (!args.do_timings_) {
    // Output input data
    std::cout << "Input buffer:\n";
    PrintIterable(host_buffer, kBufferWidth, kBufferSize);
  }

  // Set up read-only OpenCL buffer with input data
  if (SetUpReadBuffer(&context, &host_buffer, &read_buffer) != CL_SUCCESS) {
    clReleaseCommandQueue(queue);
    CleanupKernels(&kernels);
    CleanupPrograms(&programs);
    clReleaseContext(context);
    return 1;
  }

  // Set up write-only OpenCL buffer
  if (SetUpWriteBuffer(&context, &write_buffer) != CL_SUCCESS) {
    clReleaseMemObject(read_buffer);
    clReleaseCommandQueue(queue);
    CleanupKernels(&kernels);
    CleanupPrograms(&programs);
    clReleaseContext(context);
    return 1;
  }

  if (args.use_alt_kernel_) {
    if (DoCellAvgWithAlt(&context, &device_id, &queue, &programs, &kernels,
                         &read_buffer, &write_buffer, &events,
                         args.do_timings_) != CL_SUCCESS) {
      CleanupEvents(&events);
      CleanUpSubBuffers(&sub_buffers);
      clReleaseMemObject(write_buffer);
      clReleaseMemObject(read_buffer);
      clReleaseCommandQueue(queue);
      CleanupKernels(&kernels);
      CleanupPrograms(&programs);
      clReleaseContext(context);
      return 1;
    }
  } else {
    if (DoCellAvgWithSubBuffers(&context, &device_id, &queue, &programs,
                                &kernels, &read_buffer, &write_buffer,
                                &sub_buffers, &events,
                                args.do_timings_) != CL_SUCCESS) {
      CleanupEvents(&events);
      CleanUpSubBuffers(&sub_buffers);
      clReleaseMemObject(write_buffer);
      clReleaseMemObject(read_buffer);
      clReleaseCommandQueue(queue);
      CleanupKernels(&kernels);
      CleanupPrograms(&programs);
      clReleaseContext(context);
      return 1;
    }
  }

  // Read result data from "write_buffer"
  if (ReadResultData(&queue, &write_buffer, &host_out_buffer) != CL_SUCCESS) {
    CleanupEvents(&events);
    CleanUpSubBuffers(&sub_buffers);
    clReleaseMemObject(write_buffer);
    clReleaseMemObject(read_buffer);
    clReleaseCommandQueue(queue);
    CleanupKernels(&kernels);
    CleanupPrograms(&programs);
    clReleaseContext(context);
    return 1;
  }

  // Print result data
  if (!args.do_timings_) {
    std::cout << "Output buffer:\n";
    PrintIterable(host_out_buffer, kBufferWidth);
  }

  // cleanup
  CleanupEvents(&events);
  CleanUpSubBuffers(&sub_buffers);
  clReleaseMemObject(write_buffer);
  clReleaseMemObject(read_buffer);
  clReleaseCommandQueue(queue);
  CleanupKernels(&kernels);
  CleanupPrograms(&programs);
  clReleaseContext(context);
  return 0;
}
