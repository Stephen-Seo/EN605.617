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

constexpr unsigned int kBufferSize = 16;
constexpr unsigned int kBufferWidth = 4;

template <typename Iterable>
void PrintIterable(Iterable iter, unsigned int row_length) {
  for (unsigned int i = 0; i < iter.size(); ++i) {
    std::cout << std::setw(2) << iter.at(i) << ' ';
    if ((i + 1) % row_length == 0) {
      std::cout << '\n';
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
  cl_int err_num;
  *context = clCreateContext(context_properties, 1, device_id, nullptr, nullptr,
                             &err_num);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to get OpenCL context" << std::endl;
  }

  return err_num;
}

cl_int GetProgram(cl_context *context, cl_device_id *device_id,
                  cl_program *program) {
  cl_int err_num;

  // get program source
  std::string program_source;
  {
    std::ifstream ifs("assignment_src/cell_avg.cl");
    if (!ifs.is_open()) {
      std::cout << "ERROR: Failed to open \"assignment_src/cell_avg.cl\""
                << std::endl;
      return ~CL_SUCCESS;
    }
    program_source = std::string(std::istreambuf_iterator<char>(ifs),
                                 std::istreambuf_iterator<char>{});
  }
  if (program_source.empty()) {
    std::cout << "ERROR: Failed to read \"assignment_src/cell_avg.cl\""
              << std::endl;
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

cl_int GetKernel(cl_program *program, cl_kernel *kernel) {
  cl_int err_num = CL_SUCCESS;
  *kernel = clCreateKernel(*program, "cell_avg", &err_num);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to create OpenCL command queue" << std::endl;
  }

  return err_num;
}

cl_int GetQueue(cl_context *context, cl_device_id *device_id,
                cl_command_queue *queue) {
  cl_int err_num;
  *queue = clCreateCommandQueue(*context, *device_id, 0, &err_num);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to create OpenCL command queue" << std::endl;
  }

  return err_num;
}

void InitInputData(std::vector<int> *host_buffer) {
  host_buffer->clear();
  host_buffer->reserve(kBufferSize);
  for (unsigned int i = 0; i < kBufferSize; ++i) {
    host_buffer->push_back(i);
  }
}

cl_int SetUpReadBuffer(cl_context *context, std::vector<int> *host_buffer,
                       cl_mem *read_buffer) {
  cl_int err_num;
  *read_buffer =
      clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * kBufferSize, host_buffer->data(), &err_num);

  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to create OpenCL read_bufer" << std::endl;
  }

  return err_num;
}

cl_int SetUpWriteBuffer(cl_context *context, cl_mem *write_buffer) {
  cl_int err_num;
  *write_buffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY,
                                 sizeof(int) * kBufferSize, nullptr, &err_num);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to create OpenCL write_bufer" << std::endl;
  }

  return err_num;
}

cl_int SetKernelArgs(cl_kernel *kernel, cl_mem *read_buffer,
                     cl_mem *write_buffer) {
  cl_int err_num;

  err_num = clSetKernelArg(*kernel, 0, sizeof(cl_mem), read_buffer);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to set OpenCL kernel arg 0" << std::endl;
    return err_num;
  }

  err_num = clSetKernelArg(*kernel, 1, sizeof(cl_mem), write_buffer);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to set OpenCL kernel arg 1" << std::endl;
    return err_num;
  }

  err_num = clSetKernelArg(*kernel, 2, sizeof(int), &kBufferWidth);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to set OpenCL kernel arg 2" << std::endl;
    return err_num;
  }

  err_num = clSetKernelArg(*kernel, 3, sizeof(int), &kBufferSize);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to set OpenCL kernel arg 3" << std::endl;
    return err_num;
  }

  return CL_SUCCESS;
}

cl_int ExecuteKernel(cl_command_queue *queue, cl_kernel *kernel) {
  std::size_t size = kBufferSize;
  cl_int err_num = clEnqueueNDRangeKernel(*queue, *kernel, 1, nullptr, &size,
                                          &size, 0, nullptr, nullptr);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to execute OpenCL kernel" << std::endl;
  }

  return err_num;
}

cl_int ReadResultData(cl_command_queue *queue, cl_mem *write_buffer,
                      std::vector<int> *host_buffer) {
  cl_int err_num = clEnqueueReadBuffer(
      *queue, *write_buffer, CL_TRUE, 0, sizeof(int) * kBufferSize,
      host_buffer->data(), 0, nullptr, nullptr);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to read result OpenCL buffer" << std::endl;
  }

  return err_num;
}

int main(int argc, char **argv) {
  cl_int err_num;
  cl_platform_id platform_id;
  cl_device_id device_id;
  cl_context context;
  cl_program program;
  cl_kernel kernel;
  cl_command_queue queue;
  cl_mem read_buffer;
  cl_mem write_buffer;
  std::vector<int> host_buffer;

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

  // Get program
  if (GetProgram(&context, &device_id, &program) != CL_SUCCESS) {
    clReleaseContext(context);
    return 1;
  }

  // Get kernel object from program object
  if (GetKernel(&program, &kernel) != CL_SUCCESS) {
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }

  // Get command_queue "queue"
  if (GetQueue(&context, &device_id, &queue) != CL_SUCCESS) {
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }

  // Set up input data
  InitInputData(&host_buffer);

  // Output input data
  std::cout << "Input buffer:\n";
  PrintIterable(host_buffer, kBufferWidth);

  // Set up read-only OpenCL buffer with input data
  if (SetUpReadBuffer(&context, &host_buffer, &read_buffer) != CL_SUCCESS) {
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }

  // Set up write-only OpenCL buffer
  if (SetUpWriteBuffer(&context, &write_buffer) != CL_SUCCESS) {
    clReleaseMemObject(read_buffer);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }

  // Set up kernel args
  if (SetKernelArgs(&kernel, &read_buffer, &write_buffer) != CL_SUCCESS) {
    clReleaseMemObject(write_buffer);
    clReleaseMemObject(read_buffer);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }

  // Execute kernel
  if (ExecuteKernel(&queue, &kernel) != CL_SUCCESS) {
    clReleaseMemObject(write_buffer);
    clReleaseMemObject(read_buffer);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }

  // Read result data from "write_buffer"
  if (ReadResultData(&queue, &write_buffer, &host_buffer) != CL_SUCCESS) {
    clReleaseMemObject(write_buffer);
    clReleaseMemObject(read_buffer);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }

  // Print result data
  std::cout << "Output buffer:\n";
  PrintIterable(host_buffer, kBufferWidth);

  // cleanup
  clReleaseMemObject(write_buffer);
  clReleaseMemObject(read_buffer);
  clReleaseCommandQueue(queue);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseContext(context);
  return 0;
}
