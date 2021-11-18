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

  err_num = clGetPlatformIDs(1, &platform_id, nullptr);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to get OpenCL platform" << std::endl;
    return 1;
  }

  err_num =
      clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to get OpenCL device" << std::endl;
    return 1;
  }

  cl_context_properties context_properties[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
  context = clCreateContext(context_properties, 1, &device_id, nullptr, nullptr,
                            &err_num);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to get OpenCL context" << std::endl;
    return 1;
  }

  std::string program_source;
  {
    std::ifstream ifs("assignment_src/cell_avg.cl");
    if (!ifs.is_open()) {
      std::cout << "ERROR: Failed to open \"assignment_src/cell_avg.cl\""
                << std::endl;
      clReleaseContext(context);
      return 1;
    }
    program_source = std::string(std::istreambuf_iterator<char>(ifs),
                                 std::istreambuf_iterator<char>{});
  }
  if (program_source.empty()) {
    std::cout << "ERROR: Failed to read \"assignment_src/cell_avg.cl\""
              << std::endl;
    clReleaseContext(context);
    return 1;
  }

  {
    const char *program_source_cstr = program_source.c_str();
    std::size_t program_source_length = program_source.size();
    program = clCreateProgramWithSource(context, 1, &program_source_cstr,
                                        &program_source_length, &err_num);
    if (err_num != CL_SUCCESS) {
      std::cout << "ERROR: Failed to create OpenCL program" << std::endl;
      clReleaseContext(context);
      return 1;
    }
  }

  err_num = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to build OpenCL program" << std::endl;
    std::vector<char> build_log;
    build_log.resize(16384);
    build_log.at(16383) = 0;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                          build_log.size(), build_log.data(), nullptr);
    std::cout << build_log.data();
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }

  kernel = clCreateKernel(program, "cell_avg", &err_num);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to create OpenCL command queue" << std::endl;
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }

  queue = clCreateCommandQueue(context, device_id, 0, &err_num);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to create OpenCL command queue" << std::endl;
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }

  host_buffer.clear();
  host_buffer.reserve(kBufferSize);
  for (unsigned int i = 0; i < kBufferSize; ++i) {
    host_buffer.push_back(i);
  }
  std::cout << "Input buffer:\n";
  PrintIterable<decltype(host_buffer)>(host_buffer, kBufferWidth);

  read_buffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * kBufferSize, host_buffer.data(), &err_num);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to create OpenCL read_bufer" << std::endl;
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }

  write_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                sizeof(int) * kBufferSize, nullptr, &err_num);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to create OpenCL write_bufer" << std::endl;
    clReleaseMemObject(read_buffer);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }

  err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), &read_buffer);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to set OpenCL kernel arg 0" << std::endl;
    clReleaseMemObject(write_buffer);
    clReleaseMemObject(read_buffer);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }
  err_num = clSetKernelArg(kernel, 1, sizeof(cl_mem), &write_buffer);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to set OpenCL kernel arg 1" << std::endl;
    clReleaseMemObject(write_buffer);
    clReleaseMemObject(read_buffer);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }
  err_num = clSetKernelArg(kernel, 2, sizeof(int), &kBufferWidth);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to set OpenCL kernel arg 2" << std::endl;
    clReleaseMemObject(write_buffer);
    clReleaseMemObject(read_buffer);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }
  err_num = clSetKernelArg(kernel, 3, sizeof(int), &kBufferSize);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to set OpenCL kernel arg 3" << std::endl;
    clReleaseMemObject(write_buffer);
    clReleaseMemObject(read_buffer);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }

  std::size_t size = kBufferSize;
  err_num = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &size, &size, 0,
                                   nullptr, nullptr);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to execute OpenCL kernel" << std::endl;
    clReleaseMemObject(write_buffer);
    clReleaseMemObject(read_buffer);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }

  err_num = clEnqueueReadBuffer(queue, write_buffer, CL_TRUE, 0,
                                sizeof(int) * kBufferSize, host_buffer.data(),
                                0, nullptr, nullptr);
  if (err_num != CL_SUCCESS) {
    std::cout << "ERROR: Failed to read result OpenCL buffer" << std::endl;
    clReleaseMemObject(write_buffer);
    clReleaseMemObject(read_buffer);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }

  std::cout << "Output buffer:\n";
  PrintIterable<decltype(host_buffer)>(host_buffer, kBufferWidth);

  // cleanup
  clReleaseMemObject(write_buffer);
  clReleaseMemObject(read_buffer);
  clReleaseCommandQueue(queue);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseContext(context);
  return 0;
}
