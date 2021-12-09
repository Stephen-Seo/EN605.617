#include "opencl_context.h"

#include <fstream>
#include <iostream>
#include <vector>

OCLContext::OCLContext()
    : is_context_loaded_(false),
      is_queue_loaded_(false),
      is_program_loaded_(false),
      is_initialization_completed_(false),
      context_(nullptr),
      queue_(nullptr),
      program_(nullptr) {
  cl_int err_number;
  cl_uint num_platforms;
  cl_platform_id platform_id;
  std::size_t device_buffer_size = -1;
  cl_device_id device;
  std::vector<cl_device_id> devices;

  err_number = clGetPlatformIDs(1, &platform_id, &num_platforms);
  if (err_number != CL_SUCCESS || num_platforms == 0) {
    std::cout << "ERROR: Failed to get platform_id" << std::endl;
    return;
  }

  cl_context_properties context_properties[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
  context_ = clCreateContextFromType(context_properties, CL_DEVICE_TYPE_GPU,
                                     nullptr, nullptr, &err_number);
  if (err_number != CL_SUCCESS) {
    std::cout
        << "ERROR: Failed to get cl_context, trying with device type CPU..."
        << std::endl;
    context_ = clCreateContextFromType(context_properties, CL_DEVICE_TYPE_CPU,
                                       nullptr, nullptr, &err_number);
    if (err_number != CL_SUCCESS) {
      std::cout << "ERROR: Failed to get cl_context" << std::endl;
      return;
    }
  }

  is_context_loaded_ = true;

  err_number = clGetContextInfo(context_, CL_CONTEXT_DEVICES, 0, nullptr,
                                &device_buffer_size);
  if (err_number != CL_SUCCESS) {
    std::cout << "ERROR: Failed to get device count from context_" << std::endl;
    return;
  } else if (device_buffer_size == 0) {
    std::cout << "ERROR: No devices available" << std::endl;
    return;
  }

  devices.resize(device_buffer_size);
  err_number = clGetContextInfo(context_, CL_CONTEXT_DEVICES,
                                device_buffer_size, devices.data(), nullptr);
  if (err_number != CL_SUCCESS) {
    std::cout << "ERROR: Failed to get devices from context_" << std::endl;
    return;
  }

  // use first device
  device = devices.at(0);

  queue_ = clCreateCommandQueue(context_, device, 0, &err_number);
  if (err_number != CL_SUCCESS) {
    std::cout << "ERROR: Failed to create command queue" << std::endl;
    return;
  }

  is_queue_loaded_ = true;

  // set up kernel
  std::string kernel_source;
  {
    std::ifstream ifs("src/kernel.cl");
    if (!(ifs.is_open() && ifs.good())) {
      std::cout << "ERROR: Failed to open \"src/kernel.cl\"" << std::endl;
      return;
    }
    char buf[1024];
    while (ifs.good()) {
      ifs.read(buf, 1024);
      kernel_source.append(buf, ifs.gcount());
    }
    if (kernel_source.empty()) {
      std::cout << "ERROR: \"src/kernel.cl\" is empty" << std::endl;
      return;
    }
  }

  {
    const char *source_c_str = kernel_source.c_str();
    program_ = clCreateProgramWithSource(context_, 1, &source_c_str, nullptr,
                                         &err_number);
    if (err_number != CL_SUCCESS) {
      std::cout << "ERROR: Failed to create cl_program" << std::endl;
      return;
    }
  }

  is_program_loaded_ = true;

  err_number = clBuildProgram(program_, 0, nullptr, nullptr, nullptr, nullptr);
  if (err_number != CL_SUCCESS) {
    std::cout << "ERROR: Failed to build cl_program" << std::endl;
    std::vector<char> build_log;
    build_log.resize(16384);
    build_log.at(16383) = 0;
    clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                          build_log.size(), build_log.data(), nullptr);
    std::cout << build_log.data() << std::endl;
    return;
  }

  is_initialization_completed_ = true;
}

OCLContext::~OCLContext() {
  for (const auto &pair : kernel_map_) {
    clReleaseKernel(pair.second);
  }
  for (const auto &pair : buffer_map_) {
    clReleaseMemObject(pair.second);
  }

  if (is_program_loaded_) {
    clReleaseProgram(program_);
  }
  if (is_queue_loaded_) {
    clReleaseCommandQueue(queue_);
  }
  if (is_context_loaded_) {
    clReleaseContext(context_);
  }
}

cl_context OCLContext::GetContext() const { return context_; }

cl_command_queue OCLContext::GetCommandQueue() const { return queue_; }

cl_program OCLContext::GetProgram() const { return program_; }

bool OCLContext::IsValid() const {
  return is_context_loaded_ && is_queue_loaded_ && is_program_loaded_ &&
         is_initialization_completed_;
}

bool OCLContext::CreateBuffer(const std::string &name, cl_mem_flags flags,
                              std::size_t size, void *host_ptr) {
  if (!IsValid()) {
    std::cout
        << "ERROR CreateBuffer: Cannot CreateBuffer with invalid OCLContext"
        << std::endl;
    return false;
  } else if (buffer_map_.find(name) != buffer_map_.end()) {
    std::cout << "ERROR CreateBuffer: Buffer already exists with same name"
              << std::endl;
    return false;
  }

  cl_int err_number;

  cl_mem mem = clCreateBuffer(context_, flags, size, host_ptr, &err_number);
  if (err_number != CL_SUCCESS) {
    std::cout << "ERROR CreateBuffer: Failed to create buffer" << std::endl;
    return false;
  }

  buffer_map_.insert({name, mem});
  return true;
}

cl_mem OCLContext::GetBuffer(const std::string &name) const {
  if (!IsValid()) {
    std::cout << "ERROR GetBuffer: Cannot GetBuffer with invalid OCLContext"
              << std::endl;
    return nullptr;
  }
  auto iter = buffer_map_.find(name);
  if (iter == buffer_map_.end()) {
    return nullptr;
  }
  return iter->second;
}

bool OCLContext::ReleaseBuffer(const std::string &name) {
  if (!IsValid()) {
    std::cout
        << "ERROR ReleaseBuffer: Cannot ReleaseBuffer with invalid OCLContext"
        << std::endl;
    return false;
  }
  auto iter = buffer_map_.find(name);
  if (iter == buffer_map_.end()) {
    return false;
  }

  clReleaseMemObject(iter->second);
  buffer_map_.erase(iter);
  return true;
}

bool OCLContext::CreateKernel(const std::string &name) {
  if (!IsValid()) {
    std::cout
        << "ERROR CreateKernel: Cannot CreateKernel with invalid OCLContext"
        << std::endl;
    return false;
  } else if (kernel_map_.find(name) != kernel_map_.end()) {
    std::cout << "ERROR CreateKernel: kernel with same name already exists"
              << std::endl;
    return false;
  }

  cl_int err_number;
  cl_kernel kernel = clCreateKernel(program_, "Kernel", &err_number);
  if (err_number != CL_SUCCESS) {
    std::cout << "ERROR: Failed to create kernel" << std::endl;
    return false;
  }

  kernel_map_.insert({name, kernel});

  return true;
}

bool OCLContext::SetKernelArg(const std::string &kernel_name, cl_uint arg_idx,
                              const std::string &buffer_name) {
  if (!IsValid()) {
    std::cout
        << "ERROR SetKernelArg: Cannot SetKernelArg with invalid OCLContext"
        << std::endl;
    return false;
  }

  auto kernel_iter = kernel_map_.find(kernel_name);
  if (kernel_iter == kernel_map_.end()) {
    std::cout << "ERROR SetKernelArg: Cannot find kernel with name \""
              << kernel_name << '"' << std::endl;
    return false;
  }

  auto buffer_iter = buffer_map_.find(buffer_name);
  if (buffer_iter == buffer_map_.end()) {
    std::cout << "ERROR SetKernelArg: Cannot find buffer with name \""
              << buffer_name << '"' << std::endl;
    return false;
  }

  cl_int err_number = clSetKernelArg(kernel_iter->second, arg_idx,
                                     sizeof(cl_mem), &buffer_iter->second);
  if (err_number != CL_SUCCESS) {
    std::cout << "ERROR SetKernelArg: Failed to clSetKernelArg with kernel \""
              << kernel_name << "\" and buffer \"" << buffer_name << '"'
              << std::endl;
    return false;
  }

  return true;
}

bool OCLContext::ReleaseKernel(const std::string &name) {
  if (!IsValid()) {
    std::cout
        << "ERROR ReleaseKernel: Cannot ReleaseKernel with invalid OCLContext"
        << std::endl;
    return false;
  }

  auto iter = kernel_map_.find(name);
  if (iter == kernel_map_.end()) {
    std::cout << "ERROR ReleaseKernel: No kernel with name \"" << name << '"'
              << std::endl;
    return false;
  }

  clReleaseKernel(iter->second);
  kernel_map_.erase(iter);

  return true;
}

bool OCLContext::ExecuteKernel(const std::string &name,
                               unsigned int wait_events_size,
                               const cl_event *wait_events,
                               cl_event *ret_event) {
  if (!IsValid()) {
    std::cout
        << "ERROR ExecuteKernel: Cannot ExecuteKernel with invalid OCLContext"
        << std::endl;
    return false;
  }

  auto iter = kernel_map_.find(name);
  if (iter == kernel_map_.end()) {
    std::cout << "ERROR ExecuteKernel: No kernel with name \"" << name << '"'
              << std::endl;
  }

  std::size_t size = 1;
  cl_int err_number =
      clEnqueueNDRangeKernel(queue_, iter->second, 1, nullptr, &size, &size,
                             wait_events_size, wait_events, ret_event);
  if (err_number != CL_SUCCESS) {
    std::cout << "ERROR ExecuteKernel: Failed to execute with name \"" << name
              << '"' << std::endl;
    return false;
  }

  return true;
}
