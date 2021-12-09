#ifndef IGPUP_MODULE_13_EVENTS_COMMAND_QUEUE_OPENCL_CONTEXT_H_
#define IGPUP_MODULE_13_EVENTS_COMMAND_QUEUE_OPENCL_CONTEXT_H_

#include <memory>
#include <string>
#include <unordered_map>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

class OCLContext {
 public:
  OCLContext();
  ~OCLContext();

  // deny copy
  OCLContext(const OCLContext &other) = delete;
  OCLContext &operator=(const OCLContext &other) = delete;

  // allow move
  OCLContext(OCLContext &&other) = default;
  OCLContext &operator=(OCLContext &&other) = default;

  cl_context GetContext() const;
  cl_command_queue GetCommandQueue() const;
  cl_program GetProgram() const;

  bool IsValid() const;

  bool CreateBuffer(const std::string &name, cl_mem_flags flags,
                    std::size_t size, void *host_ptr);
  cl_mem GetBuffer(const std::string &name) const;
  bool ReleaseBuffer(const std::string &name);

  bool CreateKernel(const std::string &name);
  bool SetKernelArg(const std::string &kernel_name, cl_uint arg_idx,
                    const std::string &buffer_name);
  bool ReleaseKernel(const std::string &name);

  bool ExecuteKernel(const std::string &name, unsigned int wait_events_size,
                     const cl_event *wait_events, cl_event *ret_event);

 private:
  std::unordered_map<std::string, cl_mem> buffer_map_;
  std::unordered_map<std::string, cl_kernel> kernel_map_;

  bool is_context_loaded_;
  bool is_queue_loaded_;
  bool is_program_loaded_;
  bool is_initialization_completed_;

  cl_context context_;
  cl_command_queue queue_;
  cl_program program_;
};

#endif
