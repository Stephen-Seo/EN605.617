//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// HelloWorld.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.

// modified by Stephen Seo for Introduction to GPU Programming course

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "arg_parse.h"
#include "get_exe_dirname.h"

///
//  Constants
//
const int ARRAY_SIZE = 1000;

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext() {
  cl_int errNum;
  cl_uint numPlatforms;
  cl_platform_id firstPlatformId;
  cl_context context = NULL;

  // First, select an OpenCL platform to run on.  For this example, we
  // simply choose the first available platform.  Normally, you would
  // query for all available platforms and select the most appropriate one.
  errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
  if (errNum != CL_SUCCESS || numPlatforms <= 0) {
    std::cerr << "Failed to find any OpenCL platforms." << std::endl;
    return NULL;
  }

  // Next, create an OpenCL context on the platform.  Attempt to
  // create a GPU-based context, and if that fails, try to create
  // a CPU-based context.
  cl_context_properties contextProperties[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformId, 0};
  context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL,
                                    NULL, &errNum);
  if (errNum != CL_SUCCESS) {
    std::cout << "Could not create GPU context, trying CPU..." << std::endl;
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS) {
      std::cerr << "Failed to create an OpenCL GPU or CPU context."
                << std::endl;
      return NULL;
    }
  }

  return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device) {
  cl_int errNum;
  cl_device_id *devices;
  cl_command_queue command_queue = NULL;
  size_t deviceBufferSize = -1;

  // First get the size of the devices buffer
  errNum =
      clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
    return NULL;
  }

  if (deviceBufferSize <= 0) {
    std::cerr << "No devices available.";
    return NULL;
  }

  // Allocate memory for the devices buffer
  devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
  errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize,
                            devices, NULL);
  if (errNum != CL_SUCCESS) {
    delete[] devices;
    std::cerr << "Failed to get device IDs";
    return NULL;
  }

  // In this example, we just choose the first available device.  In a
  // real program, you would likely use all available devices or choose
  // the highest performance device based on OpenCL device queries
  command_queue = clCreateCommandQueue(context, devices[0], 0, NULL);
  if (command_queue == NULL) {
    delete[] devices;
    std::cerr << "Failed to create command_queue for device 0";
    return NULL;
  }

  *device = devices[0];
  delete[] devices;
  return command_queue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device,
                         const char *fileName) {
  cl_int errNum;
  cl_program program;

  std::ifstream kernelFile(fileName, std::ios::in);
  if (!kernelFile.is_open()) {
    std::cerr << "Failed to open file for reading: " << fileName << std::endl;
    return NULL;
  }

  std::ostringstream oss;
  oss << kernelFile.rdbuf();

  std::string srcStdStr = oss.str();
  const char *srcStr = srcStdStr.c_str();
  program =
      clCreateProgramWithSource(context, 1, (const char **)&srcStr, NULL, NULL);
  if (program == NULL) {
    std::cerr << "Failed to create CL program from source." << std::endl;
    return NULL;
  }

  errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (errNum != CL_SUCCESS) {
    // Determine the reason for the error
    char buildLog[16384];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          sizeof(buildLog), buildLog, NULL);

    std::cerr << "Error in kernel: " << std::endl;
    std::cerr << buildLog;
    clReleaseProgram(program);
    return NULL;
  }

  return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem mem_objects[3], float *a,
                      float *b) {
  mem_objects[0] =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * ARRAY_SIZE, a, NULL);
  mem_objects[1] =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * ARRAY_SIZE, b, NULL);
  mem_objects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  sizeof(float) * ARRAY_SIZE, NULL, NULL);

  if (mem_objects[0] == NULL || mem_objects[1] == NULL ||
      mem_objects[2] == NULL) {
    std::cerr << "Error creating memory objects." << std::endl;
    return false;
  }

  return true;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context *context, cl_command_queue *command_queue,
             cl_program *program, cl_kernel *kernel, cl_mem mem_objects[3]) {
  if (mem_objects != nullptr) {
    for (int i = 0; i < 3; i++) {
      if (mem_objects[i] != 0) clReleaseMemObject(mem_objects[i]);
      mem_objects[i] = 0;
    }
  }
  if (command_queue != nullptr) {
    if (*command_queue != 0) clReleaseCommandQueue(*command_queue);
    *command_queue = 0;
  }

  if (kernel != nullptr) {
    if (*kernel != 0) clReleaseKernel(*kernel);
    *kernel = 0;
  }

  if (program != nullptr) {
    if (*program != 0) clReleaseProgram(*program);
    *program = 0;
  }

  if (context != nullptr) {
    if (*context != 0) clReleaseContext(*context);
    *context = 0;
  }
}

bool SetUpContext(cl_context *context, cl_device_id *device,
                  cl_command_queue *command_queue) {
  // Create an OpenCL context on first available platform
  *context = CreateContext();
  if (*context == NULL) {
    std::cerr << "Failed to create OpenCL context." << std::endl;
    return false;
  }

  // Create a command-queue on the first device available
  // on the created context
  *command_queue = CreateCommandQueue(*context, device);
  if (*command_queue == NULL) {
    Cleanup(context, command_queue, nullptr, nullptr, nullptr);
    return false;
  }

  return true;
}

bool SetUpKernel(cl_context *context, cl_device_id *device,
                 cl_command_queue *command_queue, cl_kernel *kernel,
                 cl_program *program, const char *kernel_path) {
  // Create OpenCL program from HelloWorld.cl kernel source
  *program = CreateProgram(*context, *device, kernel_path);
  if (program == NULL) {
    Cleanup(context, command_queue, program, kernel, nullptr);
    return false;
  }

  // Create OpenCL kernel
  *kernel = clCreateKernel(*program, "hello_kernel", NULL);
  if (kernel == NULL) {
    std::cerr << "Failed to create kernel" << std::endl;
    Cleanup(context, command_queue, program, kernel, nullptr);
    return false;
  }

  return true;
}

bool SetUpBuffers(cl_context *context, cl_command_queue *command_queue,
                  cl_program *program, cl_kernel *kernel,
                  cl_mem mem_objects[3]) {
  // Create memory objects that will be used as arguments to
  // kernel.  First create host memory arrays that will be
  // used to store the arguments to the kernel
  float a[ARRAY_SIZE];
  float b[ARRAY_SIZE];
  for (int i = 0; i < ARRAY_SIZE; i++) {
    a[i] = (float)(i * 2);
    b[i] = (float)i;
  }

  if (!CreateMemObjects(*context, mem_objects, a, b)) {
    Cleanup(context, command_queue, program, kernel, mem_objects);
    return false;
  }

  // Set the kernel arguments (result, a, b)
  cl_int errNum = clSetKernelArg(*kernel, 0, sizeof(cl_mem), &mem_objects[0]);
  errNum |= clSetKernelArg(*kernel, 1, sizeof(cl_mem), &mem_objects[1]);
  errNum |= clSetKernelArg(*kernel, 2, sizeof(cl_mem), &mem_objects[2]);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error setting kernel arguments." << std::endl;
    Cleanup(context, command_queue, program, kernel, mem_objects);
    return false;
  }
  return true;
}

bool EnqueueTask(cl_context *context, cl_command_queue *command_queue,
                 cl_program *program, cl_kernel *kernel,
                 cl_mem mem_objects[3]) {
  size_t globalWorkSize[1] = {ARRAY_SIZE};
  size_t localWorkSize[1] = {1};

  // Queue the kernel up for execution across the array
  cl_event event;
  cl_int errNum =
      clEnqueueNDRangeKernel(*command_queue, *kernel, 1, NULL, globalWorkSize,
                             localWorkSize, 0, NULL, &event);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error queuing kernel for execution." << std::endl;
    Cleanup(context, command_queue, program, kernel, mem_objects);
    return false;
  }

  errNum = clWaitForEvents(1, &event);

  if (errNum != CL_SUCCESS) {
    std::cerr << "Error watiing on kernel to complete." << std::endl;
    Cleanup(context, command_queue, program, kernel, mem_objects);
    return false;
  }

  return true;
}

bool TimeEnqueueTask(cl_context *context, cl_command_queue *command_queue,
                     cl_program *program, cl_kernel *kernel,
                     cl_mem mem_objects[3], const char *operation_name) {
  size_t globalWorkSize[1] = {ARRAY_SIZE};
  size_t localWorkSize[1] = {1};

  cl_event event;
  cl_int errNum;

  unsigned long long total_count = 0;
  for (unsigned int i = 0; i < 25; ++i) {
    auto start_time = std::chrono::high_resolution_clock::now();

    errNum =
        clEnqueueNDRangeKernel(*command_queue, *kernel, 1, NULL, globalWorkSize,
                               localWorkSize, 0, NULL, &event);
    if (errNum != CL_SUCCESS) {
      std::cerr << "Error queuing kernel for execution." << std::endl;
      Cleanup(context, command_queue, program, kernel, mem_objects);
      return false;
    }

    errNum = clWaitForEvents(1, &event);

    if (errNum != CL_SUCCESS) {
      std::cerr << "Error watiing on kernel to complete." << std::endl;
      Cleanup(context, command_queue, program, kernel, mem_objects);
      return false;
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    if (i > 4) {
      unsigned long long count =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end_time -
                                                               start_time)
              .count();
      std::cout << "Iteration " << std::setw(2) << i - 4 << " took " << count
                << " nanoseconds\n";
      total_count += count;
    }
  }
  std::cout << "Average of 20 runs (" << operation_name
            << ") == " << total_count / 20 << " nanoseconds\n";

  return true;
}

bool GetOutput(cl_context *context, cl_command_queue *command_queue,
               cl_program *program, cl_kernel *kernel, cl_mem mem_objects[3]) {
  // Read the output buffer back to the Host
  float result[ARRAY_SIZE];
  cl_int errNum =
      clEnqueueReadBuffer(*command_queue, mem_objects[2], CL_TRUE, 0,
                          ARRAY_SIZE * sizeof(float), result, 0, NULL, NULL);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error reading result buffer." << std::endl;
    Cleanup(context, command_queue, program, kernel, mem_objects);
    return false;
  }

  // Output the result buffer
  for (int i = 0; i < ARRAY_SIZE; i++) {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;

  return true;
}

bool DoTask(cl_context *context, cl_device_id *device,
            cl_command_queue *command_queue, cl_program *program,
            cl_kernel *kernel, cl_mem mem_objects[3],
            const char *program_path) {
  if (!SetUpKernel(context, device, command_queue, kernel, program,
                   program_path)) {
    return false;
  }

  if (!SetUpBuffers(context, command_queue, program, kernel, mem_objects)) {
    return false;
  }

  if (!EnqueueTask(context, command_queue, program, kernel, mem_objects)) {
    return false;
  }

  if (!GetOutput(context, command_queue, program, kernel, mem_objects)) {
    return false;
  }

  // Cleanup the kernel and related objects for later re-invocations of DoTask
  Cleanup(0, 0, program, kernel, mem_objects);

  return true;
}

bool TimeTask(cl_context *context, cl_device_id *device,
              cl_command_queue *command_queue, cl_program *program,
              cl_kernel *kernel, cl_mem mem_objects[3],
              const char *program_path, const char *operation_name) {
  if (!SetUpKernel(context, device, command_queue, kernel, program,
                   program_path)) {
    return false;
  }

  if (!SetUpBuffers(context, command_queue, program, kernel, mem_objects)) {
    return false;
  }

  if (!TimeEnqueueTask(context, command_queue, program, kernel, mem_objects,
                       operation_name)) {
    return false;
  }

  // When timing, output is not copied from device to host and is not printed

  // Cleanup the kernel and related objects for later re-invocations of TimeTask
  Cleanup(0, 0, program, kernel, mem_objects);

  return true;
}

///
//  main() for HelloWorld example
//
int main(int argc, char **argv) {
  std::string exe_dirname = GetExeDirName(argv[0]);
  // std::cout << exe_dirname << std::endl; // DEBUG

  Args args;
  if (args.ParseArgs(argc, argv)) {
    return 0;
  } else if (!args.do_addition && !args.do_subtraction &&
             !args.do_multiplication && !args.do_division && !args.do_power) {
    std::cout << "ERROR: An operation must be specified\n";
    Args::DisplayHelp();
    return 1;
  }

  cl_context context = 0;
  cl_command_queue command_queue = 0;
  cl_program program = 0;
  cl_device_id device = 0;
  cl_kernel kernel = 0;
  cl_mem mem_objects[3] = {0, 0, 0};
  cl_int errNum;

  if (!SetUpContext(&context, &device, &command_queue)) {
    return 1;
  }

  std::string cl_path;
  if (args.do_addition) {
    cl_path = "assignment_src/ocl_addition.cl";
    if (!exe_dirname.empty()) {
      cl_path = exe_dirname + "/" + cl_path;
    }

    if (args.do_timings) {
      std::cout << "Do addition operation timings...\n";
      if (!TimeTask(&context, &device, &command_queue, &program, &kernel,
                    mem_objects, cl_path.c_str(), "addition")) {
        Cleanup(&context, &command_queue, &program, &kernel, mem_objects);
        return 1;
      }
    } else {
      std::cout << "Do addition operation...\n";
      if (!DoTask(&context, &device, &command_queue, &program, &kernel,
                  mem_objects, cl_path.c_str())) {
        Cleanup(&context, &command_queue, &program, &kernel, mem_objects);
        return 1;
      }
    }
  }
  if (args.do_subtraction) {
    cl_path = "assignment_src/ocl_subtraction.cl";
    if (!exe_dirname.empty()) {
      cl_path = exe_dirname + "/" + cl_path;
    }

    if (args.do_timings) {
      std::cout << "Do subtraction operation timings...\n";
      if (!TimeTask(&context, &device, &command_queue, &program, &kernel,
                    mem_objects, cl_path.c_str(), "subtraction")) {
        Cleanup(&context, &command_queue, &program, &kernel, mem_objects);
        return 1;
      }
    } else {
      std::cout << "Do subtraction operation...\n";
      if (!DoTask(&context, &device, &command_queue, &program, &kernel,
                  mem_objects, cl_path.c_str())) {
        Cleanup(&context, &command_queue, &program, &kernel, mem_objects);
        return 1;
      }
    }
  }
  if (args.do_multiplication) {
    cl_path = "assignment_src/ocl_multiplication.cl";
    if (!exe_dirname.empty()) {
      cl_path = exe_dirname + "/" + cl_path;
    }

    if (args.do_timings) {
      std::cout << "Do multiplication operation timings...\n";
      if (!TimeTask(&context, &device, &command_queue, &program, &kernel,
                    mem_objects, cl_path.c_str(), "multiplication")) {
        Cleanup(&context, &command_queue, &program, &kernel, mem_objects);
        return 1;
      }
    } else {
      std::cout << "Do multiplication operation...\n";
      if (!DoTask(&context, &device, &command_queue, &program, &kernel,
                  mem_objects, cl_path.c_str())) {
        Cleanup(&context, &command_queue, &program, &kernel, mem_objects);
        return 1;
      }
    }
  }
  if (args.do_division) {
    cl_path = "assignment_src/ocl_division.cl";
    if (!exe_dirname.empty()) {
      cl_path = exe_dirname + "/" + cl_path;
    }

    if (args.do_timings) {
      std::cout << "Do division operation timings...\n";
      if (!TimeTask(&context, &device, &command_queue, &program, &kernel,
                    mem_objects, cl_path.c_str(), "division")) {
        Cleanup(&context, &command_queue, &program, &kernel, mem_objects);
        return 1;
      }
    } else {
      std::cout << "Do division operation...\n";
      if (!DoTask(&context, &device, &command_queue, &program, &kernel,
                  mem_objects, cl_path.c_str())) {
        Cleanup(&context, &command_queue, &program, &kernel, mem_objects);
        return 1;
      }
    }
  }
  if (args.do_power) {
    cl_path = "assignment_src/ocl_power.cl";
    if (!exe_dirname.empty()) {
      cl_path = exe_dirname + "/" + cl_path;
    }

    if (args.do_timings) {
      std::cout << "Do power operation timings...\n";
      if (!TimeTask(&context, &device, &command_queue, &program, &kernel,
                    mem_objects, cl_path.c_str(), "power")) {
        Cleanup(&context, &command_queue, &program, &kernel, mem_objects);
        return 1;
      }
    } else {
      std::cout << "Do power operation...\n";
      if (!DoTask(&context, &device, &command_queue, &program, &kernel,
                  mem_objects, cl_path.c_str())) {
        Cleanup(&context, &command_queue, &program, &kernel, mem_objects);
        return 1;
      }
    }
  }

  std::cout << "Executed program succesfully." << std::endl;
  Cleanup(&context, &command_queue, &program, &kernel, mem_objects);

  return 0;
}
