//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//


// Convolution.cpp
//
//    This is a simple example that demonstrates OpenCL platform, device, and context
//    use.

// Modified by Stephen Seo for Module11 assignment in Intro to GPU Programming Course

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <iomanip>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Constants
const unsigned int inputSignalWidth  = 49;
const unsigned int inputSignalHeight = 49;

cl_float inputSignal[inputSignalHeight][inputSignalWidth];

const unsigned int outputSignalWidth  = 43;
const unsigned int outputSignalHeight = 43;

cl_float outputSignal[outputSignalHeight][outputSignalWidth];

const unsigned int maskWidth  = 7;
const unsigned int maskHeight = 7;

cl_float mask[maskHeight][maskWidth] =
{
    {0.0F,  0.0F,  0.0F,  0.25F, 0.0F,  0.0F,  0.0F},
    {0.0F,  0.0F,  0.25F, 0.50F, 0.25F, 0.0F,  0.0F},
    {0.0F,  0.25F, 0.50F, 0.75F, 0.50F, 0.25F, 0.0F},
    {0.25F, 0.50F, 0.75F, 1.0F,  0.75F, 0.50F, 0.25F},
    {0.0F,  0.25F, 0.50F, 0.75F, 0.50F, 0.25F, 0.0F},
    {0.0F,  0.0F,  0.25F, 0.50F, 0.25F, 0.0F,  0.0F},
    {0.0F,  0.0F,  0.0F,  0.25F, 0.0F,  0.0F,  0.0F}
};

///
// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(
    const char * errInfo,
    const void * private_info,
    size_t cb,
    void * user_data)
{
    std::cout << "Error occured during context use: " << errInfo << std::endl;
    // should really perform any clearup and so on at this point
    // but for simplicitly just exit.
    exit(1);
}

///
//  main() for Convoloution example
//
int main(int argc, char** argv)
{
    // first, initialize inputSignal with pseudo-random values
    std::srand(0); // change to use std::time(0) for less deterministic results
    for(unsigned int y = 0; y < inputSignalHeight; ++y) {
        for(unsigned int x = 0; x < inputSignalWidth; ++x) {
            inputSignal[y][x] = std::rand() % 10;
        }
    }
    // Output the input signal
    std::cout << "input signal:\n";
    for (int y = 0; y < inputSignalHeight; y++)
    {
        for (int x = 0; x < inputSignalWidth; x++)
        {
            std::cout << inputSignal[y][x] << " ";
        }
        std::cout << std::endl;
    }

    cl_int errNum = 0;
    cl_uint numPlatforms = 0;
    cl_uint numDevices = 0;
    cl_platform_id * platformIDs = nullptr;
    cl_device_id * deviceIDs = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem inputSignalBuffer = nullptr;
    cl_mem outputSignalBuffer = nullptr;
    cl_mem maskBuffer = nullptr;

    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs"); 
 
    platformIDs = (cl_platform_id *)alloca(
            sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");

    // Iterate through the list of platforms until we find one that supports
    // a CPU device, otherwise fail with an error.
    deviceIDs = NULL;
    cl_uint i;
    for (i = 0; i < numPlatforms; i++) {
        errNum = clGetDeviceIDs(
            platformIDs[i], 
            CL_DEVICE_TYPE_GPU, 
            0,
            NULL,
            &numDevices);
        if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND) {
            checkErr(errNum, "clGetDeviceIDs");
        } else if (numDevices > 0) {
            deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
            errNum = clGetDeviceIDs(
                platformIDs[i],
                CL_DEVICE_TYPE_GPU,
                numDevices, 
                &deviceIDs[0], 
                NULL);
            checkErr(errNum, "clGetDeviceIDs");
            break;
       }
    }

    // Check to see if we found at least one CPU device, otherwise return
//  if (deviceIDs == NULL) {
//      std::cout << "No CPU device found" << std::endl;
//      exit(-1);
//  }

    // Next, create an OpenCL context on the selected platform.  
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[i],
        0
    };
    context = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        &contextCallback,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");

    std::ifstream srcFile("assignment_src/convolution.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading convolution.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    // Create program from source
    program = clCreateProgramWithSource(
        context, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    // Build program
    errNum = clBuildProgram(
        program,
        numDevices,
        deviceIDs,
        NULL,
        NULL,
        NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
            program, 
            deviceIDs[0], 
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), 
            buildLog, 
            NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        checkErr(errNum, "clBuildProgram");
    }

    // Create kernel object
    kernel = clCreateKernel(
        program,
        "convolve",
        &errNum);
    checkErr(errNum, "clCreateKernel");

    // Now allocate buffers
    inputSignalBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_float) * inputSignalHeight * inputSignalWidth,
        static_cast<void *>(inputSignal),
        &errNum);
    checkErr(errNum, "clCreateBuffer(inputSignal)");

    maskBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_float) * maskHeight * maskWidth,
        static_cast<void *>(mask),
        &errNum);
    checkErr(errNum, "clCreateBuffer(mask)");

    outputSignalBuffer = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        sizeof(cl_float) * outputSignalHeight * outputSignalWidth,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer(outputSignal)");

    // Pick the first device and create command queue.
    queue = clCreateCommandQueue(
        context,
        deviceIDs[0],
        0,
        &errNum);
    checkErr(errNum, "clCreateCommandQueue");

    errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalWidth);
    errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth);
    checkErr(errNum, "clSetKernelArg");

    const size_t globalWorkSize[2] = { outputSignalWidth, outputSignalHeight };
    const size_t localWorkSize[2]  = { 1, 1 };

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(
        queue, 
        kernel, 
        2,
        NULL,
        globalWorkSize, 
        localWorkSize,
        0, 
        NULL, 
        NULL);
    checkErr(errNum, "clEnqueueNDRangeKernel");
    
    errNum = clEnqueueReadBuffer(
        queue, 
        outputSignalBuffer, 
        CL_TRUE,
        0, 
        sizeof(cl_uint) * outputSignalHeight * outputSignalHeight, 
        outputSignal,
        0, 
        NULL, 
        NULL);
    checkErr(errNum, "clEnqueueReadBuffer");

    // Output the result buffer
    std::cout << "Result buffer:\n";
    for (int y = 0; y < outputSignalHeight; y++)
    {
        for (int x = 0; x < outputSignalWidth; x++)
        {
            std::cout << std::setw(4) << std::setprecision(1) << std::fixed
                      << outputSignal[y][x] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Executed program succesfully." << std::endl;

    // cleanup
    clReleaseCommandQueue(queue);
    clReleaseMemObject(outputSignalBuffer);
    clReleaseMemObject(maskBuffer);
    clReleaseMemObject(inputSignalBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);

    return 0;
}
