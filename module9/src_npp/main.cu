#include "arg_parse.h"
#include "pgm_rw.h"

#include <iostream>

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <nppdefs.h>

int main(int argc, char **argv) {
  Args args;

  if (args.ParseArgs(argc, argv)) {
    return 0;
  }

  PGMFile inputPGM;

  if (!inputPGM.LoadImage(args.input_filename)) {
    std::cout << "Failed to LoadImage" << std::endl;
    return 1;
  }

  std::vector<std::uint8_t> host_vec(inputPGM.GetSize());

  NppiRect srcROI{0, 0, (int)inputPGM.GetWidth(), (int)inputPGM.GetHeight()};

  std::uint8_t *device_in;
  std::uint8_t *device_out;
  cudaMalloc(&device_in, inputPGM.GetSize());
  cudaMalloc(&device_out, inputPGM.GetSize());

  cudaMemcpy(device_in, inputPGM.GetImageData(), inputPGM.GetSize(),
             cudaMemcpyHostToDevice);

  NppStatus status = nppiRotate_8u_C1R(
      device_in,                                              // pSrc
      {(int)inputPGM.GetWidth(), (int)inputPGM.GetHeight()},  // oSrcSize
      inputPGM.GetWidth(),                                    // nSrcStep
      srcROI,                                                 // oSrcROI
      device_out,                                             // pDst
      inputPGM.GetWidth(),                                    // nDstStep
      srcROI,                                                 // oDstROI
      180.0,                                                  // nAngle
      inputPGM.GetWidth(),                                    // nShiftX
      inputPGM.GetHeight(),                                   // nShiftY
      NPPI_INTER_NN);

  if (status != 0) {
    std::cout << "WARNING: NPP operation did not return success" << std::endl;
  }

  // cudaDeviceSynchronize();

  cudaMemcpy(host_vec.data(), device_out, inputPGM.GetSize(),
             cudaMemcpyDeviceToHost);

  PGMFile outputPGM(host_vec.data(), host_vec.size(), inputPGM.GetWidth());
  outputPGM.SaveImage("test.pgm");

  cudaFree(device_in);
  cudaFree(device_out);

  return 0;
}
