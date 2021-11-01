#include "arg_parse.h"
#include "pgm_rw.h"

#include <cmath>
#include <iostream>

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <nppdefs.h>

#include <helper_cuda.h>

void Rotate2DPoint(double x, double y, double angle, double *out_x,
                   double *out_y) {
  static const double PI = std::acos(-1.0);
  const double radians = angle * PI / 180.0;

  *out_x = x * std::cos(radians) + y * std::sin(radians);
  *out_y = x * -std::sin(radians) + y * std::cos(radians);
}

void NPPRotate(const std::string &input_filename,
               const std::string &output_filename, double angle, bool overwrite,
               bool do_timings) {
  PGMFile pgm_handle;
  if (!pgm_handle.LoadImage(input_filename)) {
    std::cout << "ERROR: Failed to LoadImage \"" << input_filename << "\""
              << std::endl;
    return;
  }

  NppiRect fROI{0, 0, (int)pgm_handle.GetWidth(), (int)pgm_handle.GetHeight()};

  // angle management to always center output image
  double shift_x = -(double)pgm_handle.GetWidth() / 2.0;
  double shift_y = -(double)pgm_handle.GetHeight() / 2.0;
  Rotate2DPoint(shift_x, shift_y, angle, &shift_x, &shift_y);
  shift_x += pgm_handle.GetWidth() / 2.0;
  shift_y += pgm_handle.GetHeight() / 2.0;

  std::uint8_t *device_in;
  std::uint8_t *device_out;
  cudaMalloc(&device_in, pgm_handle.GetSize());
  cudaMalloc(&device_out, pgm_handle.GetSize());

  cudaMemcpy(device_in, pgm_handle.GetImageData(), pgm_handle.GetSize(),
             cudaMemcpyHostToDevice);

  NppStatus status = nppiRotate_8u_C1R(
      device_in,                                                  // pSrc
      {(int)pgm_handle.GetWidth(), (int)pgm_handle.GetHeight()},  // oSrcSize
      pgm_handle.GetWidth(),                                      // nSrcStep
      fROI,                                                       // oSrcROI
      device_out,                                                 // pDst
      pgm_handle.GetWidth(),                                      // nDstStep
      fROI,                                                       // oDstROI
      angle,                                                      // nAngle
      shift_x,                                                    // nShiftX
      shift_y,                                                    // nShiftY
      NPPI_INTER_CUBIC);  // eInterpolation

  if (status != 0) {
    std::cout << "WARNING: NPP operation did not return success (" << status
              << ") " << _cudaGetErrorEnum(status) << std::endl;
  }

  // cudaDeviceSynchronize();

  cudaMemcpy(pgm_handle.GetImageData(), device_out, pgm_handle.GetSize(),
             cudaMemcpyDeviceToHost);

  if (!pgm_handle.SaveImage(output_filename, overwrite)) {
    std::cout << "ERROR: Failed to SaveImage \"" << output_filename << "\""
              << std::endl;
  } else {
    std::cout << "Saved rotated image to \"" << output_filename << "\""
              << std::endl;
  }

  cudaFree(device_out);
  cudaFree(device_in);
}

int main(int argc, char **argv) {
  Args args;

  if (args.ParseArgs(argc, argv)) {
    return 0;
  }

  if (args.input_filename.empty()) {
    std::cout << "ERROR: input filename not specified\n";
    Args::DisplayHelp();
    return 1;
  }

  std::cout << "Rotating with angle (in degrees) " << args.angle << std::endl;

  NPPRotate(args.input_filename, args.output_filename, args.angle,
            args.overwrite, args.enable_timings);

  return 0;
}
