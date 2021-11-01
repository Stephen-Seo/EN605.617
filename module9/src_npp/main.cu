#include "arg_parse.h"
#include "pgm_rw.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
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
               const std::string &output_filename, double angle,
               bool overwrite) {
  // get input image
  PGMFile pgm_handle;
  if (!pgm_handle.LoadImage(input_filename)) {
    std::cout << "ERROR: Failed to LoadImage \"" << input_filename << "\""
              << std::endl;
    return;
  }

  NppiRect fROI{0, 0, (int)pgm_handle.GetWidth(), (int)pgm_handle.GetHeight()};
  NppiSize fSize{(int)pgm_handle.GetWidth(), (int)pgm_handle.GetHeight()};

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

  NppStatus status = nppiRotate_8u_C1R(device_in,              // pSrc
                                       fSize,                  // oSrcSize
                                       pgm_handle.GetWidth(),  // nSrcStep
                                       fROI,                   // oSrcROI
                                       device_out,             // pDst
                                       pgm_handle.GetWidth(),  // nDstStep
                                       fROI,                   // oDstROI
                                       angle,                  // nAngle
                                       shift_x,                // nShiftX
                                       shift_y,                // nShiftY
                                       NPPI_INTER_CUBIC);      // eInterpolation

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

void NPPRotateWithTimings(const std::string &input_filename) {
  // load input image
  PGMFile pgm_handle;
  if (!pgm_handle.LoadImage(input_filename)) {
    std::cout << "ERROR: Failed to LoadImage \"" << input_filename << '"'
              << std::endl;
    return;
  }

  // set up device data
  std::uint8_t *device_in;
  std::uint8_t *device_out;
  cudaMalloc(&device_in, pgm_handle.GetSize());
  cudaMalloc(&device_out, pgm_handle.GetSize());

  // set up NPP data
  NppiRect fROI{0, 0, (int)pgm_handle.GetWidth(), (int)pgm_handle.GetHeight()};
  NppiSize size{(int)pgm_handle.GetWidth(), (int)pgm_handle.GetHeight()};

  cudaMemcpy(device_in, pgm_handle.GetImageData(), pgm_handle.GetSize(),
             cudaMemcpyHostToDevice);

  // set up reusable fn for each interpolation
  const auto timing_fn = [](PGMFile &pgm_handle, std::uint8_t *device_in,
                            std::uint8_t *device_out, NppiRect fROI,
                            NppiSize size, int interpolation) {
    // same rand seed for each interpolation
    std::srand(0);
    unsigned long long count = 0;
    double angle, shift_x, shift_y;
    for (unsigned int i = 0; i < 25; ++i) {
      auto start_time = std::chrono::high_resolution_clock::now();

      // do the rotation with NPP multiple times per run
      for (unsigned int j = 0; j < 5; ++j) {
        // get angle
        angle = std::rand() % 360;

        // get shfit_x/shift_y
        // the resulting shift_x/shift_y sets the center of the rotated image
        // to the center of the output image
        shift_x = -(double)pgm_handle.GetWidth() / 2.0;
        shift_y = -(double)pgm_handle.GetHeight() / 2.0;
        Rotate2DPoint(shift_x, shift_y, angle, &shift_x, &shift_y);
        shift_x += pgm_handle.GetWidth() / 2.0;
        shift_y += pgm_handle.GetHeight() / 2.0;

        NppStatus status = nppiRotate_8u_C1R(device_in,              // pSrc
                                             size,                   // oSrcSize
                                             pgm_handle.GetWidth(),  // nSrcStep
                                             fROI,                   // oSrcROI
                                             device_out,             // pDst
                                             pgm_handle.GetWidth(),  // nDstStep
                                             fROI,                   // oDstROI
                                             angle,                  // nAngle
                                             shift_x,                // nShiftX
                                             shift_y,                // nShiftY
                                             interpolation);  // eInterpolation
        cudaDeviceSynchronize();
      }

      auto end_time = std::chrono::high_resolution_clock::now();

      if (i > 4) {
        unsigned long long nanos =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_time -
                                                                 start_time)
                .count();
        std::cout << "Iteration " << i - 4 << " took " << nanos
                  << " nanoseconds" << std::endl;
        count += nanos;
      }
    }
    std::cout << "Average (";
    switch (interpolation) {
      case NPPI_INTER_NN:
        std::cout << "NearestNeighbor Interpolation";
        break;
      case NPPI_INTER_CUBIC:
        std::cout << "Cubic Interpolation";
        break;
      case NPPI_INTER_LANCZOS:
        std::cout << "Lanczos Interpolation";
        break;
      default:
        std::cout << "UKNOWN Interpolation";
        break;
    }
    std::cout << ") of 20 runs took " << count / 20 << " nanoseconds"
              << std::endl;
  };  // end of lambda fn "timing_fn"

  // run timings with NearestNeighbor interpolation
  std::cout << "Running with NearestNeighbor interpolation...\n";
  timing_fn(pgm_handle, device_in, device_out, fROI, size, NPPI_INTER_NN);

  // run timings with Cubic interpolation
  std::cout << "Running with Cubic interpolation...\n";
  timing_fn(pgm_handle, device_in, device_out, fROI, size, NPPI_INTER_CUBIC);

  // run timings with Lanczos interpolation
  std::cout << "Running with Lanczos interpolation...\n";
  timing_fn(pgm_handle, device_in, device_out, fROI, size, NPPI_INTER_LANCZOS);

  cudaFree(device_in);
  cudaFree(device_out);
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

  if (args.enable_timings) {
    std::cout << "With timings enabled, set angle is ignored during timings.\n"
                 "Running timings..."
              << std::endl;
    NPPRotateWithTimings(args.input_filename);
  } else {
    std::cout << "Rotating with angle (in degrees) " << args.angle << std::endl;

    NPPRotate(args.input_filename, args.output_filename, args.angle,
              args.overwrite);
  }

  return 0;
}
