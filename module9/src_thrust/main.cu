#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include <cuda_device_runtime_api.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "arg_parse.h"
#include "constants.h"
#include "thrust_operations.h"

void PrepareDeviceVectorA(thrust::device_vector<int> &a,
                          const unsigned int size) {
  a.resize(size);
  thrust::sequence(a.begin(), a.end());
}

int LimitedRand() { return std::rand() % 100; }

void PrepareDeviceVectorB(thrust::host_vector<int> &host,
                          thrust::device_vector<int> &b,
                          const unsigned int size) {
  host.resize(size);
  std::srand(std::time(nullptr));
  thrust::generate(thrust::host, host.begin(), host.end(), LimitedRand);

  b = host;
}

void HandleOp(const unsigned int size, bool print_output, bool do_timings,
              bool do_addition, bool do_subtraction, bool do_multiplication,
              bool do_modulus) {
  thrust::device_vector<int> a, b, out;
  thrust::host_vector<int> host;

  // prepare device vectors
  PrepareDeviceVectorA(a, size);
  PrepareDeviceVectorB(host, b, size);
  out.resize(size);

  // do timings of each of the four operations
  if (do_timings) {
    // timings of addition
    if (do_addition) {
      ThrustOps::DoTimingsOfOp(ThrustOps::AddThrust, "addition", a, b, out);
    }

    // timings of subtraction
    if (do_subtraction) {
      ThrustOps::DoTimingsOfOp(ThrustOps::SubThrust, "subtraction", a, b, out);
    }

    // timings of multiplication
    if (do_multiplication) {
      ThrustOps::DoTimingsOfOp(ThrustOps::MultThrust, "multiplication", a, b,
                               out);
    }

    // timings of modulus
    if (do_modulus) {
      ThrustOps::DoTimingsOfOp(ThrustOps::ModThrust, "modulus", a, b, out);
    }
  } else {  // (!do_timings)
    // display contents of device vectors "a" and "b"
    if (print_output) {
      host = a;
      std::cout << "A array contents:\n";
      for (auto iter = host.begin(); iter != host.end(); ++iter) {
        std::cout << *iter << ' ';
      }
      std::cout << std::endl;

      host = b;
      std::cout << "B array contents:\n";
      for (auto iter = host.begin(); iter != host.end(); ++iter) {
        std::cout << *iter << ' ';
      }
      std::cout << std::endl;
    }  // end print_output "a" and "b"

    // do addition and print results
    if (do_addition) {
      ThrustOps::DoPrintsOfOp(ThrustOps::AddThrust, "addition", a, b, out,
                              host);
    }

    // do subtraction and print results
    if (do_subtraction) {
      ThrustOps::DoPrintsOfOp(ThrustOps::SubThrust, "subtraction", a, b, out,
                              host);
    }

    // do multiplication and print results
    if (do_multiplication) {
      ThrustOps::DoPrintsOfOp(ThrustOps::MultThrust, "multiplication", a, b,
                              out, host);
    }

    // do modulus and print results
    if (do_modulus) {
      ThrustOps::DoPrintsOfOp(ThrustOps::ModThrust, "modulus", a, b, out, host);
    }
  }  // end else (!do_timings)
}

int main(int argc, char **argv) {
  Args args;
  unsigned int data_size = DEFAULT_THRUST_DATA_SIZE;

  if (args.ParseArgs(argc, argv)) {
    return 0;
  }

  if (args.thrust_size > 0) {
    data_size = args.thrust_size;
    std::cout << "Set data_size to " << data_size << std::endl;
  } else {
    std::cout << "Defaulted data_size to " << data_size << std::endl;
  }

  if (!args.do_addition && !args.do_subtraction && !args.do_multiplication &&
      !args.do_modulus) {
    std::cout << "ERROR: At least one operation should be set to be used\n";
    Args::DisplayHelp();
    return 1;
  }

  if (!args.enable_print_output && !args.enable_timings) {
    std::cout << "ERROR: print_output and timings not enabled! Pick one to get"
                 " output"
              << std::endl;
    return 1;
  }

  HandleOp(data_size, args.enable_print_output, args.enable_timings,
           args.do_addition, args.do_subtraction, args.do_multiplication,
           args.do_modulus);

  return 0;
}
