#include <cstdlib>
#include <ctime>
#include <iostream>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "arg_parse.h"
#include "constants.h"

void AddThrust(const thrust::device_vector<int> &a,
               const thrust::device_vector<int> &b,
               thrust::device_vector<int> &out) {
  thrust::plus<int> op;
  thrust::transform(thrust::device, a.begin(), a.end(), b.begin(), out.begin(),
                    op);
}

int LimitedRand() { return std::rand() % 100; }

void PrepareDeviceVectorA(thrust::device_vector<int> &a,
                          const unsigned int size) {
  a.resize(size);
  thrust::sequence(a.begin(), a.end());
}

void PrepareDeviceVectorB(thrust::host_vector<int> &host,
                          thrust::device_vector<int> &b,
                          const unsigned int size) {
  host.resize(size);
  std::srand(std::time(nullptr));
  thrust::generate(thrust::host, host.begin(), host.end(), LimitedRand);

  b = host;
}

void HandleAdd(const unsigned int size, bool print_output, bool do_timings) {
  thrust::device_vector<int> a, b, out;
  thrust::host_vector<int> host;

  PrepareDeviceVectorA(a, size);

  PrepareDeviceVectorB(host, b, size);

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
  }

  out.resize(size);
  AddThrust(a, b, out);

  if (print_output) {
    host = out;
    std::cout << "Out array contents:\n";
    for (auto iter = host.begin(); iter != host.end(); ++iter) {
      std::cout << *iter << ' ';
    }
    std::cout << std::endl;
  }
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

  HandleAdd(data_size, args.enable_print_output, args.enable_timings);

  return 0;
}
