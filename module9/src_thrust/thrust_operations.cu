#include "thrust_operations.h"

void ThrustOps::AddThrust(const thrust::device_vector<int> &a,
                          const thrust::device_vector<int> &b,
                          thrust::device_vector<int> &out) {
  thrust::plus<int> op;
  thrust::transform(thrust::device, a.begin(), a.end(), b.begin(), out.begin(),
                    op);
  cudaDeviceSynchronize();
}

void ThrustOps::SubThrust(const thrust::device_vector<int> &a,
                          const thrust::device_vector<int> &b,
                          thrust::device_vector<int> &out) {
  thrust::minus<int> op;
  thrust::transform(thrust::device, a.begin(), a.end(), b.begin(), out.begin(),
                    op);
  cudaDeviceSynchronize();
}

void ThrustOps::MultThrust(const thrust::device_vector<int> &a,
                           const thrust::device_vector<int> &b,
                           thrust::device_vector<int> &out) {
  thrust::multiplies<int> op;
  thrust::transform(thrust::device, a.begin(), a.end(), b.begin(), out.begin(),
                    op);
  cudaDeviceSynchronize();
}

void ThrustOps::ModThrust(const thrust::device_vector<int> &a,
                          const thrust::device_vector<int> &b,
                          thrust::device_vector<int> &out) {
  thrust::modulus<int> op;
  thrust::transform(thrust::device, a.begin(), a.end(), b.begin(), out.begin(),
                    op);
  cudaDeviceSynchronize();
}

void ThrustOps::DoTimingsOfOp(ThrustFn fn, const char *op_name,
                              const thrust::device_vector<int> &a,
                              const thrust::device_vector<int> &b,
                              thrust::device_vector<int> &out) {
  unsigned long long count = 0;
  for (unsigned int i = 0; i < 25; ++i) {
    auto start_time = std::chrono::high_resolution_clock::now();
    fn(a, b, out);
    auto end_time = std::chrono::high_resolution_clock::now();
    if (i > 4) {
      unsigned long long nanos =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end_time -
                                                               start_time)
              .count();
      std::cout << "Iteration " << i - 4 << " of " << op_name << ", took "
                << nanos << " nanoseconds\n";
      count += nanos;
    }
  }
  std::cout << "Average of 20 " << op_name << " of size " << a.size()
            << " runs == " << count / 20 << " nanos\n";
}

void ThrustOps::DoPrintsOfOp(ThrustFn fn, const char *op_name,
                             const thrust::device_vector<int> &a,
                             const thrust::device_vector<int> &b,
                             thrust::device_vector<int> &out,
                             thrust::host_vector<int> &host) {
  fn(a, b, out);

  host = out;
  std::cout << "Out array contents after " << op_name << ":\n";
  for (auto iter = host.begin(); iter != host.end(); ++iter) {
    std::cout << *iter << ' ';
  }
  std::cout << std::endl;
}
