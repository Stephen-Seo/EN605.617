#ifndef IGPUP_MODULE_9_THRUST_OPERATIONS_H
#define IGPUP_MODULE_9_THRUST_OPERATIONS_H

#include <chrono>

#include <thrust/device_vector.h>

namespace ThrustOps {

void AddThrust(const thrust::device_vector<int> &a,
               const thrust::device_vector<int> &b,
               thrust::device_vector<int> &out);

void SubThrust(const thrust::device_vector<int> &a,
               const thrust::device_vector<int> &b,
               thrust::device_vector<int> &out);

void MultThrust(const thrust::device_vector<int> &a,
                const thrust::device_vector<int> &b,
                thrust::device_vector<int> &out);

void ModThrust(const thrust::device_vector<int> &a,
               const thrust::device_vector<int> &b,
               thrust::device_vector<int> &out);

typedef void (*ThrustFn)(const thrust::device_vector<int> &a,
                         const thrust::device_vector<int> &b,
                         thrust::device_vector<int> &out);

void DoTimingsOfOp(ThrustFn fn, const char *op_name,
                   const thrust::device_vector<int> &a,
                   const thrust::device_vector<int> &b,
                   thrust::device_vector<int> &out);

void DoPrintsOfOp(ThrustFn fn, const char *op_name,
                  const thrust::device_vector<int> &a,
                  const thrust::device_vector<int> &b,
                  thrust::device_vector<int> &out,
                  thrust::host_vector<int> &host);

}  // namespace ThrustOps

#endif
