#include <fstream>
#include <iostream>

#include "arg_parse.h"
#include "csv_parser.h"
#include "dependencies.h"
#include "opencl_context.h"

int main(int argc, char **argv) {
  Args args{};

  if (!args.ParseArgs(argc, argv)) {
    return 1;
  } else if (args.help_printed_) {
    return 0;
  } else if (args.input_filename_.empty()) {
    std::cout << "ERROR: Input filename is an empty string" << std::endl;
    Args::PrintUsage();
    return 2;
  }

  Dependencies deps = DepsCSVParser::GetDepsFromCSV(args.input_filename_);
  if (deps.IsEmpty()) {
    std::cout << "ERROR: Got emtpy Dependencies object from CSV \""
              << args.input_filename_ << '"' << std::endl;
    return 3;
  }

  auto cycle_start = deps.HasCycle();
  if (cycle_start) {
    std::cout << "ERROR: Dependencies object has a cycle starting at "
              << *cycle_start << std::endl;
    Args::PrintUsage();
    return 4;
  }

  ReverseDependencies reverseDeps = deps.GenerateReverseDependencies();
  if (reverseDeps.IsEmpty()) {
    std::cout << "ERROR: reverseDeps is empty (internal error?)" << std::endl;
    Args::PrintUsage();
    return 5;
  }

  auto stages = reverseDeps.GetDependenciesOrdered();

  std::cout << "Printing stages:" << std::endl;
  for (unsigned int i = 0; i < stages.size(); ++i) {
    std::cout << "Stage " << i << std::endl;
    for (unsigned int j = 0; j < stages.at(i).size(); ++j) {
      std::cout << "  " << stages.at(i).at(j) << std::endl;
    }
  }

  OCLContext ocl_context{};

  // create shared buffer
  unsigned int temp = 0;
  if (!ocl_context.CreateBuffer("shared",
                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(unsigned int), &temp)) {
    std::cout << "ERROR: Failed to create \"shared\" buffer" << std::endl;
    return 6;
  }

  // create individual kernels and buffers
  for (unsigned int stage_idx = 0; stage_idx < stages.size(); ++stage_idx) {
    for (unsigned int idx : stages.at(stage_idx)) {
      temp = idx;
      std::string idx_name = std::to_string(temp);
      if (!ocl_context.CreateKernel(idx_name)) {
        std::cout << "ERROR: Failed to CreateKernel \"" << idx_name << '"'
                  << std::endl;
        return 7;
      }
      if (!ocl_context.CreateBuffer(idx_name,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(unsigned int), &temp)) {
        std::cout << "ERROR: Failed to CreateBuffer \"" << idx_name << '"'
                  << std::endl;
        return 8;
      }

      if (!ocl_context.SetKernelArg(idx_name, 0, idx_name)) {
        std::cout << "ERROR: Failed to set first kernel arg for \"" << idx_name
                  << '"' << std::endl;
        return 9;
      }
      if (!ocl_context.SetKernelArg(idx_name, 1, "shared")) {
        std::cout << "ERROR: Failed to set second kernel arg for \"" << idx_name
                  << '"' << std::endl;
        return 10;
      }
    }
  }

  std::vector<cl_event> events{};
  std::vector<cl_event> next_events{};
  for (unsigned int stage_idx = 0; stage_idx < stages.size(); ++stage_idx) {
    for (unsigned int id : stages.at(stage_idx)) {
      cl_event event;
      std::string name = std::to_string(id);
      if (!ocl_context.ExecuteKernel(name, events.size(),
                                     events.empty() ? nullptr : events.data(),
                                     &event)) {
        std::cout << "ERROR: Failed to Execute kernel \"" << name << '"'
                  << std::endl;
        return 11;
      }
      next_events.push_back(event);
    }
    events = std::move(next_events);
    next_events = {};

    if (args.print_intermediate_steps_) {
      cl_int err_number = clWaitForEvents(events.size(), events.data());
      if (err_number != CL_SUCCESS) {
        std::cout << "ERROR: Failed to wait for events while printing "
                     "intermediate steps"
                  << std::endl;
        return 12;
      }
      unsigned int value = 0;
      err_number = clEnqueueReadBuffer(
          ocl_context.GetCommandQueue(), ocl_context.GetBuffer("shared"),
          CL_TRUE, 0, sizeof(unsigned int), &value, 0, nullptr, nullptr);
      if (err_number != CL_SUCCESS) {
        std::cout << "ERROR: Failed to read shared buffer" << std::endl;
        return 13;
      }
      std::cout << "Shared value after stage " << stage_idx << " is " << value
                << std::endl;
    }
  }

  std::cout << "Program executed successfully" << std::endl;
  return 0;
}
