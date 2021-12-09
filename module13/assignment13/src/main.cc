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
  } else if (args.input_filename.empty()) {
    std::cout << "ERROR: Input filename is an empty string" << std::endl;
    return 2;
  }

  Dependencies deps = DepsCSVParser::GetDepsFromCSV(args.input_filename);
  if (deps.IsEmpty()) {
    std::cout << "ERROR: Got emtpy Dependencies object from CSV \""
              << args.input_filename << '"' << std::endl;
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
        return 9;
      }
    }
  }

  std::cout << "Program executed successfully" << std::endl;
  return 0;
}
