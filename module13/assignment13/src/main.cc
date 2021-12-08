#include <fstream>
#include <iostream>

#include "arg_parse.h"
#include "csv_parser.h"
#include "dependencies.h"

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

  std::cout << "Program executed successfully" << std::endl;
  return 0;
}
