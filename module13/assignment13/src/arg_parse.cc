#include "arg_parse.h"

#include <cstring>
#include <iostream>

Args::Args() : input_filename() {}

bool Args::ParseArgs(int argc, char **argv) {
  --argc;
  ++argv;
  if (argc == 1) {
    input_filename = argv[0];
  } else {
    std::cout
        << "ERROR: Expected one argument <input_csv> (should be a filename)"
        << std::endl;
    PrintUsage();
    return false;
  }
  return true;
}

void Args::PrintUsage() {
  std::cout << "Usage: <input_csv>\n"
               "  <input_csv> must be a csv file where each entry is a "
               "dependency list of integers\nExample csv entry:\n"
               "  50, 23, 1, 7\n50 depends on 23, 23 depends on 1, etc...\n"
               "Multiple lists define more dependencies, but there must not be "
               "a cycle"
            << std::endl;
}
