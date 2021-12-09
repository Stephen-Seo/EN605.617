#include "arg_parse.h"

#include <cstring>
#include <iostream>

Args::Args()
    : input_filename_(),
      help_printed_(false),
      print_intermediate_steps_(false) {}

bool Args::ParseArgs(int argc, char **argv) {
  --argc;
  ++argv;

  while (argc > 0) {
    if (argc > 1 && (std::strcmp(argv[0], "-i") == 0 ||
                     std::strcmp(argv[0], "--input") == 0)) {
      input_filename_ = argv[1];
      --argc;
      ++argv;
    } else if (std::strcmp(argv[0], "-p") == 0 ||
               std::strcmp(argv[0], "--print") == 0) {
      print_intermediate_steps_ = true;
    } else if (std::strcmp(argv[0], "-h") == 0 ||
               std::strcmp(argv[0], "--help") == 0) {
      PrintUsage();
      help_printed_ = true;
      return true;
    } else {
      std::cout << "Ignoring invalid arg \"" << argv[0] << '"' << std::endl;
    }
    --argc;
    ++argv;
  }

  return true;
}

void Args::PrintUsage() {
  std::cout << "Usage: [-h | --help] [-i <input_csv> | --input <input_csv>] "
               "[-p | --print]\n"
               "  <input_csv> must be a csv file where each entry is a "
               "dependency list of integers\nExample csv entry:\n"
               "  50, 23, 1, 7\n50 depends on 23, 23 depends on 1, etc...\n"
               "Multiple lists define more dependencies, but there must not be "
               "a cycle\n"
               "  -h | --help\tPrint this usage text\n"
               "  -p | --print\tPrint current value in between stages"
            << std::endl;
}
