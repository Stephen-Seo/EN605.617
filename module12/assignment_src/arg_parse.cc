#include "arg_parse.h"

#include <cstring>
#include <iostream>

Args::Args() : do_timings_(false), use_alt_kernel_(false) {}

Args::~Args() {}

bool Args::ParseArgs(int argc, char **argv) {
  --argc;
  ++argv;
  while (argc > 0) {
    if (std::strcmp(argv[0], "-h") == 0 ||
        std::strcmp(argv[0], "--help") == 0) {
      DisplayHelp();
      return true;
    } else if (std::strcmp(argv[0], "-t") == 0 ||
               std::strcmp(argv[0], "--timings") == 0) {
      do_timings_ = true;
    } else if (std::strcmp(argv[0], "-a") == 0 ||
               std::strcmp(argv[0], "--alt") == 0) {
      use_alt_kernel_ = true;
    } else {
      std::cout << "WARNING: Ignoring invalid argument \"" << argv[0] << '"'
                << std::endl;
    }
    --argc;
    ++argv;
  }

  return false;
}

void Args::DisplayHelp() {
  std::cout << "[-h | --help] [-t | --timings] [-a | --alt]\n"
               "  -h | --help\t\tDisplay this usage text\n"
               "  -t | --timings\t\tGet timings of kernel execution(s)\n"
               "  -a | --alt\t\tUse alternate kernel"
            << std::endl;
}
