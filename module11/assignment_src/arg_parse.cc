#include "arg_parse.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

Args::Args() : randomize_signal_(false), do_timings_(false) {}

Args::~Args() {}

bool Args::ParseArgs(int argc, char **argv) {
  --argc;
  ++argv;
  while (argc > 0) {
    if (std::strcmp(argv[0], "-h") == 0 ||
        std::strcmp(argv[0], "--help") == 0) {
      Args::DisplayHelp();
      return true;
    } else if (std::strcmp(argv[0], "-r") == 0 ||
               std::strcmp(argv[0], "--randomize") == 0) {
      randomize_signal_ = true;
    } else if (std::strcmp(argv[0], "-t") == 0 ||
               std::strcmp(argv[0], "--timings") == 0) {
      do_timings_ = true;
    } else {
      std::cout << "Ignoring invalid argument \"" << argv[0] << "\"\n";
    }
    --argc;
    ++argv;
  }

  return false;
}

void Args::DisplayHelp() {
  std::cout << "Usage:\n"
               "  -r | --randomize\t\tRandomize signal with seed from time\n"
               "  -t | --timings\t\tDo timings of kernel execution\n"
               "\n    If using run.sh, prefix flags with \"clc\"\n";
}
