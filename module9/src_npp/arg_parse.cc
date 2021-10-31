#include "arg_parse.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

Args::Args()
    : enable_print_output(false),
      enable_timings(false),
      input_filename(),
      do_rotation(false) {}

Args::~Args() {}

bool Args::ParseArgs(int argc, char **argv) {
  --argc;
  ++argv;
  while (argc > 0) {
    if (std::strcmp(argv[0], "-h") == 0 ||
        std::strcmp(argv[0], "--help") == 0) {
      Args::DisplayHelp();
      return true;
    } else if (std::strcmp(argv[0], "-p") == 0 ||
               std::strcmp(argv[0], "--print-output") == 0) {
      enable_print_output = true;
    } else if (std::strcmp(argv[0], "-t") == 0 ||
               std::strcmp(argv[0], "--timings") == 0) {
      enable_timings = true;
    } else if ((std::strcmp(argv[0], "-f") == 0 ||
                std::strcmp(argv[0], "--filename") == 0) &&
               argc > 1) {
      input_filename = std::string(argv[1]);
      --argc;
      ++argv;
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
               "  -h | --help\t\t\t\tprint this help text\n"
               "  -p | --print-output\t\t\tprint result outputs when running "
               "algorithms\n"
               "  -t | --timings\t\t\tprint timings\n"
               "  -f <filename> | --filename <filename>\tset input file\n"
               "    Note that -p and -t are mutually exclusive.\n"
               "    If both are specified, -t takes precedence.\n"
               "\n    If using run.sh, prepend flag(s) with \"npp\"\n";
}
