#include "arg_parse.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

Args::Args()
    : enable_print_output(false),
      enable_timings(false),
      thrust_size(0),
      do_addition(false),
      do_subtraction(false),
      do_multiplication(false),
      do_modulus(false) {}

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
    } else if ((std::strcmp(argv[0], "-s") == 0 ||
                std::strcmp(argv[0], "--size") == 0) &&
               argc > 1) {
      thrust_size = std::strtoul(argv[1], nullptr, 10);
      --argc;
      ++argv;
    } else if (std::strcmp(argv[0], "-a") == 0 ||
               std::strcmp(argv[0], "--add") == 0) {
      do_addition = true;
    } else if (std::strcmp(argv[0], "-u") == 0 ||
               std::strcmp(argv[0], "--sub") == 0) {
      do_subtraction = true;
    } else if (std::strcmp(argv[0], "-m") == 0 ||
               std::strcmp(argv[0], "--mult") == 0) {
      do_multiplication = true;
    } else if (std::strcmp(argv[0], "-o") == 0 ||
               std::strcmp(argv[0], "--mod") == 0) {
      do_modulus = true;
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
               "  -h | --help\t\t\tprint this help text\n"
               "  -p | --print-output\t\tprint result outputs when running "
               "algorithms\n"
               "  -t | --timings\t\tprint timings\n"
               "  -s <int> | --size <int>\tset size of array used by thrust\n"
               "  -a | --add\t\t\tRun add algorithm\n"
               "  -u | --sub\t\t\tRun subtract algorithm\n"
               "  -m | --mult\t\t\tRun multiply algorithm\n"
               "  -o | --mod\t\t\tRun modulus algorithm\n"
               "    Note that -p and -t are mutually exclusive.\n"
               "    If both are specified, -t takes precedence.\n"
               "\n    If using run.sh, prepend flag(s) with \"thrust\"\n";
}
