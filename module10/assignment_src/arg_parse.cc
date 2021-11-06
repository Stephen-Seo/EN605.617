#include "arg_parse.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

Args::Args()
    : do_addition(false),
      do_subtraction(false),
      do_multiplication(false),
      do_modulus(false),
      do_power(false) {}

Args::~Args() {}

bool Args::ParseArgs(int argc, char **argv) {
  --argc;
  ++argv;
  while (argc > 0) {
    if (std::strcmp(argv[0], "-h") == 0 ||
        std::strcmp(argv[0], "--help") == 0) {
      Args::DisplayHelp();
      return true;
    } else if (std::strcmp(argv[0], "-a") == 0 ||
               std::strcmp(argv[0], "--add") == 0) {
      do_addition = true;
    } else if (std::strcmp(argv[0], "-s") == 0 ||
               std::strcmp(argv[0], "--sub") == 0) {
      do_subtraction = true;
    } else if (std::strcmp(argv[0], "-m") == 0 ||
               std::strcmp(argv[0], "--mult") == 0) {
      do_multiplication = true;
    } else if (std::strcmp(argv[0], "-o") == 0 ||
               std::strcmp(argv[0], "--mod") == 0) {
      do_modulus = true;
    } else if (std::strcmp(argv[0], "-p") == 0 ||
               std::strcmp(argv[0], "--pow") == 0) {
      do_power = true;
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
               "  -a | --add\t\t\tRun add operation\n"
               "  -s | --sub\t\t\tRun subtract operation\n"
               "  -m | --mult\t\t\tRun multiply operation\n"
               "  -o | --mod\t\t\tRun modulus operation\n"
               "  -p | --pow\t\t\tRun power operation\n";
}
