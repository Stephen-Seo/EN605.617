#include "arg_parse.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "constants.h"

Args::Args()
    : enable_print_output(false),
      enable_timings(false),
      input_filename(),
      output_filename(DEFAULT_OUTPUT_FILENAME),
      angle(DEFAULT_ROTATION_DEGREES),
      overwrite(false) {}

Args::~Args() {}

bool Args::ParseArgs(int argc, char **argv) {
  --argc;
  ++argv;
  while (argc > 0) {
    if (std::strcmp(argv[0], "-h") == 0 ||
        std::strcmp(argv[0], "--help") == 0) {
      Args::DisplayHelp();
      return true;
    } else if (std::strcmp(argv[0], "-t") == 0 ||
               std::strcmp(argv[0], "--timings") == 0) {
      enable_timings = true;
    } else if (std::strcmp(argv[0], "--input-filename") == 0 && argc > 1) {
      input_filename = std::string(argv[1]);
      --argc;
      ++argv;
    } else if (std::strcmp(argv[0], "--output-filename") == 0 && argc > 1) {
      output_filename = std::string(argv[1]);
      --argc;
      ++argv;
    } else if ((std::strcmp(argv[0], "-a") == 0 ||
                std::strcmp(argv[0], "--angle") == 0) &&
               argc > 1) {
      angle = strtod(argv[1], nullptr);
      --argc;
      ++argv;
    } else if (std::strcmp(argv[0], "--overwrite") == 0) {
      overwrite = true;
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
               "  -t | --timings\t\t\tprint timings\n"
               "  --input-filename <filename>\t\tset input file\n"
               "  --output-filename <filename>\t\tset output file\n"
               "  --overwrite\t\t\t\tenable overwriting output-filename\n"
               "  -a <degrees> | --angle <degrees>\tset rotation angle\n"
               "\n    If using run.sh, prepend flag(s) with \"npp\"\n";
}
