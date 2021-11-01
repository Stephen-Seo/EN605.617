#include "arg_parse.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

Args::Args()
    : enable_print_output(false),
      enable_timings(false),
      randomize_weights(false),
      seed_random_with_time(false),
      use_alternate_graph(false) {}

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
               std::strcmp(argv[0], "--print") == 0) {
      enable_print_output = true;
    } else if (std::strcmp(argv[0], "-t") == 0 ||
               std::strcmp(argv[0], "--timings") == 0) {
      enable_timings = true;
    } else if (std::strcmp(argv[0], "-r") == 0 ||
               std::strcmp(argv[0], "--randomize") == 0) {
      randomize_weights = true;
    } else if (std::strcmp(argv[0], "-s") == 0 ||
               std::strcmp(argv[0], "--seed") == 0) {
      seed_random_with_time = true;
    } else if (std::strcmp(argv[0], "-a") == 0 ||
               std::strcmp(argv[0], "--alternate") == 0) {
      use_alternate_graph = true;
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
               "  -h | --help\t\tprint this help text\n"
               "  -p | --print\t\tprint outputs\n"
               "  -t | --timings\tprint timings\n"
               "  -r | --randomize\trandomize weights\n"
               "  -s | --seed\t\tseed random with current time (not 0)\n"
               "  -a | --alternate\tuse alternate (bigger) graph\n"
               "    If both -p and -t are specified, -t takes precedence\n"
               "\n    If using run.sh, prepend flag(s) with \"nvgraph\"\n";
}
