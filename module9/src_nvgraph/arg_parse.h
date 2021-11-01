#ifndef IGPUP_MODULE_9_NVGRAPH_ARG_PARSE_H_
#define IGPUP_MODULE_9_NVGRAPH_ARG_PARSE_H_

#include <string>

struct Args {
  Args();
  ~Args();

  // enable copy
  Args(const Args &other) = default;
  Args &operator=(const Args &other) = default;

  // enable move
  Args(Args &&other) = default;
  Args &operator=(Args &&other) = default;

  /// Returns true if help was printed
  bool ParseArgs(int argc, char **argv);
  static void DisplayHelp();

  bool enable_print_output;
  bool enable_timings;
  bool randomize_weights;
  bool seed_random_with_time;
  bool use_alternate_graph;
};

#endif
