#ifndef IGPUP_MODULE_13_EVENTS_COMMAND_QUEUES_ARG_PARSE_H_
#define IGPUP_MODULE_13_EVENTS_COMMAND_QUEUES_ARG_PARSE_H_

#include <string>

struct Args {
  Args();

  /// Returns true on success
  bool ParseArgs(int argc, char **argv);

  static void PrintUsage();

  std::string input_filename_;
  bool help_printed_;
  bool print_intermediate_steps_;
  bool do_timings_;
};

#endif
