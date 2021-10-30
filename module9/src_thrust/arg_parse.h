#ifndef IGPUP_MODULE_9_THRUST_ARG_PARSE_H
#define IGPUP_MODULE_9_THRUST_ARG_PARSE_H

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
  unsigned int thrust_size;
};

#endif
