#ifndef IGPUP_MODULE_12_OPENCL_ARG_PARSE_H
#define IGPUP_MODULE_12_OPENCL_ARG_PARSE_H

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

  bool do_timings_;
  bool use_alt_kernel_;
};

#endif
