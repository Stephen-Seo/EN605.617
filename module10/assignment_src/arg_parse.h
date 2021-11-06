#ifndef IGPUP_MODULE_10_OPENCL_ARG_PARSE_H
#define IGPUP_MODULE_10_OPENCL_ARG_PARSE_H

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

  bool do_addition;
  bool do_subtraction;
  bool do_multiplication;
  bool do_modulus;
  bool do_power;
};

#endif
