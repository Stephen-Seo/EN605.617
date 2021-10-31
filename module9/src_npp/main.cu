#include "arg_parse.h"
#include "pgm_rw.h"

#include <iostream>

int main(int argc, char **argv) {
  Args args;

  if (args.ParseArgs(argc, argv)) {
    return 0;
  }

  PGMFile pgm;

  if (!pgm.LoadImage(args.input_filename)) {
    std::cout << "Failed to LoadImage" << std::endl;
    return 1;
  }

  if (!pgm.SaveImage("test.pgm")) {
    std::cout << "Failed to SaveImage" << std::endl;
    return 1;
  }

  return 0;
}
