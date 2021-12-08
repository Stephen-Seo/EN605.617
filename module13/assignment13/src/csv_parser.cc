#include "csv_parser.h"

#include <fstream>
#include <iostream>
#include <memory>

Dependencies DepsCSVParser::GetDepsFromCSV(const std::string &filename) {
  Dependencies deps{};

  std::ifstream ifs(filename);
  if (!(ifs.is_open() && ifs.good())) {
    std::cout << "ERROR DepsCSVParser: Failed to open filename" << std::endl;
    return {};
  }

  // std::optional would be better, but using C++11 not C++17
  std::unique_ptr<unsigned int> prev_number{};
  std::unique_ptr<unsigned int> number{};
  int char_buf;
  unsigned int line_number = 1;
  while (ifs.good()) {
    char_buf = ifs.get();
    if (char_buf == decltype(ifs)::traits_type::eof()) {
      break;
    } else if (char_buf >= '0' && char_buf <= '9') {
      if (!number) {
        number = std::unique_ptr<unsigned int>(new unsigned int(0));
      }
      *number = *number * 10 + char_buf - '0';
      continue;
    } else if (char_buf == ',') {
      if (prev_number) {
        deps.AddDependency(*prev_number, *number);
      }
      prev_number = std::unique_ptr<unsigned int>(new unsigned int(*number));
      number.reset();
    } else if (char_buf == '\n') {
      if (prev_number && number) {
        deps.AddDependency(*prev_number, *number);
      } else if ((prev_number && !number) || (!prev_number && number)) {
        std::cout
            << "ERROR DepsCSVParser: Got newline but only one number in line "
            << line_number << std::endl;
        return {};
      }
      prev_number.reset();
      number.reset();
      ++line_number;
    } else if (char_buf == ' ' || char_buf == '\t') {
      continue;
    } else {
      // invalid character
      std::cout << "ERROR DepsCSVParser: Got invalid character " << char_buf
                << " at line number " << line_number << std::endl;
      return {};
    }
  }
  if (ifs.eof()) {
    if (prev_number && number) {
      deps.AddDependency(*prev_number, *number);
    } else if ((prev_number && !number) || (!prev_number && number)) {
      std::cout << "ERROR DepsCSVParser: EOF, but only one number in last line"
                << std::endl;
      return {};
    }
  }

  return deps;
}
