#include "get_exe_dirname.h"

#include <iostream>
#include <regex>

std::string GetExeDirName(const char *argv_zero) {
  std::regex re("^(.+)/[^/]+$");
  std::cmatch match;

  if (std::regex_match(argv_zero, match, re) && !match.empty() &&
      match.size() > 1 && match[1].first == argv_zero) {
    return std::string(match[1].first, match[1].second);
  }

  return {};
}
