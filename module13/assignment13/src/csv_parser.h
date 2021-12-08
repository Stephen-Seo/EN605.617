#ifndef IGPUP_MODULE_13_EVENTS_COMMAND_QUEUES_CSV_PARSER_H_
#define IGPUP_MODULE_13_EVENTS_COMMAND_QUEUES_CSV_PARSER_H_

#include <string>

#include "dependencies.h"

namespace DepsCSVParser {
Dependencies GetDepsFromCSV(const std::string &filename);
}  // namespace DepsCSVParser

#endif
