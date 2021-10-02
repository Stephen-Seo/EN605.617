#include "arg_parse.h"

#include <cstring>
#include <iostream>

Args::Args() :
    runShared(false),
    runConstant(false),
    enablePrintOutput(false),
    enableTimings(false)
{}

Args::~Args() {}

bool Args::parseArgs(int argc, char **argv) {
    --argc; ++argv;
    while (argc > 0) {
        if (std::strcmp(argv[0], "-h") == 0
                || std::strcmp(argv[0], "--help") == 0) {
            Args::displayHelp();
            return true;
        } else if (std::strcmp(argv[0], "-s") == 0
                || std::strcmp(argv[0], "--shared") == 0) {
            runShared = true;
        } else if (std::strcmp(argv[0], "-c") == 0
                || std::strcmp(argv[0], "--constant") == 0) {
            runConstant = true;
        } else if (std::strcmp(argv[0], "-p") == 0
                || std::strcmp(argv[0], "--print-output") == 0) {
            enablePrintOutput = true;
        } else if (std::strcmp(argv[0], "-t") == 0
                || std::strcmp(argv[0], "--timings") == 0) {
            enableTimings = true;
        } else {
            std::cout << "Ignoring invalid argument \"" << argv[0] << "\"\n";
        }
        --argc; ++argv;
    }

    return false;
}

void Args::displayHelp() {
    std::cout << "Usage:\n"
        "  -h | --help\t\tprint this help text\n"
        "  -s | --shared\t\trun algorithm using shared memory\n"
        "  -c | --constant\trun algorithm using constant memory\n"
        "  -p | --print-output\tprint result outputs when running algorithms\n"
        "  -t | --timings\tprint timings\n"
        "    Note that -s and -c are mutually exclusive.\n"
        "    If both are specified, -s takes precedence.\n"
        "    Note that -p and -t are mutually exclusive.\n"
        "    If both are specified, -t takes precedence.\n";
}
