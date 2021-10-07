#include "arg_parse.h"

#include <cstring>
#include <iostream>

Args::Args() :
    runRegisterBasedMemory(false),
    runSharedBasedMemory(false),
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
        } else if (std::strcmp(argv[0], "-r") == 0
                || std::strcmp(argv[0], "--register") == 0) {
            runRegisterBasedMemory = true;
        } else if (std::strcmp(argv[0], "-s") == 0
                || std::strcmp(argv[0], "--shared") == 0) {
            runSharedBasedMemory = true;
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
        "  -r | --register\trun algorithm using registers\n"
        "  -s | --shared\t\trun algorithm using shared memory\n"
        "  -p | --print-output\tprint result outputs when running algorithms\n"
        "  -t | --timings\tprint timings\n"
        "    Note that -r and -s are mutually exclusive.\n"
        "    If both are specified, -r takes precedence.\n"
        "    Note that -p and -t are mutually exclusive.\n"
        "    If both are specified, -t takes precedence.\n";
}
