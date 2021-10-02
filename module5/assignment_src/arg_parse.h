#ifndef IGPUP_MODULE_5_ARG_PARSE_H
#define IGPUP_MODULE_5_ARG_PARSE_H

struct Args {
    Args();
    ~Args();

    // enable copy
    Args(const Args &other) = default;
    Args& operator=(const Args &other) = default;

    // enable move
    Args(Args &&other) = default;
    Args& operator=(Args &&other) = default;

    /// Returns true if help was printed
    bool parseArgs(int argc, char **argv);
    static void displayHelp();

    bool runShared;
    bool runConstant;
    bool enablePrintOutput;
    bool enableTimings;
};

#endif
