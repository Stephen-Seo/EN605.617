#ifndef IGPUP_MODULE_5_CONSTANTS_H
#define IGPUP_MODULE_5_CONSTANTS_H

// This fits 64KiB of data for constant case.
// Therefore, increasing these constants may break the const algorithm.
constexpr unsigned int NUM_BLOCKS = 64;
constexpr unsigned int NUM_THREADS = 128;

#endif
