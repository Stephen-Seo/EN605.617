#!/usr/bin/env bash

function on_exit() {
    #make clean || true
    popd &>/dev/null || true
}

function print_usage() {
    echo "Usage:"
    echo "  clc_test_harness"
    echo "  clc [-h] [-r] [-t]"
    echo "    -h  Print help text"
    echo "    -r  Use time to seed the randomized signal"
    echo "    -t  Time the executions of the OpenCL kernel"
    echo -n "    Note that by default, using \"clc\" with no flags will run "
    echo -n "the default behavior, which is to print the signal, run the "
    echo "convolution, and output the resulting data"
}

function opencl_assignment() {
    make assignment11
    ./assignment11 "$@"
}

trap on_exit EXIT

set -e

pushd "$(dirname "$0")" &>/dev/null

if [[ "$1" == "clc_test_harness" ]]; then
    make assignment11
    for ((i=0; i < 5; ++i)); do
        ./assignment11 -t -r | grep Average
    done
elif [[ "$1" == "clc" ]]; then
    shift 1
    opencl_assignment "$@"
elif (( $# == 0 )); then
    echo "ERROR: Run this script with \"clc_test_harness\" or \"clc\""
    print_usage
    exit 1
else
    echo "Invalid arg(s) \"$@\""
    print_usage
fi
