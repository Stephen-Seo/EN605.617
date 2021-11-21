#!/usr/bin/env bash

function on_exit() {
    #make clean || true
    popd &>/dev/null || true
}

function print_usage() {
    echo "Usage:"
    echo "  test_harness"
    echo "  bsb [-h] [-t] [-a]"
    echo "    -h  Print help text"
    echo "    -t  Time the executions of the OpenCL kernel"
    echo "    -a  Use the alternate kernel"
    echo -n "    Note that by default, using \"bsb\" with no flags will run "
    echo -n "the default behavior, which is to print the inputs/outputs with "
    echo "the kernel using \"sub-buffers\""
}

function opencl_assignment() {
    make assignment12
    ./assignment12 "$@"
}

trap on_exit EXIT

set -e

pushd "$(dirname "$0")" &>/dev/null

if [[ "$1" == "test_harness" ]]; then
    make assignment12
    for ((i=0; i < 5; ++i)); do
        ./assignment12 -t | grep Average
    done
    for ((i=0; i < 5; ++i)); do
        ./assignment12 -t -a | grep Average
    done
elif [[ "$1" == "bsb" ]]; then
    shift 1
    opencl_assignment "$@"
elif (( $# == 0 )); then
    echo "ERROR: Run this script with \"test_harness\" or \"bsb\""
    print_usage
    exit 1
else
    echo "Invalid arg(s) \"$@\""
    print_usage
fi
