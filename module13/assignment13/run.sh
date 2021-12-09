#!/usr/bin/env bash

function on_exit() {
    #make clean || true
    popd &>/dev/null || true
}

function print_usage() {
    echo "Usage:"
    echo "  test_harness"
    echo "  ecq [-h] [-i <input_csv>] [-p]"
    echo "    -h              Print help text"
    echo "    -i <input_csv>  use the given csv to define dependencies"
    echo "    -p              print intermediate steps"
    echo "    -t              do timings"
    echo -n "Note: -p and -t are mutually exclusive. If both are specified, "
    echo "then -t will take precedence."
}

function opencl_assignment() {
    make assignment13
    ./assignment13 "$@"
}

trap on_exit EXIT

set -e

pushd "$(dirname "$0")" &>/dev/null

if [[ "$1" == "test_harness" ]]; then
    make assignment13
    echo -e "\nTesting 5 runs of example.csv"
    for ((i = 0; i < 5; ++i)); do
        ./assignment13 -i example.csv -t | grep '=='
    done
    echo -e "\nTesting 5 runs of example2.csv"
    for ((i = 0; i < 5; ++i)); do
        ./assignment13 -i example2.csv -t | grep '=='
    done
    echo -e "\nTesting 5 runs of example3.csv"
    for ((i = 0; i < 5; ++i)); do
        ./assignment13 -i example3.csv -t | grep '=='
    done
    echo -e "\nTesting 5 runs of example4.csv"
    for ((i = 0; i < 5; ++i)); do
        ./assignment13 -i example4.csv -t | grep '=='
    done
elif [[ "$1" == "ecq" ]]; then
    shift 1
    opencl_assignment "$@"
elif (( $# == 0 )); then
    echo "ERROR: Run this script with \"test_harness\" or \"ecq\""
    print_usage
    exit 1
else
    echo "Invalid arg(s) \"$@\""
    print_usage
fi
