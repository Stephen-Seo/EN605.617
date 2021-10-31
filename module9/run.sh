#!/usr/bin/env bash

function on_exit() {
    #make clean || true
    popd &>/dev/null || true
}

function print_usage() {
    echo "Usage:"
    echo "  thrust_test_harness"
    echo "  thrust [-h | --help] [-p | -t] [-s <int>] [-a | -u | -m | -o]"
    echo "    -p is print outputs"
    echo "    -t is print timings"
    echo "    -s <int> is set size of array used"
    echo "    -a is use addition"
    echo "    -u is use subtraction"
    echo "    -m is use multiplication"
    echo "    -o is use modulus"
}

function thrust_assignment() {
    make thrust_assignment9
    ./thrust_assignment9 "$@"
}

trap on_exit EXIT

set -e

pushd "$(dirname "$0")" &>/dev/null

if [[ "$1" == "thrust_test_harness" ]]; then
    make thrust_assignment9
    echo -e "Thrust test harness: size 1024 and $((2 ** 20)) of each operation"
    echo -e "\nFirst test: testing with size 1024...\n"
    ./thrust_assignment9 -s 1024 -t -a -u -m -o | grep Average
    echo -e "\nSecond test: testing with size $((2 ** 20))...\n"
    ./thrust_assignment9 -s $((2 ** 20)) -t -a -u -m -o | grep Average
    echo -e "\nTest harness of thrust finished"
elif [[ "$1" == "thrust" ]]; then
    shift 1
    thrust_assignment "$@"
else
    echo "Invalid arg(s) \"$@\""
    print_usage
fi
