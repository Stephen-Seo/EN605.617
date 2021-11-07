#!/usr/bin/env bash

function on_exit() {
    #make clean || true
    popd &>/dev/null || true
}

function print_usage() {
    echo "Usage:"
    echo "  excode_test_harness"
    echo "  excode [-h] [-t] [-a] [-s] [-m] [-d] [-p]"
    echo "    -h     Print this help text"
    echo "    -t     Time the operation being run"
    echo "    -a     Run add operation"
    echo "    -s     Run subtract operation"
    echo "    -m     Run multiply operation"
    echo "    -d     Run division operation"
    echo "    -p     Run power operation"
}

function opencl_assignment() {
    make assignment10
    ./assignment10 "$@"
}

trap on_exit EXIT

set -e

pushd "$(dirname "$0")" &>/dev/null

if [[ "$1" == "excode_test_harness" ]]; then
    make assignment10
    ./assignment10 -t -a -s -m -d -p | grep Average
elif [[ "$1" == "excode" ]]; then
    shift 1
    opencl_assignment "$@"
elif (( $# == 0 )); then
    echo "ERROR: Run this script with \"excode_test_harness\" or \"excode\""
    print_usage
    exit 1
else
    echo "Invalid arg(s) \"$@\""
    print_usage
fi
