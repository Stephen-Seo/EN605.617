#!/usr/bin/env bash

function run_simple_helloworld() {
    if ! [ -e ./helloworld_cpp ]; then
        make helloworld_cpp
    fi
    ./helloworld_cpp
}

function run_cuda_helloworld() {
    if ! [ -e ./helloworld_cuda ]; then
        make helloworld_cuda
    fi
    ./helloworld_cuda
}

function run_opencl_helloworld() {
    if ! [ -e ./helloworld_opencl ]; then
        make helloworld_opencl
    fi
    ./helloworld_opencl
}

function print_help() {
    echo "Usage:"
    echo "  -a      run all helloworld programs"
    echo "  -s      run simple helloworld program"
    echo "  -c      run cuda helloworld program"
    echo "  -o      run opencl helloworld program"
    echo "  -h      print this help text"
}

SCRIPT_DIR="$(dirname "$0")"
cd $SCRIPT_DIR

set -e

while getopts "ascoh" arg; do
    case "$arg" in
        'a')
            echo "Executing all helloworld programs..."
            run_simple_helloworld
            run_cuda_helloworld
            run_opencl_helloworld
            make clean
            exit 0;;
        's')
            echo "Executing simple helloworld program..."
            run_simple_helloworld
            make clean
            exit 0;;
        'c')
            echo "Executing cuda helloworld program..."
            run_cuda_helloworld
            make clean
            exit 0;;
        'o')
            echo "Executing opencl helloworld program..."
            run_opencl_helloworld
            make clean
            exit 0;;
        'h')
            print_help
            exit 0;;
        '?')
            print_help
            exit 1;;
    esac
done

print_help
