#!/usr/bin/env bash

SCRIPT_DIR="$(dirname "$0")"
cd $SCRIPT_DIR

set -ve

if ! [ -e ./helloworld_cpp ]; then
    make helloworld_cpp
fi
./helloworld_cpp

if ! [ -e ./helloworld_cuda ]; then
    make helloworld_cuda
fi
./helloworld_cuda

if ! [ -e ./helloworld_opencl ]; then
    make helloworld_opencl
fi
./helloworld_opencl

make clean
