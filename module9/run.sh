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
    echo "  npp_test_harness"
    echo "  npp [-h | --help] [-t] [-a <degrees>] [--input-filename <filename>] [--output-filename <filename>] [--overwrite]"
    echo "    -t is print timings"
    echo "    -a <degrees> is set rotation angle"
    echo "    --intput-filename <filename> sets the input image filename (must be .pgm)"
    echo "    --output-filename <filename sets the output image filename (must end in .pgm)"
    echo "    --overwrite allows overwriting output filename"
    echo "  nvgraph_test_harness"
    echo "  nvgraph [-h | --help] [-p | -t] [-r] [-s] [-a]"
    echo "    -p prints output of shortest path nvgraph algorithm"
    echo "    -t times usage of nvgraph algorithm"
    echo "    -r uses random edge weights (default seeded by 0)"
    echo "    -s uses the time to seed the random weights (instead of 0)"
    echo "    -a uses an alternate (bigger) graph"
}

function thrust_assignment() {
    make thrust_assignment9
    ./thrust_assignment9 "$@"
}

function npp_assignment() {
    make npp_assignment9
    ./npp_assignment9 "$@"
}

function nvgraph_assignment() {
    make nvgraph_assignment9
    ./nvgraph_assignment9 "$@"
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
elif [[ "$1" == "npp_test_harness" ]]; then
    make npp_assignment9
    echo -e "\nNPP test harness: using timings with louie_lq.pgm image\n"
    ./npp_assignment9 -t --input-filename louie_lq.pgm | grep Average
    echo -e "\nNPP test harness: using timings with louie_hq.pgm (larger) image\n"
    ./npp_assignment9 -t --input-filename louie_hq.pgm | grep Average
elif [[ "$1" == "npp" ]]; then
    shift 1
    npp_assignment "$@"
elif [[ "$1" == "nvgraph_test_harness" ]]; then
    make nvgraph_assignment9
    echo -e "\nnvgraph test harness: using timings with regular graph with random edge weights\n"
    ./nvgraph_assignment9 -t -r -s | grep Average
    echo -e "\nnvgraph test harness: using timings with alternate bigger graph with random edge weights\n"
    ./nvgraph_assignment9 -t -r -s -a | grep Average
elif [[ "$1" == "nvgraph" ]]; then
    shift 1
    nvgraph_assignment "$@"
else
    echo "Invalid arg(s) \"$@\""
    print_usage
fi
