#!/usr/bin/env bash

function print_help() {
    echo "Usage:"
    echo "  -a          run paged memory algorithm (use with \"-p\" or \"-t\")"
    echo "  -i          run pinned memory algorithm (use with \"-p\" or \"-t\")"
    echo "  -p          enable printing result output"
    echo "  -t          enable timings when running paged/pinned"
    echo "  -c          run caesar cipher algorithm (using \"-p\" is advised)"
    echo "  -o <offset> use <offset> when running caesar cipher algorithm"
}

function on_exit() {
    make clean || true
    popd &>/dev/null || true
}

trap on_exit EXIT

set -e

pushd "$(dirname "$0")" >&/dev/null

USE_PAGED_MEMORY_ALGO=0
USE_PINNED_MEMORY_ALGO=0
PRINT_RESULTS=0
USE_TIMINGS=0
USE_CAESAR_CIPHER_ALGO=0
CIPHER_ALGO_OFFSET=3

while getopts "aiptco:h" arg; do
    case "$arg" in
        'a') USE_PAGED_MEMORY_ALGO=1;;
        'i') USE_PINNED_MEMORY_ALGO=1;;
        'p') PRINT_RESULTS=1;;
        't') USE_TIMINGS=1;;
        'c') USE_CAESAR_CIPHER_ALGO=1;;
        'o') CIPHER_ALGO_OFFSET="$OPTARG";;
        'h') print_help; exit 0;;
    esac
done

if (( !USE_PAGED_MEMORY_ALGO && !USE_PINNED_MEMORY_ALGO \
        && !USE_CAESAR_CIPHER_ALGO )); then
    print_help
    exit 0
fi

make assignment.exe

if (( USE_PAGED_MEMORY_ALGO )); then
    if (( USE_TIMINGS )); then
        ./assignment.exe --use-paged --enable-timings
    elif (( PRINT_RESULTS )); then
        ./assignment.exe --use-paged --print-results
    else
        ./assignment.exe --use-paged
    fi
fi

if (( USE_PINNED_MEMORY_ALGO )); then
    if (( USE_TIMINGS )); then
        ./assignment.exe --use-pinned --enable-timings
    elif (( PRINT_RESULTS )); then
        ./assignment.exe --use-pinned --print-results
    else
        ./assignment.exe --use-pinned
    fi
fi

if (( USE_CAESAR_CIPHER_ALGO )); then
    if (( PRINT_RESULTS )); then
        ./assignment.exe --use-cipher \
            --use-cipher-offset "$CIPHER_ALGO_OFFSET" \
            --print-results
    else
        ./assignment.exe --use-cipher \
            --use-cipher-offset "$CIPHER_ALGO_OFFSET"
    fi
fi
