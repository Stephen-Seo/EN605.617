#!/usr/bin/env bash

function on_exit() {
    #make clean || true
    popd &>/dev/null || true
}

trap on_exit EXIT

set -e

pushd "$(dirname "$0")" &>/dev/null

make assignment6

./assignment6 "$@"
