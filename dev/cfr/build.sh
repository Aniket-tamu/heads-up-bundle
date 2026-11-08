#!/bin/sh
# Build the offline feature library dev/cfr/out/libfeatures.so (gcc, C11, OpenMP).
# Offline only: nothing compiled is shipped with the bot.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HERE/out"
gcc -std=c11 -O3 -march=native -fopenmp -fPIC -shared -Wall -Wextra \
    -o "$HERE/out/libfeatures.so" "$HERE/features.c"
echo "built $HERE/out/libfeatures.so"
