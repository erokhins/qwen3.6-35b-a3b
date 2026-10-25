#!/usr/bin/env bash
# build.sh — build the q4k_dequant device binary and its host-probe twin.
#
#   ./build.sh            build both (objects out-of-tree under build/)
#   ./build.sh --check    compile only, touch no device, run nothing (the same compiles)
#   ./build.sh --clean    remove build/ and both binaries
#
# Build facts that are not optional on this box (pitfalls/50 §"Build and toolchain facts that are
# not optional"): -O3, and -Wl,-rpath,/opt/rocm/lib on the link lines. The device arch is gfx1151
# (Strix Halo) and exactly one arch is built, matching tools/build_llama.sh's pinned triple.
# Compiling is CPU work and takes no lock; the device work belongs to ./gate.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD="$HERE/build"
HIPCC=${HIPCC:-hipcc}
CXX=${CXX:-g++}
ARCH=${Q36_GPU_ARCH:-gfx1151}
RPATH="-Wl,-rpath,/opt/rocm/lib"
BIN="$HERE/q4k_dequant"
PROBE="$HERE/host_probe"

case "${1:-}" in
  --clean)
    rm -rf "$BUILD" "$BIN" "$PROBE"
    echo "CLEAN removed build/ q4k_dequant host_probe"
    exit 0
    ;;
esac

mkdir -p "$BUILD"
echo "BUILD hipcc=$HIPCC arch=$ARCH out=$BUILD"
"$HIPCC" -x hip --offload-arch="$ARCH" -O3 -I"$HERE" -c "$HERE/q4k_dequant.hip" \
        -o "$BUILD/q4k_dequant.o"
"$HIPCC" -O3 $RPATH "$BUILD/q4k_dequant.o" -o "$BIN"
"$CXX" -O2 -I"$HERE" -std=c++11 -Wall -Wextra -c "$HERE/host_probe.cpp" -o "$BUILD/host_probe.o"
"$CXX" -O2 $RPATH "$BUILD/host_probe.o" -o "$PROBE"
echo "BUILD ok binaries=$BIN $PROBE"
echo "BUILD note compile only; no device was touched and no timing taken"
