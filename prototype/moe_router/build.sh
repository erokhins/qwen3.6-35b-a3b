#!/usr/bin/env bash
# build.sh — build the moe_router device binary and its host-probe twin.
#
#   ./build.sh            build both (objects out-of-tree under build/)
#   ./build.sh --clean    remove build/ and both binaries
#
# Build facts that are not optional on this box (pitfalls/50 §"Build and toolchain facts that are
# not optional"): -O3 on the hipcc lines, and -Wl,-rpath,/opt/rocm/lib on the link lines. The device
# arch is gfx1151 (Strix Halo) and exactly one arch is built, matching tools/build_llama.sh's pinned
# triple. Compiling is CPU work, touches no device and takes no lock; the device work is ./gate.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD="$HERE/build"
HIPCC=${HIPCC:-hipcc}
CXX=${CXX:-g++}
ARCH=${Q36_GPU_ARCH:-gfx1151}
RPATH="-Wl,-rpath,/opt/rocm/lib"
BIN="$HERE/moe_router"
PROBE="$HERE/host_probe"

case "${1:-}" in
  --clean)
    rm -rf "$BUILD" "$BIN" "$PROBE"
    echo "CLEAN removed build/ moe_router host_probe"
    exit 0
    ;;
esac

mkdir -p "$BUILD"
echo "BUILD hipcc=$HIPCC arch=$ARCH out=$BUILD"
"$HIPCC" -x hip --offload-arch="$ARCH" -O3 -I"$HERE" -c "$HERE/moe_router.hip" \
        -o "$BUILD/moe_router.o"
"$HIPCC" -O3 $RPATH "$BUILD/moe_router.o" -o "$BIN"
"$CXX" -O2 -I"$HERE" -std=c++17 -Wall -Wextra -c "$HERE/host_probe.cpp" -o "$BUILD/host_probe.o"
"$CXX" -O2 $RPATH "$BUILD/host_probe.o" -o "$PROBE"
echo "BUILD ok binaries=$BIN $PROBE"
echo "BUILD note compile only; no device was touched and no timing taken"
