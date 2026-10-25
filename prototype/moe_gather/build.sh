#!/usr/bin/env bash
# build.sh — build the moe_gather device binary and its g++ twin.
#
#   ./build.sh            build both (objects out-of-tree under build/)
#   ./build.sh --check    compile only, touch no device, run nothing (the same compiles)
#   ./build.sh --clean    remove build/ and both binaries
#
# Build facts that are not optional here (pitfalls/50 §"Build and toolchain facts that are not
# optional", plus one specific to this lane):
#   * -O3 and -Wl,-rpath,/opt/rocm/lib on the link lines, and exactly one offload arch — gfx1151,
#     the triple tools/build_llama.sh pins;
#   * -ffp-contract=off on BOTH arms. The two builds are compared bit for bit, so the two compilers
#     must round the same sequence of operations; letting either fuse a mul into an add makes the
#     comparison a question about contraction rather than about the device.
#   * -fno-fast-math / no -ffast-math anywhere: expf and the divisions must stay IEEE.
# Compiling is CPU work and takes no lock; the device work belongs to ./gate.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD="$HERE/build"
HIPCC=${HIPCC:-hipcc}
CXX=${CXX:-g++}
ARCH=${Q36_GPU_ARCH:-gfx1151}
RPATH="-Wl,-rpath,/opt/rocm/lib"
BIN="$HERE/moe_gather"
PROBE="$HERE/host_probe"
COMMON="-O3 -ffp-contract=off -fno-unsafe-math-optimizations -I$HERE"

case "${1:-}" in
  --clean)
    rm -rf "$BUILD" "$BIN" "$PROBE"
    echo "CLEAN removed build/ moe_gather host_probe"
    exit 0
    ;;
esac

mkdir -p "$BUILD"
echo "BUILD hipcc=$HIPCC cxx=$CXX arch=$ARCH out=$BUILD contract=off"
"$HIPCC" -x hip --offload-arch="$ARCH" $COMMON -c "$HERE/moe_gather.hip" -o "$BUILD/moe_gather.o"
"$HIPCC" -O3 $RPATH "$BUILD/moe_gather.o" -o "$BIN"
"$CXX" -O2 -std=c++11 -Wall -Wextra -ffp-contract=off -I"$HERE" \
      -c "$HERE/host_probe.cpp" -o "$BUILD/host_probe.o"
"$CXX" -O2 $RPATH "$BUILD/host_probe.o" -o "$PROBE"
echo "BUILD ok binaries=$BIN $PROBE"
echo "BUILD note compile only; no device was touched and no timing taken"
