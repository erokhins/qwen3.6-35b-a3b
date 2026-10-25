#!/usr/bin/env bash
# build.sh — build the lm_head device binary and its host twin.
#
#   ./build.sh            build both (objects out-of-tree under build/)
#   ./build.sh --check    compile only, touch no device, run nothing (the same compiles)
#   ./build.sh --clean    remove build/ and both binaries
#
# Build facts that are not optional on this box (pitfalls/50-harness-and-shell.md §"Build and
# toolchain facts that are not optional"): -O3, and -Wl,-rpath,/opt/rocm/lib on the link lines. The
# device arch is gfx1151 (Strix Halo) and exactly one arch is built, matching
# tools/build_llama.sh's pinned triple.
#
# NO -ffast-math, and that is load-bearing rather than stylistic: lm_head_kernels.h states its
# accumulation order in the source, and the bit-exact claims (device vs host twin, batched vs
# per-row, gather vs dense) only hold while neither compiler reassociates the f32 chain. The host
# twin is built at -O2 with -Wall -Wextra so a silently-unused parameter or a narrowing conversion
# in the shared headers is a build-time finding, not a runtime surprise.
#
# Compiling is CPU work and takes no lock; the device work belongs to ./gate.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD="$HERE/build"
HIPCC=${HIPCC:-hipcc}
CXX=${CXX:-g++}
ARCH=${Q36_GPU_ARCH:-gfx1151}
RPATH="-Wl,-rpath,/opt/rocm/lib"
BIN="$HERE/lm_head"
PROBE="$HERE/host_probe"

case "${1:-}" in
  --clean)
    rm -rf "$BUILD" "$BIN" "$PROBE"
    echo "CLEAN removed build/ lm_head host_probe"
    exit 0
    ;;
esac

mkdir -p "$BUILD"
echo "BUILD hipcc=$HIPCC arch=$ARCH out=$BUILD"
"$HIPCC" -x hip --offload-arch="$ARCH" -O3 -I"$HERE" -c "$HERE/lm_head.hip" \
        -o "$BUILD/lm_head.o"
"$HIPCC" -O3 $RPATH "$BUILD/lm_head.o" -o "$BIN"
"$CXX" -O2 -I"$HERE" -std=c++11 -Wall -Wextra -c "$HERE/host_probe.cpp" -o "$BUILD/host_probe.o"
"$CXX" -O2 $RPATH "$BUILD/host_probe.o" -o "$PROBE"
echo "BUILD ok binaries=$BIN $PROBE"
echo "BUILD note compile only; no device was touched and no timing taken"
