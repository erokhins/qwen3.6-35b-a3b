#!/usr/bin/env bash
# build.sh — build the shared-expert device binary and its g++ twin.
#
#   ./build.sh            build both (objects out-of-tree under build/)
#   ./build.sh --check    compile only, touch no device, run nothing (the same compiles)
#   ./build.sh --clean    remove build/ and both binaries
#
# Non-optional build facts (pitfalls/50 §"Build and toolchain facts that are not optional"):
#   * -O3 and -Wl,-rpath,/opt/rocm/lib on the link lines, exactly one offload arch (gfx1151, the
#     triple tools/build_llama.sh pins);
#   * -ffp-contract=off on BOTH arms. The two arms are compared bit for bit, so both compilers must
#     round the same sequence of operations; letting either fuse a multiply into an add turns the
#     cross-arm comparison into a question about contraction. The declared arithmetic depends on it
#     (`shared_expert_kernels.h` §FMA).
#   * no -ffast-math anywhere: expf and the divisions must stay IEEE, because the transcendental
#     route is attributed in this lane, not banded.
# Compiling is CPU work and takes no lock; the device work belongs to ./gate.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD="$HERE/build"
HIPCC=${HIPCC:-hipcc}
CXX=${CXX:-g++}
ARCH=${Q36_GPU_ARCH:-gfx1151}
RPATH="-Wl,-rpath,/opt/rocm/lib"
BIN="$HERE/shared_expert"
PROBE="$HERE/host_probe"
COMMON="-O3 -ffp-contract=off -fno-unsafe-math-optimizations -I$HERE"

case "${1:-}" in
  --clean)
    rm -rf "$BUILD" "$BIN" "$PROBE"
    echo "CLEAN removed build/ shared_expert host_probe"
    exit 0
    ;;
esac

mkdir -p "$BUILD" "$HERE/work"
echo "BUILD hipcc=$HIPCC cxx=$CXX arch=$ARCH out=$BUILD contract=off"
"$HIPCC" -x hip --offload-arch="$ARCH" $COMMON -c "$HERE/shared_expert.hip" -o "$BUILD/shared_expert.o"
# The census needs the DEVICE object, not the host object. --save-temps leaves the unwrapped
# gfx1151 object next to the -o file, named after the source; the census must price the SAME source
# and flags as the binary, not a second copy compiled differently (pitfalls/20 census rules).
mkdir -p "$HERE/work/isa"
"$HIPCC" -x hip --offload-arch="$ARCH" $COMMON --save-temps -c "$HERE/shared_expert.hip" \
        -o "$HERE/work/isa/se.o"
[ -f "$HERE/work/isa/shared_expert-hip-amdgcn-amd-amdhsa-gfx1151.o" ] \
  || { echo "BUILD FAIL no device object for the census; census.py would then price nothing"; exit 1; }
"$HIPCC" -O3 $RPATH "$BUILD/shared_expert.o" -o "$BIN"
"$CXX" -O2 -std=c++11 -Wall -Wextra -ffp-contract=off -I"$HERE" \
      -c "$HERE/host_probe.cpp" -o "$BUILD/host_probe.o"
"$CXX" -O2 $RPATH "$BUILD/host_probe.o" -o "$PROBE"
# The act floor probe is part of the build, not an ad-hoc compile: it evaluates the SAME
# `se_swiglu` the kernel runs, so it must be built with the same contraction flags or its "leg" is a
# different route. gate.py also builds it on demand if the binary is absent (a missing probe must
# fail loudly, never default a band).
"$CXX" -O2 -std=c++11 -ffp-contract=off -I"$HERE" "$HERE/work/floor_probe_act.c" -o "$HERE/work/floor_probe_act" -lm
echo "BUILD ok binaries=$BIN $PROBE probe=$HERE/work/floor_probe_act"
echo "BUILD note compile only; no device was touched and no timing taken"
