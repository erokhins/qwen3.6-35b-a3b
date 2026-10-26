#!/usr/bin/env bash
# build.sh — build slice 1's two compiled arms: the g++ host twin and the hipcc device TU.
#
#   ./build.sh            build both (objects under build/)
#   ./build.sh --check    compile only, run nothing (the same compiles)
#   ./build.sh --broken   ALSO build build/host_probe_broken: the same source with
#                         -DGDN_BROKEN_BF16_TRUNC, the control that shows the self-test and the
#                         read/decode checks can fail. Every check in this lane is witnessed
#                         failing before it is trusted; this is that witness's compiler.
#   ./build.sh --clean    remove build/ and both binaries
#
# Build facts that are not optional on this box (pitfalls/50-harness-and-shell.md §"Build and
# toolchain facts that are not optional"): -O3 on the hip link line and -Wl,-rpath,/opt/rocm/lib.
# The device arch is gfx1151 (Strix Halo) and exactly one arch is built, matching
# tools/build_llama.sh's pinned triple.
#
# NO -ffast-math, and `-ffp-contract=off` on BOTH arms, and that is load-bearing rather than
# stylistic: gdn_kernels.h states its accumulation order in the source, and the zero-tolerance
# claim between the two compiled arms only holds while neither compiler reassociates the f32 chain
# or fuses a multiply-add into an FMA. The host twin is built with -Wall -Wextra so a silently
# unused parameter or a narrowing conversion in the shared headers is a build-time finding.
#
# Compiling is CPU work: it touches no device, takes no lock and reads no timer.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD="$HERE/build"
HIPCC=${HIPCC:-hipcc}
CXX=${CXX:-g++}
ARCH=${Q36_GPU_ARCH:-gfx1151}
RPATH="-Wl,-rpath,/opt/rocm/lib"
FP="-ffp-contract=off -fno-fast-math"
BIN="$HERE/gdn"
PROBE="$HERE/host_probe"

case "${1:-}" in
  --clean)
    rm -rf "$BUILD" "$BIN" "$PROBE"
    echo "CLEAN removed build/ gdn host_probe"
    exit 0
    ;;
esac

mkdir -p "$BUILD"
echo "BUILD hipcc=$HIPCC arch=$ARCH out=$BUILD"
"$HIPCC" -x hip --offload-arch="$ARCH" -O3 $FP -I"$HERE" -c "$HERE/gdn.hip" -o "$BUILD/gdn.o"
"$HIPCC" -O3 $RPATH "$BUILD/gdn.o" -o "$BIN"
"$CXX" -O2 -I"$HERE" -std=c++11 -Wall -Wextra $FP -c "$HERE/host_probe.cpp" -o "$BUILD/host_probe.o"
"$CXX" -O2 $RPATH "$BUILD/host_probe.o" -o "$PROBE"
echo "BUILD ok binaries=$BIN $PROBE"

if [ "${1:-}" = "--broken" ]; then
    "$CXX" -O2 -I"$HERE" -std=c++11 $FP -DGDN_BROKEN_BF16_TRUNC \
          -c "$HERE/host_probe.cpp" -o "$BUILD/host_probe_broken.o"
    "$CXX" -O2 $RPATH "$BUILD/host_probe_broken.o" -o "$BUILD/host_probe_broken"
    echo "BUILD broken control=$BUILD/host_probe_broken (bf16 conversion truncates instead of rounding)"
fi
if [ "${1:-}" = "--check" ]; then
    echo "BUILD note --check: compile only; no device was touched and no timing was taken"
else
    echo "BUILD note compile only; no device was touched and no timing was taken"
fi
