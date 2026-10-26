#!/usr/bin/env bash
# build.sh — compile M2's device stages for both arms: the hipcc/gfx1151 arm and the g++ twin.
#
#   ./build.sh            build every pair (objects and binaries under engine/device/build/)
#   ./build.sh --check    compile only, run nothing (the same compiles, still no device)
#   ./build.sh --clean    remove build/ and all binaries
#
# Two pairs today: `m2_se_*` (the shared expert) and `m2_rt_*` (the router). Each pair is one header of
# arithmetic plus one `.inc` of run order, included by both compilers. `engine/block_device.py` runs
# each pair on one host-visible plan and requires byte-identical records, so a host/device disagreement
# is attributable to the device compiler instead of to two spellings (`prototype/gdn/build.sh` and
# `prototype/shared_expert/build.sh` are the pattern, and `pitfalls/40`
# §a-check-whose-reference-side-is-derived-from-the-thing-under-test is the reason).
#
# NON-OPTIONAL BUILD FACTS (pitfalls/50 §“Build and toolchain facts that are not optional”):
#   * `-O3` and `-Wl,-rpath,/opt/rocm/lib` on the HIP link lines, and exactly one offload arch —
#     gfx1151, the triple `tools/build_llama.sh` pins. A binary linked without the rpath does not find
#     `libamdhip64` at run time and fails in a way that looks like a device fault and is not one.
#   * `-ffp-contract=off -fno-fast-math` on BOTH arms. The declared dot in
#     `shared_expert_kernels.h` states one mul then one add per step; letting either compiler fuse them
#     into an FMA makes the pair comparison a question about contraction rather than about codegen.
#   * the prototype lanes' headers are INCLUDED, never copied: `-I prototype/shared_expert` and
#     `-I prototype/moe_router`. The lanes are read-only material for M2, and a copy would be a second
#     ruler that goes stale on its own schedule.
#
# Compiling is CPU work: this script touches no device, takes no lock, reads no timer, and never
# launches a binary. Execution — host or device — belongs to `engine/block_device.py`.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
BUILD="$HERE/build"
HIPCC=${HIPCC:-hipcc}
CXX=${CXX:-g++}
ARCH=${Q36_GPU_ARCH:-gfx1151}
RPATH="-Wl,-rpath,/opt/rocm/lib"
FP="-ffp-contract=off -fno-fast-math"
INC="-I$HERE -I$ROOT/prototype/shared_expert -I$ROOT/prototype/moe_router"
BIN_HIP="$BUILD/m2_se_arm"
BIN_HOST="$BUILD/m2_se_host"
RT_HIP="$BUILD/m2_rt_arm"
RT_HOST="$BUILD/m2_rt_host"

case "${1:-}" in
  --clean)
    rm -rf "$BUILD"
    echo "CLEAN removed engine/device/build"
    exit 0
    ;;
esac

if [ ! -f "$ROOT/prototype/shared_expert/shared_expert_kernels.h" ]; then
    echo "BUILD FAIL missing prototype/shared_expert/shared_expert_kernels.h; the lane's arithmetic is"
    echo "     included, not copied, so a missing header is a missing dependency and not a fallback"
    exit 1
fi
if [ ! -f "$ROOT/prototype/moe_router/moe_router_kernels.h" ]; then
    echo "BUILD FAIL missing prototype/moe_router/moe_router_kernels.h; R2..R5 are included from that"
    echo "     lane verbatim, so a missing header means the router arm has no softmax to run"
    exit 1
fi

mkdir -p "$BUILD"
echo "BUILD hipcc=$HIPCC cxx=$CXX arch=$ARCH out=$BUILD contract=off fast_math=off"

"$CXX" -O2 -std=c++11 -Wall -Wextra $FP $INC -c "$HERE/m2_se_host.cpp" -o "$BUILD/m2_se_host.o"
"$CXX" -O2 $RPATH "$BUILD/m2_se_host.o" -o "$BIN_HOST"
echo "BUILD ok host_twin=$BIN_HOST"

"$HIPCC" -x hip --offload-arch="$ARCH" -O3 $FP $INC -c "$HERE/m2_se_arm.hip" -o "$BUILD/m2_se_arm.o"
"$HIPCC" -O3 $RPATH "$BUILD/m2_se_arm.o" -o "$BIN_HIP"
echo "BUILD ok hip_arm=$BIN_HIP"

"$CXX" -O2 -std=c++11 -Wall -Wextra $FP $INC -c "$HERE/m2_rt_host.cpp" -o "$BUILD/m2_rt_host.o"
"$CXX" -O2 $RPATH "$BUILD/m2_rt_host.o" -o "$RT_HOST"
echo "BUILD ok router_host_twin=$RT_HOST"

"$HIPCC" -x hip --offload-arch="$ARCH" -O3 $FP $INC -c "$HERE/m2_rt_arm.hip" -o "$BUILD/m2_rt_arm.o"
"$HIPCC" -O3 $RPATH "$BUILD/m2_rt_arm.o" -o "$RT_HIP"
echo "BUILD ok router_hip_arm=$RT_HIP"

if [ "${1:-}" = "--check" ]; then
    echo "BUILD note --check: compiled only; no binary was executed, no device was touched,"
    echo "BUILD note         and the pair comparison is engine/block_device.py's job, not this script's"
    exit 0
fi
echo "BUILD note compile only; both arms exist and neither has been run. no timing taken"
