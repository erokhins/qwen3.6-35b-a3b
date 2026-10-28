#!/usr/bin/env bash
# llama_tokens_build.sh — compile tools/llama_tokens_dump.cpp against the instrumented CPU-only
# libllama, and prove two things from the artifact rather than from the source:
#
#   1. CPU-ONLY. ldd on the binary contains no ROCm/HIP library, no unresolved dependency, and the
#      libllama it resolves is the one the CPU-only gate in tools/llama_instrument_apply.sh approved.
#   2. NOT LINKED TO THE TEMPLATE MACHINERY. llama.cpp keeps chat templating and tokenization in
#      libllama-common (src/llama-chat.cpp, src/llama-vocab.cpp + common/common.cpp of the front ends).
#      This program links libllama and libggml only, so the template/tokenize code is not merely
#      uncalled, it is absent from the process image. The gate prints both halves: llama-tokens-dump
#      must NOT carry libllama-common and llama-cli MUST, so the check is witnessed firing and cannot
#      pass by reading nothing.
#
# Nothing here touches the iGPU: no device compiler is invoked (plain ${CXX} against a CPU-only library
# set), no lockdir path appears in this file, and no duration or rate is printed. If a future tree makes
# any of the guards below require a device build, this script refuses instead of guessing.
#
# Usage: tools/llama_tokens_build.sh            # configure-free single-TU compile + gates
# Env overrides: WORK, SRC, CXX, PY.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)

WORK=${WORK:-/tmp/llama-instrument-vendored}
SRC=${SRC:-$REPO/engines/llama.cpp}
BUILD=$WORK/build-cpu
CXX=${CXX:-g++}
PY=${PY:-/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python}

die() { echo "llama_tokens_build: $*" >&2; exit 1; }

LLAMALIB=$BUILD/bin/libllama.so
[ -f "$LLAMALIB" ] || die "no instrumented libllama at $LLAMALIB — run: WORK=$WORK tools/llama_instrument_apply.sh build"
[ -f "$SRC/include/llama.h" ] || die "no $SRC/include/llama.h"
git -C "$SRC" rev-parse HEAD | grep -q '^d327bf1f0780f87b8c01b367b61853b0c291d7c4$' \
    || die "vendored tree is not at the pinned commit d327bf1f — every quote about the oracle is a claim about that commit"

# The dump hooks must be in the library this program links, or the run would produce no manifest.
# `|| true` on nm, not on the grep: nm exits non-zero on a shared object whose static symtab is
# stripped even while it prints the dynamic symbols, and a pipeline under pipefail would otherwise
# report "uninstrumented" from a successful symbol listing.
DUMP_SYMS=$(nm -D --defined-only "$LLAMALIB" 2>/dev/null || true)
case $DUMP_SYMS in
    *llama_dump_before_graph*) echo "  ok   libllama carries the dump hooks" ;;
    *) die "libllama at $LLAMALIB has no llama_dump_before_graph — it is an uninstrumented build" ;;
esac

OUT=$BUILD/bin/llama-tokens-dump
mkdir -p "$BUILD/bin"
"$CXX" -std=c++17 -O2 -DNDEBUG \
    -I "$SRC/include" -I "$SRC/ggml/include" \
    "$REPO/tools/llama_tokens_dump.cpp" \
    -L "$BUILD/bin" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$BUILD/bin" \
    -o "$OUT"
echo "built: $OUT"

# ---- gates (counted, so a gate that stops checking fails) -----------------------------------------
checks=0; expected=0
chk() { expected=$((expected+1)); if eval "$2"; then checks=$((checks+1)); echo "  ok   $1"; else echo "  FAIL $1"; fi; }

ldd_lines() { ldd "$1" 2>/dev/null; }

chk "binary exists and is executable" "[ -x '$OUT' ]"
chk "ldd resolves everything (no 'not found')" "! ldd_lines '$OUT' | grep -q 'not found'"
chk "ldd carries zero ROCm/HIP libraries" \
    "! ldd_lines '$OUT' | grep -Eiq 'libhip|libamdhip|librocclr|libhsa-runtime|librocm|libggml-hip|rocm'"
chk "libllama resolves inside the CPU-only build dir" \
    "ldd_lines '$OUT' | grep -F 'libllama.so' | grep -qF '$BUILD/bin/libllama.so'"
chk "template/tokenize machinery is NOT linked (libllama-common absent)" \
    "! ldd_lines '$OUT' | grep -q 'libllama-common'"
chk "control: libllama-common IS present in llama-cli (the check can see it)" \
    "ldd_lines '$BUILD/bin/llama-cli' | grep -q 'libllama-common'"
chk "control: llama-cli resolves the same libllama" \
    "ldd_lines '$BUILD/bin/llama-cli' | grep -F 'libllama.so' | grep -qF '$BUILD/bin/libllama.so'"
chk "no device fatbin or device .co object in the build tree" \
    "! find '$BUILD' -maxdepth 3 \\( -name '*.hip_fatbin' -o -name '*.co' \\) -print -quit | grep -q ."

echo "checks ran: $checks / expected $expected"
[ "$checks" -eq "$expected" ] || die "gate failed"

# A build record so a dump can name the binary that made it, and bind it by digest.
REC=$BUILD/tokens-build-record.json
{
    echo "# tokens-build-record.json — how tools/llama-tokens-dump was made."
    echo "# Written by tools/llama_tokens_build.sh; a manifest binds it by sha256."
    echo "generated_by=tools/llama_tokens_build.sh"
    echo "source=$REPO/tools/llama_tokens_dump.cpp"
    echo "source_sha256=$("$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$REPO/tools/llama_tokens_dump.cpp")"
    echo "binary=$OUT"
    echo "binary_sha256=$("$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$OUT")"
    echo "libllama=$LLAMALIB"
    echo "libllama_sha256=$("$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$LLAMALIB")"
    echo "vendored_commit=$(git -C "$SRC" rev-parse HEAD)"
    echo "compiler=$("$CXX" --version | head -1)"
} > "$REC"
echo "record: $REC"
