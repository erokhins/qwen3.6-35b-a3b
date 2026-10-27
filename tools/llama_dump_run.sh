#!/usr/bin/env bash
# llama_dump_run.sh — run the instrumented CPU-only llama.cpp once and pin what it produced.
#
# Plan of record: tools/LLAMA_DUMP_NOTES.md. The instrumentation is tools/llama-instrument.patch,
# applied and built by tools/llama_instrument_apply.sh.
#
# Device discipline (AGENTS.md §GPU execution is serialized and belongs to the main agent): this
# script runs the CPU build only — `-ngl 0`, a binary whose link line contains no HIP or ROCm
# library (tools/llama_instrument_apply.sh gates on that), and no lock of any kind: this file
# contains no lockdir path and no acquire/release call, so a caller that already holds the
# machine-wide lock cannot deadlock against it (pitfalls/50 §self-locking drivers). It measures nothing
# of how long anything took: no duration, no rate, anywhere in this file or in the log it writes.
#
# Usage: tools/llama_dump_run.sh <tag> <prompt>
#   <tag> becomes the dump leaf directory data/dumps/2026-10-27-llama/<tag>/ and the manifest
#   suffix checkpoints/llama-dump-manifest-2026-10-27-<tag>.json. Both are write-once.
#
# Env overrides: BIN, GGUF, BUILD_RECORD, PINFILE, PY, THREADS, NCTX, MAX_GRAPHS, UNNAMED,
#                DUMP_ROOT, MANIFEST_ROOT.
#
# MAX_GRAPHS defaults to 4 because llama-cli computes a graph of its own before the prompt: the
# captured graphs are llama-cli's init graph, the prompt prefill (which the conversation mode splits
# into two prefills), and the first decode step. The token ids
# of every captured graph are recorded in the manifest's "graphs" list, which is how you tell them
# apart. UNNAMED=1 (the default) also writes the bytes of tensors the graph builder left unnamed —
# in the fused GDN path that includes the node carrying the post-update recurrent state.
set -euo pipefail

die() { echo "llama_dump_run: $*" >&2; exit 1; }

TAG=${1:-}; PROMPT=${2:-}
[ -n "$TAG" ]    || die "usage: $0 <tag> <prompt>"
[ -n "$PROMPT" ] || die "usage: $0 <tag> <prompt>"
case $TAG in *[!A-Za-z0-9._-]*) die "tag must be [A-Za-z0-9._-]+";; esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)

BIN=${BIN:-${WORK:-/tmp/llama-instrument}/build-cpu/bin/llama-cli}
GGUF=${GGUF:-$REPO/models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
BUILD_RECORD=${BUILD_RECORD:-${WORK:-/tmp/llama-instrument}/build-cpu/build-record.json}
PATCH_APPLY_LOG=${PATCH_APPLY_LOG:-${WORK:-/tmp/llama-instrument}/apply-record.json}
PINFILE=${PINFILE:-$REPO/checkpoints/Qwen3.6-35B-A3B.gguf-pin.json}
PY=${PY:-/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python}
THREADS=${THREADS:-8}
NCTX=${NCTX:-64}
MAX_GRAPHS=${MAX_GRAPHS:-4}
UNNAMED=${UNNAMED:-1}
DUMP_ROOT=${DUMP_ROOT:-$REPO/data/dumps/2026-10-27-llama}
MANIFEST_ROOT=${MANIFEST_ROOT:-$REPO/checkpoints}

[ -x "$BIN" ]          || die "no binary at $BIN — run tools/llama_instrument_apply.sh build"
[ -f "$GGUF" ]         || die "no GGUF at $GGUF"
[ -f "$BUILD_RECORD" ] || die "no build record at $BUILD_RECORD"
[ -f "$PINFILE" ]      || die "no pin file at $PINFILE"

# The pin, read from the file that holds it — never typed here, so this script cannot certify a
# different artifact than the one checkpoints/ names.
PIN_SHA=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['sha256'])" "$PINFILE")
if [ "${#PIN_SHA}" != 64 ]; then die "pin sha256 is not 64 chars in $PINFILE"; fi

DUMP_DIR=$DUMP_ROOT/$TAG
MANIFEST=$MANIFEST_ROOT/llama-dump-manifest-2026-10-27-$TAG.json
# The log lives outside the dump directory on purpose: the dumper refuses a non-empty dump dir, so
# anything this script wrote in there would either trip that guard or make it vacuous.
LOG_DIR=$DUMP_ROOT/logs
RUN_LOG=$LOG_DIR/$TAG.log

# Append-only, write-once, checked here as well as in the dumper.
if [ -e "$MANIFEST" ]; then die "refusing to overwrite $MANIFEST (write-once)"; fi
if [ -e "$RUN_LOG" ];   then die "refusing to overwrite $RUN_LOG (write-once)"; fi
if [ -d "$DUMP_DIR" ] && [ -n "$(ls -A "$DUMP_DIR" 2>/dev/null)" ]; then
  die "refusing to write into non-empty $DUMP_DIR (append-only)"
fi
mkdir -p "$DUMP_DIR" "$LOG_DIR"

# -ngl 0 keeps every tensor on the CPU. --no-warmup matters: a warmup forward would become graph 0
# of the dump. -st/--single-turn avoids the interactive-mode trap recorded in
# quantization/evidence/llama-cpp-smoke-gguf-2026-10-26.md. -s / --temp / -t are pinned because
# the determinism control compares two runs of this exact line.
ARGS=( -m "$GGUF" -p "$PROMPT" -n "$MAX_GRAPHS" -ngl 0 -c "$NCTX" -b 64 -ub 32 -t "$THREADS"
       --temp 0 -s 42 --no-display-prompt --no-warmup -st -co off )

CMDLINE=$("$PY" -c "import sys,shlex; print(' '.join(shlex.quote(a) for a in sys.argv[1:]))" "$BIN" "${ARGS[@]}")

# The flag lines come straight out of the CMake cache copy kept in the build record, so the
# manifest carries the flags that were actually used rather than the ones this script means to pass.
# The record keeps them inside configure_command=, not as one key per flag, so that is what is read.
BUILD_FLAGS=$(sed -n 's/^configure_command=//p' "$BUILD_RECORD" | tr '\n' ' ' || true)

LLAMA_DUMP_DIR="$DUMP_DIR" \
LLAMA_DUMP_MANIFEST="$MANIFEST" \
LLAMA_DUMP_TAG="$TAG" \
LLAMA_DUMP_PROMPT="$PROMPT" \
LLAMA_DUMP_CMDLINE="$CMDLINE" \
LLAMA_DUMP_GGUF_PIN_SHA256="$PIN_SHA" \
LLAMA_DUMP_GGUF_EXPECTED_PATH="$GGUF" \
LLAMA_DUMP_BUILD_RECORD="$BUILD_RECORD" \
LLAMA_DUMP_BUILD_FLAGS="$BUILD_FLAGS" \
LLAMA_DUMP_MAX_GRAPHS="$MAX_GRAPHS" \
LLAMA_DUMP_UNNAMED="$UNNAMED" \
  "$BIN" "${ARGS[@]}" > "$RUN_LOG" 2>&1
rc=$?

"$PY" "$REPO/tools/llama_dump_provenance.py" "$DUMP_DIR/provenance.json" "$TAG" "$PROMPT" \
      "$BIN" "$GGUF" "$PINFILE" "$rc" "$CMDLINE" "$BUILD_RECORD" "$PATCH_APPLY_LOG"

if [ ! -f "$MANIFEST" ]; then
  echo "llama_dump_run: no manifest written (rc=$rc); see $RUN_LOG" >&2
  exit 1
fi
grep -q "^LLAMA-DUMP DONE" "$RUN_LOG" || die "log has no DONE line; the dump did not complete: $RUN_LOG"
grep -q "^LLAMA-DUMP DONE verdict=PASS" "$RUN_LOG" \
  || die "the dump's own verdict is not PASS; see $RUN_LOG"

echo "ran tag=$TAG rc=$rc"
echo "  dump     : $DUMP_DIR"
echo "  manifest : $MANIFEST"
echo "  log      : $RUN_LOG"
exit $rc
