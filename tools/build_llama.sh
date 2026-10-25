#!/usr/bin/env bash
# build_llama.sh — reproducible HIP (gfx1151) build of the pinned llama.cpp checkout that
# DECISIONS.md §D-12 adopted as the cross-check oracle.
#
# What it does: configures + builds `engines/llama.cpp` out-of-tree in
# `engines/llama.cpp/build-hip` with the HIP backend for exactly one GPU arch, then runs an
# integrity gate over the artifacts (artifacts present, HIP really linked, device code really
# present, every device compile really aimed at gfx1151).
#
# Pinned checkout: engines/llama.cpp @ d327bf1f0780f87b8c01b367b61853b0c291d7c4 (tag b11869),
#                  from https://github.com/ggml-org/llama.cpp — do not update it from here.
# The commit guard below is the point of this script: an oracle whose version drifted is worse
# than no oracle, because the four claims in
# quantization/evidence/llama-cpp-build-and-flags-2026-10-26.md are claims about *that* revision.
#
# Toolchain facts this build needs, recorded here so the script runs from this repo alone
# (rule: AGENTS.md §Self-containment): ROCm 7.15.26333-0000000 at /opt/rocm, `hipcc` and
# `rocminfo` on PATH, device arch per /opt/rocm/bin/offload-arch == gfx1151, and **cmake is NOT
# on PATH on this box** — install one (`uv tool install cmake` or `python3 -m pip install cmake`)
# or pass CMAKE=/path/to/cmake. The arch knob this CMake version honours is GPU_TARGETS /
# AMDGPU_TARGETS, forwarded to CMAKE_HIP_ARCHITECTURES by ggml/src/ggml-hip/CMakeLists.txt
# §"Forward (AMD)GPU_TARGETS to CMAKE_HIP_ARCHITECTURES"; we pass all three because that
# forwarding is guarded by `if(GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)` and because the
# whole branch is skipped when hipcc is used as the C++ compiler.
#
# Device discipline (AGENTS.md §GPU execution is serialized and belongs to the main agent):
# this script only compiles. It loads no model, allocates no device memory, takes no timing, and
# deliberately takes NO lock of any kind: this file contains no lockdir path and no acquire/release
# call, so a caller that already holds the machine-wide lock cannot deadlock against it
# (pitfalls/50-harness-and-shell.md §Self-locking gate drivers deadlock silently). It shells out to
# cmake, git, readelf, ldd, llvm-objcopy, awk and python3 only — check by reading, not by grepping
# this script for a lock name: grepping it for the lock's own name would match this very sentence
# (pitfalls/40 §A check that can produce no output cannot fail).
#
# Usage:
#   tools/build_llama.sh              # guards, configure, build, verify
#   tools/build_llama.sh --check      # guards + verification gate only (no compile)
# Env overrides: CMAKE, JOBS, ROCM_PATH, LLAMA_HIP_ARCH, TARGETS, LOG, BUILD_DIR, SRC,
#                BUILD_LLAMA_PINNED_COMMIT (test hook for the guard — see §pinned revision).
set -euo pipefail

# ---------------------------------------------------------------- pinned revision -----------
# BUILD_LLAMA_PINNED_COMMIT exists only so the guard's refusal path can be witnessed firing
# (pitfalls/40: a check that can produce no output cannot fail). Never set it for a real build.
PINNED_COMMIT="${BUILD_LLAMA_PINNED_COMMIT:-d327bf1f0780f87b8c01b367b61853b0c291d7c4}"
PINNED_TAG="b11869"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
SRC=${SRC:-$REPO/engines/llama.cpp}
BUILD=${BUILD_DIR:-$SRC/build-hip}
ARCH=${LLAMA_HIP_ARCH:-gfx1151}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
TARGETS=${TARGETS:-"llama-cli llama-quantize"}
NPROC=$(nproc)
JOBS=${JOBS:-$(( NPROC > 16 ? 16 : NPROC ))}

MODE=build
if [ "${1:-}" = "--check" ]; then
  MODE=check
elif [ $# -gt 0 ]; then
  echo "usage: $0 [--check]" >&2
  exit 2
fi

LOG=${LOG:-$REPO/logs/llama-build-$(date +%F).log}
mkdir -p "$(dirname "$LOG")"

FAILED=0
RAN=0
EXPECTED_CHECKS=17   # every pass/fail call site in the gate below, counted by hand; bump together
pass() { printf 'PASS: %s\n' "$*"; RAN=$((RAN + 1)); }
fail() { printf 'FAIL: %s\n' "$*" >&2; FAILED=$((FAILED + 1)); RAN=$((RAN + 1)); }

# ---------------------------------------------------------------- guard: pinned commit ------
if [ ! -d "$SRC/.git" ]; then
  echo "FATAL: $SRC is not a git checkout (clone ggml-org/llama.cpp there first)" >&2
  exit 2
fi
HEAD=$(git -C "$SRC" rev-parse HEAD)
if [ "$HEAD" != "$PINNED_COMMIT" ]; then
  echo "GUARD FIRED: $SRC is not the pinned oracle revision." >&2
  echo "  pinned : $PINNED_COMMIT ($PINNED_TAG)" >&2
  echo "  found  : $HEAD ($(git -C "$SRC" describe --tags --always 2>/dev/null || echo no-tag))" >&2
  echo "  An unpinned cross-check oracle is worse than none: the four verdicts in" >&2
  echo "  quantization/evidence/llama-cpp-build-and-flags-2026-10-26.md are claims about the pinned" >&2
  echo "  revision, and llama.cpp moves weekly. Either check out the pinned commit, or re-read those" >&2
  echo "  four sources at this revision, update that note, and update PINNED_COMMIT here." >&2
  exit 3
fi
echo "guard: HEAD == $HEAD ($(git -C "$SRC" describe --tags --always 2>/dev/null || echo no-tag)) — pinned revision matches."

# An edit inside the checkout invalidates the quoted evidence lines just as much as a new commit.
if ! git -C "$SRC" diff --quiet -- .; then
  echo "GUARD FIRED: tracked files in $SRC are modified; quoted evidence would not be what upstream says." >&2
  git -C "$SRC" status --porcelain -- . | sed 's/^/  /' >&2
  exit 4
fi

# ---------------------------------------------------------------- guards: toolchain ---------
if [ ! -x "$ROCM_PATH/bin/hipcc" ]; then
  echo "FATAL: no hipcc under ROCM_PATH=$ROCM_PATH" >&2
  exit 2
fi
ROCM_VER=$(cat "$ROCM_PATH/.info/version-alt" 2>/dev/null || true)
[ -n "$ROCM_VER" ] || ROCM_VER=$("$ROCM_PATH/bin/hipconfig" --version 2>/dev/null || true)
OFFLOAD_ARCH=$("$ROCM_PATH/bin/offload-arch" 2>/dev/null | tr -d '[:space:]' || true)

# The arch spelling must be one this ROCm and this part agree on. rocminfo prints one `Name:`
# field per device and per ISA target; counted with an explicit if (pitfalls/50 §awk boolean
# count — `n + ($0 ~ /a/)` silently stays 0 in this box's awk).
ARCH_HITS=$(rocminfo 2>/dev/null | awk -v a="$ARCH" 'BEGIN{n=0} { if ($1 == "Name:" && $2 == a) n++ } END { print n + 0 }' || echo 0)
if [ "${ARCH_HITS:-0}" -lt 1 ]; then
  echo "FATAL: rocminfo does not list arch '$ARCH'; refusing to build for an arch this box did not report." >&2
  echo "       rocminfo arch names: $(rocminfo 2>/dev/null | awk '{ if ($1 == "Name:" && $2 ~ /^gfx/) printf "%s ", $2 }' || true)" >&2
  exit 2
fi
if [ -n "$OFFLOAD_ARCH" ] && [ "$ARCH" != "$OFFLOAD_ARCH" ]; then
  echo "note: LLAMA_HIP_ARCH=$ARCH differs from offload-arch=$OFFLOAD_ARCH (a cross-compile; not a measurement)"
fi

CMAKE=${CMAKE:-$(command -v cmake || true)}
if [ -z "$CMAKE" ]; then
  for c in "$HOME/.local/bin/cmake" /usr/local/bin/cmake /usr/bin/cmake; do
    if [ -x "$c" ]; then CMAKE=$c; break; fi
  done
fi
if [ -z "$CMAKE" ]; then
  echo "FATAL: no cmake found — this box has none on PATH. Install one ('uv tool install cmake'" >&2
  echo "       or 'python3 -m pip install cmake') or run with CMAKE=/path/to/cmake." >&2
  exit 2
fi

HIP_COMPILER="$ROCM_PATH/lib/llvm/bin/clang++"
if [ ! -x "$HIP_COMPILER" ]; then HIP_COMPILER=""; fi

if [ "$MODE" = build ]; then
  echo "configure: $("$CMAKE" --version | head -1)"
  echo "toolchain: ROCm ${ROCM_VER:-unknown} at $ROCM_PATH, HIP compiler=${HIP_COMPILER:-cmake-default}, arch=$ARCH, jobs=$JOBS"
  echo "log: $LOG"
fi

# --------------------------------------------------------- expected device-source count ----
# Derived from the same globs ggml/src/ggml-hip/CMakeLists.txt §"file(GLOB GGML_SOURCES_ROCM ...)"
# uses, so the verification below has a denominator that exists before the build runs.
CUDASRC="$SRC/ggml/src/ggml-cuda"
shopt -s nullglob
# unquoted expansion on purpose: the argument is a glob, not a literal path
count_glob() { local -a arr; arr=($1); printf '%s' "${#arr[@]}"; }
N_TOP=$(count_glob "$CUDASRC/*.cu")
N_TILE=$(count_glob "$CUDASRC/template-instances/fattn-tile*.cu")
N_MMA=$(count_glob "$CUDASRC/template-instances/fattn-mma*.cu")
N_MMQ=$(count_glob "$CUDASRC/template-instances/mmq*.cu")
N_MMF=$(count_glob "$CUDASRC/template-instances/mmf*.cu")
shopt -u nullglob
# fattn-vec instances are selected by GGML_CUDA_FA_QUANTS (ggml/cmake/common.cmake
# §"Determines which FlashAttention vector kernel template instances to compile"); read the cache
# if it exists, else the ggml/CMakeLists.txt default.
FA_QUANTS=$(sed -n 's/^GGML_CUDA_FA_QUANTS:STRING=//p' "$BUILD/CMakeCache.txt" 2>/dev/null | tr -d '[:space:]' || true)
if [ -z "$FA_QUANTS" ]; then FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16"; fi
N_VEC=0
IFS=';' read -r -a FA_COMBOS <<< "$FA_QUANTS"
for combo in "${FA_COMBOS[@]}"; do
  if [ -f "$CUDASRC/template-instances/fattn-vec-instance-$combo.cu" ]; then N_VEC=$((N_VEC + 1)); fi
done
N_EXPECT_HIP=$((N_TOP + N_TILE + N_MMA + N_MMF + N_MMQ + N_VEC))

# ---------------------------------------------------------------- configure + build --------
if [ "$MODE" = build ]; then
  mkdir -p "$BUILD"
  echo "--- configure (output appended to $LOG) ---"
  {
    echo "=== configure $(date -Is) ==="
    "$CMAKE" -S "$SRC" -B "$BUILD" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_HIP=ON \
      -DAMDGPU_TARGETS="$ARCH" \
      -DGPU_TARGETS="$ARCH" \
      -DCMAKE_HIP_ARCHITECTURES="$ARCH" \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
      ${HIP_COMPILER:+-DCMAKE_HIP_COMPILER=$HIP_COMPILER} \
      -DGGML_CCACHE=OFF \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DLLAMA_CURL=OFF \
      -DLLAMA_OPENSSL=OFF \
      -DLLAMA_USE_PREBUILT_UI=OFF \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,$ROCM_PATH/lib" \
      -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,$ROCM_PATH/lib" 2>&1
  } | tee -a "$LOG"

  echo "--- build targets: $TARGETS ---"
  # shellcheck disable=SC2086
  {
    echo "=== build $(date -Is) targets=$TARGETS arch=$ARCH jobs=$JOBS ==="
    "$CMAKE" --build "$BUILD" --parallel "$JOBS" -- $TARGETS 2>&1
  } | tee -a "$LOG" | grep -vE '^\[ *[0-9]+%\] (Building|Linking)' || true
fi

# ---------------------------------------------------------------- verification gate -------
echo "--- verification gate (arch=$ARCH) ---"
BIN=$BUILD/bin

# The rpath is not decoration: /opt/rocm/lib is NOT in this box's ld.so config (ldconfig -p has no
# hipblas entry), so a build without -Wl,-rpath links fine and then dies at exec with
# "error while loading shared libraries: libhipblas.so.3" (pitfalls/50 §Build and toolchain facts).
# RUNPATH on the executable alone does not fix it either — RUNPATH is not inherited by the deps of
# a dependency, so libggml-hip.so needs its own. Both linker flags are therefore required, and the
# two checks below are what a bare "cmake exited 0" cannot substitute for.
for t in $TARGETS; do
  if [ -x "$BIN/$t" ]; then
    pass "$t present and executable"
  else
    fail "$t missing from $BIN"
    fail "$t cannot be invoked — binary missing"
    fail "$t ldd cannot be checked — binary missing"
    continue
  fi
  case "$t" in
    llama-quantize) probe_flag=--help ;;
    *)              probe_flag=--version ;;
  esac
  OUT=$("$BIN/$t" "$probe_flag" 2>&1 || true)
  if [ "$t" = "llama-quantize" ]; then
    # Witness the D-12 §3 claim ("policy A artifact production is upstream's problem") against the
    # shipped artifact, not only against the source: all four per-tensor type flags must be there.
    NHIT=0
    for f in --tensor-type --tensor-type-file --output-tensor-type --token-embedding-type; do
      hit=$(printf '%s\n' "$OUT" | awk -v p="$f" 'BEGIN{n=0} { if (index($0, p) > 0) n++ } END { print n + 0 }')
      if [ "$hit" -ge 1 ]; then NHIT=$((NHIT + 1)); fi
    done
    if [ "$NHIT" -eq 4 ]; then
      pass "llama-quantize --help advertises all 4 per-tensor type flags (--tensor-type[-file], --output-tensor-type, --token-embedding-type)"
    else
      fail "llama-quantize --help advertises only $NHIT/4 per-tensor type flags — DECISIONS.md §D-12 §3 would be false at this revision"
    fi
  else
    first=$(printf '%s\n' "$OUT" | awk 'BEGIN{r=""} { if (r == "" && index($0, "version") > 0) r = $0 } END { print r }')
    if [ -n "$first" ]; then
      pass "$t $probe_flag starts and prints: ${first:0:70}"
    else
      fail "$t $probe_flag printed no version line — the binary does not start (first line: $(printf '%s\n' "$OUT" | head -1))"
    fi
  fi
  MISSING=$(ldd "$BIN/$t" 2>&1 | awk 'BEGIN{n=0} { if (index($0, "not found") > 0) n++ } END { print n + 0 }')
  if [ "$MISSING" -eq 0 ]; then
    pass "ldd $t: every shared library resolves"
  else
    fail "ldd $t: $MISSING unresolved librar(ies): $(ldd "$BIN/$t" 2>&1 | awk '{ if (index($0, "not found") > 0) printf "%s ", $1 }') — add -Wl,-rpath,$ROCM_PATH/lib"
  fi
done

# libggml-hip.so must be a real shared object: BUILD_SHARED_LIBS=ON is what makes the HIP backend
# inspectable, and a static fallback would otherwise pass a bare "target built" check.
HIPSO=$(find "$BUILD" -name 'libggml-hip.so*' -type f 2>/dev/null | sort | head -1 || true)
if [ -n "$HIPSO" ]; then
  pass "libggml-hip.so: $HIPSO ($(stat -c %s "$HIPSO") bytes)"
else
  fail "no libggml-hip.so under $BUILD — the HIP backend was not built shared (or not at all)"
fi

if [ -n "$HIPSO" ]; then
  NEEDS=$(readelf -d "$HIPSO" 2>/dev/null | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' || true)
  for lib in amdhip64 rocblas hipblas; do
    hit=$(printf '%s\n' "$NEEDS" | awk -v l="lib$lib" 'BEGIN{n=0} { if (index($0, l) > 0) n++ } END { print n + 0 }')
    if [ "$hit" -ge 1 ]; then pass "libggml-hip.so NEEDED lib$lib.so"
    else fail "libggml-hip.so does not NEEDED lib$lib.so (NEEDED: $(printf '%s' "$NEEDS" | tr '\n' ' '))"; fi
  done

  # Device code really embedded, and really for $ARCH — read three independent ways that must
  # agree: readelf's Size column, the bytes llvm-objcopy extracts from that section, and the target
  # strings the packed bundle declares about itself. A CPU-only fallback has no .hip_fatbin at all;
  # a bundle for another arch declares a different target string. (The first version of this check
  # read the Address column instead of the Size column and printed an address as a byte count, and
  # the check before it read the runtime device enumeration instead of the file — both are in the
  # note §Corrections. A check whose number is wrong is a check that was not run.)
  TMPD=$(mktemp -d)
  "$ROCM_PATH/lib/llvm/bin/llvm-objcopy" --dump-section=.hip_fatbin="$TMPD/fatbin.bin" "$HIPSO" /dev/null 2>/dev/null || true
  FAT=$(python3 - "$HIPSO" "$TMPD/fatbin.bin" "$BUILD" "$ARCH" <<'PY'
import os, re, subprocess, sys
hipso, dump, build, arch = sys.argv[1:5]
sizes = []
out = subprocess.run(['readelf', '-S', '-W', hipso], capture_output=True, text=True).stdout
for line in out.splitlines():
    if ']' not in line:
        continue
    f = line.split(']', 1)[1].split()          # [Nr] Name Type Address Off Size ES Flg Lk Inf Al
    if f and f[0] == '.hip_fatbin':
        sizes.append(int(f[4], 16))
extracted = os.path.getsize(dump) if os.path.exists(dump) else -1
blob = open(dump, 'rb').read() if extracted > 0 else b''
targets = {}
for m in re.finditer(rb'hipv\d+-amdgcn-amd-amdhsa--(gfx[0-9a-z]+)', blob):
    targets[m.group(1).decode()] = targets.get(m.group(1).decode(), 0) + 1
n_tu = 0
n_obj = 0
for root, _, files in os.walk(build):
    if 'ggml-hip.dir' not in root:
        continue
    for f in files:
        if f.endswith('.o') and ('--' + arch).encode() in open(os.path.join(root, f), 'rb').read():
            n_tu += 1
        if f.endswith('.o'):
            n_obj += 1
other = {k: v for k, v in targets.items() if k != arch}
print(f"{len(sizes)} {sum(sizes)} {extracted} {targets.get(arch, 0)} {sum(other.values())} "
      f"{','.join(sorted(other)) or 'none'} {n_tu} {n_obj}")
PY
)
  set -- $FAT
  N_SEC=$1; SEC_BYTES=$2; EXT_BYTES=$3; N_DEV=$4; N_OTHER=$5; OTHER_LIST=$6; N_TU=$7; N_OBJ_ALL=$8
  echo "record: .hip_fatbin sections=$N_SEC size=$SEC_BYTES extracted=$EXT_BYTES; device code objects=$N_DEV (other arches: $N_OTHER, listed: $OTHER_LIST); HIP sources carrying one=$N_TU of $N_OBJ_ALL objects"
  if [ "$N_SEC" -eq 1 ] && [ "$SEC_BYTES" -eq "$EXT_BYTES" ] && [ "$SEC_BYTES" -ge 1000000 ]; then
    pass ".hip_fatbin present: 1 section of $SEC_BYTES bytes, byte-for-byte what llvm-objcopy extracted"
  else
    fail ".hip_fatbin missing or wrong in $HIPSO (sections=$N_SEC size=$SEC_BYTES extracted=$EXT_BYTES) — no device code embedded, or the two readers disagree"
  fi
  if [ "$N_OTHER" -eq 0 ] && [ "$N_DEV" -ge 1 ]; then
    pass "the packed bundle declares exactly one device target, $ARCH ($N_DEV code objects, 0 for any other arch)"
  else
    fail "the packed bundle is not a single-$ARCH build: $N_DEV $ARCH code objects and $N_OTHER for [$OTHER_LIST]"
  fi
  if [ "$N_DEV" -eq "$N_TU" ] && [ "$N_DEV" -ge 100 ]; then
    pass "code objects linked == HIP sources that compiled device code ($N_DEV == $N_TU of $N_OBJ_ALL; the rest are host-side dispatchers)"
  else
    fail "code objects $N_DEV != HIP sources carrying device code $N_TU (of $N_OBJ_ALL objects) — device code was dropped between compile and link, or the count is vacuous"
  fi

  # The shipped binary must actually reach the backend at run time (pitfalls/50 §-rpath).
  LDD_HIP=$(ldd "$BIN/llama-cli" 2>/dev/null | awk 'BEGIN{n=0} { if (index($0, "libggml-hip") > 0) n++ } END { print n + 0 }')
  if [ "$LDD_HIP" -ge 1 ]; then pass "ldd llama-cli reaches libggml-hip.so"
  else fail "ldd llama-cli does not reach libggml-hip.so — the HIP backend is not in the dependency chain, a silent CPU fallback"; fi
else
  for why in "NEEDED libamdhip64" "NEEDED librocblas" "NEEDED libhipblas" ".hip_fatbin section" \
             "packed-bundle device target" "code-object count vs HIP sources" "ldd reachability"; do
    fail "cannot check $why — no libggml-hip.so"
  done
fi

# Every HIP device compile must carry an --offload-arch= and the only value allowed is $ARCH.
CCJ=$BUILD/compile_commands.json
if [ -f "$CCJ" ]; then
  PARSED=$(python3 - "$CCJ" <<'PY'
import json, sys
entries = json.load(open(sys.argv[1]))
def args(e):
    a = e.get('arguments')
    return a if a else e.get('command', '').split()
hip = [e for e in entries if any(a.startswith('--offload-arch=') for a in args(e))]
archs = sorted({a.split('=', 1)[1] for e in hip for a in args(e) if a.startswith('--offload-arch=')})
files = sorted({e['file'] for e in hip})
print(f"{len(entries)} {len(hip)} {len(files)} {'/'.join(archs) or 'NONE'}")
PY
)
  set -- $PARSED
  N_ALL=$1; N_HIP_CMD=$2; N_HIP_FILE=$3; HIP_ARCHES=$4
  echo "record: compile_commands.json $N_ALL entries; $N_HIP_CMD device compiles over $N_HIP_FILE distinct files; arch(es)=$HIP_ARCHES"
  if [ "$HIP_ARCHES" = "$ARCH" ]; then pass "the only --offload-arch value in the build is $ARCH"
  else fail "offload arch set is '$HIP_ARCHES', expected exactly '$ARCH' — the build fell back or built extra arches"; fi

  if [ "$N_HIP_FILE" -eq "$N_EXPECT_HIP" ] && [ "$N_HIP_CMD" -eq "$N_EXPECT_HIP" ]; then
    pass "device-compile count $N_HIP_FILE == glob-derived expectation $N_EXPECT_HIP (top $N_TOP + tile $N_TILE + mma $N_MMA + mmq $N_MMQ + mmf $N_MMF + fattn-vec $N_VEC)"
  else
    fail "device-compile count $N_HIP_CMD/$N_HIP_FILE != derived $N_EXPECT_HIP — a source set drifted from the ggml-hip globs"
  fi

  N_OBJ=$(find "$BUILD" -path '*ggml-hip.dir*' -name '*.o' -type f 2>/dev/null | awk 'BEGIN{n=0}{n++}END{print n+0}')
  if [ "$N_OBJ" -gt 0 ] && [ "$N_OBJ" -eq "$N_HIP_FILE" ]; then
    pass "objects built for ggml-hip.dir: $N_OBJ == $N_HIP_FILE compiled sources"
  else
    fail "ggml-hip.dir objects $N_OBJ != compiled sources $N_HIP_FILE (0 objects means the backend never compiled)"
  fi
else
  fail "the only --offload-arch value in the build is $ARCH — no $CCJ to read"
  fail "device-compile count vs derived $N_EXPECT_HIP — no $CCJ to read"
  fail "ggml-hip.dir objects vs compiled sources — no $CCJ to read"
fi

echo "checks ran: $RAN / expected $EXPECTED_CHECKS (failures: $FAILED)"
if [ "$RAN" -ne "$EXPECTED_CHECKS" ]; then
  echo "FATAL: the gate ran $RAN checks but expects $EXPECTED_CHECKS — a check stopped running. Fix the gate; do not lower the number." >&2
  exit 1
fi
if [ "$FAILED" -ne 0 ]; then
  echo "GATE RED: $FAILED check(s) failed. Raw log: $LOG" >&2
  exit 1
fi
echo "GATE GREEN: $RAN/$EXPECTED_CHECKS checks passed. Artifacts in $BIN, raw log $LOG."
echo "no timing taken (this script compiles; the main agent owns every device run)"
