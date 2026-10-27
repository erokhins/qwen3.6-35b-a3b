#!/usr/bin/env bash
# llama_instrument_apply.sh — apply tools/llama-instrument.patch to the vendored llama.cpp
# checkout, build the CPU-only instrumented binary, and be able to put the checkout back exactly
# as it was found.
#
# Why a patch file instead of edits in engines/llama.cpp: the vendored tree is a read-only
# cross-check (DECISIONS.md §D-12) and every quote in quantization/evidence/ is a claim about
# commit d327bf1f as it came from upstream. A patch plus this script means the instrumentation
# survives a re-clone and the tree is provably pristine when the work is done.
#
# CPU-only, always. This script never configures a HIP/device variant and never touches the iGPU
# (AGENTS.md §GPU execution is serialized and belongs to the main agent). The gate below proves
# the claim from the artifacts rather than from the flags: no ROCm library in ldd, no device
# compile in compile_commands.json, no libggml-hip in the build dir.
#
# Usage:
#   tools/llama_instrument_apply.sh apply     # guards + expected-header check + git apply
#   tools/llama_instrument_apply.sh build     # configure + build + CPU-only gate
#   tools/llama_instrument_apply.sh restore   # reverse-apply, delete generated header, prove clean
#   tools/llama_instrument_apply.sh status    # is the patch currently applied?
#
# Env overrides: SRC, WORK, CMAKE, JOBS, PINNED_COMMIT, PY.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)

SRC=${SRC:-$REPO/engines/llama.cpp}
WORK=${WORK:-/tmp/llama-instrument}
BUILD=$WORK/build-cpu
CLONE_HINT="$WORK/clone"
PATCH=$REPO/tools/llama-instrument.patch
EXPECTED_H=$REPO/tools/llama-dump-expected.h
GEN=$REPO/tools/llama_dump_expected_gen.py
PY=${PY:-/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python}
PINNED_COMMIT=${PINNED_COMMIT:-d327bf1f0780f87b8c01b367b61853b0c291d7c4}
NPROC=$(nproc)
JOBS=${JOBS:-$(( NPROC > 32 ? 32 : NPROC ))}

GENERATED_IN_TREE=( src/llama-dump.cpp src/llama-dump.h src/llama-dump-expected.h )

die() { echo "llama_instrument_apply: $*" >&2; exit 1; }

find_cmake() {
    if [ -n "${CMAKE:-}" ]; then return 0; fi
    for c in "$(command -v cmake 2>/dev/null || true)" "$HOME/.local/bin/cmake" \
             /usr/local/bin/cmake /usr/bin/cmake; do
        if [ -n "$c" ] && [ -x "$c" ]; then CMAKE=$c; return 0; fi
    done
    die "cmake not found on PATH or in ~/.local/bin — install one (uv tool install cmake) or pass CMAKE=/path/to/cmake"
}

git_in_src() { git -C "$SRC" "$@"; }

guard_checkout() {
    [ -d "$SRC/.git" ] || die "no git checkout at $SRC"
    local head
    head=$(git_in_src rev-parse HEAD)
    if [ "$head" != "$PINNED_COMMIT" ]; then
        die "GUARD FIRED: $SRC is at $head, not the pinned $PINNED_COMMIT. Every quote in
quantization/evidence/llama-cpp-*.md is a claim about the pinned revision; re-read them and
update PINNED_COMMIT before instrumenting a different one."
    fi
    if [ -n "$(git_in_src status --porcelain)" ]; then
        die "GUARD FIRED: $SRC has modified or untracked tracked files. An edited checkout
invalidates the quotes in quantization/evidence/ as much as a new commit would, and it makes
'restore' unable to prove the tree is pristine. Run '$0 restore' or clean it up first."
    fi
}

patch_applied() {
    [ -f "$SRC/src/llama-dump.cpp" ] && git_in_src apply --reverse --check "$PATCH" >/dev/null 2>&1
}

cmd_status() {
    echo "src            : $SRC"
    echo "HEAD           : $(git_in_src rev-parse HEAD)"
    echo "porcelain      : $(git_in_src status --porcelain | wc -l) line(s)"
    echo "patch applied  : $(patch_applied && echo yes || echo no)"
    echo "build dir      : $BUILD"
    [ -f "$BUILD/bin/llama-cli" ] && echo "binary         : $BUILD/bin/llama-cli"
}

cmd_apply() {
    [ -f "$PATCH" ]      || die "no patch at $PATCH"
    [ -f "$EXPECTED_H" ] || die "no derived header at $EXPECTED_H — run tools/llama_dump_expected_gen.py"
    "$PY" "$GEN" --check || die "the committed expected header does not match the vendored sources"

    if patch_applied; then
        # Idempotent path: the tree is deliberately dirty, so the clean-tree guard cannot apply.
        # What still has to hold is that it is dirty in exactly the files the patch touches and
        # nothing else — the dirt set is derived from the patch header, not typed.
        local want have
        want=$(grep -E '^\+\+\+ b/' "$PATCH" | sed 's|^+++ b/||' | sort)
        want=$(printf '%s\n%s\n' "$want" "src/llama-dump-expected.h" | sort)
        have=$(git_in_src status --porcelain | awk '{print $2}' | sort)
        if [ "$want" != "$have" ]; then
            echo "expected dirt (from the patch) | actual dirt" >&2
            diff <(printf '%s\n' "$want") <(printf '%s\n' "$have") >&2 || true
            die "the checkout is dirty beyond what $PATCH accounts for"
        fi
        echo "patch already applied at $SRC, and only the patch's files are dirty"
    else
        guard_checkout
        git_in_src apply --check "$PATCH" || die "patch does not apply cleanly to $SRC"
        git_in_src apply "$PATCH" || die "git apply failed"
        cp "$EXPECTED_H" "$SRC/src/llama-dump-expected.h"
        echo "applied $PATCH to $SRC (+ src/llama-dump-expected.h generated header)"
    fi
    mkdir -p "$WORK"
    {
        echo "# apply-record.json — the instrumentation state of the vendored checkout."
        echo "generated_by=tools/llama_instrument_apply.sh apply"
        echo "src=$SRC"
        echo "commit=$PINNED_COMMIT"
        echo "patch=$PATCH"
        echo "patch_sha256=$(sha256sum "$PATCH" | cut -d' ' -f1)"
        echo "expected_header=$EXPECTED_H"
        echo "expected_header_sha256=$(sha256sum "$EXPECTED_H" | cut -d' ' -f1)"
        echo "generator=$GEN"
        echo "generator_sha256=$(sha256sum "$GEN" | cut -d' ' -f1)"
        echo "porcelain_lines=$(git_in_src status --porcelain | wc -l)"
    } > "$WORK/apply-record.json"
    git_in_src status --porcelain | sed 's/^/  /'
}

cmd_restore() {
    [ -d "$SRC/.git" ] || die "no git checkout at $SRC"
    if patch_applied; then
        git_in_src apply --reverse "$PATCH" || die "reverse apply failed"
        echo "reverse-applied $PATCH"
    fi
    # the generated header is not in the patch (it is derived), so it is removed by explicit name —
    # never a recursive delete of a variable path (pitfalls/50).
    for f in "${GENERATED_IN_TREE[@]}"; do
        [ -f "$SRC/$f" ] && rm -f "$SRC/$f"
    done
    local n
    n=$(git_in_src status --porcelain | wc -l)
    if [ "$n" != "0" ]; then
        git_in_src status --porcelain | sed 's/^/  /' >&2
        die "tree is NOT pristine after restore ($n porcelain line(s))"
    fi
    echo "restore: git -C $SRC status is clean"
}

write_build_record() {
    find_cmake
    mkdir -p "$BUILD"
    {
        echo "# build-record.json — how the instrumented CPU-only llama.cpp was made."
        echo "# Written by tools/llama_instrument_apply.sh; the manifest binds it by sha256."
        echo "generated_by=tools/llama_instrument_apply.sh"
        echo "src=$SRC"
        echo "commit=$PINNED_COMMIT"
        echo "patch=$PATCH"
        echo "patch_sha256=$(sha256sum "$PATCH" | cut -d' ' -f1)"
        echo "expected_header_sha256=$(sha256sum "$EXPECTED_H" | cut -d' ' -f1)"
        echo "build_dir=$BUILD"
        echo "jobs=$JOBS"
        echo "cmake=$CMAKE"
        echo "configure_command=cmake -S $SRC -B $BUILD -G \"Unix Makefiles\" -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_OPENMP=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_FLAGS=-DLLAMA_DUMP_GIT_COMMIT=\\\"$PINNED_COMMIT\\\""
        echo "build_command=cmake --build $BUILD --target llama-cli --parallel $JOBS"
        grep -E '^(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|CMAKE_CXX_FLAGS|CMAKE_CXX_FLAGS_RELEASE|GGML_HIP|GGML_CUDA|GGML_OPENMP|GGML_NATIVE|GGML_CCACHE|BUILD_SHARED_LIBS|LLAMA_CURL|LLAMA_OPENSSL|CMAKE_EXPORT_COMPILE_COMMANDS):' \
            "$BUILD/CMakeCache.txt" 2>/dev/null || true
    } > "$BUILD/build-record.json"
}

gate() {
    # Counted checks, same discipline as tools/build_llama.sh: the expected count is a constant the
    # gate refuses to disagree with, so a check that stops running is a failure, not a shorter
    # green print (pitfalls/40 §a check that can produce no output cannot fail).
    local -a names=(
        "binary exists"
        "ldd: zero ROCm/HIP libraries"
        "ldd: zero 'not found'"
        "compile_commands.json: zero device compiles"
        "CMakeCache: GGML_HIP=OFF and GGML_CUDA=OFF"
        "build dir: no libggml-hip artifact"
        "llama-cli --version reports the pinned commit"
        "instrument linked: llama_dump_after_graph in libllama"
    )
    local expected=${#names[@]} ran=0 fails=0 i=0
    local ok line

    # The checks below are probes whose non-zero status is data, not a failure of this script, so
    # errexit is off for the block and the counters are the only thing that decides PASS/FAIL.
    set +e

    check() { # check <pass:0|1> <name> <detail...>
        local rc=$1; shift
        local nm=$1; shift
        ran=$((ran + 1))
        if [ "$rc" = "0" ]; then
            printf 'PASS: %s\n' "$nm"
        else
            fails=$((fails + 1))
            printf 'FAIL: %s\n' "$nm"
        fi
        for line in "$@"; do printf '      %s\n' "$line"; done
    }

    [ -f "$BUILD/build-record.json" ] || write_build_record

    # 1 binary exists
    [ -x "$BUILD/bin/llama-cli" ]; check $? "${names[$i]}: $BUILD/bin/llama-cli"; i=$((i+1))

    # 2 no ROCm in the link line — read the artifact, do not trust the flag
    local ldd_hip
    ldd_hip=$(ldd "$BUILD/bin/llama-cli" 2>/dev/null | grep -Eci 'hip|rocblas|hipblas|amdhip|hsa-runtime' || true)
    if [ "${ldd_hip:-1}" = "0" ]; then
        check 0 "${names[$i]} (matched lines: $ldd_hip)"
    else
        check 1 "${names[$i]} (matched lines: $ldd_hip)"
        ldd "$BUILD/bin/llama-cli" | grep -Ei 'hip|rocblas|amdhip|hsa' | sed 's/^/      /'
    fi
    i=$((i+1))

    local ldd_miss
    ldd_miss=$(ldd "$BUILD/bin/llama-cli" 2>/dev/null | grep -c 'not found' || true)
    [ "${ldd_miss:-1}" = "0" ]; check $? "${names[$i]} (missing: $ldd_miss)"; i=$((i+1))

    # 4 the compile database holds no device compilation at all
    local dev=0 dev2=0 cc
    cc=$BUILD/compile_commands.json
    if [ -f "$cc" ]; then
        dev=$(grep -c -- '--offload-arch' "$cc" || true)
        dev2=$(grep -cE '"file": "[^"]*\.(cu|hip)"' "$cc" || true)
    else
        dev=-1
    fi
    [ "$dev" = "0" ] && [ "$dev2" = "0" ]
    check $? "${names[$i]} (offload-arch lines: $dev, device sources: $dev2)"; i=$((i+1))

    # 5 the cache says what we asked for
    grep -q '^GGML_HIP:BOOL=OFF$' "$BUILD/CMakeCache.txt" 2>/dev/null \
      && grep -q '^GGML_CUDA:BOOL=OFF$' "$BUILD/CMakeCache.txt" 2>/dev/null
    check $? "${names[$i]}"; i=$((i+1))

    # 6 no HIP object anywhere in the build tree
    local n_hip
    n_hip=$(find "$BUILD" -name 'libggml-hip*' -o -name '*.hip_fatbin' -o -name '*.co' 2>/dev/null | wc -l)
    [ "$n_hip" = "0" ]; check $? "${names[$i]} (found: $n_hip)"; i=$((i+1))

    # 7 the binary self-reports the revision the quotes in quantization/evidence/ are about.
    # llama_print_build_info() writes to stderr, so 2>/dev/null here would capture nothing.
    local ver
    ver=$("$BUILD/bin/llama-cli" --version 2>&1 | tr -d '\r' | head -3 | tr '\n' ' ')
    case $ver in
        *"$(printf %.7s "$PINNED_COMMIT")"*) check 0 "${names[$i]} [$ver]" ;;
        *)                                   check 1 "${names[$i]} [$ver]" ;;
    esac
    i=$((i+1))

    # 8 the instrumentation is actually in the library the binary loads (not compiled out). grep -q
    # inside a pipeline can SIGPIPE nm, and pipefail would then report the pipeline as failed, so
    # the symbol table is written out first.
    local sym syms
    sym=$(find "$BUILD/bin" -maxdepth 1 -name 'libllama.so*' -type f -print -quit)
    syms=$WORK/.gate-symbols.txt
    if [ -n "$sym" ]; then nm -D --defined-only "$sym" > "$syms" 2>/dev/null; else : > "$syms"; fi
    grep -q llama_dump_after_graph "$syms"
    check $? "${names[$i]} ($sym)"; i=$((i+1))

    set -e
    echo "checks ran: $ran / expected $expected (failures: $fails)"
    if [ "$ran" != "$expected" ] || [ "$fails" != "0" ]; then
        echo "GATE RED"
        return 1
    fi
    echo "GATE GREEN"
    write_build_record
    echo "build record: $BUILD/build-record.json"
    return 0
}

cmd_build() {
    find_cmake
    patch_applied || die "patch is not applied — run '$0 apply' first"
    mkdir -p "$WORK"
    if [ -f "$BUILD/CMakeCache.txt" ]; then
        local cached
        cached=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$BUILD/CMakeCache.txt" | head -1)
        if [ -n "$cached" ] && [ "$cached" != "$SRC" ]; then
            die "$BUILD was configured from source dir $cached, not $SRC. Point WORK somewhere else
  (cmake refuses to reuse a cache from a different source tree, and silently reusing one would
  build a binary that does not match the checkout the manifest claims)."
        fi
    fi
    if [ ! -f "$BUILD/CMakeCache.txt" ]; then
        "$CMAKE" -S "$SRC" -B "$BUILD" -G "Unix Makefiles" \
            -DCMAKE_BUILD_TYPE=Release \
            -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_OPENMP=ON \
            -DBUILD_SHARED_LIBS=ON \
            -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            -DCMAKE_CXX_FLAGS="-DLLAMA_DUMP_GIT_COMMIT=\\\"$PINNED_COMMIT\\\"" \
            > "$WORK/configure.log" 2>&1 || { tail -30 "$WORK/configure.log" >&2; die "configure failed"; }
    fi
    "$CMAKE" --build "$BUILD" --target llama-cli --parallel "$JOBS" > "$WORK/build.log" 2>&1 \
        || { tail -40 "$WORK/build.log" >&2; die "build failed"; }
    gate
}

cmd_regen() {
    # Regenerate tools/llama-instrument.patch from the instrumented checkout. The patch is the
    # artifact of record, so it is produced by diffing a clean clone of the pinned HEAD against the
    # current instrumented files, and then proved to apply to a *second* clean clone before it is
    # kept. The vendored checkout's index is never touched.
    [ -d "$SRC/.git" ] || die "no git checkout at $SRC"
    [ -f "$PATCH" ]    || die "no patch at $PATCH to read the file list from"
    local tmp
    tmp=$(mktemp -d /tmp/llama-regen.XXXXXX) || die "mktemp failed"
    local files
    files=$(grep -E '^\+\+\+ b/' "$PATCH" | sed 's|^+++ b/||')
    [ -n "$files" ] || die "patch $PATCH lists no files"
    git clone -q --no-hardlinks -s "$SRC" "$tmp/base" || die "clone failed"
    git clone -q --no-hardlinks -s "$SRC" "$tmp/verify" || die "clone failed"
    ( cd "$tmp/base" && git checkout -q -- . && git clean -qfd )
    local f
    for f in $files; do
        [ -f "$SRC/$f" ] || die "$SRC/$f is missing — the patch names a file that is not in the tree"
        mkdir -p "$tmp/base/$(dirname "$f")"
        cp "$SRC/$f" "$tmp/base/$f"
    done
    (
        cd "$tmp/base"
        git add -A -- $files
        git diff --cached
    ) > "$tmp/patch" || die "diff failed"
    [ -s "$tmp/patch" ] || die "generated patch is empty"
    # prove it: a clone that has never seen the instrumentation must take it
    ( cd "$tmp/verify" && git apply --check "$tmp/patch" ) \
        || { cat "$tmp/patch" >&2; die "generated patch does not apply to a clean clone"; }
    cp "$tmp/patch" "$PATCH"
    echo "regenerated $PATCH ($(wc -l < "$PATCH") lines, $(printf '%s\n' $files | wc -w) files)"
    echo "scratch kept for inspection: $tmp"
}

main() {
    case ${1:-status} in
        apply)   cmd_apply ;;
        build)   cmd_build ;;
        restore) cmd_restore ;;
        regen)   cmd_regen ;;
        status)  cmd_status ;;
        *)       die "unknown subcommand '${1:-}' (apply|build|restore|regen|status)" ;;
    esac
}

main "$@"
