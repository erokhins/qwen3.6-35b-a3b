# Building and verifying the pinned llama.cpp oracle (HIP / `gfx1151`)

**Provenance.** Written 2026-10-26 (box clock read 2026-10-25 CEST at build time) by W3, from a build
of `engines/llama.cpp` at commit **`d327bf1f0780f87b8c01b367b61853b0c291d7c4`** (tag **`b11869`**,
"jinja : fix select and selectattr argument handling"), from `https://github.com/ggml-org/llama.cpp`.
Box: Framework Desktop, Ryzen AI Max+ 395, `gfx1151`, ROCm **7.15.26333-0000000** at `/opt/rocm`,
HIP compiler `/opt/rocm/lib/llvm/bin/clang++` (**Clang 23.0.0**), C/C++ **GNU 13.3.0**, generator
**Unix Makefiles** (`/usr/bin/gmake`). Reproduce with `tools/build_llama.sh`; raw log
`logs/llama-build-2026-10-25.log` (gitignored), two invocations at 02:09:09 and 02:24:41.
Everything below is a claim about **that** revision; llama.cpp moves weekly, so re-read before
relying on it (`AGENTS.md` §Self-containment — the point of this file is that a reader who cannot
open `engines/llama.cpp` can still check every verdict here).

**No device timing was taken anywhere in this work.** Build wall time is reported; the only device
contact was compilation and one `--list-devices` query. §Deferred lists what the main agent must run.

---

## §Artifacts

All in `engines/llama.cpp/build-hip/bin/` (out-of-tree build dir inside the checkout, ignored by
llama.cpp's own `/build*` rule, and `engines/` is gitignored here):

| artifact | bytes | note |
|---|---|---|
| `llama-cli` | — | `--version` → `version: 0.7.0-dev (build 1, commit d327bf1)`, `built with GNU 13.3.0 for Linux x86_64` |
| `llama-quantize` | — | `--help` prints the type table (§Verdict 1) |
| `libggml-hip.so.0.27.0` | 72,910,832 | SONAME `libggml-hip.so.0`; carries the device code (§§HIP proof) |
| `libggml.so.0.27.0`, `libggml-base.so.0.27.0`, `libggml-cpu.so.0.27.0` | — | dispatcher + CPU backend |
| `libllama.so.0.7.0`, `libllama-common.so.0.7.0`, `libllama-cli-impl.so`, `libllama-quantize-impl.so`, `libllama-server-impl.so`, `libmtmd.so.0.7.0` | — | llama side |

## §The build recipe (exact commands)

`tools/build_llama.sh` runs these; they are repeated here so the recipe survives the script.

```bash
cmake -S engines/llama.cpp -B engines/llama.cpp/build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151 -DGPU_TARGETS=gfx1151 -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DGGML_CCACHE=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
  -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,/opt/rocm/lib \
  -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath,/opt/rocm/lib

cmake --build engines/llama.cpp/build-hip --parallel 16 -- llama-cli llama-quantize
```

Five facts that are not obvious and cost real time if rediscovered:

1. **`cmake` is not on PATH on this box.** Neither is `ninja`. Installed for this build with
   `uv tool install cmake` → cmake **4.4.4** at `~/.local/bin/cmake` (user-level, `uv tool uninstall
   cmake` to undo). `tools/build_llama.sh` takes `CMAKE=/path/to/cmake` and otherwise searches PATH
   then `~/.local/bin/cmake`, `/usr/local/bin/cmake`, `/usr/bin/cmake`, and fails with that advice.
   Generator `Unix Makefiles` + `/usr/bin/gmake`, `-j16`.
2. **The arch knob this CMake version honours is `GPU_TARGETS` / `AMDGPU_TARGETS`,** forwarded to
   `CMAKE_HIP_ARCHITECTURES` by `engines/llama.cpp/ggml/src/ggml-hip/CMakeLists.txt` §"Forward (AMD)GPU_TARGETS to
   CMAKE_HIP_ARCHITECTURES". Pass **all three**: the forwarding is guarded by
   `if(GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)` and the whole branch is skipped when `hipcc` is
   used as the C++ compiler (`CXX_IS_HIPCC`, which also draws a "legacy behavior" warning).
   Canonical arch spelling for this part: `/opt/rocm/bin/offload-arch` prints `gfx1151`, and
   `rocminfo` lists `Name: gfx1151` (plus `amdgcn-amd-amdhsa--gfx11-generic`).
   `hipcc --list-gpu-arch` **no longer exists** in ROCm 7 (`clang++: error: unknown argument`), so
   `offload-arch` + `rocminfo` are the spellings to use.
3. **`-Wl,-rpath,/opt/rocm/lib` is required on BOTH the executable and the shared-library link
   lines** (`pitfalls/50-harness-and-shell.md` §Build and toolchain facts, reproduced fresh). Without
   it everything links and then dies at exec: `./llama-cli: error while loading shared libraries:
   libhipblas.so.3: cannot open shared object file`. This box has **no** ROCm entry in
   `/etc/ld.so.conf.d/` (`ldconfig -p | grep -c hipblas` → 0). Setting it on the executable alone is
   not enough: `RUNPATH` is not consulted for the *dependencies of a dependency*, so
   `libggml-hip.so` needs its own — that is why `CMAKE_SHARED_LINKER_FLAGS` is there too.
   Both are now in the script, and the gate checks `ldd` for zero `not found` lines.
4. **`-DLLAMA_USE_PREBUILT_UI=OFF`** — with the default ON, the build silently downloads
   `llama-ui/dist.tar.gz` from Hugging Face (`engines/llama.cpp/scripts/ui-assets.cmake`), i.e. a network dependency
   inside a "reproducible" build. The first invocation of this build did exactly that; the flag is
   now pinned OFF.
5. **`GGML_CUDA_FA_QUANTS` defaults to** `"q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16"`
   (`engines/llama.cpp/ggml/CMakeLists.txt` §option list), i.e. 4 FlashAttention vector instances, not all 49. Left at
   the default, and the build script *reads it back out of the CMake cache* to derive its expected
   device-source count (§§HIP proof), so the number is not hardcoded.

Configure headline lines (from the log): `The HIP compiler identification is Clang 23.0.0`,
`HIP and hipBLAS found`, `Including HIP backend`, `FlashAttention K-V type combinations:
f16-f16;q4_0-q4_0;q8_0-q8_0;bf16-bf16`, `ggml version: 0.27.0`, `ggml commit: d327bf1`,
`Adding CPU backend variant ggml-cpu: -march=native` (`GGML_NATIVE` left at its default ON).
Warnings that matter: one `CMake Warning at scripts/ui-assets.cmake:582` (the UI download, now
avoided by flag 4). No compile warnings survived into the log.

## §Commit guard, and the three negative controls witnessed

The guard is the point of the script — an oracle whose revision drifted is worse than no oracle,
because every verdict in this file is a claim about `d327bf1`. Verified by **making it fire**, not
by trusting it to stay quiet:

```
$ BUILD_LLAMA_PINNED_COMMIT=000…0 tools/build_llama.sh --check      # exit 3
GUARD FIRED: …/engines/llama.cpp is not the pinned oracle revision.
  pinned : 0000000000000000000000000000000000000000 (b11869)
  found  : d327bf1f0780f87b8c01b367b61853b0c291d7c4 (b11869)
  An unpinned cross-check oracle is worse than none: the four verdicts in
  quantization/evidence/llama-cpp-build-and-flags-2026-10-26.md are claims about the pinned
  revision, and llama.cpp moves weekly. Either check out the pinned commit, or re-read those
  four sources at this revision, update that note, and update PINNED_COMMIT here.
```

`BUILD_LLAMA_PINNED_COMMIT` exists **only** to witness that refusal; never set it for a real build.
A second guard refuses a checkout with modified tracked files (exit 4) — an edit inside the checkout
invalidates the quotes below as much as a new commit. `git status --porcelain` in the checkout is
empty as of this writing.

* **Bogus arch is rejected, not ignored.** Configuring with `-DGPU_TARGETS=gfx99999` fails at
  `engines/llama.cpp/ggml/src/ggml-hip/CMakeLists.txt` §"Forward (AMD)GPU_TARGETS" runs `enable_language(HIP)` and blames that file, with
  `clang++: error: unsupported HIP gpu architecture: gfx99999`, and the failing command line shows
  `--offload-arch=gfx99999` — direct proof the knob reaches the compiler rather than being
  decorative. `tools/build_llama.sh` additionally refuses any `LLAMA_HIP_ARCH` that `rocminfo` does
  not list (exit 2), so a typo'd arch cannot quietly become a different arch.
* **The verification gate can go red.** Run against an empty build dir: 17 named `FAIL:` lines,
  `checks ran: 17 / expected 17 (failures: 17)`, `GATE RED`, exit 1. The counter is a constant
  (`EXPECTED_CHECKS=17`) that the gate refuses to disagree with, so a check that stops running is a
  hard failure rather than a shorter, greener print (`AGENTS.md` §A check that can produce no output
  cannot fail).
* **Two checks in the first version of that gate were themselves vacuous and were replaced** — worth
  recording because they passed while telling the truth by accident:
  1. `.hip_fatbin` "size" was read from the wrong `readelf -S` column (Address, not Size) and printed
     1,196,032 as a byte count; the real size is 56,112,304. A trivially small section would have
     passed.
  2. The per-arch check called `/opt/rocm/lib/llvm/bin/llvm-offload-device-info`, which prints the
     **runtime device enumeration** — it printed `Name: gfx1151` because a gfx1151 GPU is attached,
     not because the file contains gfx1151 code. It would have passed on a CPU-only build.
  Both now read the artifact (§§HIP proof). Generalised trap: a tool that can answer from the machine
  will answer from the machine.

## §The HIP path really is in the artifact chain

This section exists because a plausible-looking check said otherwise. Three separate readings, and
the two commands that print *nothing* and why.

**Where the files are.** `libggml-hip.so` is at **`engines/llama.cpp/build-hip/bin/libggml-hip.so.0.27.0`**
(symlinks `libggml-hip.so` and `libggml-hip.so.0` beside it). There is **no**
`engines/llama.cpp/ggml/libggml-hip.so` — `readelf -d` on a nonexistent path prints nothing at all on
stdout, which is what "readelf printed no NEEDED/RUNPATH lines" means. And `find build-hip -name
'*.hip_fatbin*' -o -name '*.co' -o -name '*.hsaco'` returns **nothing**: the device code is *embedded
in the `.so`*, not loose — under the section name **`.hip_fatbin`** (underscore; `.hip.fatbin` with a
dot matches nothing).

**1. The backend library is a real HIP object** — `readelf -d build-hip/bin/libggml-hip.so.0.27.0`:

```
(RUNPATH) Library runpath: [/opt/rocm/lib:/…/engines/llama.cpp/build-hip/bin:]
(NEEDED)  libggml-base.so.0     (NEEDED)  libhipblas.so.3
(NEEDED)  librocblas.so.5       (NEEDED)  libamdhip64.so.7
(SONAME)  libggml-hip.so.0
```

**2. It is wired into what `llama-cli` loads** — this is the step that is easy to miss. `llama-cli`
NEEDs only `libllama-cli-impl.so`, so **`readelf -d bin/llama-cli` legitimately shows no
`libggml-hip`**: it is a second-order dependency. Read it one level down,
`readelf -d build-hip/bin/libggml.so.0.27.0`:

```
(NEEDED)  libggml-cpu.so.0   (NEEDED)  libggml-hip.so.0   (NEEDED)  libggml-base.so.0
```

or take the transitive closure, `ldd bin/llama-cli` (12 HIP-related lines, none `not found`):

```
libggml.so.0        => …/build-hip/bin/libggml.so.0
libggml-hip.so.0    => …/build-hip/bin/libggml-hip.so.0
libhipblas.so.3     => /opt/rocm/lib/libhipblas.so.3
librocblas.so.5     => /opt/rocm/lib/librocblas.so.5
libamdhip64.so.7    => /opt/rocm/lib/libamdhip64.so.7
… hipblaslt, hsa-runtime64, amd_comgr, LLVM.so.23.0git …
```

So the HIP backend is **linked in at load time, not dlopened** (`GGML_BACKEND_DL` is OFF): if
`libggml-hip.so.0` were missing or unresolvable, `llama-cli` would not start at all rather than
quietly becoming CPU-only.

**3. Device code is present, and its arch is `gfx1151` — census, not sample.** `readelf -S -W` gives
one `.hip_fatbin` section of `0x35834b0` = **56,112,304 bytes**; `llvm-objcopy
--dump-section=.hip_fatbin=…` extracts exactly that many bytes. Carving the ELF members out of it
(`e_machine == 224`, `EM_AMDGCN`) yields **141 code objects**, and LLVM's own reader names the arch
each one declares:

```
$ for f in /tmp/w3_co_all/*.elf; do llvm-readobj --notes "$f"; done | grep -oE 'amdgcn-amd-amdhsa--gfx[0-9a-z]+' | sort | uniq -c
    141 amdgcn-amd-amdhsa--gfx1151
code objects carved: 141 | declaring an arch: 141 | declaring anything but gfx1151: 0
```

(`llvm-readobj --file-headers` on the same members reports only `Arch: amdgcn` /
`Machine: EM_AMDGPU (0xE0)` and `OS/ABI: AMDGPU_HSA (0x40)`, `ABIVersion: 4` — for this part the
specific arch lives in the code object's `.note` metadata key `amdhsa.target`, not in the file
header, so `--file-headers` alone cannot settle the arch question.)

**4. Every HIP source was compiled for that arch, and the count has a denominator.**
`compile_commands.json` holds 473 entries, of which **146** are device compiles over 146 distinct
files, and the *only* `--offload-arch=` value anywhere is `gfx1151`. 146 is not what the build
produced — it is derived beforehand from the globs in
`engines/llama.cpp/ggml/src/ggml-hip/CMakeLists.txt` §`file(GLOB GGML_SOURCES_ROCM …)`: 71 top-level `ggml-cuda/*.cu`
+ 12 `fattn-tile*` + 21 `fattn-mma*` + 22 `mmq*` + 16 `mmf*` + 4 fattn-vec instances (read from
`GGML_CUDA_FA_QUANTS` in the cache). Objects under `ggml-hip.dir` also number 146. Of those 146
sources, **141 embed a gfx1151 code object and 141 members were found** — an exact match; the 5
without device code are `fattn-tile.cu`, `mmf.cu`, `mmq.cu`, `set.cu`, `sum.cu`, host-side
dispatchers whose kernels live in the template-instance files.

Note that a `grep` for the literal `gfx1250` inside the fatbin is *not* evidence of a second arch:
those hits are the assembler metadata key `.gfx1250_revision` (an erratum/revision tag), and the
bundle's own entry table declares `hipv4-amdgcn-amd-amdhsa--gfx1151` and no other device target.

## §Would `llama-cli` pick HIP at runtime? Yes — and here is the no-model command

`bin/llama-cli --list-devices` (no model argument, no `-ngl`, nothing loaded; the flag exists at
this commit — `engines/llama.cpp/common/arg.cpp` §`{"--list-devices"}`):

```
$ engines/llama.cpp/build-hip/bin/llama-cli --list-devices      # rc=0, wall 0 s
Available devices:
  ROCm0: AMD Radeon 8060S Graphics (112640 MiB, 34841 MiB free)
```

The ROCm backend is registered and enumerated, so `llama-cli` will select HIP when asked for GPU
layers and is not falling back to CPU for want of a backend. Two honest limits on that line: it
proves *enumeration*, not that a matmul runs on-device (only a model load does that, and that is the
main agent's); and `112640 MiB / 34841 MiB free` is the driver's own self-report from a query, not a
measurement by me — it is the ~110 GiB `totalGlobalMem` illusion plus ~32–34 GiB usable that
`pitfalls/10-device-memory.md` §"The pool is ~32–34 GiB per process" warns about, so do not quote it
as headroom. `tools/gpu_lock.sh util` read **99 %** immediately after this query (someone else's
activity, not mine — I held no lock and took no timing); whoever runs the deferred checks below must
take a clean window.

## §Deferred to the main agent (device runs I was not allowed to take)

Nothing here was run by me. Suggested commands, in the main agent's own halogen-down, zero-workers,
`gpu_lock`-held window:

1. **The smoke test that proves on-device compute** — the published Q4_K_M, small layer count, and
   *the smallest thing that answers the question*:
   `engines/llama.cpp/build-hip/bin/llama-cli -m models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -ngl 1 -p "hi" -n 4 --verbose 3 2>&1 | grep -E "offload|HIP|ROCm|load_tensors"`
   — expect the routed-expert tensors reported as offloaded to `ROCm0`, and expect a *failure*, not a
   fallback, if the backend is broken. `-ngl 1` is the ceiling for this check; `-ngl 99` only after
   §D-11's residency arithmetic is respected, since the model is far above the per-process pool.
2. `--list-devices` again as the window's canary (cheap, proves the backend still registers).
3. The perf ruler of D-12 §2 (`llama-bench`) — **only** in the main agent's window, and only after
   §Verdict 1's gotchas have been read, because a policy-A artifact and a `Q4_K_M` artifact are not
   the same bytes.
4. Policy-A artifact production itself (D-12 §3) is runnable — see §Verdict 1's two gotchas first;
   `--tensor-type` patterns are **regexes matched by `regex_search`**, so an unanchored name is a
   substring match across tensor families.

---

## §Verdict 1 — `llama-quantize` per-tensor type control: **CONFIRMED** (with two gotchas that are not in our docs)

DECISIONS.md §D-12 item 3 — "`llama-quantize` supports `--tensor-type name=ggml_type`,
`--tensor-type-file`, `--output-tensor-type` and `--token-embedding-type`, so 'experts Q4_K,
attention/GDN/shared Q8_0, `lm_head`/embeddings Q8_0' is expressible in the stock tool" — is **true
at this revision**. All four are in `engines/llama.cpp/tools/quantize/quantize.cpp`: declared in `usage()`, parsed in
`llama_quantize()`'s option loop (§the `strcmp` chain at `--output-tensor-type`,
`--token-embedding-type`, `--tensor-type`, `--tensor-type-file`), with `parse_tensor_type()`
(`name=type`, name lowercased, invalid type → usage) and `parse_tensor_type_file()` (same format,
"separated by spaces or newlines"). Witnessed against the shipped artifact, not just the source —
the gate in `tools/build_llama.sh` fails unless the built binary's own `--help` advertises all four:

```
PASS: llama-quantize --help advertises all 4 per-tensor type flags
      (--tensor-type[-file], --output-tensor-type, --token-embedding-type)
```

`--tensor-type` "may be specified multiple times"; the type table also offers `Q4_K`/`Q4_K_M`,
`Q5_K*`, `Q6_K`, `Q8_0`, `MXFP4_MOE`, `BF16`, `COPY`. "Policy A artifact production is upstream's
problem" therefore still holds. Two things our docs do not say, both of which can silently produce
a wrong artifact:

1. **Overrides are regexes, matched by `std::regex_search` — substring matching.** In
   `engines/llama.cpp/src/llama-quant.cpp` §`llama_tensor_get_type`, the parsed names become
   `std::vector<std::pair<std::regex, ggml_type>>` (§`quantize_state_impl`'s constructor,
   "compile regex patterns once") and each is applied with `regex_search(tensor_name, pattern)`,
   first match wins. `--tensor-type attn_q=q8_0` therefore also hits `blk.N.attn_qkv.weight` and
   `blk.N.attn_q_norm.weight`. Policy-A tensor files must be anchored, e.g.
   `^blk\.[0-9]+\.attn_q\.weight$`. Names are lowercased at parse time but matched against GGUF
   names as stored, so write patterns lowercase.
2. **An override is ignored unless the base type is itself quantized.** The same function wraps the
   whole override path in `if (ggml_is_quantized(default_type))`, and `default_type` comes from
   `get_type_from_ftype()` (§`case LLAMA_FTYPE_MOSTLY_Q4_K_S: case LLAMA_FTYPE_MOSTLY_Q4_K_M: return
   GGML_TYPE_Q4_K;`). Quantizing with `F16`/`BF16`/`F32`/`COPY` and passing `--tensor-type` yields
   the base type with no warning. Policy A must be produced from an f16 source at a quantized ftype
   (`Q4_K` is the natural base) with the overrides on top.
3. Also worth knowing: an override cannot force a tensor that
   `tensor_allows_quantization()` refuses to be quantized — that check runs first and returns the
   tensor's existing type. And every requested type still passes through
   `tensor_type_fallback()` (§next), which **downgrades** `Q4_K → Q5_0`, `Q5_K → Q5_1`,
   `Q6_K → Q8_0` when `ne[0] % 256 != 0`, and to `F16` if even the 32-block fallback does not fit —
   loudly (`LLAMA_LOG_WARN … "falling back to"`), so read stderr after a quantize run rather than
   trusting the exit code. For this model's shapes the fallback should never fire: with `hidden=2048`,
   `moe_intermediate=512`, `head_dim=256 × 16 heads`, `value_dim=128 × 32`, every family policy A
   puts in a K-quant has `ne[0] ∈ {512, 1024, 2048, 4096}` — all multiples of 256
   (`checkpoints/Qwen3.6-35B-A3B.config.json`; shapes as declared by `engines/llama.cpp/src/models/qwen35moe.cpp`
   §`load_block_trunk`). The one tensor that *would* have tripped it, `blk.N.ssm_conv1d.weight`
   (`{ssm_d_conv=4, conv_dim}`), is never quantized at all (§Verdict 2).

## §Verdict 2 — the tensor policy note vs the code: **PARTLY — one load-bearing claim is FALSE at this commit**

`quantization/evidence/llama-cpp-tensor-policy-2026-10-26.md` was read at this same tag, and most of
it still matches. What no longer matches is the sentence that DECISIONS.md quotes as evidence for a
user decision.

* **FALSE — "Categories `FFN_LATENT` and `ATTENTION_GATE`: if the chosen type is below Q8_0, it is
  raised to Q8_0."** That bump is now **architecture-gated**, verbatim from
  `engines/llama.cpp/src/llama-quant.cpp` §`llama_tensor_get_type_impl`:

  ```c
  // bumping to q8_0 is preferred since these tensors are not really big
  // TODO: nemotron 3 super shares ffn_latent and afmoe, bailingmoe3, dots3note, hy-v4, kimi-k3 and laguna share attn_gate.
  //       Widening the guard is likely a win there too, but it changes their files, so it wants a perplexity comparison per architecture first.
  if (arch == LLM_ARCH_MUSE_SPARK &&
      (category == tensor_category::FFN_LATENT || category == tensor_category::ATTENTION_GATE)) {
  ```

  For `LLM_ARCH_QWEN35MOE` (`engines/llama.cpp/src/llama-model.cpp` §`LLM_ARCH_QWEN35MOE → llama_model_up_qwen35moe`
  / `llama_model_qwen35moe`) the guard is false, so **llama.cpp does not raise our latent/gate
  tensors to Q8_0** — under `Q4_K_M` they stay Q4_K. The upstream comment says widening the guard is
  deliberately waiting on a per-architecture perplexity study.
  **Why this matters beyond a footnote:** DECISIONS.md §D-01a justifies the user's Q8_0 choice in
  part with "llama.cpp's own policy forces the latent/gate categories to ≥ Q8_0 (evidence
  `quantization/evidence/llama-cpp-tensor-policy-2026-10-26.md`)". That support is gone at this
  revision. D-01a survives on its *other* legs — the unsloth per-tensor sweep calling `attn_*` the
  most sensitive class and GDN `out_proj` the worst single tensor, and the fact that the whole
  attention/GDN family is 1.975 GiB in bf16 so Q8_0 is cheap — but the llama.cpp citation must not
  be repeated as support. Not my decision to make: flagged for the user, and this file is the
  evidence for it.
* **Drift, cosmetic but cite-rot:** the note cites `get_quant_type_use()` (≈line 500). That function
  **no longer exists** anywhere in `engines/llama.cpp/src/` or `engines/llama.cpp/common/`. The type selection now lives in
  `llama_tensor_get_type_impl()` (§line 459) and its wrapper `llama_tensor_get_type()` (§line 743),
  plus `tensor_type_fallback()` (§line 403) and `get_type_from_ftype()`. `tensor_allows_quantization()`
  (§line 295) is still there and still ≈where the note said.
* **FALSE, minor — the `--help` labels.** The note says `quantize --help` calls Q4_K_M
  "*recommended*", Q5_K_S/Q5_K_M "the recommended levels". At this commit `QUANT_OPTIONS` carries no
  "recommended" text at all — descriptions are bare size/ppl strings. And the ppl ladder in the note
  (Q4_0 +0.2499 … Q8_0 +0.0004, "at 7 B") is a *different table* from the one this binary prints
  (`@ Llama-3-8B`: Q4_0 +0.4685, Q4_K_S +0.2689, Q4_K_M +0.1754, Q5_K_M +0.0569, Q6_K +0.0217, Q8_0
  +0.0026). Both may be historically true of their own sources; the note attributes them to
  `quantize --help` at this tag, and that attribution is wrong.
* **CONFIRMED — what it refuses to quantize.** `tensor_allows_quantization()` still begins
  `if (ggml_n_dims(tensor) < 2) return false;` ("quantize only 2D and 3D tensors (experts)"), still
  requires the name to end in `weight` (`name.rfind("weight") == name.size() - 6`), still exempts
  `*_norm.weight`, `ffn_gate_inp.weight` ("do not quantize expert gating tensors" — this family's
  router), `ffn_gate_tid2eid.weight`, `altup`/`laurel`, `per_layer_model_proj`, positional
  embeddings/token types, `ssm_conv1d`, `shortconv.conv.weight`, MiniMax's `indexer.*_proj.weight`,
  and the `LLAMA_FTYPE`-independent `output.weight` unless `quantize_output_tensor`.
  Two consequences specific to us, both good news and both worth stating precisely:
  - **Our GDN conv1d really is protected — but by the name rule, not the dimension rule.** The loader
    creates it as `tn(LLM_TENSOR_SSM_CONV1D, "weight", il)` with shape
    `{ hparams.ssm_d_conv, conv_dim }` (`engines/llama.cpp/src/models/qwen35moe.cpp` §linear-attention branch), and
    `engines/llama.cpp/src/llama-arch.cpp` §tensor-name table maps `LLM_TENSOR_SSM_CONV1D → "blk.%d.ssm_conv1d"`. So the
    GGUF name is `blk.N.ssm_conv1d.weight` — 2-D, hence *not* caught by `n_dims < 2`, but caught by
    the `ssm_conv1d` substring test. The note's parenthetical "(the analogue of this family's
    `linear_attn.conv1d.weight`)" is correct.
  - `blk.N.ssm_a` (1-D) and `ssm_dt.bias` are excluded twice over (below 2-D **and** not ending in
    `weight`). But `ssm_beta.weight` / `ssm_alpha.weight` (`{n_embd, n_v_heads}`) and
    `ssm_out.weight` are 2-D, end in `weight`, and match no exemption — llama.cpp *does* quantize
    them to the base width, which is one more reason policy A must be an explicit artifact and not
    "whatever `Q4_K_M` does".
* **CONFIRMED — `output` → Q6_K.** §`llama_tensor_get_type_impl`, OUTPUT branch: `Q8_0` for
  `MXFP4_MOE`, Falcon, or `nx % qk_k != 0`; `Q5_K` for the IQ1/IQ2/IQ3_XXS family; otherwise
  `else if (new_type != GGML_TYPE_Q8_0) { new_type = GGML_TYPE_Q6_K; }`. Tied
  `token_embd` follows OUTPUT.
* **CONFIRMED, and correctly caveated in the note — the 8-expert bumps.** `attn_v`-family
  (`category_is_attn_v`) and `ATTENTION_K` get `GGML_TYPE_Q8_0` only under
  `if (qs.model.hparams.n_expert == 8)` with the comment "for the 8-expert model, bumping this to
  Q8_0 trades just ~128MB". With 256 experts they do not fire, exactly as the note's closing
  paragraph says — llama.cpp's own `Q4_K_M` for this model is policy E, the +40 % option.
* **CONFIRMED — `MXFP4_MOE` is policy A with MXFP4 in place of Q4_K.** `else if (ftype ==
  LLAMA_FTYPE_MOSTLY_MXFP4_MOE) { … if (tensor->ne[2] > 1 && (arch != LLM_ARCH_BAILINGMOE3 ||
  is_bailingmoe3_expert)) new_type = GGML_TYPE_MXFP4; else … Q8_0 }` — 3-D routed-expert tensors to
  MXFP4, everything else to Q8_0, with `output` forced Q8_0 earlier in the function. Note the
  3-D test is `ne[2] > 1`, not a category test, with an explicit MLA carve-out — relevant to us
  because our expert tensors are 3-D and our attention tensors are not.
* **CONFIRMED — "everything else → Q4_K"** for `Q4_K_S`/`Q4_K_M`, via
  `get_type_from_ftype()`: `case LLAMA_FTYPE_MOSTLY_Q4_K_S: case LLAMA_FTYPE_MOSTLY_Q4_K_M: return
  GGML_TYPE_Q4_K;`, with the k-quant mixtures layered on top in `llama_tensor_get_type_impl`
  (`use_more_bits()` for `ffn_down` / `attn_v` under `Q4_K_M`, first four `attn_v` under `Q4_K_S`, etc.).

## §Verdict 3 — K-quant block layouts in `ggml-common.h`: **PARTLY — the description fits Q4_K and Q5_K only**

The brief's one-sentence description — "256-weight super-block, 8 sub-blocks of 32, 6-bit scale AND
6-bit min under fp16 super-block scales for the K-quants" — is true of **`block_q4_K` and
`block_q5_K`** and false of **`block_q6_K` and `block_q8_0`**. W1 is writing a reader from this same
header right now, so the names and the differences matter more than the summary.

Shared macros (§"QK = number of values after dequantization"): **`#define QK_K 256`**,
**`#define K_SCALE_SIZE 12`**, and **`#define QK8_0 32`** (a separate constant — Q8_0 does not use
`QK_K`).

| type | struct | fields, verbatim | size / 256 elts | bpw |
|---|---|---|---|---|
| Q4_K | `block_q4_K` | `union { struct { ggml_half d; /* super-block scale for quantized scales */ ggml_half dmin; /* … mins */ } ; ggml_half2 dm }` ; `uint8_t scales[K_SCALE_SIZE]` /* scales and mins, quantized with 6 bits */ ; `uint8_t qs[QK_K/2]` | 144 | 4.5 |
| Q5_K | `block_q5_K` | same `d`/`dmin`/`dm` union ; `uint8_t scales[K_SCALE_SIZE]` /* 6 bits */ ; `uint8_t qh[QK_K/8]` /* quants, high bit */ ; `uint8_t qs[QK_K/2]` /* quants, low 4 bits */ | 176 | **5.5** |
| Q6_K | `block_q6_K` | `uint8_t ql[QK_K/2]` /* lower 4 bits */ ; `uint8_t qh[QK_K/4]` /* upper 2 bits */ ; `int8_t scales[QK_K/16]` /* scales, quantized with **8** bits */ ; `ggml_half d` /* super-block scale */ | 210 | 6.5625 |
| Q8_0 | `block_q8_0` | `ggml_half d` /* delta */ ; `int8_t qs[QK8_0]` | 34 per **32** | 8.5 |

Corrections to the description, stated exactly:

* **Q6_K has no `dmin`, no 6-bit packing, and 16 sub-blocks of 16, not 8 of 32.** Its own header
  comment is "// 6-bit quantization / // weight is represented as x = a * q / // 16 blocks of 16
  elements each / // Effectively 6.5625 bits per weight", and its scales are `int8_t scales[QK_K/16]`
  — one 8-bit scale per 16 weights, `x = a*q` with **no min**. A reader written for the Q4_K/Q5_K
  6-bit scale+min unpack will not decode Q6_K.
* **Q8_0 is not a super-block type at all:** 32 weights, one fp16 `d`, no min, no sub-blocks, no
  scale quantization. The `QK8_0 32` constant is what a reader must use, not `QK_K`.
* Q4_K/Q5_K do match the description: 256-weight super-block, "8 blocks of 32 elements each",
  "weight is represented as x = a * q + b", 6-bit scales **and** mins packed into `K_SCALE_SIZE` 12
  bytes, with the two fp16 super-block scales `d` (for the scales) and `dmin` (for the mins) — also
  reachable as the `ggml_half2 dm` member of the union. The fields sit inside
  `GGML_EXTENSION union … GGML_COMMON_AGGR_U` / `GGML_COMMON_AGGR_S` wrappers, which expand to
  nothing in C++ but are required for portable anonymous unions — a hand-written reader should name
  `.d`, `.dmin` (or `.dm`) and not assume anonymous-union layout.
* **Sizes are pinned by `static_assert` in the header**, which is the check to copy rather than
  trusting field comments: `sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2`,
  `sizeof(block_q5_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 + QK_K/8`,
  `sizeof(block_q6_K) == sizeof(ggml_half) + QK_K/16 + 3*QK_K/4`,
  `sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0`.
* **`Q5_K` is 5.5 bpw, confirming W1 — and the header is the witness, not any table.** 176 B / 256
  weights × 8 = 5.5 exactly, and the header says "Effectively 5.5 bits per weight". The number
  `5.5625` appears nowhere in `engines/llama.cpp/ggml/src/ggml-common.h` — its only three bpw comments
  are the 4.5 (Q4_K), 5.5 (Q5_K) and 6.5625 (Q6_K) lines quoted in the table above — so `"Q5_K":
  5.5625` was the header's **Q6_K** figure shifted onto the wrong type. That is what a hand-typed
  bits-per-weight table does. `tools/gguf_header_types.py` carried exactly such a dict (and
  `tools/quant_scenarios.py` shared the number); as of this writing the dict is **gone**, and bpw is
  derived from the pinned struct sizes in `tools/gguf_reader.py` with each size cross-checked against
  that struct's own `static_assert` — the right fix, and the same arithmetic W1's
  `sum(row_bytes) + data_start == file size` witness (`tools/PROVENANCE.md` §`gguf_reader.py`).
  Recorded here because the wrong number had already propagated into two tools, and because the
  `static_assert` lines quoted above are the thing to transcribe, not a bpw table.

## §Verdict 4 — how the Qwen35MoE loader reads the fused experts: **CONFIRMED, one tensor preferred and both accepted**

`engines/llama.cpp/src/models/qwen35moe.cpp` §`load_block_trunk` (identical block in §`load_block_mtp`) declares the
routed experts as:

```c
layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);
```

and `llama_model_base::create_tensor_gate_up_exps` (§`engines/llama.cpp/src/llama-model.cpp`) is
**fused-first, split-fallback**:

```c
layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", bid), {n_embd_, n_ff_ * 2, n_expert_}, TENSOR_NOT_REQUIRED);
if (layer.ffn_gate_up_exps == nullptr) {
    layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", bid), {n_embd_, n_ff_, n_expert_}, flags);
    layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", bid), {n_embd_, n_ff_, n_expert_}, flags);
}
```

GGUF names expected, from `engines/llama.cpp/src/llama-arch.cpp` §tensor-name table (each gains `.weight` from the
`tn(..., "weight", …)` call): **`blk.%d.ffn_gate_up_exps.weight`** or the pair
**`blk.%d.ffn_gate_exps.weight` / `blk.%d.ffn_up_exps.weight`**, plus
**`blk.%d.ffn_down_exps.weight`** and router **`blk.%d.ffn_gate_inp.weight`**.

Shapes are in **ggml `ne` order — the reverse of the HF order in AGENTS.md prediction 3**: the fused
gate/up is `{n_embd, n_ff*2, n_expert}` = `[H, 2I, E]` (HF's `[E, 2I, H]` transposed), and
`ffn_down_exps` is `{n_ff_exp, n_embd, n_expert}` = `[I, H, E]`. The gate|up pair therefore lies
along **`ne[1]`, the output dim**, exactly as prediction 3 says; a SwiGLU kernel written for two
separate `[I, H]` matrices needs the stride/offset check, not a re-derivation.

That the split is along the output dim, with **gate first**, is confirmed twice more:

* graph side, `engines/llama.cpp/src/llama-graph.cpp` §`build_moe_ffn`: `if (gate_up_exps) { // merged gate_up path:
  one mul_mat_id, then split into gate and up views … }` with the result annotated
  `[n_ff*2, n_expert_used, n_tokens]` — one `mul_mat_id` over the whole `2I`, then two views.
  `engines/llama.cpp/src/models/qwen35moe.cpp` §`build_ffn_layers` passes both forms in: `build_moe_ffn(…, 
  model.layers[il].ffn_up_exps, model.layers[il].ffn_gate_exps, model.layers[il].ffn_down_exps, …,
  nullptr, model.layers[il].ffn_gate_up_exps, …)`.
* writer side, `engines/llama.cpp/conversion/base.py` §`modify_tensors`: with fusion on, `fused_data = torch.cat([gate_data,
  up_data], dim=1)` over tensors of shape `(n_expert, n_ff, n_embd)`, written as
  `MODEL_TENSOR.FFN_GATE_UP_EXP` = `blk.{bid}.ffn_gate_up_exps`
  (`engines/llama.cpp/gguf-py/gguf/constants.py` §MODEL_TENSOR).

**For the reader spec, the operative fact is that fusion is opt-in and off by default:**
`engines/llama.cpp/convert_hf_to_gguf.py` §`--fuse-gate-up-exps` (`action="store_true"`, "Fuse gate_exps and up_exps
tensors into a single gate_up_exps tensor for MoE models") feeds
`engines/llama.cpp/conversion/base.py` §`fuse_gate_up_exps: bool = False`. So an HF→GGUF conversion this repo runs
produces **split** `ffn_gate_exps`/`ffn_up_exps` unless the flag is passed, while the *published*
artifact's choice is an empirical question answered by the per-tensor type census in
`quantization/evidence/q4km-published-width-map-2026-10-26.md` — a reader that only handles the
fused tensor will fail on our own conversion, and one that only handles the split pair will fail on
anyone else's. Accept both; the loader does.

---

## §What this contradicts in committed docs

1. `quantization/evidence/llama-cpp-tensor-policy-2026-10-26.md` §"What it forces upward", first
   bullet — the `FFN_LATENT`/`ATTENTION_GATE` → Q8_0 rule is `arch == LLM_ARCH_MUSE_SPARK`-only at
   this commit (§Verdict 2).
2. `DECISIONS.md` §D-01a — quotes that same rule as support for the user's Q8_0 choice ("llama.cpp's
   own policy forces the latent/gate categories to ≥ Q8_0"). The citation no longer supports the
   decision at this revision; the decision's other legs are untouched. **User's call whether to amend
   the wording; I have not edited `DECISIONS.md`.**
3. Same note §"The named types it ships" — `--help` has no "recommended" labels at this commit, and
   its ppl ladder is the Llama-3-8B table with different numbers than the ones quoted (§Verdict 2).
4. Same note §provenance — `get_quant_type_use()` no longer exists; the functions to cite are
   `llama_tensor_get_type_impl()` / `llama_tensor_get_type()` / `tensor_type_fallback()` /
   `get_type_from_ftype()`.
5. ~~`tools/gguf_header_types.py`'s width table — `"Q5_K": 5.5625` should be 5.5.~~ **Already fixed**
   while this audit was in flight: the hand-typed bpW dict is deleted from `tools/gguf_header_types.py`
   and bpw now comes from the pinned struct sizes via `tools/gguf_reader.py` (§Verdict 3). Kept as a
   line item because the same wrong number lived in two tools at once, which is the argument for
   deriving widths from the header rather than typing them.
6. `quantization/evidence/llama-cpp-tensor-policy-2026-10-26.md` §"What it refuses to quantize at
   all" reads as if the conv1d protection is generic; it is the `ssm_conv1d` **name** rule that
   protects our GDN conv (the tensor is 2-D), and `ssm_beta`/`ssm_alpha`/`ssm_out` are *not*
   protected (§Verdict 2).

## §Reproducing this

```bash
tools/build_llama.sh              # guard → configure → build → 17-check gate; log to logs/llama-build-<date>.log
tools/build_llama.sh --check      # guards + gate only, no compile
```

Green state as of this writing: `checks ran: 17 / expected 17 (failures: 0)`, `GATE GREEN`, and the
record lines `compile_commands.json 473 entries; 146 device compiles over 146 distinct files;
arch(es)=gfx1151` and `.hip_fatbin sections=1 size=56112304 extracted=56112304; device code
objects=141 (other arches: 0, listed: none); HIP sources carrying one=141 of 146 objects`.
