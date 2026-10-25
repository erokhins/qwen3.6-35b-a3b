# HANDOFF — prototype/moe_gather (routed-expert gather), paused as a draft

**Status: DRAFT, never executed on the device. Paused 2026-10-25 ~14:35 by the operator taking the box
down for maintenance.** Nothing in this lane has touched the iGPU. `evidence/` is empty and
`build/` is empty. No commit was made and nothing was staged.

This file is the source of truth for what exists. Read it before trusting any line below it, and
read it before running anything: the lane's own suite has **not** been watched fail as a whole, and a
green-looking `RESULT PASS` from `--selftest` means less than it looks (see §6).

---

## 1. Files present, and the honest status of each

| file | lines | compiled? | executed? |
|---|---|---|---|
| `moe_gather_kernels.h` | ~1300 | yes, via g++ (as included by `host_probe.cpp`) | only through the selftest path, never through a plan |
| `moe_gather_selftest.h` | ~430 | yes, g++ | yes, all three arms, CPU (`--selftest --arm 0/1/2`) |
| `host_probe.cpp` | 250 | yes, g++ `-O2 -std=c++11 -Wall -Wextra -ffp-contract=off`, zero warnings | **`--selftest` only.** Its plan/slab/expect/report path has never run — there is no plan file on disk, because the thing that writes plans (`gate.py`) does not exist yet |
| `moe_gather.hip` | ~330 | **NO — has never compiled.** Last attempt failed on missing `#include <hip/hip_runtime.h>`; that include was added, and then two more edits landed (Q6_K shift fix, comment fix). Per the pause order, hipcc was not re-run | no |
| `build.sh` | 55 | n/a | run once; it succeeded for g++ and **failed at the hipcc step** (the missing include above) |
| `work/probe1.py` … `probe8.py` | — | n/a | yes, all run; they hold the decisive evidence in §5 |
| `evidence/`, `build/` | — | — | empty |

**Files the task asked for that do not exist yet:** `gate.py`, `gate` (the bash wrapper and sole
`tools/gpu_lock.sh` acquirer), `oracle_arm.py`, `results.md`, and the `.gitignore` lines. There is
therefore **no way to run this lane end-to-end today** — `host_probe` needs `--plan/--slab/--expect`
files and the driver needs the same, and nothing generates them.

## 2. Exact build commands

```sh
cd prototype/moe_gather
./build.sh                 # hipcc --offload-arch=gfx1151 -O3 + g++ -O2, both with -ffp-contract=off
./build.sh --check         # compiles, then runs the g++ selftest
./build.sh --clean
```

**I did not run `build.sh` after my last three edits.** The g++ half is known-good (recompiled by
hand at 14:33, clean, exit 0 — see §6). The hipcc half is unverified since the include fix; expect it
to need one or two more iterations before it links, and expect `-Wl,-rpath,/opt/rocm/lib` to be
load-bearing on the link line (`pitfalls/50-harness-and-shell.md`).

## 3. Verified vs written-but-unverified

**Verified by execution (CPU only, this session):**

- The g++ build compiles warning-free with `-Wall -Wextra`.
- The 62-check selftest passes in the honest arm, and both sabotage arms are **witnessed to fire**:
  - `--arm 0` honest: `checks=62 equals=57 differs=5 failures=0 changed=0 verdict=ok`
  - `--arm 1` stub: `changed=5 changed_differs=5` — five DIFFERS checks stop differing
  - `--arm 2` crossed: `changed=53 changed_equals=50 changed_differs=3`
- Six hand-worked constants in that selftest were **wrong while the kernel was right**, and one was
  wrong while the *kernel* was wrong. The one real kernel bug: the Q6_K per-sub-block high-bit shift
  was `qh[32g+l] >> 2*(k>>1)`, correct is `>> 2k` (`tools/gguf_reader.py`'s `a/b/c/d` build uses
  `>>0, >>2, >>4, >>6`). Caught by `q6k/elt32_*`, which cannot be right under the wrong shift. Fixed
  in `moe_gather_kernels.h:342,375` — **after** the last g++ compile that shipped a green line, so
  recompile before believing §6.
- The five constants I had wrong, and what they actually are (each is arithmetic, not a band):
  `get_scale_min_k4` mg1=21 mg2=38 (I wrote 22/39); a zero-quant Q6_K element is **not** 0.0 but
  `(0-32)·sc[e>>4]`, i.e. −32.0 where `sc=1`; that product's zero is **−0.0** (`0x80000000`), and the
  comparator distinguishes it from `+0.0`; `bf16(3.0×1.0078125) = 3.03125` not 3.0234375 (the bf16
  step at 3.0 is 2⁻⁶, and an exact tie goes to the even *bit pattern*, 0x4042); and a bf16 sum is
  order-sensitive only for the right multiset — my first choice collapsed to 2.0 either way.

**Written but never executed (do not call any of this verified):**

- Every device path: allocation, the two launches per block, the `atomicAdd` launch counter, the
  D2H read-backs, `--per-slot` mode. The `.hip` has not compiled once.
- The whole plan/slab/expect/report pipeline in **both** arms: `mg_read_plan`, `mg_verify_plan`,
  `mg_run_case`, the per-case counters, the host-vs-device agreement claim. Zero plan files exist.
- The oracle arm (`oracle_arm.py` does not exist), so "host oracle judged against committed
  `speccheck/`" is **not** satisfied — what is satisfied is the probe-level evidence in §5.
- The three-arm gate over real rows, the negative-control injections over real rows, the class
  populations (`MATCH/MISMATCH/ALT/IMMUNE/REPORT`), the `CUT` line, the sample declaration, and
  `results.md`. All unwritten or unrun.
- The two cast arms *on real rows*. The selftest discriminates them on hand-worked constants
  (3.03125 vs 3.015625), and §5 gives the real-row magnitude, but no gate has compared an arm built
  from real pinned rows against an independently computed expectation.
- Prediction 2's answer. The grouped shape is implemented (one gather launch per block carrying every
  selected slot of every sampled row, one combine launch per block) and `--per-slot` exists to count
  the alternative, but **no launch count has ever been printed**. results.md must not claim P2 until
  `LAUNCHES mode=grouped … device_counted=N` and `mode=per-slot … device_counted=M` both exist.
- `HOST_SYNC d2h_total=3 d2h_during_launches=0` — the counter exists and is honest by construction
  (the only D2H copies are the two read-backs plus the launch-count read), but it has never printed.

## 4. Exact commands to re-verify

CPU-only, safe for any agent:

```sh
cd prototype/moe_gather
g++ -O2 -std=c++11 -Wall -Wextra -ffp-contract=off -I. host_probe.cpp -o work/host_probe
./work/host_probe --selftest --arm 0     # expect failures=0 changed=0
./work/host_probe --selftest --arm 1     # expect changed>0  (stub fired)
./work/host_probe --selftest --arm 2     # expect changed>0  (crossed fired)
/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python work/probe3.py   # speccheck ref vs pinned dumps
/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python work/probe6.py   # combine-form discrimination
```

**Device / main-agent-only** (needs the iGPU, a `gpu_lock` hold, and halogen down — none of which a
worker may arrange; and after this pause, a maintenance window):

```sh
./build.sh                       # hipcc --offload-arch=gfx1151
./gate                           # once it exists: sole acquirer of tools/gpu_lock.sh
```

`gate.py` must contain **zero** references to `gpu_lock`; the wrapper `gate` is the only acquirer,
must use `acquire-noblock`, must refuse rather than queue, and must `trap release EXIT INT TERM`.
Copy the shape from `prototype/q4k_dequant/gate` and `prototype/moe_router/gate`, both already landed.

## 5. Witnessed facts this lane established (all CPU, all reproducible from `work/`)

1. **`speccheck/moe_ref.py`'s `ref` arm reproduces the pinned MoE dumps bit-exactly.** For
   `layer0` (32 tokens × 8 slots): `expert_raw ndiff 0 / 524 288`, `routed_sum ndiff 0 / 65 536`
   (`work/probe3.py`). This is the anchor the whole lane was supposed to be judged against.
2. **The combine form is load-bearing, measured on all four locally available blocks**
   (`layer0`, `layer1`, `layer39`, `mtp0`, `work/probe6.py`). Only this form reproduces the pinned
   `routed_sum` bit-exactly: product rounded to bf16 → **bf16 accumulator** → **addends walked in
   ascending expert-id order** → weight indexed by **rank**:
   - fp32-accumulate-once: ≈38 500 / 65 536 elements differ per block
   - cast-after (multiply in f32, cast the product afterwards): ≈40 700 / 65 536 differ per block
   That ≈40 700 is the real-row magnitude behind README prediction 4; the selftest proves the
   comparator can see the same distinction on hand-worked values.
3. **Widths confirmed against the published Q4_K_M GGUF**
   (`models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`): `ffn_gate_exps` Q4_K ×40,
   `ffn_up_exps` Q4_K ×40, `ffn_down_exps` **Q5_K ×37 + Q6_K ×3**, the Q6_K ones at layers **34, 38,
   39**. GGUF dims: gate/up `(2048, 512, 256)`, down `(512, 2048, 256)` — i.e. per-expert contiguous,
   H fastest, `I = 512`, `H = 2048`, `E = 256`, `K = 8`, 40 layers.
4. **Local safetensors coverage:** shards 1, 2, 25, 26 are on disk ⇒ full MoE blocks available for
   `layer0`, `layer1`, `layer39`, `mtp0`. **`layer2` is missing** (shard 3 absent) — any sample plan
   must not silently assume it. All 20 dumped weight tensors match the local shards (`work/probe2.py`).
5. **Q6_K is not exact in f32**: f16 `d` (~11 significand bits) × signed scale (≤8) × q offset
   (≤6) ≈ 25 bits. The lane therefore judges device dequant against the audited reader through a
   **double** route rounded once to f32, and keeps the plain-f32 route only as a **counted census per
   width** — never as a tolerance. Q4_K/Q5_K are expected to diverge 0; Q6_K divergence is reported,
   not banded. No typed tolerance exists anywhere in this lane.
6. The llama.cpp ruler (`44.9 / 49.1 / 51.4 t/s`) is **context only**. No timing of any kind was
   taken by this lane; no timer is constructed in `moe_gather.hip`.

## 6. Checks actually run this session, with output

```
$ g++ -O2 -std=c++11 -Wall -Wextra -ffp-contract=off -I. host_probe.cpp -o work/host_probe_selftest
compile_exit=0                                     # zero warnings

$ ./work/host_probe_selftest --selftest
SELFTEST arm=0 name=honest checks=62 equals=57 differs=5 failures=0 changed=0 changed_equals=0 changed_differs=0 verdict=ok
HOSTPROBE SELFTEST arms=1 failures=0
RESULT PASS host_probe selftest arms=1
selftest_exit=0
```

⚠ **That `RESULT PASS` is weaker than it reads.** `arg_value` defaults `--arm` to `"0"`, so a bare
`--selftest` runs *only the honest arm* and prints `arms=1`. The sabotage arms had to be requested
individually (`--arm 1`, `--arm 2`) and are what actually prove the checks bite. Fix the default to
run all three, or make `RESULT PASS` conditional on having executed all three (§7).

## 7. Defects in my own harness that the next session must fix first

1. **`arg_value` silently ignores `--arm=1`.** It only matches `argv[i] == "--arm"` and reads
   `argv[i+1]`, so `--arm=1` / `--arm=-1` fall through to the default `0` — I ran four "arm"
   invocations that were all arm 0 before noticing. Same trap class as
   `pitfalls/50-harness-and-shell.md` (argv parsing that fails open). Accept both forms, or refuse an
   unrecognised `--*`.
2. **`--selftest` alone runs one arm and prints PASS.** The driver-level `RESULT PASS` must require
   all three arms executed, with `changed==0` on honest and `changed>0` on both saboteurs. Today a
   reader can print PASS while never having watched the suite fail.
3. **The crossed arm's per-check `FAIL` lines print `got == want`**, because they print the honest
   expectation while the verdict was computed against the crossed one. Confusing enough to be
   dangerous in a log a human re-reads; print the expectation actually used.
4. `run_group`'s unused `per_slot`/`slot` params were removed, but the `--per-slot` path still shares
   `d_caseidx` with the grouped path; harmless today (no concurrency), worth a comment.
5. `moe_gather.hip` has never been through hipcc, so treat every line of it as unproven — including
   whether `MgPlan` can be passed to the kernel **by value** (it holds plain pointers, so it should,
   but `sizeof(MgPlan)` in a kernel parameter space is exactly the kind of thing that fails at 4 KB of
   kernel args, and the fallback is one flat device buffer plus offsets).

## 8. What remains, in order — and what I would do first

1. **`oracle_arm.py` + the P3 candidate probe.** First, because everything else is judged against it
   and because it is pure CPU (no device, no lock, no maintenance window needed). It must, per
   README prediction 3, evaluate the candidate readings against the pinned `expert_raw`/`routed_sum`
   and **prove they are different functions**: `output_chunk_gate_first` (gate = rows `0:I`,
   up = rows `I:2I`), `output_chunk_up_first`, `input_chunk`, `row_interleaved`, `stack_transposed`.
   Write `UNRESOLVED` if the data cannot separate them — do not guess.
2. **`gate.py`** — build the sample/plan/slab/expect files, run the host probe, run the driver only
   via the wrapper, declare the sample *before* any device work (cases, arms, rows real vs synthetic,
   field comparisons, elements compared, launches, `CUT` if cut), judge device-vs-host and
   device-vs-oracle, report every arm's counts and every injection's fired/did-not-fire, and assert
   parsed-count == printed-count for every parser (`pitfalls/40-gates-and-admissions.md`).
3. **`gate`** wrapper (copy `prototype/q4k_dequant/gate`), then `./build.sh` + one correctness-only
   device gate, <60 s wall, PASS/FAIL, no `ms` anywhere, evidence carrying the full lock record and
   the literal line `no timing taken`.
4. **`results.md`** with the RESULT lines, class populations, the two cast arms, every injection, the
   real-vs-synthetic row split, elements compared, launch counts and the P2/P3 verdicts.
5. **`.gitignore`**, by exact path, never a directory ignore:
   `prototype/moe_gather/moe_gather`, `prototype/moe_gather/host_probe`,
   `prototype/moe_gather/build/moe_gather.o`, `prototype/moe_gather/work/host_probe_selftest`.
   Note `work/` is currently **not** ignored, so the draft commit will include `work/probe*.py` — that
   is desirable (they hold the §5 evidence); only the compiled artefacts must be excluded.

## 9. Prediction 3 — UNRESOLVED

**Written as the task requires: `UNRESOLVED`.** But the reason has changed and the next session must
not re-derive it:

- `prototype/moe_router/results.md` recorded P3 as unresolvable because no candidate reading of the
  safetensors reproduced the pinned `expert_raw`. **That probe had an 8-byte bug**:
  `prototype/moe_router/work/probe15.py` memmapped the file at `header_json_len + data_offsets[0]`,
  and the data region actually begins at `8 + header_json_len + data_offsets[0]` — every value was
  shifted by four bf16 elements, so no reading *could* match.
- With the correct offset, the pinned `layer0.expert_raw` reproduces **bit-exactly, 0 of 524 288**
  (`work/probe3.py`, via `speccheck/moe_ref.py`). So the dump *does* carry the layout information, and
  the earlier UNRESOLVED was an artefact.
- What is still missing: I never wrote the probe that evaluates the *other* candidate readings under
  the corrected offset and shows they differ from the gate-first one. Without that witness I cannot
  say the gate-first reading is *distinguished* from its neighbours, only that it *reproduces*. That
  is exactly the gap prediction 3 asks to close, so it stays **UNRESOLVED** until item 1 of §8 runs.
- Caveat on generality: the bit-exact reproduction was witnessed for **`layer0` only** (all 32 tokens,
  all 8 slots). `layer1`, `layer39`, `mtp0` are locally available and unprobed for `expert_raw`;
  `layer2` is not available at all (§5.4).

## 10. Gates I would refuse to call passing today

| gate | why it cannot pass yet | what makes it pass |
|---|---|---|
| host oracle vs `speccheck/` | `oracle_arm.py` does not exist; the reproduction exists only inside `work/probe3.py`, not inside the lane's judgement path | oracle arm emits records with counts; gate compares them to the device output and asserts parsed == printed |
| three-arm selftest (honest/stub/crossed) over the *lane*, not just the header | arms run but the driver's `RESULT PASS` does not require them (§7.2) | PASS conditional on all three arms + `changed>0` on both saboteurs |
| cast-before vs cast-after on real rows | only hand-worked constants + a probe magnitude (≈40 700/65 536) | a real-row case whose expectation comes from the oracle in the *other* arm, with the two arms counted separately |
| every injection has a witnessed negative control | the 14 modes and `mg_reaches` exist; nothing has run them over real rows | per-injection fired/did-not-fire table with non-zero denominators; a counter reading exactly 0 is vacuous until witnessed non-zero |
| P2 grouped launches | no launch count has ever been printed | `LAUNCHES mode=grouped … device_counted=N` and `mode=per-slot … device_counted=M`, `N != M`, and `device_counted == issued` |
| single device gate, correctness-only, <60 s, wrapper-only lock | `gate`, `gate.py` do not exist; `evidence/` is empty | the wrapper run, plus an `evidence/` file with the lock record and `no timing taken` |
| `.gitignore` exact-path binary ignores | not written; a `git add -A` today would stage nothing bad *only because* I deleted `work/host_probe_selftest` and `build/` is empty | §8.5 |

## 11. Provenance notes

- Every `work/probeN.py` was run this session; the numbers in §5 are copied from their stdout.
- No file outside `prototype/moe_gather/` was modified by this lane. In particular
  `quantization/evidence/audit-ordering-2026-10-26.log` shows 11 uncommitted added lines reading
  `section=ssm_a ulp bound` — **not this lane's**; do not attribute them to `moe_gather`.
- The self-containment rule was respected: widths, dims, pool figures and the combine form are quoted
  here with their numbers and their probe file, not pointed at another repo.
