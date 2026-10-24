# Agent notes — qwen3.6-35b-a3b

Rules for work in this repo. The facts are in `README.md`; the map to reusable code is in
`reference-implementation.md`; the user decisions still open are in `DECISIONS.md`.

## SCOPE: one model, and this time capacity IS a constraint

The only target is **`Qwen/Qwen3.6-35B-A3B`, text-only** (prefill + decode + MTP draft) on the
Framework Desktop (Strix Halo, `gfx1151`). Purpose: land the **MoE block** as a gated
subsystem, because it is the only part of `Qwen/Qwen3.8-Flash-Next`'s design that is
reachable on a model this box can actually run. Lineage and evidence: `README.md`.

Read this before importing the 0.8B project's scope statement — **one of its load-bearing
paragraphs does not transfer.** `../qwen3.5-0.8b/AGENTS.md` says memory capacity is not a
constraint (the 0.8B is 4 % of the device-visible pool, so no paging/eviction/quantize-to-fit).
Here the model is **64.6 GiB bf16 against a measured ~32–34 GiB pool**, so:

- capacity engineering is in scope for the first time: residency layout, shard staging,
  expert placement, state-pool sizing under a tighter budget;
- **quantization to fit is in scope only after the user says so in those words** (`DECISIONS.md`
  D-01). Until then: develop and gate in bf16 against pinned reference dumps, which works —
  the CPU oracle path (torch 5.19 `qwen3_5_moe`) runs in 123 GiB of host RAM and does not need
  the weights on the GPU at all;
- still out of scope: the vision tower, multi-user batched serving, vLLM API parity, YaRN
  context extension, and the three Flash-Next-only subsystems (Gated Residual /
  hyperconnections, QSA indexer, n-gram PLE) — see D-04.

## Before you write anything, decide nothing alone

`DECISIONS.md` holds the questions that are the user's, not the agent's (quantization bands,
the residency probe, whether MTP ships in the first milestone). Answering one of those by
yourself is the expensive kind of helpful.

## GPU execution is serialized and belongs to the main agent (inherited verbatim in force)

Every agent on this box — this repo's and `../qwen3.5-0.8b`'s alike — is served by one
endpoint running on this same iGPU. Any device run concurrent with another agent's work, or
with our own serving, is contaminated, and a contaminated number is worse than no number.

- **Subagents may run short correctness-only device gates**: under ~60 s wall, PASS/FAIL only,
  never a timing, no `ms` figure in a report, one at a time, and the report must carry the
  command line, wall time, `tools/gpu_lock.sh util` and an explicit "no timing taken".
- **The main agent owns every measurement**, with zero subagents active (stopped, not quiet),
  in the foreground. Backgrounded runs (`&`, `nohup`, `bg_wait`) keep the token stream alive
  and make the measurement contend with itself.
- **`tools/gpu_lock.sh` is copied here and is machine-wide by design** — its lockdir is
  `/tmp/pi_gpu.lock`, so a worker in this repo and a worker in the 0.8B repo exclude each
  other correctly. Do not "fix" it to a repo-local path: two locks is one lock too many.
  Before acquiring anything, `grep -c gpu_lock <script>`: a driver that self-acquires the lock
  and is called by a caller already holding it **deadlocks silently** (no CPU, no GPU, no message,
  no exit — and `gpu_lock.sh status` afterwards reads `STALE (holder pid dead)`). At least a dozen
  such drivers exist in the sibling repo; check rather than remember.
- **One subagent at a time**, 2 h deadline passed explicitly on every launch
  (`options: {timeoutMs: 7200000, checkpointBeforeDeadlineMs: 600000}`), and a worker is
  stopped before the main agent measures. Full rationale:
  `../qwen3.5-0.8b/AGENTS.md` §"Subagent run deadlines".

## Standing rules (inherited)

- **Never loosen a gate to make it pass.** Bands, floors and must-catch sets are derived from a
  floor probe and from geometry, never from what the current build happens to produce.
- **`speccheck/` is the independent implementation and the oracle**; `engine/` is the
  deliverable; `prototype/<op>/` graduates into `engine/` only after its own gate passes.
- **Budget verification to what a change can affect.** Re-running a green suite is the silent
  failure mode; if the result is on disk and nothing feeding it changed, cite it.
- **Commit by topic**, header ≤ 72 chars in the form `area: what the commit does`, then a body
  saying what changed and why, including the number or failure that forced it and a pointer to
  the file holding the evidence.
- **Citation hygiene.** Prefer `path §label` to `path:NNN`; a line-number cite is a claim about
  a file's line count. When editing a cited file: `wc -l` before and after, and read each cited
  coordinate against the previous revision — a checker whose baseline is `HEAD` cannot see drift
  introduced by the commit that contains it.
- **A check that can produce no output cannot fail.** Applies to every parser, gate driver,
  negative control and counter written here: require the parsed record count to equal the
  printed record count; derive case lists from geometry; treat a counter that reads exactly 0
  as vacuous until witnessed non-zero somewhere it must fire.
- **Never print chat-template special tokens literally** — not in replies, commit messages,
  gate output, logs or docs. Every agent here is served by a checkpoint whose stop list is its
  own eos id, so a literal end-of-turn token silently truncates the message or kills the tool
  call it landed in. Refer to them by id (this family: 248044 eos/bos, 248046 end-of-turn,
  248068/248069 think-boundary — re-derive per checkpoint, see `README.md`).

## Inherited traps that bind this project (detail lives at the path given)

| trap | where it is written up |
|---|---|
| a bandwidth number above ~240 GB/s means skipped work, not a fast kernel | `../qwen3.5-0.8b/prototype/README.md` |
| the first GPU pass over fresh pages runs ~35 GB/s (7× slow) — warm before measuring | `../qwen3.5-0.8b/devices/strix-halo-unified-memory.md` |
| device-visible pool is ~32–34 GiB per process and `GPU_MAX_HEAP_SIZE` does not move it; `hipMallocManaged` degrades **silently** near it while `hipMalloc` fails loudly → `hipMalloc` + explicit `hipMemcpy` for weights | same file, §"The two real limits" — **this one has gone from [FUTURE] to load-bearing** |
| host-pinned reads run at GTT speed, so residency choice is about control and address space, not throughput | same file, §"What this does not buy" |
| dispatch/launch-bound timings are bimodal on this box (best-of-N spreads 158–269 %); only bandwidth-bound numbers repeat; a paired A/B must take both sides inside one lock hold | `../qwen3.5-0.8b/engine/performance-status.md` §6, §"a ruler re-taken in a different window is not an A/B" |
| `hipEventElapsedTime` returns exactly 0.000 ms for a 2.1 ms kernel — burst + steady_clock is the only sub-ms timer | `../qwen3.5-0.8b/engine/performance-status.md` §6 |
| a poll loop that `pgrep -f`s a script named in its own command line matches itself — wait on a sentinel the job writes | `../qwen3.5-0.8b/AGENTS.md` traps index |
| one argv slot caps at `MAX_ARG_STRLEN` = 131,072 B (`getconf ARG_MAX` is not the limit): long prompts go in by file, and the harness asserts the field count it wrote. At 32k tokens a space-joined id list is 191,588 B and `exec` fails in 0 s with exit 126, which looks like "the engine crashes at 32k" | `../qwen3.5-0.8b/AGENTS.md` traps index |
| `--ids-file` splits on **commas**; a space-joined fixture reads as one token and the run still succeeds, printing a plausible table for a run that computed nothing | same |
| `Tensor.to(torch.uint16)` casts, `.view(torch.uint16)` bitcasts; torch refuses `bf16.numpy()` | `../qwen3.5-0.8b/prototype/README.md` |
| a BF16 ULP comparator on a `uint16` view has a dead `x < 0` branch and passes vacuously; correct form is `np.where(bits & 0x8000, bits ^ 0xFFFF, bits ^ 0x8000)` | `../qwen3.5-0.8b/prototype/bf16_ulp.py` (has a selftest — run it) |
| max-ULP bands are blind to whole-tensor 1-ULP error and ndiff is blind to chains: use ndiff for stage-local near-exact ops, cos/meanrel for chains, and size any cancellation band with a floor probe | `../qwen3.5-0.8b/prototype/mlp_gemv/{floor_probe.py,results.md}` — **the expert GEMV is a long dot, so it needs exactly this** |
| a negative control with an empty must-catch set is not a control; derive the case list from the geometry that can make the bug fire | `../qwen3.5-0.8b/prototype/attn_tile_model.py` (`neg_expect`), `prototype/mlp_gemv/run.py --negative` |
| a gate driver whose parser silently matches nothing reports PASS | `../qwen3.5-0.8b/prototype/gdn_collapse/run.py` |
| an admission over a fixture list typed from memory is not an admission: derive the population and assert `cases == |population|`; treat any `PASS=0` as an unclassified run | `../qwen3.5-0.8b/AGENTS.md` traps index |
| a device fault on a path only reachable at rows > 1 presents as a **hang**, not a crash | `../qwen3.5-0.8b/engine/results.md` (Q5 / P8) — relevant the moment a batched multi-token MoE path exists |
| byte-identity claims name their width; longer-context judgement is decision-level aggregates plus pooled 1−cos | `../qwen3.5-0.8b/prototype/attn_wide/a13_identity_metrics.py` |
| a read dump's dtype off its file size (F32 read as BF16 yields NaN cos plus a plausible-looking ndiff) | `../qwen3.5-0.8b/prototype/attn_wide/A13-RESULT.md` |
| a rate must be compared against the ceiling its own instruction class can occupy (scalar `v_fma_f32` 7.30 → `v_dot2_f32_bf16` 23.28 → WMMA 52.63 TFLOP/s), and occupancy/LDS/SIMD counts come from KFD/`rocminfo`, never from `hipGetDeviceProperties` on this SKU | `../qwen3.5-0.8b/prototype/issue_rate_probe/results.md`, `tools/kernel_occupancy_probe.cpp` |
| a static launch bill is not a body price: price a body by differencing two device arms at the same width, never by summing launch records | `../qwen3.5-0.8b/prototype/mtp_llama/LLAMA-MTP-RESULT.md` §2 |
| a hoisted scalar constant is a different number, and a CPU mirror is blind to it — when a form claims bit-identity with a shipped kernel, transcribe the shipped expressions character for character, association included | `../qwen3.5-0.8b/prototype/gdn_collapse/results.md` §2 |
| HIP builds need `-Wl,-rpath,/opt/rocm/lib`; the `.venv` torch is CPU-only; `__shfl_down_sync`'s mask is 64-bit here | `../qwen3.5-0.8b/prototype/README.md` |
| before writing a probe or census tool, list the existing ones — a census that cannot fail is worse than none | `../qwen3.5-0.8b/prototype/gdn_walk/BET.md` §2, `tools/` |

## Traps that do NOT transfer (0.8B-specific — do not re-litigate, do not import)

The sibling engine's numerics bands, route allowances, `MAXSC` wave geometry, attn-tile wave
rules, GDN collapse variants and MTP accept rates are facts about **its** pinned dumps and
**its** geometry. New model ⇒ new pins ⇒ new floors. Copy the *method* of sizing a band (floor
probe + derived must-catch set), never the band.

## Predictions to falsify early in this project (written as predictions, not facts)

1. **The router is the first per-token serial point the engine has had.** `softmax(fp32)` over
   256 logits, then `topk(8)`, then a normalisation, gates every expert read. If it serialises
   behind the host (`torch.where`/`nonzero` in the HF reference does exactly that), decode
   becomes host-bound at ~40 tok/s and the whole roofline is moot. Measure host-sync count per
   decode step before optimising any expert kernel.
2. **An expert is small (6.0 MiB) but not too small**: 8 gathers × 40 layers = 240 dependent
   6 MiB reads per token, so the gather must be issued as 40 independent grouped launches (one
   per layer, all selected experts) or the machine idles between dependent reads.
3. **`gate_up_proj` is `[E, 2I, H]` with `gate, up = chunk(2)`** — the fused gate|up pair is
   laid out along the output dim, so a SwiGLU kernel written for two separate `[I, H]` matrices
   must be checked for a stride/offset swap, not re-derived.
4. **Routing-weight dtype order matters**: the HF form casts the renormalised top-k weights
   back to the logits dtype *before* multiplying the expert output. Doing the multiply in F32
   and casting after is a different number and will move the argmax of a near-tie row.
5. **Untied `lm_head` costs 1.0 GiB/token** — a fifth of the per-token read. `lm_head` batching
   (already prototyped in the sibling for a *tied* table) needs re-validation against an untied
   table where the row-gather and the full GEMV read different memory.
