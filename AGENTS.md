# Agent notes — qwen3.6-35b-a3b

Rules for work in this repo. The facts are in `README.md`, the trap library is in `pitfalls/`, the
map to reusable code is in `reference-implementation.md`, and the user decisions still open are in
`DECISIONS.md`. Everything needed to do a task lives in this repo.

## SCOPE: one model, and this time capacity IS a constraint

The only target is **`Qwen/Qwen3.6-35B-A3B`, text-only** (prefill + decode + MTP draft) on the
Framework Desktop (Strix Halo, `gfx1151`). Purpose: land the **MoE block** as a gated
subsystem, because it is the only part of `Qwen/Qwen3.8-Flash-Next`'s design that is
reachable on a model this box can actually run. Lineage and evidence: `README.md`.

Read this before importing the 0.8B project's scope statement — **one of its load-bearing
paragraphs does not transfer.** That project's `AGENTS.md` states that memory capacity is *not* a
constraint (its model is 4 % of the device-visible pool, so it forbade paging, eviction and
quantize-to-fit). Here the model is **65.72 GiB bf16 against a device-visible pool of ≈55.5–56.5 GiB
for the whole box — ≈32 GiB free while halogen is resident, and host-pinning does not raise it**
(`devices/measured-facts.md` §"Residency above the pool"), so:

- capacity engineering is in scope for the first time: residency layout, shard staging,
  expert placement, state-pool sizing under a tighter budget;
- **quantization to fit is decided and load-bearing**: the user authorized 4-bit routed experts on
  2026-10-26 (`DECISIONS.md` D-01) and confirmed the attention width the same day (D-01a: Q8_0), so the
  shipping shape — experts Q4_K, attention/GDN/shared Q8_0, router and norms exact — is **policy A** in
  `quantization/README.md`. int8/int4/int-fp anything *beyond* what those two decisions cover still
  needs the user's explicit words; the decisions are recorded, never inferred. Note the arithmetic that
  forced them: bf16 (65.72 GiB) is above the whole device-visible pool, and Q8-throughout (34.97 GiB)
  is above what one process gets while halogen is resident (`DECISIONS.md` D-11).
- developing and gating in bf16 against pinned reference dumps still works and is still the first
  step — the CPU oracle path (installed `transformers` `qwen3_5_moe`) runs in 123 GiB of host RAM and
  does not need the weights on the GPU at all. The 4-bit kernel is not the first thing to write;
  the bf16 MoE block is.
- **MTP is in scope** (`DECISIONS.md` D-05), sequenced after the dense path is attributable.
- still out of scope: the vision tower, multi-user batched serving, vLLM API parity, YaRN
  context extension, and the three Flash-Next-only subsystems (Gated Residual /
  hyperconnections, QSA indexer, n-gram PLE) — decided out, `DECISIONS.md` D-04.

## Before you write anything, decide nothing alone

`DECISIONS.md` holds the questions that are the user's, not the agent's (which width a tensor family
gets, whether to download the checkpoint, what counts as a band). Answering one of those by
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
  (`options: {timeoutMs: 7200000, checkpointBeforeDeadlineMs: 600000}`). The deadline, not
  `toolTimeoutMs`, is the knob that matters here: a long CPU-side analysis or one long device run
  can occupy a worker for tens of minutes. Census over the 0.8B project's 73 recorded subagent
  runs (copied here as a fact, 2026-10-26): **21 died on the run deadline** — 16 at 90 min, 2 at
  80, 2 at 30, 1 at 120 — while *successful* runs went up to 161 min, seven finishing at ≥80.
  90 min therefore sits inside the productive distribution, not above it, and the killed runs were
  still emitting tool results at the deadline (one died one line after its own gate printed
  `SELFTEST PASS`). Pass `checkpointBeforeDeadlineMs` so a long worker hands back changed files
  and remaining work instead of being cut mid-tool. A worker is stopped before the main agent
  measures.

## Self-containment: look elsewhere, but write it here

This repo is the whole working set of a fresh session, so a fact that lives only in another repo
is a fact no agent here can use.

> **You may *look at* another project on this box — `../qwen3.5-0.8b` for engine shape and
> precedent, `../qwen3.8-flash-next-old` for the failed route. You may not *reference* it.** Any
> number, trap, band, protocol or measurement a worker needs to do a task correctly is **copied
> into this repo**, with its numbers, its date and a one-line provenance note. A pointer may
> accompany a copy; it may never replace one.

Where copies go: traps in `pitfalls/`, platform numbers in `devices/measured-facts.md`, copied
executables in `tools/` (recorded in `tools/PROVENANCE.md`), model facts in `README.md`.
`reference-implementation.md` is a *reading list for code shape* and is explicitly not where a
fact is stored. Full rationale and the citation rules that follow from it: `pitfalls/60-docs-and-citations.md`.

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
  introduced by the commit that contains it. No fact needed for a task may be cited out of another
  repo (see §Self-containment).
- **A check that can produce no output cannot fail.** Applies to every parser, gate driver,
  negative control and counter written here: require the parsed record count to equal the
  printed record count; derive case lists from geometry; treat a counter that reads exactly 0
  as vacuous until witnessed non-zero somewhere it must fire.
- **Never print chat-template special tokens literally** — not in replies, commit messages,
  gate output, logs or docs. Every agent here is served by a checkpoint whose stop list is its
  own eos id, so a literal end-of-turn token silently truncates the message or kills the tool
  call it landed in. Refer to them by id (this family: 248044 eos/bos, 248046 end-of-turn,
  248068/248069 think-boundary — re-derive per checkpoint, see `README.md`).

## Traps: read them in-repo

Everything that binds work here is in `pitfalls/` — copied in, with its numbers and its correct
form, so no task requires opening another project. Index: `pitfalls/README.md`.

| when you are about to… | read first |
|---|---|
| take any timing, or trust one | `pitfalls/00-measurement.md` (blocking runs, warm pages, bimodality, the sub-ms timer lie, A/B inside one window + a behavioural canary, the two-rulers problem, the >240 GB/s rule and its ~32 MiB exception) |
| allocate, lay out or stream weights | `pitfalls/10-device-memory.md` (**~32–34 GiB pool, and nothing moves it**; silent `Managed` degradation; `hipMalloc` + bulk copy recipe; first-touch ~35 GB/s; host-pinned == GTT speed) |
| price a kernel, read an occupancy number, or write a census | `pitfalls/20-rates-and-occupancy.md` (rate ladder, geometry from KFD not HIP, register-file-per-SIMD, launch bill ≠ body price, workgroups per launch, census rules) |
| compare bf16 output or size a band | `pitfalls/30-numerics.md` (ULP comparator dead branch, `.view` vs `.to`, dtype-by-filesize, cancellation bands + floor probes, the 0.8876 chaos floor, association/contraction, byte-identity width claims, what is a user decision vs a green gate) |
| write or extend a gate, control or admission | `pitfalls/40-gates-and-admissions.md` (empty parse prints PASS, the unnamed denominator, derived must-catch sets, derived fixture populations, two self-consistent layouts, attribution, three-arm A/Bs) |
| write harness, shell, or a build | `pitfalls/50-harness-and-shell.md` (131,072 B argv slot, comma `--ids-file`, self-matching `pgrep`, self-locking driver deadlock, awk boolean count, `-rpath`, `-O0`, the special-token rule) |
| write docs or cites | `pitfalls/60-docs-and-citations.md` (the self-containment rule, line-cite rot, the `HEAD`-baseline blind spot, `§label` must name its file, doc moves are two jobs) |

## Traps that do NOT transfer (0.8B-specific — do not re-litigate, do not import)

That engine's numerics bands, route allowances, `MAXSC` wave geometry, attn-tile wave rules, GDN
collapse variants and MTP accept rates are facts about **its** pinned dumps and **its** geometry.
New model ⇒ new pins ⇒ new floors. Take the *method* (floor probe + geometry-derived must-catch
set, both written up in `pitfalls/30-numerics.md` and `pitfalls/40-gates-and-admissions.md`), never the band.

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
