# prototype/moe_gather — lane status 2026-10-26 (worker `halogen`, lane `moe_gather`)

No timing appears in this file. The `WALL` lines in `evidence/` are lock records required by
`AGENTS.md`, not measurements.

## 1. What this turn was for

`HANDOFF.md` §7 listed three harness defects that made the lane's `RESULT PASS` untrustworthy: an
argv parser that failed open, a selftest that could pass from one arm, and a sabotage whose FAIL line
printed `got == want`. §8's first substantive item was prediction 3 (the `gate_up_proj` fused
layout). All of §7 is fixed and witnessed; prediction 3 is closed with data. §8's expert-path work
then stopped on a real defect, reported in §6 rather than papered over.

## 2. §7 fixes, each with the sabotage witnessed firing

**§7.1 argv parser.** `moe_gather_args.h` is one parser shared by `host_probe.cpp` and
`moe_gather.hip` (the two copies had drifted: the `.hip` accepted `--arm N`, the probe accepted
neither form). Witnessed by `./gate --selftest`:

- `argv---arm=9 refused ok=1 rc=2`, `argv---arm=x refused ok=1`, `argv---nonsense refused ok=1`,
  `argv---arm refused ok=1` (missing value) — for **both** binaries.
- ``argv-`--arm=1` == `--arm 1` ok=1`` — identical stdout and exit code, so the two spellings select
  the same arm rather than silently selecting different ones.

**§7.2 three-arm selftest.** `mg_selftest_all()` runs honest / stub / crossed and PASS requires
`arms_run == 3`, honest `changed == 0` and `failures == 0`, stub `changed > 0`, crossed `changed > 0`,
equal check counts across arms, and both an `equals` and a `differs` population non-empty. Witnessed:

- `selftest-host_probe arms=3 honest_changed=0 stub_changed>0 crossed_changed>0 ok=1`
  (`arms_run=3 honest_changed=0 stub_changed=5 crossed_changed=53`), same line for `moe_gather`.
- The control that makes it falsifiable, run per arm: `selftest-… --arm=0 alone must FAIL ok=1`
  (and `--arm=1`, `--arm=2`): a single arm now prints `arms_run=1` and exits 1. Before this fix a
  single clean arm printed `RESULT PASS`.
- The stub arm fires 5 checks, the crossed arm 53 — both saboteurs are observed moving numbers, and
  `mg_selftest_all` also fails an arm that "passes" by judging fewer checks.

**§7.3 the misleading FAIL line.** The crossed arm computes its verdict against the *next* check's
expectation; it used to print the honest one, so a re-reader saw `got == want` on a failing line.
Now `want_used[]`/`src_name[]` are what the verdict used, and the witness is
`crossed-arm FAIL lines print the borrowed want ok=1 n=53` — all 53 crossed FAIL lines name a
`want_src` that is not their own check. (`got == want` can still appear on such a line and is honest:
a borrowed answer can coincide. The claim the check tests is that the printed want is the used want.)

## 3. Kernel/harness defects found while making the above true

- **`slot_bytes` vs `block_bytes`.** `mg_unit_elt` verified `slot_bytes == block_bytes + 1` and then
  addressed blocks with `block_bytes`, so the sampled-unit geometry the header promised was not the
  geometry used. Now `slot_bytes` is the single stride expression and `mg_verify_plan` requires
  `slot_bytes == block_bytes` (sampled units are contiguous runs of real tensor blocks).
- **`EXPERT_ROTATE` identity.** Rotate compared its array index `r->block` against `b->index`, the MoE
  block number; for any block whose MoE number differs from its array index the resolve returned NULL.
  It now compares the intended same-block identity.

## 4. Prediction 3 — closed with data: **PREDICTED_AND_CONFIRMED**

`oracle_arm.py` (new) evaluates five readings of the fused `gate_up_proj` against the pinned
`expert_raw` dumps, on 4 tokens × 8 slots per block, for every block available locally —
`layer0`, `layer1`, `layer39`, `mtp0`. `layer2` is not on disk and is listed as unprobed, not skipped
quietly.

The candidate address maps were checked for distinctness first, because five "candidate readings" that
are the same function would make the comparison a tautology: `P3DISTINCT pairs_of_candidates=10
distinguished=10 identical=0` in every block, so the interleaved spellings really are different
functions and not duplicate spellings.

The reproduction bound is measured, not typed: it is the largest bit-difference any *legal*
re-association of the reference's own arithmetic produces on the same data, over
`canonical_grouped`, `canonical_per_slot` and the hand-rolled single-row path the candidates use.

| block | elements | bound (grouped / per_slot / hand-rolled) | gate-first | up-first | in-chunk even | in-chunk odd | stack-transposed |
|---|---|---|---|---|---|---|---|
| layer0 | 65536 | 1 (1/1/1) | **1** | 65251 | 65503 | 65498 | 65502 |
| layer1 | 65536 | 91 (91/91/91) | **91** | 65307 | 65467 | 65475 | 65471 |
| layer39 | 65536 | 49 (49/49/49) | **49** | 65344 | 65473 | 65489 | 65466 |
| mtp0 | 65536 | 56 (23/56/56) | **56** | 65383 | 65477 | 65470 | 65486 |

`P3VERDICT … consistent_candidates=[output_chunk_gate_first] verdict=PREDICTED_AND_CONFIRMED` in all
four blocks; `P3OVERALL verdicts=PREDICTED_AND_CONFIRMED blocks=4 unprobed_blocks=layer2(not on disk)`.
The four rejected readings sit 700–1337× above the bound, so the verdict is not a tolerance artifact.
Records: `work/oracle_p3.tsv`, `work/oracle_p3_matrix.tsv`.

This also repairs the mistake that made the router lane's P3 read `UNRESOLVED`: safetensors data starts
at `8 + header_json_len + data_offsets[0]`, and the 8-byte-short read corrupts 523841/524288 elements
across all 256 rows. `oracle_arm.py` reads tensors only through `speccheck/moe_ref.py`, so it cannot
repeat that error. `f32`-then-round differs in 308076/524288 elements, so f32 accumulation is not the
reference math and no tolerance was invented to cover it.

## 5. What the sample declares (printed before any device work)

`SAMPLE cases=97 expert=70 combine=27 modes=15 classes=MG_EXP_MATCH=29 MG_EXP_MISMATCH=6
MG_EXP_ALT=43 MG_EXP_IMMUNE=3 MG_EXP_REPORT=16`, 3 blocks (`layer0`, `layer39`, `mtp0`), 2 pinned real
rows and 17 synthetic rows, `slab_bytes=38011872`, `ELEMENTS judged=306176`, plus
`MODECOVERAGE modes=15 covered=15` (the enum is scraped from the header's own `enum { … }` blocks, so
an unexercised mode is a failure rather than an absent line) and
`COMBINEFORM rows=2 elements=4096 differing=0` — the combine ruler reproduces the pinned
`routed_sum` bit-for-bit. `--max-rows` cuts sampled rows only and prints a `CUT` line; it removes no
mode, no class, and no assertion. Every `parsed == printed` equality in `gate.py` is enforced
(`PLAN`, `SELFTEST`×3, `CLASSES` vs `cases.tsv`, report records vs cases).

## 6. Superseded: the diagnosis below is WRONG in its central claim — see §9

> **Read §9 before believing a line of this section.** §6 claimed the expert records were *all zero* and
> inferred that the weight decode returned zeros. They were all `MG_NAN_BITS`, the decode was never the
> problem, and the cause was one inverted success test in the caller of the reader. The section is kept
> verbatim because the wrong reading is part of the record of how the defect hid; it is not the finding.

The host arm runs end to end and **all 27 combine cases pass** (bit-exact against the oracle, including
the cast-after / f32-combine / rank-order injections firing), but **all 70 expert cases write an
all-zero record**: for a case whose span is 3584 floats, the output span is zero in every word (real
row and synthetic row alike), so

- the 24 expert `MG_EXP_MATCH` cases fail bit-exactness,
- the 16 `MG_EXP_REPORT` cases fail the `gate`/`up` field checks (`differing=512/512`), while `s`/`raw`
  are printed as `COUNT … counted not banded` exactly as designed,
- 19 of 49 expert-side injections do not fire — they cannot, because the honest and sabotaged outputs
  are both all-zero (`MG_MODE_HALVES_SWAPPED differs from the honest output ok=0`, "differs from honest
  in 0 words"). `INJECTIONS host checked=49 fired=30`.

`tally: CHECKS ran=263 failed=101`, all 101 traceable to that one cause. The combine path is fully
green, and the kernel's per-case counters agree with the gate's independent recount in every case
(`counter==recount ok=1` for all 97), so this is not a bookkeeping mismatch. The expert kernel does run
and does write: an unwritten span would still hold the `mg_nan()` the driver initialises `out` with, and
these spans are exact zeros. Exact zeros in `gate`, `up` **and** `raw` is what `acc += x[j] * w` produces
when every `w` comes back 0, which localises the defect to the weight-reading path —
`mg_resolve_unit` / `mg_unit_elt` and the block-0-relative addressing I corrected this turn — not to the
plan files, the row table or the output layout. The x vectors are present in the slab (real rows: 2048
non-zero fp32; synthetic fixture rows: exactly the two live columns, by design) and the unit byte runs
are present and non-zero.

The behaviour that makes this quiet is that a failed unit resolve contributes zeros rather than
`MG_NAN_BITS` where the resolve itself returns NULL is caught, but a *resolved* unit whose decode yields
zeros is indistinguishable from a real zero dot. **That should become a hard failure —
`mg_verify_plan` refusing a case whose expert has no unit, and a witness that a known-nonzero fixture
row produces a nonzero `gate` — before any further expert work is judged**, otherwise the next round of
green means the same nothing this round meant.

Because of that, **no device run was taken this turn**: the host arm is the same code path and it fails,
so a device `RESULT` would add a lock record and no information. The halogen stream lock was released
at handback.

## 7. Honest residual — unverified after this turn

- Everything device-side: the HIP kernel has never been compared to the host arm, nor to the oracle.
  `./gate` (device) is written and unrun; `LAUNCHES`/`HOST_SYNC` (prediction 2's grouped-vs-per-slot
  launch count and the zero-host-sync claim) are asserted by code that has not executed.
- The expert path at all, per §6 — including whether the `slot_bytes` and rotate fixes are right, which
  can only be shown once weights actually decode.
- `MG_MODE_SCALE_MIN_NAIVE`, `MG_MODE_WIDTH_MISREAD`, `MG_MODE_BLOCK_OFF_1`, `MG_MODE_INPUT_CHUNK`
  etc. as *fired* injections on expert cases: structurally covered, not yet observed firing.
- `layer2` for P3 (weights not on disk), and Q5_K `down` beyond `layer0`.
- Whether the 5-candidate set exhausts the plausible layouts: P3 shows exactly one of the five is
  consistent, not that no sixth exists.
- `mtp0` is synthetic-rows-only (`main_n=1`), so the MTP block's real-row behaviour is untested.

## 8. Files

`oracle_arm.py` (CPU-only oracle: sample + P3 + floor), `gate.py`, `gate` (sole lock acquirer,
`acquire-noblock`, prints the lock record and `no timing taken`), `moe_gather_args.h`,
`moe_gather_kernels.h`, `moe_gather_selftest.h`, `host_probe.cpp`, `moe_gather.hip`,
`evidence/moe-gather-{host,device}-2026-10-26.log`, `work/` (generated, gitignored: `plan.tsv`,
`slab.bin`, `expect.f32`, `cases.tsv`, `rows.tsv`, `units.tsv`, `oracle_p3*.tsv`, `oracle_floor.tsv`,
`*_report.tsv`, logs).

Added by the repair turn (§9), all un-ignored by exact path because §9 cites them:
`work/diag_wspan.py` (per-field word census that falsifies §6's "all zero"), `work/diag_unit_elt.cpp`
(the three-candidate separator), `work/diag_residual.py` (the four residual groups, measured),
`work/diag_alt_coincides.py` (which derived alternatives coincide with the honest one),
`work/diag_armdiff.py` (attribution of the cross-arm word differences), `work/mutate.py` +
`work/negctl.sh` (the witnessed negative controls, with a restore trap).

---

## 9. The repair turn (2026-10-26, lane `moe_gather-expert`): §6's defect closed, six more behind it

No timing appears here. The `WALL` line in `evidence/moe-gather-device-2026-10-26.log` is a lock record.

### 9.1 Root cause, and the witness that separates it from the other candidates

`mg_unit_elt`'s contract is **0 = the element is defined, non-zero = this plan cannot answer for it**.
Both phases of `mg_expert_case` tested `if (!mg_unit_elt(…))`, i.e. read success as a refusal: every
element of every expert case was treated as undefined, `acc` was NaN-ed on the first `j`, and the width
census below it was unreachable. One defect, both arms (one header, two compilers), all 70 expert cases.

`work/diag_unit_elt.cpp` (CPU, same `plan.tsv`/`slab.bin`, calls each function directly) separates the
three candidate causes on the committed build:

| candidate | measured |
|---|---|
| `mg_resolve_unit` returns NULL | no — 9/9 role resolves returned a unit |
| `mg_unit_elt` refuses | no — `refused=0` of 9,437,184 element reads, 6,522,739 decoding non-zero; `elt(0,0) rc=0 value=0.0156894922` |
| the read succeeds and decodes 0.0 | no — same program runs the identical case through `mg_run_case`: `span=3584 nan=3584 zero=0 counter_route_n=0` |

Written-but-NaN versus never-written was settled before theorising, from the **counters**, not the words:
`out` is pre-filled with `mg_nan()`, so an unwritten word and a written NaN are the same word
(`0x7FC00000`), but `bad=3584 nan=1 route_n=0` can only come from a case that ran. `work/diag_wspan.py`
holds the per-field census that falsifies §6's "all-zero" claim.

### 9.2 The silence wirings (§6's unfinished business), each watched firing

1. **`mg_verify_plan` refuses an expert case with no unit**, per role, on the plan, before a case runs.
   Witness (`work/mutate.py` mutation B, restore guaranteed by the trap in `work/negctl.sh`):
   `FAIL plan: case 70 mode 3 resolves no unit for role 0: an expert with no weights must be refused,
   not run and write a silent record` ×6, `rc=1`, no verdict printed.
2. **`record carries a value`** per case: NaN-ness must agree with the oracle word for word; a span the
   oracle gives a value for must contain at least one non-zero finite word; the kernel's `nan` counter
   may not contradict the words. Zeros are not declared illegal in general — only absent where the oracle
   has a value. Aggregate line `INTEGRITY … silent_records=0`; honest build clean, mutation A reads
   `silent_records=82 of which no_value=79 nan-moved=82`.
3. **`weight reads == geometry`** per correct expert case: `route_n == 2I·H + H·I = 3145728`. Derived
   from geometry, *not* from the oracle, so it is the expectation-independent witness that the expert
   path actually read weights. The shipped state printed `route_n=0` and nothing asserted on it.
4. **the device arm's poison pre-fill**: `hipMemset(d_out, 0, …)` made an unwritten device span read
   exact zero — the one place `mg_nan()` was not. It is now an H2D copy of the host's `mg_nan()` fill.
   `HOST_SYNC` counts device→host reads and is unaffected (`d2h_total=3 d2h_during_launches=0`).

Mutation A (the shipped defect, re-introduced) now produces `RESULT FAIL … findings=246` with
`INJECTIONS host expert checked=29 fired=9` — a *non-zero* denominator showing the vacuous firing,
instead of the old `checked=49 fired=30` where the vacuity was invisible.

### 9.3 The five defects that only became visible once the expert path produced values

Each was a disagreement between the kernel and the oracle, or a sample that could not fail; none was a
tolerance, and no band was typed anywhere.

| # | defect | fix | how it showed |
|---|---|---|---|
| 1 | `MG_MODE_ROW_OFF_1` rotated the row index modulo `MG_TWO_I` while `row` is a **half**-matrix index, so row `I-1` of gate and of up pointed past the unit: two refused reads, and because the NaN lands in `s[i]`, all `H` raw elements went NaN with it (2051 NaN words per case) | rotate modulo the unit's own `rows`, one expression for both roles | `case 29 record carries a value … out nan=2051 … kernel_nan=1` |
| 2 | `MG_MODE_WIDTH_MISREAD` was applied to gate and up as well as down; Q4_K has no defined swap, so the whole record refused and "fired" for a reason unrelated to width — contradicting the mode's own `mg_reaches` entry (RAW only) | restrict the swap to the role the mode claims to misread | `case 35 … out nan=3584 zero=0 value=0` |
| 3 | the synthetic row's four combine injections read the PINNED slice (all-zero for a synthetic row) while their honest case read `KERNEL_RAW`: they judged two different inputs, and the IMMUNE one could never equal the honest output | the injections inherit the honest case's `raw` and `combine_src` | `case 25 MG_MODE_RANK_ACCUM equals the honest output … differs in 2048 words` |
| 4 | three class assignments were must-not-catch cases mislabelled as must-fire: `MG_MODE_CAST_AFTER` on the pinned rows (both `topk_weights` blocks are bf16-exact: `bf16-rounding-changes=0` of 256 words, `work/diag_residual.py` §A) and four cases whose **derived alternative is bit-identical to the honest expectation** (`work/diag_alt_coincides.py`: cases 23/61/83/86) | both immunities derived, not typed: a row-level `weights_already_bf16`, and the coincidence test on the two derived spans; `gate.py` re-derives the same predicate from `expect.f32` and refuses a plan that disagrees, and refuses any mode that is immune on *every* one of its cases (`ALTREACH … immune_everywhere=none`) | `case 9 MG_MODE_CAST_AFTER differs from the honest output … in 0 words` |
| 5 | `MG_MODE_SCALE_MIN_NAIVE` could not fire at all: both live fixture columns sat at `e ≡ 7 (mod 256)`, sub-block `j=0`, the one sub-block where `get_scale_min_k4` and the naive contiguous-6-bit reading agree on the fixture's bytes | live columns moved to 33/289 (`e=33 → j=1`, where the two unpacks read 1 and 4), still 256 apart so `BLOCK_OFF_1` still lands on the other one | `case 34 MG_MODE_SCALE_MIN_NAIVE FIRED (must differ) differing=0` |

Two more, in the oracle rather than the kernel: `MG_MODE_HALVES_SWAPPED`'s alternative was computed with
**un**swapped units, i.e. the expectation was the honest output (on the integer fixture `s` and `raw` are
invariant under the swap, so gate and up were the only fields that could distinguish the readings —
`differing=1024`); and an ALT case that reads *published* bytes crosses the same transcendental as a
REPORT case, so it is now judged by the same declared field rule (`field_exact=all|gate_up` in
`cases.tsv`, derived from the `cls` of the units the case actually reads, printed as
`FIELDEXACT all=79 gate_up=18`, and refused for any `MG_EXP_MATCH` case).

### 9.4 Populations and RESULT lines, before and after

| | before (committed) | after |
|---|---|---|
| `RESULT` host arm | `RESULT FAIL plan check (no device) checks=263 findings=101` | `RESULT PASS plan check (no device) checks=456 findings=0` |
| expert cases producing a value | 0 of 70 (all `MG_NAN_BITS`) | 70 of 70; `INTEGRITY host cases=97 silent_records=0` |
| expert cases bit-exact | 0 | all 24 synthetic `MATCH` cases bit-exact; all 16 `REPORT` cases bit-exact on `gate` and `up` |
| injections | `INJECTIONS host checked=49 fired=30` (19 could not fire) | `INJECTIONS host expert checked=29 fired=29`, `INJECTIONS host combine checked=14 fired=14`, total `43/43` |
| classes | `MATCH=29 MISMATCH=6 ALT=43 IMMUNE=3 REPORT=16` | `MATCH=29 MISMATCH=6 ALT=37 IMMUNE=9 REPORT=16` |
| three arms | `arms=3 honest_changed=0 stub_changed=5 crossed_changed=53`, single arm fails, argv refusals `rc=2` | unchanged, `RESULT PASS host selftest checks=20 findings=0` |
| `route_n` census | 0 | `route_n=125829120 route_bad=0` — the f32 dequant route diverged from the double route **0** times over 125.8M elements, Q6_K included |

The four new IMMUNE cases are the derived coincidences of §9.3 row 4; no injection was dropped from a
denominator to make the arm green (`host injection denominator matches the plan … plan-derived=43
counted=43`).

## 10. The device gate, first run ever, and what is still NOT ESTABLISHED

`./gate` (wrapper as sole `gpu_lock` acquirer, `acquire-noblock`, `LOCKREF … = 0`), correctness only,
lock record `WALL started=…T15:27:50Z finished=…T15:28:04Z (lock record only; no timing taken)` —
14 s wall, `UTIL_BEFORE gpu use: 92%`, `no timing taken`, `LOCK released`.

- `RESULT FAIL moe_gather device gate checks=908 findings=1`.
- **Every device-side check passes**: device `INTEGRITY … silent_records=0`, device injections
  `expert checked=29 fired=29` / `combine checked=14 fired=14`, device `MATCH`/`ALT` bit-exactness,
  device `gate`/`up` exact on the published-byte cases, `counter==recount` for all 97 device cases,
  `LAUNCHES mode=grouped issued=6 expected=6 device_counted=6`,
  `LAUNCHES mode=per-slot issued=27 … device_counted=27` (prediction 2's grouped-vs-per-slot counts,
  first time printed), the per-slot shape producing the same words, and
  `HOST_SYNC d2h_total=3 d2h_during_launches=0`.
- **The one finding**: `host arm and device arm agree word for word ok=0 differing words=6631 of
  306176`. Attributed completely by `work/diag_armdiff.py`: all 6631 are in `s` (206) and `raw` (6425),
  and only in the 18 cases declared `field_exact=gate_up`. In those fields `gate` and `up` differ
  **0** times between the arms (and 0 times from the oracle in both arms), while `s` differs
  host-vs-oracle 2001 / device-vs-oracle 2025 / host-vs-device 206, and `raw` 24631 / 24747 / 6425.
  That is the transcendental route — `expf` in glibc against `expf` in ROCm's libdevice — which this
  lane declares counted-and-never-banded; the cross-arm check was written before any device run existed
  and compares the whole buffer, so it spans the fields the lane refuses to band. **Left red on
  purpose**: making that check respect the declared field rule is a change to a band, which is the
  user's call, not a repair's, and it cannot be verified without a second device run.

**NOT ESTABLISHED after this turn**

- Whether the cross-arm whole-buffer check should be scoped to the declared field rule (see above). It
  is the only open finding and it is a decision, not a bug.
- Anything about Q6_K `down` beyond the sampled layers, `layer2` at all (weights not on disk), and the
  MTP block's **real**-row behaviour (`mtp0` is synthetic-rows-only).
- That the f32-route census of 0 divergences generalises: it is 125.8M elements of `layer0`/`layer39`
  real bytes plus two synthetic units, and Q6_K divergence was predicted but not seen here.
- Whether `mg_reaches`' YES/NO table is consistent with the outputs: `MG_MODE_HALVES_SWAPPED` declares
  it reaches `s` and `raw`, and on this fixture it provably does not (the product is commutative), which
  is now explained in the oracle but is not asserted by any check.
- That the six-immune-case classification is exhaustive for *other* fixtures: the coincidence rule is
  per-row arithmetic, so a different pinned row can move a case back into ALT. The guard is that no mode
  may be immune everywhere, not that any case stays immune.
- `MG_MODE_INPUT_CHUNK`'s NaN region is modelled from the run's own block count; whether that is the only
  sensible reading of "the fused pair chunked along the INPUT dim" is untested against any other source.
