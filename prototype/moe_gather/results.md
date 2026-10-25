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

## 6. Open defect, found by the new gate, NOT fixed: every expert case computes from no weights

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

`oracle_arm.py` (new, CPU-only oracle: sample + P3 + floor), `gate.py` (new), `gate` (new, sole lock
acquirer, `acquire-noblock`, prints the lock record and `no timing taken`), `moe_gather_args.h` (new),
`moe_gather_kernels.h`, `moe_gather_selftest.h`, `host_probe.cpp`, `moe_gather.hip`,
`evidence/moe-gather-{host,device}-2026-10-26.log`, `work/` (generated, gitignored: `plan.tsv`,
`slab.bin`, `expect.f32`, `cases.tsv`, `rows.tsv`, `units.tsv`, `oracle_p3*.tsv`, `oracle_floor.tsv`,
`*_report.tsv`, logs).
