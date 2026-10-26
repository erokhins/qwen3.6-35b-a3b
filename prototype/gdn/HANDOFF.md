# prototype/gdn — HANDOFF (living file, written early on purpose)

Written ~minute 38 of the lane. This file is updated as things execute; it is not a summary
written at the end, because the previous lane at this slice died at the 90-minute deadline with
three headers and no executable verdict. Checkpoint log is at the bottom. The slice-2 lane continued
this file on 2026-10-26 from ~minute 25 of its own deadline. The slice-3 lane wrote this section at
~minute 52 of its own deadline, before its full-gate run finished; the numbers it could not yet know
are marked and filled in below.

## SLICE 3 — every banded row cites its licence, and the multi-chunk question gets a field name

**Why the slice exists.** Slice 2 named two limits as its own: (1) its 84 accepted decode rows are
cross-form (the manifest dispatches decode as *recurrent*, so what agreed with the pinned bytes was
a chunked kernel at C=64 with 63 zero pads) and `band_licenses` was never consulted for that pair;
(2) a multi-chunk prefill had never been checked against pinned bytes. Both are closed here — one by
a licence citation on every row, one by a census that names the manifest field that makes it
unadjudicable plus a carry test at the only width the pinned material allows.

**JOB 2's answer, which is the deliverable: UNLICENSED.** `chunked|pinned` is licensed at
`layer{0,1}/pinned/prefill` for all seven stages and at **no decode pass, for any stage**. The
reason is not an omission, it is the dispatch table: `gdn_ref.probe_arms_for("decode", "pinned")`
returns `recurrent, reassoc, pinned` — §04b.1 gives a cached decode step the recurrent program, so
the probe never ran a chunked arm at a decode pass and `band_from_probe` cannot license a pair it
never measured. (1,136 licensed rows in the band file, all accounted for; the slice-2 total of 84
accepted decode rows sits inside the 82-candidate-per-stage refuse set that never had a record.)
So there was **nothing to cite and nothing to loosen**: the gate now prints, per banded row,

    LICENCE row=layer0/pinned/decode0|chunked|pinned plan=pinned_state pass=layer0/pinned/decode0 \
        stage=q_hat licensed=0 form_probed=0 form_chunk_width=reference \
        arms_at_pass=recurrent,reassoc,pinned verdict=UNLICENSED credited=0 \
        why=no probe record for one of the two arms at this pass

on all 84 rows, plus `LICENCETALLY`/`CARRYVERDICT ... credited=0 unlicensed=42 verdict=UNLICENSED`
per plan, and `GATE PASS` is now unreachable while any banded row lacks a licence — the gate prints
`GATE QUALIFIED ... uncredited=N` and exits 1 instead. The acceptances were NOT deleted, the band
was NOT touched, and the two `q_hat` findings are still the only two findings (`findings=2`). The
fix is a probe record, which only `--floor-probe` may write, and §04b.1 currently refuses to run a
chunked arm at a decode pass to get one; that is a user/oracle decision and it is reported, not
routed around.

The same citation runs on the prefill rows and they come out `LICENSED credited=1` (14 rows), which
is what makes the counter a measurement and not a constant. A third state exists, named
`LICENSED_FORM_UNPROBED`, for slice 3's own re-chunked rows — see below.

**JOB 1's answer: the fixture cannot carry multiple chunks, and the field is `dumps[].shape[1]`.**
Every pinned pass's `q_pre` is `[1,T,...]` with T=32 for the two prefills and T=1 for the six decode
steps, against `geometry.chunk_width=64`, so `chunks=1` for all eight pinned passes
(`input.n_tokens=32`, `input.n_decode_steps=3`, `counts.passes_per_block=4`, and `dispatch.decode`
= "recurrent, cached conv window + recurrent state (§04b.1)"). The census prints one `CHUNKSRC`
line per pass, cross-checks the manifest's T against the T of the tensor the dump reader handed back
(so it is anchored to bytes, not only to the manifest's claim), and reports
`MULTICHUNK ... max_chunks=1 verdict=NO_PINNED_MULTICHUNK_PASS`. No fixture is invented.
Two things follow, both witnessed:
* the arm itself refuses a case that claims more than one chunk (`T=65` and `T=128` refused,
  `T=32` accepted — both directions, so neither a parser that refuses everything nor one that
  accepts everything can pass it);
* the census is not decoration: the SAME census over an in-memory copy of the record whose
  `shape[1]` is set to 128 prints `verdict=ADJUDICATION_REQUIRED` and the absence check goes RED
  (`CONTROL check=chunk_census_fires_on_a_multichunk_pass fired=RED`).

**And the carry mechanism is still tested against pinned bytes, at the only width available.** The
pinned 32-token prefill is split into `n_chunks=2` equal chunks of `T//2 = 16` (two because two is
the smallest number that forces a carry; 16 because the split is equal and 32 is the manifest's
field — neither is typed). Chunk 1's incoming state is the state chunk 0 **emitted in the same arm
run** (`carry=0`), and the final state is judged against the pinned `layer{0,1}.prefill.state`:

    CHUNKCARRY block=layer0 pass=prefill chunks=2 chunk_width=16 carried=last<-previous \
        pinned_field=layer0.prefill.state relL2=2.15464e-07 ndiff=438500/524288 band=ACCEPTED \
        verdict=RECONSTITUTED

18 banded rows (the five stages `downstream_closure(scan_out,state)` reaches × 2 blocks × 2 chunks,
minus the two interior `state` boundaries that have no pinned reference — printed as `SPLITSKIP
verdict=NO_PINNED_REFERENCE`, not skipped silently), all ACCEPTED; plus 8 zero-tolerance rows showing
`q_hat`/`k_hat` are bit-identical to the single-chunk arm's own rows, which is also what certifies
the slicing (the token axis is derived per stage from `gdn_ref.HEAD_AXIS` and the pass T, because
`q_hat` is `[1,32,32,128]` with `h_v == T == 32` and a guessed axis silently compares heads against
tokens — an early version of this slice got exactly that wrong and the bit-check is what caught it).
These 18 rows are `LICENSED_FORM_UNPROBED`: the pair IS licensed at a prefill pass, but chunk width
16 is not a width any probe arm ran (`probed_chunk_widths=32,64`, read out of `gdn_ref.Form`), so
they are measured, printed, banded, and **not credited**.

**Checks added, and how each was seen failing.** The committed slice-2 run printed 0 `CHECK` lines
and 9 `CONTROL` lines that no counter watched; this run prints 11 `CHECK` lines and 13 `CONTROL`
lines, of which 4 are named sabotage witnesses, all 15 named in `COUNTS named_checks=15
named_checks_red=0`. Every one of them goes RED in a mutated world and green in this one:

| check | what it asserts (data, never a constant) | witness that it fires |
|---|---|---|
| `licence_cited_band_single_chunk` | 14 cites == blocks×stages | the same counter over a carry scope where nothing is licensed goes RED (`--only carry` before the per-plan tally was fixed printed `rows=84 derived=42 verdict=FAIL`) |
| `licence_cited_carry_pinned_state` / `_arm_carried` | 42 == blocks×passes×stages per plan | same as above |
| `carry_rows_unlicensed_eq_dispatch` | unlicensed count == a SECOND derivation from `probe_arms_for`, not from the band file | `CONTROL check=licence_census_fires_on_a_poisoned_table fired=RED` — an in-memory band copy granted `layer0/pinned/decode0\|chunked\|pinned`; 41 vs 42 derived |
| `licence_census_fires_on_a_dropped_row` | the honest predicate re-evaluated over a row set with one row dropped is False | fires (83 vs 84) |
| `chunk_census_covers_pinned_passes` | 8 == blocks × `counts.passes_per_block` | short by construction if a pass lacks `q_pre` |
| `chunk_no_pinned_multichunk_pass` | no pinned pass has chunks>1 | `fired=RED` at shape[1]=128 |
| `chunk_case_refused_by_arm` | T=65 and T=128 refused AND T=32 accepted | the accept leg is the second direction |
| `split_carry_premise` | each carried chunk's `state_nonzero` == the non-zero count of the state its predecessor emitted, read back out of `out.bin` | — (premise of the section) |
| `split_banded_rows_eq_derived`, `split_row_local_stages_bit_identical` | 18 and 8 rows, derived from the closure and the stage list | a wrong slice axis moves the 8 to tens of thousands of ndiff (measured in the scratch run) |
| `split_carry_is_load_bearing` (witness) | — | `fired=RED`: replacing the carry with a zero state makes the final `state` row REJECT at relL2 0.566 (against 2.15e-07 carried) |
| `chunk_census_fires_on_a_multichunk_pass` (witness) | — | `fired=RED` at shape[1]=128, see above |

The pad census is **printed as a fact and not counted as a check**, on purpose: chunk-pad rows are
synthesised inside the arm (`gdn_item_enter` writes 0 for `i >= t`) and are not bytes the driver can
reach, so no driver-side mutation can move it and a check on it would be a check that cannot fail.
The `q_hat` band is untouched and still red.

**What got bigger, in numbers.** Committed slice-2 run (`evidence/gdn-slice2-gate-2026-10-26.log`)
→ this slice's run (`evidence/gdn-slice3-gate-2026-10-26.log`): `STAGE` rows 158 → 178 (the 158 are
byte-identical; `evidence/gdn-slice3-stage-line.diff`: 20 lines added, 0 removed, one insertion
hunk), named checks (`CHECK` + `CONTROL check=` lines, counted in `COUNTS named_checks`) 0 → 15,
with 0 of them red, `COUNTS` keys 14 → 22, banded rows carrying a licence citation 0 → 116 (14
LICENSED, 84 UNLICENSED, 18 LICENSED_FORM_UNPROBED), and the determinism section's reach 158 → 178
lines (`DETERMINISM here=178 run1=178 run2=178 run1_eq_run2=1 both_eq_this_run=1
verdict=IDENTICAL`) — the new rows are inside the lane's reproducibility claim, not beside it. The
only thing that did NOT go up is the finding count: still exactly 2, still both `q_hat`.

The determinism section's child command used to be a typed `--only band,carry`; it is now DERIVED
from the sections this run actually ran (`[s for s in (band, chunkcarry, carry) if s in ran]`), for
the reason in `pitfalls/40`: a typed list compares whatever it names, which silently becomes the
wrong population the moment a section is added or `--only` is narrowed.

**Files changed by slice 3:** `prototype/gdn/gate.py` (the `Licence`/`Checks` objects, the LICENCE
citation in `band`/`carry`, the new `chunkcarry` section, the `GATE QUALIFIED` verdict path) and this
file. No C source, no header, no build change, nothing under `speccheck/`, `checkpoints/` or
`data/`. `grep -c gpu_lock prototype/gdn/gate.py` = 0; no device run, no timer read.

**Remaining after slice 3:** unchanged from slice 2 except that the licence state is now named
instead of assumed — the decode rows stay uncredited until the oracle licenses them (a user call),
the engine-shape multi-chunk prefill (T>64, C=64 chunks) still needs pinned material **and** a chunk
loop in the arm, batching/MTP/§04.2/§04.3/device codegen are still out, and `q_hat` is still the
user's call.

## SLICE 2 — the state is live, and the state read is now a witnessed check

**Why the slice exists.** Slice 1's `gdn_kernels.h` read the recurrent state as `S^T` (the oracle's
own `state_stored_transposed` trap). The lane fixed it, but the fix was UNTESTABLE there: every state
slice 1 ran was exactly zero, at which the honest read and the `S^T` read return the same number.
`d_k == d_v == 128`, so a transposed read changes no shape, raises nothing, and cannot move a band
that nothing moves it. Slice 2 makes the state non-zero by construction and proves the fix.

**The material is pinned, not invented.** `checkpoints/dump-manifest-gdn-2026-10-26.json`
§reading_rules.decode_rows: `decode{s}` is step s of the cached sequence, fed AFTER the pinned
prefill produced `prefill.state`. So prefill → decode0 → decode1 → decode2 is a real state-carry
chain, every incoming state in it is non-zero (509,952 and 498,944 of 524,288 words, per block), and
every pass has a pinned `state` reference. Nothing in this slice synthesises a fixture.

**What changed in the code.**
* `gdn_state_dot()` is the ONE place the incoming state is contracted; §04b.4's delta rule and
  §04b.5's `q_decayed S` term both go through it. The state WRITE (item H) copies row-major and
  deliberately does not — which is why the witness names the two contractions and not the carry term.
* `build.sh --broken` now also builds `build/host_probe_xpose` with `-DGDN_BROKEN_STATE_TRANSPOSE`,
  the same source with the read put back to `S^T`. It is a control BUILD and not a scratch copy of
  the headers, so it cannot drift from the shipped arithmetic; the honest build never defines it.
* The plan is `v2` and gained `carry=<idx>`: that case's incoming state is the `state` stage the
  named case emitted IN THIS RUN (a pointer into the arm's own out buffer — no copy, no second
  buffer). The arm REFUSES a self or forward carry, a cross-block carry, and a case whose `T`
  differs from the plan's `T`; `gate.py` witnesses all three refusals firing.
* The zero-state REFUSAL is gone (that was slice 1's boundary). `GDN_ST_STATE_NONZERO` survives and is
  now load-bearing: `gate.py` requires it to equal the non-zero count of the pinned tensor the state
  came from (pinned-state plan), or of the state the previous case emitted, read back out of the
  arm's own `out.bin` (carried plan). A counter that can only read 0 would be vacuous; this one is
  witnessed at 498,944 and 509,952 in the same run.
* New self-test checks (21 → 23): a non-zero hand-built `[d_k,d_v]` state with the expected value
  spelled independently of `gdn_state_dot`, plus a clause requiring the transposed sum to DIFFER from
  the honest one, so the fixture cannot pass vacuously. `host_probe_xpose --selftest` FAILS it.

**What the checks say (full log: `evidence/gdn-slice2-gate-2026-10-26.log`).**
* `carry`: 6 cases × 7 stages × 2 plans = **84 rows, 84 ACCEPTED, zero rejects**. Plan
  `pinned_state` reads each pass's pinned predecessor state; plan `arm_carried` makes cases 1 and 2 of
  each block read what the previous case emitted. The `state` stage sits at relL2 3.3e-08…7.0e-08
  against the pinned bytes, `ndiff` 183,335…327,798 of 524,288. `maxabs`/`cos` are printed and
  labelled `[reported_only]`; the verdict is `band_check_row`'s.
* The compiled pair agrees on the carried plan at zero tolerance: `out.bin` 88,670,208 B and
  `echo.bin` 299,520 B, `ndiff_bytes=0`.
* **The transposition sabotage.** `host_probe_xpose` on the ZERO-state prefill plan: `ndiff_bytes=0`
  over all 35,143,680 B of `out.bin` — indistinguishable from the honest arm, which is the measured
  statement that slice 1 could not have caught this. On the live plans it moves 8,548,098 and
  8,717,959 bytes; `SABOTAGE_BAND rows=60 rejected=60` on the derived must-catch set (the five stages
  the oracle's own `downstream_closure` says the read reaches × 6 cases × 2 plans); `q_hat`/`k_hat`
  — 24 rows — are byte-identical under it, asserted and not assumed. Per case the `state` row is
  relL2 4e-08 honest against 0.056…0.137 transposed (ratio 1.1e6…2.3e6) and ndiff 190,952 against
  488,539. The predicate is the ordering `relL2_honest < relL2_xpose AND ndiff_honest < ndiff_xpose`
  on all 60 live-stage rows — an ordering, not a typed threshold, with the absolute verdict left to
  the band.
* `determinism` is a named section: it re-runs `gate.py --only band,carry` as two further processes
  and requires their STAGE lines to match each other AND this run's (158 lines, `verdict=IDENTICAL`,
  logs in `evidence/gdn-determinism-run{1,2}-2026-10-26.log`). Two FULL gate runs were also compared
  outside the gate: 158 STAGE lines, sha256 `3951cdb1d97df92c…`, `diff` empty
  (`evidence/gdn-slice2-gate-second-run-2026-10-26.log`, `gdn-slice2-stage-line.diff` 0 bytes).
* `controls` is now 9: stub 14/14 red on the derived count, crossed 14/14, 2-arm refused, the
  `--broken` bf16 control fires (self-test RED, 8 of 14 rows red), stub on the live-state cases 42/42
  red, crossed on the live-state cases 42/42 (fields AND live state crossed together), and the three
  bad-plan shapes refused.
* Everything slice 1 had is still green: 12 of 14 prefill rows ACCEPTED, `q_hat` still the only two
  reds at `max_abs=2.980e-08 > licensed 2.900e-08`, exact reads 16/16 BIT_IDENTICAL, the oracle CLI
  arm `CROSS-FORM PASS` with digest `27a68c7e82c9aa05…`, the `norm_weight` ladder reproducing the
  oracle's accepted case at k=2 with 190 of 131,072 elements moved. **The band was not touched.**

One display change is worth knowing about: the prefill rows now print `perhead=…(h7/32)` where
slice 1's log printed `(h84657/131072)`. Slice 1 flattened the pinned reference to 1-D before
`aggregates`, so its "per-head maximum" was a per-element maximum wearing the name; the reference is
now passed in its own orientation and the per-head maximum is per HEAD (32 of them). `per_head_max`
equals `max_abs` for every row either way, so no verdict moved — checked line by line against the
committed slice-1 log, whose judged fields are identical to the new prefill rows.

## Inventory of the dead lane's three headers (step 0)

| file | verdict | why |
|---|---|---|
| `gdn_kernels.h` | KEPT, with four fixes | the arithmetic transcription is correct and matches `speccheck/gdn_ref.py` `scan_chunked`/`gated_norm`/`out_proj` line for line; the geometry constants match the pinned manifest (h_k=16, h_v=32, d_k=d_v=128, C=64); its dump-orientation comments match `data/dumps/2026-10-26-gdn` |
| `gdn_case.h` | KEPT, with renames | the item decomposition and the derived per-stage item denominators are right; its stage NAMES were internal item names (`enter`,`cum`,`pair`,…) and not the oracle's, so the emitted-stage table is rewritten to the seven `gdn_ref` stage names |
| `gdn_arm.h` | REWRITTEN | it did not compile: `GdnArgs` had no `read_field_dummy`; it called `gdn_fault_id`, `gdn_field_id`, `gdn_field_name`, `gdn_echo_hash`, `gdn_bind`, `gdn_case_run_device`, `c->field_bad_total` — none of which exist in any of the three files. There was no driver, no build, no gate, so nothing in it had ever been executed |

The four fixes to `gdn_kernels.h`, each of which is a real defect and not cosmetics:

1. `gdn_item_vnew` and `gdn_item_core` read the recurrent state as `S + col*GDN_D_V` and dotted
   over `d_k`. In the `[d_k, d_v]` layout that is `S^T`, i.e. the oracle's own
   `state_stored_transposed` injection. It is invisible in slice 1 only because the initial state
   is zero — which is exactly the kind of latent wrongness that survives into slice 2. Fixed to a
   strided read of `S[k, col]`.
2. §04.7's product was not rounded to the storage dtype. `gated_norm` computes
   `weight(bf16) * x.to(bf16)` → the PRODUCT is bf16, then it is multiplied by the f32 gate. The
   header multiplied in f32. Fixed to `bf16_rt(w * bf16_rt(x̂))`.
3. The q_hat/k_hat STAGE was never emitted (only per-head scratch). The oracle's `q_hat`/`k_hat`
   stages are `[B,h_v,T,d]` f32 in the scan dtype; the header had no output for them, so two of
   the seven banded stages could not have been compared at all.
4. The in-kernel read-sabotage ladder (`gdn_sabotage_*`, `field_bad`) is removed: the perturbation
   belongs in the driver, in the bytes handed to the arm, the way `gdn_ref.py`'s own sabotage
   witnesses perturb pinned tensors in memory after `read_dump_tensor` has hashed them. It also
   lets the driver derive the expected moved-count exactly instead of counting it inside the arm.

## What runs

    ./build.sh                                   g++ host twin + hipcc device TU (CPU work, no device touched)
    ./build.sh --broken                          ALSO the two control builds: the bf16-truncation one
                                                 and build/host_probe_xpose, the S^T state-read sabotage
    ./gate.py                                    the whole gate; CPU only, no device, no lock, no timer
    ./gate.py --only three-arm|band|carry|exact|controls|witness|oracle|determinism
    ./bf16_vectors.py                            derives the self-test's bf16 table from torch

Build artefacts: objects and the two control binaries live in `prototype/gdn/build/`, the arms at
`prototype/gdn/{gdn,host_probe}`, and the regenerable planes in `prototype/gdn/work/` (~1.8 GB of
`.bin`, never committed; deleted at the end of this lane, which took it from 1.8 GB to 336 KB). The
first three paths are gitignored by exact path (`git check-ignore -v` proves all four). NOTE: slice 1
accidentally COMMITTED some of `work/` — `work/*.tsv`, `work/plan*.tsv`, `work/run-*.log` and
`work/asan.echo` (a 3,163,136-byte binary) are tracked. This lane regenerated the tracked `.tsv`
outputs (the plan and report grammar gained `carry=`) and restored `work/asan.echo` after cleaning,
so the only tracked deletions here are none. Fixing that hygiene means editing `.gitignore`, which is
outside this slice's boundary, so it is written down instead of done.

## What passes (executed 2026-10-26; full logs in `evidence/gdn-slice2-gate-2026-10-26.log` and
`evidence/gdn-slice1-gate-2026-10-26.log`)

* `SELFTEST PASS` on BOTH compiled arms, 23 checks each (slice 1 had 21; the two new ones are the
  state read and its non-vacuity clause). Both control builds FAIL their self-test, which is the point.
* Compiled pair at zero tolerance: `out.bin` 35,143,680 B and `echo.bin` 3,163,136 B, `ndiff_bytes=0`.
  The hip arm announced `arm=hip_hostpath hip_calls=0`; gate refuses it otherwise.
* Band verdicts, `band_check_row` only, against the pinned dump, `band_digest = 27a68c7e82c9aa05…`:
  **12 of 14 rows ACCEPTED.** `k_hat, scan_out, core_attn, state, normed, y` ACCEPTED on both
  blocks. **`q_hat` REJECTED on both blocks: `max_abs = 2.980e-08 > licensed 2.900e-08`** — one f32
  ULP over a band whose own `mean_rel` (7.0e-08 vs licensed 1.16e-07) and `1-cos` (2.4e-14 vs
  1.6e-13) sit comfortably inside. See "open" below; the band was NOT touched.
* Exact read/decode checks: **16 of 16 BIT_IDENTICAL** (7 echoed fields per block at zero tolerance,
  plus the 8,388,608-word `out_weight` positional fingerprint per block).
* Controls: `stub` 14/14 rows red on the derived count, `crossed` 14/14 red on the derived count,
  a 2-arm agreement REFUSED, and the `--broken` control build (bf16 truncation) FIREs the
  self-test and turns 8 of 14 stage rows red — every new check was witnessed failing.
* The oracle CLI arm: `speccheck/gdn_ref.py --cross-form` exited 0 with `CROSS-FORM PASS` and the
  same band digest, in this lane, in this window (`evidence/gdn-cross-form-in-lane-2026-10-26.txt`).
* `pad_nonzero=0` and `state_nonzero=0` on every case: §04b.2's zero padding and slice 1's zero
  initial state are counted facts of the run, not assumptions.

## The `norm_weight` witness — what it actually showed (`gate.py --only witness`)

The accepted case reproduces the oracle's own number **exactly**: blending this kernel's honest
`normed` toward its `norm_weight_f32` `normed` at 1/2**2 is ACCEPTED by `band_check_row` with
**190 of 131,072 stored elements still changed** (meanrel 1.944e-06 against a licensed 3.941e-06,
maxabs 3.906e-03 against 3.125e-02) — the same row `speccheck/evidence/gdn-cross-form-2026-10-26.txt`
records. So the hole is real and it is reproduced here rather than cited.

But the brief's expected pairing does **not exist** at this field, and that is a finding, not a
shortfall in the check. Perturbing `norm_weight` in memory (bf16 LSB flip on 1 word in 2**rung):
at rung 7 — a single word out of 128 — the exact check goes RED (1 word moved, exactly the derived
count) and `band_check_row` on `normed` also REJECTS (meanrel 9.324e-06 > 3.941e-06). Every rung
0..7 is RED on both sides. A `normed` band cannot hide a *byte* change to the weight because one
weight word owns 1,024 output elements; what it hides is the *width* defect at k=2, which changes
190 elements and whose weight bytes are honest. The two checks are therefore **complementary, not
nested**: byte-exactness is strictly more sensitive on the read path (1 word of 128), the band is
the only thing that sees the arithmetic defect — and on the accepted case the exact check is
GREEN, which is reported rather than hidden.

## NOT ESTABLISHED by slice 2 — read this before trusting the carry

* **The decode passes are cross-form.** The manifest's §dispatch says prefill is chunked and
  **decode is recurrent**, so this slice's 84 accepted rows are a *chunked kernel at C=64 with 63
  zero pad rows* agreeing with *pinned recurrent bytes*. The agreement is licensed by the band's
  per-stage limits (`band_check_row`), which the floor probe derived over both forms — but the
  oracle's per-pass licence (`band_licenses`) was NOT consulted, and `chunked|pinned` is not a
  licensed row at a decode pass. The agreement measured here (relL2 ~4e-08 on `state`) is the
  evidence; a per-pass licence would be the oracle's job, not this lane's.
* **A multi-chunk PREFILL is still untested against pinned bytes.** The pinned prefill is 32 tokens =
  one chunk, so a prefill→prefill carry has no pinned reference; the chain this slice proves is
  prefill→decode→decode→decode. `prefill_2chunk_*` fixtures exist in the oracle's probe population,
  but this lane does not generate fixtures, so a three-chunk prefill carry is a later slice.
* **Carry error growth is bounded, not characterised.** The carried plan is judged against the pinned
  bytes with the same band, so a diverging carry would show up as a rejection — but three steps is
  three steps. No growth curve, no long-cache test, and nothing about a cache that is evicted, reused
  or re-shaped.
* **Nothing has run on the iGPU.** Both arms announce `hip_calls=0`; the state read has never been
  issued as a device kernel, and there is no device codegen for it yet.
* **Out of slice, unchanged:** batching (B>1), MTP, §04.2/§04.3 (the projections and the conv, whose
  decode-side window carry this slice depends on but does not test), and every performance question.
* **The `q_hat` ULP is still the user's call** (the three ways out are listed under "Open" below);
  slice 2 neither fixes it nor hides it, and it does not touch `state`: the decode `q_hat` rows are
  inside the band at 1.49e-08.
* **The `norm_weight` side-by-side pair is still not found**, re-confirmed unchanged from slice 1
  (`side_by_side_pair_found=NO`): every perturbation rung is red on both the exact check and the band.

## Open (the only thing standing between this lane and a green gate)

`q_hat`'s `max_abs` is one f32 ULP over the licensed value. Suspected cause, measured not assumed:
§04.4's l2norm takes a sum of 128 squares, and torch's `.sum(-1)` is a blocked reduction while
`gdn_kernels.h` declares an ascending single-accumulator reduction (the declaration that makes the
two compiled arms agree bit for bit). `1/sqrt` vs `torch.rsqrt` was already removed as a cause:
switching to a double-precision reciprocal square root took max_abs from 3.725e-08 to 2.980e-08 and
no further. The three ways out are all someone else's call, and none of them is tightening or
widening a band from here:
  1. reproduce torch's reduction order for that one sum (and then this arm's spread has to be
     re-measured, because the arm's spelling is what the band's `max_abs` describes);
  2. re-run the oracle's `--floor-probe` with a non-torch-reduction arm so the band has a record for
     this class of implementation (only `--floor-probe` writes the band; this lane never ran it);
  3. accept `q_hat` as a documented out-of-band stage for slice 1 and revisit it when the engine's
     real `l2norm` kernel exists.

## Checkpoint log (slice 2, 2026-10-26)

- min 5 — claimed `/tmp/pi_halogen.lock` (no live holder existed); wrote the lane name and pid into
  `holder`. No device run was taken all lane, so `gpu_lock` was never needed and `gate.py` contains
  zero `gpu_lock` references (`grep -c gpu_lock prototype/gdn/gate.py` = 0).
- min 18 — read the oracle's `scan_chunked`/`scan_exact`/`SCAN_WIDTH_STAGES`, confirmed from the
  manifest that decode0/1/2 are a cached chain off `prefill.state`, and that the band's own probe
  population covers `layer{0,1}/pinned/decode{0,1,2}`.
- min 20 — `gdn_state_dot` + the `--broken` xpose control + plan `v2 carry=` + the state-read
  self-test check (23 checks; both honest arms PASS, both control builds FAIL). `--only band` still
  reproduces slice 1's 12/14 rows, `q_hat` the only red, band untouched.
- min 22 — `--only carry` first verdict: 84/84 live-state rows ACCEPTED, compiled pair identical on
  the carried plan, sabotage invisible at zero state and visible on both live plans. One inverted
  predicate found and fixed (a sabotage row was being reported as a failure for being REJECTED).
- min 24 — controls extended to the live-state cases and to three bad-plan shapes; determinism
  section added. `--only three-arm,band,carry,exact,controls,determinism` → findings = the two q_hat
  rows and nothing else.
- min 26 — full gate three times (the third from a cleaned `work/`): rc=1 with exactly the two q_hat
  findings each time; the 158 STAGE lines carry sha256 `3951cdb1d97df92c…` in ALL FIVE sets — the
  three whole-gate runs and the determinism section's two nested re-runs. Evidence copied to
  `evidence/`; `work/*.bin` cleaned; `work/asan.echo` restored from HEAD; halogen lock released.

## Checkpoint log (slice 1, continued)

- min 68 — `gate.py` in progress.
- min 78 — first verdict: 12/14 banded rows ACCEPTED, `q_hat` one ULP over. ASan build used to find
  a heap overflow (the scratch formula undercounted `kcd` by 262,144 words — a silent overrun that
  no band would ever have shown).
- min 85 — exact checks 16/16; controls all firing; the accepted `norm_weight` case reproduced at
  exactly the oracle's recorded numbers; oracle CLI arm `CROSS-FORM PASS`.
- min 90 — evidence copied to `evidence/`, 1.2 GB of regenerable `work/*.bin` deleted, halogen
  stream lock released. Nothing staged; nothing outside `prototype/gdn/` touched.

## Checkpoint log (slice 1, first entries)

- min 38 — inventory done; band digest read through `load_band()` = `27a68c7e82c9aa05…`, matches
  the authoritative value; the seven banded stages confirmed present in the band file.
- min 52 — `build.sh` builds both arms clean (`g++ -O2 -ffp-contract=off` host twin, `hipcc
  -x hip --offload-arch=gfx1151 -O3` device TU). No device touched; compile is CPU work.
- min 66 — `--selftest` on both arms: **PASS, 21 checks each**. Getting there found four real
  defects, all in the dead lane's never-executed code:
  * `gdn_bf16_of_f32` tested `((u>>23) & 0x1ff) == 0x1ff`, which can never fire for a **positive**
    float (u>>23 ≤ 0xff), so a positive signalling NaN was converted to **+inf**. Mask is `0xff`.
    This is the exact corruption §04b.3 exists to prevent, arriving through the cast.
  * the bf16 expectation table was hand-typed and four rows were wrong (65504 → 0x4780 not 0x477e;
    1e-8 → 0x322c not 0x3243; a NaN round-trip asserted identity where quieting is correct; the
    hand solve residual had a sign flip). `bf16_vectors.py` now DERIVES the table from torch and
    checks the C rule against torch over 200k random f32 patterns: **0 mismatches**.
  * `gdn_item_cum` did `(float)g_decay[i]` — an integer-to-float conversion of an f32 *bit image*
    instead of a bitcast. Now goes through `gdn_load`.
  * the emitted-stage table (`gdn_stage_name`) is asserted against the oracle's seven names inside
    the self-test, because a rename there silently detaches every band verdict from its stage.
- min 68 — `gate.py` (the driver) in progress. Nothing has produced a numerics verdict yet; that is
  the next thing.

## Checkpoint log (slice 3)

- min 8-20 — read the four pitfall files, this file, `gate.py`, `gdn_arm.h`, `gdn_case.h`,
  `gdn_kernels.h`, `speccheck/gdn_ref.py`. Established the two facts the whole slice turns on, both
  by reading `gdn_ref.py` rather than by guessing: `probe_arms_for("decode","pinned")` returns
  `recurrent,reassoc,pinned` (so `chunked|pinned` has no probe record at a decode pass), and
  `dumps[].shape[1]` is 32/1 against `geometry.chunk_width=64` (so no pinned pass is multi-chunk).
  Scratch probes for both live in `work/` during the lane and are deleted at the end; the numbers are
  here instead, per `pitfalls/60-docs-and-citations.md`.
- min 20-24 — first re-chunking attempt measured in a scratch driver and it was WRONG in a way worth
  recording: the token axis was sliced on axis 1 because that is where a token axis usually is, but
  `q_hat`/`k_hat` are `[1,32,32,128]` with `h_v == T == 32`, so axis 1 is heads there. The symptom
  was a `scan_out` row that agreed with the pinned bytes and a `q_hat` row that differed from the
  honest arm by 65,532 of 65,536 words. The fix derives the axis per stage from
  `gdn_ref.HEAD_AXIS` plus the pass T, refuses on an ambiguous axis, and adds a zero-tolerance
  arm-vs-arm row that is now the thing that would catch a recurrence (`split_row_local_stages_
  bit_identical`, measured at ndiff=0 on all 8 rows).
- min 24-38 — `Licence`/`Checks` objects, LICENCE citation on all 14 band rows + all 84 carry rows +
  the split rows, `GATE PASS` made unreachable while any banded row is uncredited.
- min 38-46 — `chunkcarry` section: census, refusal, witness, split carry, zero-state sabotage.
  The first three runs died on real defects (a 2-tuple unpack over `recs`, a `fails` name out of
  scope, a tally whose mark advanced late enough to double-count 84 rows as 42) — the third was
  caught because the check comparing cites to its derived denominator went RED, which is the reason
  it exists.
- min 46-52 — witnesses for the licence census (poisoned in-memory band table, dropped row). Full
  gate green except the two known `q_hat` findings; `findings=2`, same two rows as slice 2, `q_hat`
  still RED on purpose and the band file untouched.
- min 52 — this file updated, before the final full-gate run finished.
- min 60 — final full gate: `GATE FAIL findings=2 uncredited_banded_rows=102`, `DETERMINISM
  here=178 run1=178 run2=178 run1_eq_run2=1 both_eq_this_run=1 verdict=IDENTICAL` (the determinism
  children now re-run `chunkcarry` too, so the 20 new rows are covered), `ORACLE rc=0 lines=2696`, `named_checks=15 named_checks_red=0`, `stage_rows=14 split_rows=18 split_bit_rows=8
  chunk_census_passes=8`. Evidence: `evidence/gdn-slice3-gate-2026-10-26.log`,
  `evidence/gdn-slice3-chunkcarry-only-2026-10-26.log`, and `evidence/gdn-slice3-stage-line.diff`
  (which proves the 158 pre-existing STAGE rows are byte-identical and only the 20 new rows were
  added). Scratch cleaned from `work/`; nothing staged; nothing outside `prototype/gdn/` written;
  no device run and no timer read, so no timing appears anywhere in this file or in the logs.
