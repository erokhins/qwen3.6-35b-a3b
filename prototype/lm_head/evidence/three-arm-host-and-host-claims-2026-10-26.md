# evidence/three-arm-host-and-host-claims-2026-10-26.md — the host-side three-arm agreement, and the
# two claims this lane exists for (disjointness, amortisation), witnessed firing AND passing.

Written 2026-10-26 (box clock `2026-10-25 22:11`). CPU only: no iGPU, no `gpu_lock`, no `halogen`,
no timer, and **no `ms` / `tok/s` / `GB/s` figure anywhere in this file**. Where an arm is the device
binary, it is the device binary's **host** code and says so on its own line (`hip_calls=0`).

## commands and exit codes

| command | rc | printed |
|---|---|---|
| `... .venv/bin/python three_arm.py` | 0 | `RESULT PASS lm_head three-arm host agreement groups=48 field_rows=13 compiled_pair_bit_identical=1 oracle_inside_band=1` |
| `... python gate.py --check-only` | 0 | `CHECKS ran=1513 failed=0` |
| `... python gate.py --selftest` | 0 | `CHECKS ran=31 failed=0` |
| `./host_probe --selftest` / `./lm_head --selftest` | 0 / 0 | `checks=66 failures=0` each (identical roster) |
| `./build.sh` | 0 | "compile only; no device was touched and no timing taken" |

Full output: `work/three-arm.log` (61 records: 48 `THREEARM` word groups + 13 `THREEARMF` report
fields + 2 `THREEARMRANK` + 1 `THREEARMAGG`).

## the three arms

| arm | binary / file | compiler | device touched |
|---|---|---|---|
| `host_twin` | `host_probe` | g++ `-O2`, no ROCm on the link line | no |
| `hip_hostpath` | `lm_head --host-path` (new this run) | **hipcc**, `--offload-arch=gfx1151` — the device translation unit, running `lmh_run_case_host` | **no**: `RUN arm=device-hostpath cases=192 skipped=0 items=1186 out_floats=27488 hip_calls=0` |
| `oracle` | `work/expect.f32` from `oracle_arm.py` | numpy (venv), bf16 by `<<16`, Q8_0 through `tools/gguf_reader.py`'s audited dequant, `math.fsum` over exact products | no |

The third arm is what the lane was missing: without it the "three-arm" agreement was g++ vs numpy, and
the claim being made is about what **the device compiler** does to this arithmetic. `three_arm.py`
refuses to run if that arm does not announce `arm=device-hostpath` **and** `hip_calls=0`, so a
mislabelled arm cannot be mistaken for the device.

## the metric policy, and why (this is the section the task asked for)

Load-bearing: **`relL2`** and **`relL2_bandnorm = sqrt(mean((|a-b|/band)^2))`** (the band is the one in
`band-derivation-2026-10-26.md`), and `ndiff` over bit images. Reported, never load-bearing:
**`maxabs`**. Printed alongside, never load-bearing: **`cos`**. The line carries that policy itself:
`THREEARMAGG ... load_bearing=relL2_bandnorm+ndiff maxabs_load_bearing=0 cos_load_bearing=0`.

Two measurements in this run make the policy concrete rather than rhetorical.

1. **`cos` cannot see 1263 differing words.** Against the oracle, `all_honest` reads
   `n=1884 bit_identical=0 ndiff=1263 maxabs=2.86102e-06 relL2=2.44258e-10 relL2_bandnorm=0.0001627
   worst_word=0.001646 cos=1`. `cos` prints `1` (to 9 digits) while **67 % of the words differ in
   their bit image**. A cosine-similarity gate would call that a pass without ever looking at `ndiff`.
2. **`maxabs` ranks the population differently from the primary, i.e. it would triage to the wrong
   table.** `THREEARMRANK pair=host_twin_vs_oracle relL2_worst_group=table2_enc1
   maxabs_worst_group=all_honest ranking_agrees=0 note=maxabs_is_reported_only` (same for the hipcc
   arm). `table7_enc1` carries the largest absolute deviation (`maxabs=2.86102e-06`) while
   `table2_enc1` — a Q8_0 table, `maxabs=1.2666e-07`, 23× *smaller* in absolute terms — is the worst
   group on the band-normalised primary (`0.0002346` vs `7.179e-05`). A maxabs-ordered triage spends
   the day on the wrong family.
3. **The precedent, copied in rather than cited out** (so this file is readable without opening the
   sibling lane): in `prototype/shared_expert/work/derive.log` (run 2026-10-26, same box) the honest
   row map and the wrong row map were scored against the pinned eager output —
   `LAYOUTRT candidate=gate_first shared_raw_vs_pinned_eager maxabs=0.96875 cos=0.785072923` and
   `LAYOUTRT candidate=up_first shared_raw_vs_pinned_eager maxabs=0.961426 cos=0.707297027`, i.e.
   **the wrong candidate was *nearer* on maxabs (0.961426 vs 0.96875)**. That lane's own conclusion,
   `the row map the kernel implements is not distinguishable from a wrong one: honest maxabs=0.96875,
   closest wrong=0.961426`, is why an absolute metric is not allowed to decide anything here: an
   absolute metric that cannot see what you are claiming is not conservative, it is blind.
4. Honest negative result: `maxabs_blind_words=0` in all 48 groups of *this* population — no word here
   has a >0.1 % relative deviation that hides below 1 % of its group's worst. So the blindness argument
   in this lane rests on (1), (2) and the sibling precedent in (3), not on a local example.

## the per-field table (from `work/three-arm.log`)

Compiled pair (`host_twin` vs `hip_hostpath`) — the compiler-comparison claim, zero tolerance:

| group | n | bit_identical | ndiff | maxabs | relL2 |
|---|---|---|---|---|---|
| all_honest | 1884 | **1** | 0 | 0 | 0 |
| table0_enc0 … table7_enc1 (8 groups) | 976/180/200/180/64/80/104/100 | **1** | 0 | 0 | 0 |
| assoc_seq / assoc_rev / assoc_split8 | 1724/128/32 | **1** | 0 | 0 | 0 |
| path_dense / path_gather | 1592/292 | **1** | 0 | 0 | 0 |
| batch_batched / batch_perrow | 1372/512 | **1** | 0 | 0 | 0 |
| report fields `bytes loads xorb vsum bsum lo hi cross peer_hits lanes bover oxor osum` | 192 each | **1** | 0 | 0 | 0 |

`hipcc` and `g++` produced the same bit images for every word and every traced counter, on all 192
cases. (This is *not* the device claim; it is the same-source/two-compilers claim, and it is the
closest thing available to the device claim without touching the iGPU.)

Each compiled arm vs the oracle (the arm rows are identical between the two compiled arms, which is
itself the row above):

| group | n | ndiff | maxabs | relL2 | relL2_bandnorm | worst_word | cos |
|---|---|---|---|---|---|---|---|
| all_honest | 1884 | 1263 | 2.86102e-06 | 2.44258e-10 | 1.627e-04 | 1.646e-03 | 1 |
| table0_enc0 | 976 | 794 | 7.45058e-08 | 3.9815e-08 | 1.496e-04 | 9.653e-04 | 1 |
| table1_enc0 | 180 | 85 | 1.67638e-08 | 2.20711e-08 | 2.102e-04 | 8.592e-04 | 1 |
| table2_enc1 (Q8_0) | 200 | 115 | 1.2666e-07 | 2.88515e-08 | **2.346e-04** | 1.646e-03 | 1 |
| table3_enc1 (Q8_0) | 180 | 92 | 2.98023e-08 | 2.58864e-08 | 1.964e-04 | 1.187e-03 | 1 |
| table4_enc0 (tied) | 64 | 56 | 2.98023e-08 | 3.28188e-07 | 1.040e-04 | 3.665e-04 | 1 |
| table5_enc0 (tied) | 80 | 70 | 2.98023e-08 | 3.30779e-07 | 1.065e-04 | 3.665e-04 | 1 |
| table6_enc0 (synth) | 104 | **0** | 0 | 0 | 0 | 0 | 1 |
| table7_enc1 (synth Q8_0) | 100 | 51 | 2.86102e-06 | 3.43314e-10 | 7.179e-05 | 4.558e-04 | 1 |
| assoc_seq / rev / split8 | 1724/128/32 | 1186/55/22 | 2.86102e-06 / 1.2666e-07 / 9.31323e-09 | 2.44113e-10 / 1.64157e-08 / 1.45197e-07 | 1.576e-04 / 2.328e-04 / 5.623e-05 | 9.653e-04 / 1.646e-03 / 1.335e-04 | 1 |
| path_dense / path_gather | 1592/292 | 1041/222 | 9.53674e-07 / 2.86102e-06 | 1.26623e-10 / 2.03164e-07 | 1.628e-04 / 1.621e-04 | 1.646e-03 / 8.592e-04 | 1 |
| batch_batched / batch_perrow | 1372/512 | 821/442 | 2.86102e-06 / 9.53674e-07 | 2.27399e-10 / 1.22229e-07 | 1.604e-04 / 1.687e-04 | 1.646e-03 / 9.653e-04 | 1 |

Verdict rule actually applied: the compiled pair must be `bit_identical` on every group and every
report field; the oracle pairs must satisfy `relL2_bandnorm <= 1` **and** `worst_word <= 1` (per-word
deviation inside its own derived band) on every group. `maxabs` and `cos` are printed and consulted by
nothing. 61 records printed, 61 parsed back (`THREEARMAGG records_printed=61`), `words_compared=28260`.

## disjointness — both colours witnessed (`gate.py --check-only`, tag=host)

```
DISJOINT tag=host witness untied_cases=175 tied_cases=17 tied_peer_nonzero=17 tied_cross_max=0
                      untied_cross_arms=4 arms_with_cross>0=4
CHECK name=host no untied case reads outside its own table unless its mode says it must ok=1 n=192 no stray reads
CHECK name=host the tied control's peer read is witnessed non-zero (peer_hits is not vacuous) ok=1 n=17 tied cases=17 with peer_hits>0=17
CHECK name=host the cross-table-read witness moves `cross` on every untied arm ok=1 n=4 arms=[(97, 4, 4), (120, 4, 4), (145, 4, 4), (169, 4, 4)]
DISJOINT tag=host real_tables=5 overlaps=0 lmhead_files=['model-00026-of-00026.safetensors'] embd_files=['model-00001-of-00026.safetensors'] same_file=0
CHECK name=host the real tables occupy disjoint slab ranges ok=1 n=5 disjoint
CHECK name=host lm_head and the embeddings come from DIFFERENT shard files (the untiedness) ok=1 n=2 ...
```

The untiedness is the shard files, and they stay in the print: `lm_head.weight` comes from
`models/qwen36-src/model-00026-of-00026.safetensors`, `model.language_model.embed_tokens.weight` from
`model-00001-of-00026.safetensors`, `same_file=0`, and their slab ranges do not overlap
(`real_tables=5 overlaps=0`). Peer extents: tables 0↔1 (bf16 lm_head ↔ embeddings), 2↔3 (Q8_0 pair),
6↔7 (synth pair) are **disjoint**; only tables 4/5 share a range (`base=401408, nbytes=131072`), which
is the tied control.

* **`peer_hits == 0` for every untied case** — 175 untied cases, none with a stray `peer_hits` or
  `cross` outside the three modes whose own rule sanctions a foreign range.
* **`peer_hits > 0` for the tied control** — all 17 tied cases have `peer_hits == loads` (4 for an
  honest tied case, 8 for the cross-read witness on a tied pair) and `cross == 0`
  (`tied_cross_max=0`). The counter is *witnessed moving*, not merely non-contradicted: the tracer
  (`lm_head_kernels.h` §`lmh_trace_range`) tests the own range before the peer range, which is why a
  tied peer read is a byte read and not a cross.
* **`cross` fired by the witness mode** — cases 97, 120, 145, 169 (untied bf16/q8 × lm_head/embeddings)
  each carry `cross=4`, `peer_hits=4` where their twins carry 0/0, i.e. `cross == n_out*nlanes` as the
  extents say. The tied arm of the same mode (case 173) is the must-NOT-move arm: `cross=0/0`,
  `peer_hits=8/8`.

## amortisation — no timer anywhere (`gate.py --check-only`, tag=host)

`bytes` independent of `bcount` when batched, `bcount×` when per-row, and the re-looping kernel caught
on bytes alone:

```
CHECK name=host batching amortises the weight read by exactly bcount ok=1 n=10 all groups exact
AMORTISE tag=host group=62 rows=32 bcount=4  batched_bytes=131072 perrow_bytes_sum=524288  ratio=4.00 geometry=131072
AMORTISE tag=host group=67 rows=32 bcount=8  batched_bytes=131072 perrow_bytes_sum=1048576 ratio=8.00 geometry=131072
CHECK name=host addressed bytes are independent of bcount for the same row set ok=1 n=5 flat
AMORTISE tag=host shape=table0/rows0+n4 bytes=16384 constant across bcounts [1, 2, 4]
AMORTISE tag=host shape=table0/rows0+n32 bytes=131072 constant across bcounts [4, 8]
AMORTISE tag=host shape=table1/rows0+n4 bytes=16384 constant across bcounts [1, 4]
AMORTISE tag=host shape=table2/rows0+n4 bytes=8704  constant across bcounts [1, 4]     (Q8_0)
AMORTISE tag=host shape=table3/rows0+n4 bytes=8704  constant across bcounts [1, 4]     (Q8_0)
CHECK name=host the re-looping batched kernel is caught on bytes alone (words bit-identical) ok=1 n=4 witnessed=4/4; clean
AMORTISE tag=host reloops case=95  bcount=4 bytes=65536 twin_bytes=16384 ratio=4 words_identical=1 xorb_unchanged=1 vsum_unchanged=1
AMORTISE tag=host reloops case=118 bcount=4 bytes=34816 twin_bytes=8704  ratio=4 words_identical=1 xorb_unchanged=1 vsum_unchanged=1
AMORTISE tag=host reloops case=143 bcount=4 bytes=65536 twin_bytes=16384 ratio=4 words_identical=1 xorb_unchanged=1 vsum_unchanged=1
AMORTISE tag=host reloops case=167 bcount=4 bytes=34816 twin_bytes=8704  ratio=4 words_identical=1 xorb_unchanged=1 vsum_unchanged=1
```

* Batched `bytes` equals the row-set geometry (`rows × row_b`), and does not move when `bcount` goes
  1→2→4 or 4→8 for the same row set — five shapes, `flat`.
* The sum of the per-row arms' `bytes` is exactly `bcount ×` the batched case's, in all 10 complete
  families (ratio 4.00 and 8.00); an incomplete family is a finding, not a skipped group.
* **`BATCH_RELOOPS` is caught on bytes alone**: 4 cases, words bit-identical to the twin
  (`ndiff == 0`), `xorb` and `vsum` unchanged, `bytes` and `loads` exactly `bcount×`. The immunity half
  already lives in the driver roster (`batch-reloops-immune-at-bcount-1`, both binaries, 66 checks
  each), so the two halves do not overlap: the roster owns "at `bcount=1` the injection is invisible by
  construction and must be classed IMMUNE", this check owns "at `bcount>1` only the byte counter can
  see it, and it does".

## what had to change in the judge for these claims to be checkable (nothing was loosened)

Three of the checks above were **incapable of passing** before this run, and the reports that read them
as "the kernel is wrong" were reading a broken instrument:

1. `every injection that must fire fired` recorded the untied cross-table-read arms under the
   **negation** of their own claim (`cross == twin.cross and peer_hits == twin.peer_hits`), so the four
   arms that correctly reported `cross=4/4 peer_hits=4/4` were logged `fired=0`. One predicate now
   feeds both the verdict and the roll-up.
2. `every derived-immune case really is immune` demanded of the *tied* cross-read arm that `peer_hits`
   stand still, which the tracer cannot do (`peer_hits == loads` for a tied pair). The tied claim is
   now what the geometry says: `cross` unchanged **and** `peer_hits == loads`.
3. `no case reads outside its own table unless its mode says it must` demanded `peer_hits == 0` of the
   tied control too, so cases 32-35 (`honest-bf16-tied-a-*`, `peer_hits=4`) made it permanently red. It
   is now role-aware and *strict in both directions*, with the two positive witnesses added so a
   counter that never moves cannot satisfy it.
4. `injection denominator matches the plan` counted list entries, not cases (79 judged against 77
   declared, because the gather branch and the FIRES branch both register the same case). It now
   compares **sets**: `never_judged` and `judged_but_not_injection` must both be empty.
5. `--control scale-q8` exited 0 while printing 103 red inner checks (a control a driver walks past),
   and its `vsum` predicate's second conjunct (`moved_v <= n_traced`) was true by construction. It now
   exits 1 with a named refusal, and the predicate is per-case with a geometry-derived exemption set
   `{119, 120, 122, 176}` — cases whose read-set leaves the tampered table (`cross > 0`) or whose
   injection throws the delta away (`Q8_DROP_SCALE`). 31 of 31 non-exempt traced cases moved `vsum`.

## NOT ESTABLISHED here

* any device claim. `hip_hostpath` is hipcc's codegen on the CPU; the iGPU arm (`three_way` in
  `gate.py`) still requires the main agent's exclusive window.
* that `relL2` is *sufficient* for the Q8_0 families — see the `gamma_{2K-1}` finding in
  `band-derivation-2026-10-26.md`.
* that `cos` is a safe metric anywhere: in this population it printed 1 with 1263 words apart, which is
  a refutation of it as a gate, not a support for it.
