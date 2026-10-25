# prototype/shared_expert — results

Correctness only. Nothing here is a measurement: no `ms`, no `tok/s`, no `GB/s`, no warm loop. The
wrapper's `WALL started=… finished=… (lock record only; no timing taken)` line is the lock record
`AGENTS.md` asks a device run to carry, and `no timing taken` is printed by the driver itself.

## Verdicts, verbatim

```
host witnesses:  RESULT PASS (host witnesses only): 12 checks, three arms exercised, no timing taken
host gate:       RESULT PASS (host gate): 67 checks, three arms exercised, no timing taken
device gate:     RESULT PASS (device gate): 101 checks, three arms exercised, no timing taken

device gate BEFORE this task, from evidence/shared-expert-device-2026-10-26-before-band-and-ownership:
device gate:     RESULT FAIL (device gate): 2 of 79 checks failed: every counted field stays inside
                 the band the floor probe measured; every counted field stays inside the band the
                 floor probe measured
```

101 checks, 0 failing. The device gate was **2 of 79 red** before this task — both failures the same
item, `case 2 field act: 3 ULP exceeds the probe's 2`, the one honest `UNVERIFIED by probe domain` item
recorded below — and the 22 checks this task added are all green with their controls shown firing. The
band that moved did so because a probe now measures the domain the field lives in (act probe section
below); nothing was widened by hand, and the refusal that moved is exercised on every run by two named
controls (`the band refusal still fires …`, `the stage-N (…) ownership control fires …`). The new checks
are: 6 from the act probe (band derivation, coverage, population identity, triangle, census identity,
build), 8 per-stage ownership controls, 8 per-arm ownership coverage checks (4 per arm) and 2
band-refusal control checks. Full per-check table for the older 76-check run: `evidence/check-
enumeration-2026-10-26.tsv` (derived by `work/enumerate_checks.py`; the total has moved twice since —
the tail fix and this task — so that table is a snapshot of the run it names, not a live total).

Transcripts: `evidence/shared-expert-device-2026-10-26.log` (the device run, tee'd, includes the width
anchor and every control), `evidence/shared-expert-host-2026-10-26.log` (host witnesses, no device, no
lock), `evidence/floor-probe-act-2026-10-26.log` (the act floor probe's own output),
`evidence/device-ownership-stages-2026-10-26.log` (the four per-stage lines the device arm printed),
`evidence/control-ownership-{device,host}-stage{1,2,3,4}-2026-10-26.log` (each stage's leaked-write
control, shown firing 43 times), `evidence/shared-expert-device-2026-10-26-before-band-and-ownership.log`
(the 2-of-79 transcript this task replaced).

## What the lane had to fix before the device arm could run at all

Four predicates were dead or the fixture was not the claimed width. Each is recorded with the witness
that shows it can fire.

| # | defect | why it was dead | fix | witness that the replacement bites |
|---|---|---|---|---|
| 1 | `q8_quantize` packed `scale = amax/512` and called it Q8_0 | not a dead check — a wrong premise: with the normal ±127 clamp this clips every element above 24.8 % of its block max | pack `f16(amax/127)` with the step taken from the f16-rounded scale, `±127` clamp, per `tools/gguf_reader.py §_qref_q8_0` and `tools/moe_stats.py §q8_0_proxy` | `--q8-rule 512` control prints `clamp_eq127=551575 clamp_bound_from_blockmax=34778 relL2_vs_dump=0.5911241` then refuses, rc=1 (`evidence/control-q8-rule-512-2026-10-26.log`) |
| 2 | `route_discriminates`: `worst_wrong <= honest → refuse` | `layout_rows("pair_swap", j) == layout_rows("gate_first", j)`, so the minimum over "wrong" candidates was always bit-identical to honest: `nearest_wrong_ratio=1.0x` for **every possible build**; `--mode derive` exited non-zero and held the device arm hostage | score each candidate on the route its claim spells (pair_swap puts silu on the other half), print the row-map-only reading as a blindness witness, rank on relL2, and make the *claim* an input so a wrong claim is refused | honest wins at `nearest_wrong_ratio=53.661x`; `--claim-candidate up_first` refused rc=1 (`evidence/control-claim-up-first-2026-10-26.log`); an intermediate version that ranked pair_swap on its row map alone wrongly passed `--claim-candidate pair_swap` and is what forced the semantic-route scoring |
| 3 | `span_diff(words, a, b, off, n)` accepted `off` and never used it | every field-level comparison silently compared from element 0 of the case, so the pair_swap blindness check printed `gate=0 up=0 act=0` while 507 of 512 act words in fact differed | honour `off`; add `buffers_diff` for the two-buffer comparisons that were being (mis)passed to it, and fix `counted_fields` shadowing its own function name in `cross_arm_gate` (UnboundLocalError on the second case) | blindness now prints `case 6 gate=0 up=0 act=507` |
| 4 | two unsatisfiable predicates | `diverging * 2 < n` with `n == 1` (a 1-element `prob` field can never be "isolated"), and `bad == 0` demanded of cases whose own header declares a counted field | a 1-element field is judged by the band and says `isolation=not-defined-for-a-1-element-field`; the arm's `bad` must equal the driver's counted-field divergence per case (two sites, one truth) instead of being told to be zero | case 1 `bad=1` now reconciles against the driver's own `1/1` prob divergence; a divergence in a declared-exact field still trips the exact-field check |

Device harness crash (first-ever device execution, rc=-6 SIGABRT, `malloc(): invalid size
(unsorted)`): `shared_expert.hip` allocated `calloc(p.n_c, 4)` — one int32 array — and then wrote the
second array it uses (`bhi = blo + n_c`) past the end. Fixed to `calloc(2 * n_c, sizeof(int32_t))`.
The host twin does not use that layout, which is why the host arm was clean and the device arm aborted.

## The Q8_0 width claim, anchored to the artifact

Printed by `--mode derive` and echoed into the device transcript:

```
Q8RULEPUB family=gate  tensor=blk.0.ffn_gate_shexp.weight type=Q8_0 rule=f16(amax/127)*int8 relL2_vs_pinned_hf=0.006126833 maxabs=0.00079155
Q8RULEPUB family=up    … relL2_vs_pinned_hf=0.005874989 maxabs=0.000524521
Q8RULEPUB family=down  … relL2_vs_pinned_hf=0.006036182 maxabs=0.00118637
Q8RULE family=gate  divisor=127 clamp_eq127=33138 clamp_bound_from_blockmax=34778 elements_eq_own_blockmax=33121 clamp_rate=0.031603 relL2_vs_dump=0.006123959 relL2_published=0.006126833 f32_vs_f64_inexact=0 max_sig_bits=15 max_sig_bits_f32_has=24
```

* The width was read from the file, never typed: `WIDTH role=gate … dtype=Q8_0` for gate/up/down and
  `dtype=F32` for `ffn_gate_inp_shexp` (policy A: shared expert Q8_0, gate row exact).
* The published tensor decodes to the pinned bf16 dump at relL2 **0.00613**; this lane's own pack now
  sits at **0.00612**. Before the fix it sat at **0.58983** — 96.3× worse — with **551575/1048576**
  elements on the clamp where geometry allows ~1 in 32. Negative control kept at
  `evidence/control-q8-rule-512-2026-10-26.log`; those numbers are proof the check bites, not a verdict.
* **Clamp rate is now a check, with a geometrically derived bound**: the count of elements equal to
  their own block max (`elements_eq_own_blockmax=33121`, i.e. 3.159 %, where a block max in a
  32-element block predicts 1/32 = 3.125 %) × 1.05 slack for f16 step rounding = `clamp_bound…=34778`.
  The wrong divisor lands at 52.6 % and is refused.
* **The "≤11+7 significand bits, EXACT in f32" header claim was re-measured, not rewritten**:
  `f32_vs_f64_inexact=0` over all three families and `max_sig_bits=15` against the 24 f32 has. The
  exactness claim holds. What was false before was the *scale*, not the exactness.

## The four questions

### (a) the fused `gate|up` layout — ESTABLISHED, by candidate distinction, with the control caught

The published file stores `ffn_gate_shexp` and `ffn_up_shexp` as **two** tensors
(`WIDTH fused_gate_up_in_file=no`), so the `[2I,H]` slab is this project's assembly and the row map is
a question, not a fact. Answered against the pinned HF eager `shared_raw`, on relL2 (see the metric
note below), 32 pinned rows, block layer0:

```
LAYOUTRT candidate=gate_first             relL2=0.01004416 cos=0.999952078 maxabs=0.00976562  (row map (0,512)..(511,1023))
LAYOUTRT candidate=up_first               relL2=0.538981   cos=0.848566592 maxabs=0.527344   (row map (512,0)..(1023,511))
LAYOUTRT candidate=row_interleaved        relL2=1.272461   cos=0.010499730 maxabs=1.25488    (row map (0,1)..(1022,1023))
LAYOUTRT candidate=pair_swap              relL2=0.538981   cos=0.848566592 maxabs=0.527344   (row map (0,512)..(511,1023) silu_on=up)
LAYOUTRT candidate=pair_swap_rowmap_only  relL2=0.01004416 … (the reading byte-identical to gate_first)
LAYOUTRT candidate=row_shift1_control     relL2=1.391529   cos=-0.045516506
LAYOUTRT candidate=col_shift1_control     relL2=1.355625   cos=0.000337727
LAYOUTRT winner=gate_first claimed=gate_first metric=relL2 nearest_wrong_ratio=53.661x control_also_worse=yes
```

* **Which candidates match**: only `gate_first`. Every wrong candidate is ≥53.7× further from the
  pinned eager `shared_raw`, and the two off-by-one controls (one row early, one row late) are worse
  still — so a layout bug and an off-by-one are distinguishable, which is what the control is for.
* **Wrong-candidate control caught**: `--claim-candidate up_first` is REFUSED with rc=1
  (`evidence/control-claim-up-first-2026-10-26.log`), and `--claim-candidate pair_swap` is refused by
  the silu-side scoring after the intermediate version was caught passing it.
* Independent second ruler: nearest-row preservation over 128 HF rows —
  `LAYOUTROW candidate=gate_first rows_nearest_own_hf_row=128/128`, `up_first 0/128`,
  `row_interleaved 1/128`, margin nearest-vs-runner-up min 91.712.
* **Declared blindness**: `pair_swap` shares `gate_first`'s row map, so no nearest-ROW measure can see
  it; it is separated by which half gets silu (`act` differs in 507 of 512 words; the driver asserts
  `gate=0 up=0 act=507` as a fact rather than trusting a pass-by-not-looking).
* **Why the metric matters, recorded as a trap**: under the mis-scaled pack the honest candidate was
  the *loser* on maxabs (0.96875 vs up_first's 0.961426, ratio 0.992×) while relL2 and cos already
  disagreed in the right direction. An absolute metric that cannot see the thing you are claiming is
  not conservative, it is blind. relL2 is the ruler here, cos is the second opinion, and
  `maxabs_is_load_bearing=no` is printed on the winner line.

### (b) `shared_expert_gate` — SCALAR per token. ESTABLISHED from the installed reference

Installed reference, `transformers` 5.19.0,
`.venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`:

* line 900 (`Qwen3_5MoeSparseMoeBlock.__init__`):
  `self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)` — `out_features=1`,
  so one logit per token, not one per channel.
* line 909: `shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) *
  shared_expert_output` — a `[T,1]` sigmoid broadcast over the `H` outputs.

The kernel is testable both ways and both are exercised: the plan carries `sgate=scalar` (honest) and
`sgate=perchan` (`perchan_gate` case), and the per-channel form **poisons its `prob` field on purpose**
so a reader that assumed per-channel cannot silently pass. The scalar reading is what reproduces the
pinned `shared_gate_prob` bit-for-bit (`PINNED block=… shared_gate_prob=32/32` for all four blocks).

### (c) multiply before or after the cast — the reference casts BEFORE the multiply; ESTABLISHED

Reference line 909 computes `F.sigmoid(gate(h))` at the module dtype (bf16 under a bf16 model) and
only then multiplies, i.e. `p = bf16(sigmoid(bf16(t)))`, `out = bf16(p * raw)` — cast **before**.
The wrong spelling (`out = bf16(sigmoid(bf16(t)) * raw)` with the product in f32) is kept as a live
arm, `cast_after`, not a paragraph.

The comparator separates them: for case 1 every field is bit-identical to the honest case **except**
`out` (596 of 2048 words differ) and `prob` (1 of 1, the f32-carrying form). Measured with
`work/dbg_case_fields.py`. So the claim is not "both spellings look the same": the field layout makes
the spelling visible, and the census prints it.

### (d) host ↔ device divergence — attributed, with the transcendental band now measured in its own domain

1. **`raw[2047]` / `out[2047]` — the last output column, on the device only. CLOSED by the committed
   tail fix (`980fac7`), recorded here because this file predates it.** That revision root-caused the
   moving divergence count as a leaked sentinel item — a thread computing column `-1` — and its
   before/after prediction logs are `evidence/tail-ownership-prediction-{prefix,postfix}-2026-10-26.log`.
   In the current transcript the device arm is bit-identical to the oracle in every declared field
   (`256 fields / 241706 words exact`), host and device agree bit-for-bit with only the two counted f32
   fields exempt, and the whole-buffer cross-arm difference is 8 words — all of them inside the counted
   fields. The leaked-sentinel defect class is now a standing probe, not a fixed-and-forgotten bug: see
   “Write ownership, per stage”.
2. **`reads_sg = 34816` where geometry says 2048 — attributed by construction.** The prob stage
   launches `1 + H/THREADS = 17` blocks per case and every one of them counts the full `sg` row, so
   the device reads the 2048-element shared-expert-gate row 17 times per case. The counter is telling
   the truth; the stage is doing redundant work. That is a real per-token cost finding for whoever
   prices the block, not a counter bug.
3. **`f32_act` `act` field at 3 ULP — CLOSED by measurement, not by widening.** The band used to come
   from `work/floor_probe_exp.c` + `work/floor_probe_exp_np.py`, which measure `expf`/`sigmoid`
   disagreement between glibc and numpy over **every bf16 input** (34194 values, `max_ulp=2`,
   `sigmoid_diff_after_bf16_store=0`). That domain does not contain this field: `act` carries
   `silu(gate)·up`, so the transcendental sees an **f32 dot product**, not a bf16 value. The band was
   therefore re-measured where the field lives, in its own probe `work/floor_probe_act.c` +
   `work/floor_probe_act_np.py`, and the driver now maps each counted field to the probe that covers it
   (domain `bf16_input` → the exp probe, domain `f32_dot` → the act probe; the header declares the
   domain, there is no fallback).

   *Input domain, and how it was derived*: the population is **every f32 `(gate, up)` dot pair the
   fixture carries** — 43 cases × 512 columns = 22016 pairs, of which the 512 pairs belonging to the
   `f32_act` case are the judged subset. Those pairs are the exact f32 products the field is built from,
   taken from the oracle's own expect dump, and the probe refuses to measure anything until its emitted
   table is asserted bit-identical to the arm's words: `FLOOREXP_ACT_C pairs_read=22016 pairs_emitted=
   22016 malformed=0 hull_gate=C08F5D21..40437671` (−4.48011827 .. 3.05410409) and `FLOOREXP_ACT_POP
   cases_probed=43 pairs=22016 judged_pairs=512 arm_words_identical_to_oracle=22016 distinct_gate=18425
   distinct_up=18431`. It is exhaustive over the field, not over the bf16 grid: the continuum inside the
   gate×up hull is ~10⁹ patterns, so the claim is “every value this fixture actually carries”, and the
   driver re-derives the field's census independently and requires the probe's numbers to be it
   (`the census it reports on the field's pairs IS the field's census`, judged=512 differing=80 max=3).

   *Factor, derived and printed* (the same shape the GDN band derivation uses: two probe legs to one
   anchor, licensed divergence is their **sum**, because the two legs' worst cases need not coincide —
   that is the triangle-inequality slack, and it is stated instead of hidden):
   ```
   FLOOREXP_ACT_DERIVE derivation=triangle-inequality-on-floor-probe leg_c_all=3 leg_np_all=3 band_all=6
                                              leg_c_field=2 leg_np_field=2 band_field=4 licensed=band_field
   FLOOREXP_ACT_TRI rows=22016 per_row_violations=0 direct_max=4 band_all=6 direct_field=3 band_field=4
   FLOOREXP_ACT max_ulp_f32=4 legs=2+2 domain=every f32 (gate,up) dot pair the fixture carries
   ```
   leg 1 is the shipping C kernel against an anchor (the f64 exact value rounded to f32): max 3 over the
   domain, **2 over the field's own pairs**. leg 2 is the oracle's **own** `silu_f32` (`oracle_arm.py`,
   imported, not re-implemented) against the same anchor: max 3, **2** over the field. The band is
   2+2=**4** for the judged field; the whole-fixture figures (3+3=6) are printed but only the judged
   field's band is licensed, and a probe whose population did not contain the judged field refuses. No
   typed factor exists anywhere, and the driver re-derives each counted band from the legs the probe
   prints — a hand-edited band that does not equal its legs is a red check.

   The measured field then sits inside it: `field=act judged=512 identical=432 differing=80
   ulp_histogram[1:54 2:24 3:2] max_ulp=3 … band=4 domain=f32_dot` → `POLICYLINE case=2 field=act
   verdict=counted-and-attributed`. The refusal machinery is intact and shown firing every run: the
   `each-field-gets-the-other-probe-band` control judges `act` by the bf16 probe and the same line
   prints `verdict=REFUSED` — `BANDCONTROL wrong_domain_refused=1 arm_refused=0 scratch_red=1
   verdict_rows_moved=0`, with two named checks requiring that refusal to appear and requiring the
   control's own board to go red while the verdict table stays unmoved.
4. **The `cast_after` `prob` word at 2 ULP — counted and attributed**: same expf route (C `expf` vs
   numpy), at the probe's measured ceiling, on a 1-element field where the band is the only possible
   ruler (`isolation=not-defined-for-a-1-element-field(band_judges_it)`).
5. **bf16-carrying fields stay exact because the store is coarser than the disagreement**, which is a
   measured statement, not an assumption: `sigmoid_diff_after_bf16_store=0` over 34194 inputs.

## Claims, per field

| claim | status | what the digest provably detects |
|---|---|---|
| gate, up (f32 dots) vs oracle | **bit-exact** (256 exact fields / 241706 words, host arm) | any change of dot order, of Q8 scale rule, or of row map: 1 of 2048 words differing is reported |
| `raw`, `out` (bf16) host arm vs oracle | **bit-exact** on host **and** device (the device `h=2047` failure is closed by `980fac7`, see (d)(1)) | same, plus the poison-hole count |
| `prob` (bf16, reference cast order) | **bit-exact** | the cast-order claim itself: `cast_after` moves it 1/1 |
| `act` of the `f32_act` case | **counted-and-attributed**, band 4 measured on the fixture's own f32 dot pairs (legs 2+2); observed 3 | an expf/silu libm disagreement, now priced in the domain the field actually carries |
| `prob` of the `cast_after` case | **counted-and-attributed** at 2 ULP | the same, at the ceiling |
| Q8_0 width of the fixture bytes | **bit-exact rule against the published tensor's rule** (relL2 0.00612 vs 0.00613), clamp rate inside the geometric bound | any other scale divisor: 52.6 % clamp rate, relL2 0.59, refused |
| device output as a whole | **bit-exact to the oracle in every declared field** (256 fields / 241706 words); host↔device differ in 8 words, all of them in the two counted f32 fields | any write outside the stage that owns a span — the per-stage ownership probe, shown firing on all 43 cases per stage | |

## Write ownership, per stage (was: stage 3 only)

The launch set is split into **four attributable groups** — `1_gateup` (gate|up|act), `2_down` (raw),
`3_prob` (prob), `4_out` (out) — and the whole output buffer is snapshotted immediately before and after
each group, on **both** binaries. The ownership table lives in `shared_expert_ownership.h`, included by
the device driver and the host probe so the two cannot drift into two self-consistent layouts, and the
driver re-checks that table against the case layout every run (`partition=ok`): if two stages owned the
same span, or a field were owned by nobody, the probe would still print a tidy 0 and nothing else would
notice. `compared` is the whole record span × the number of stages — not the owned span — because a stage
that watched only its own field could not see a write across a boundary, and all 43 records are poisoned
with a sentinel before any launch so an untouched hole is a wrong value rather than a plausible zero.

```
OWNERSHIPSTAGE stage=1 group=1_gateup owns=gate|up|act compared=242219 violations=0 lines=0
OWNERSHIPSTAGE stage=2 group=2_down owns=raw compared=242219 violations=0 lines=0
OWNERSHIPSTAGE stage=3 group=3_prob owns=prob compared=242219 violations=0 lines=0
OWNERSHIPSTAGE stage=4 group=4_out owns=out compared=242219 violations=0 lines=0
OWNERSHIPSUMMARY stages=4 cases=43 compared=968876 violations=0 lines=0 partition=ok attributable=yes mode=grouped inject=0
```

(device arm, `evidence/device-ownership-stages-2026-10-26.log`; the host arm prints the same four lines
with `mode=grouped`; the per-case launch mode prints `attributable=no mode=per-case` and the driver
refuses to accept it as evidence, because there stage 1 of case *n+1* runs after stage 4 of case *n*.)

Every stage was then **shown able to fire** by reintroducing the leaked-sentinel defect at its own launch
boundary: one extra item per case whose `dst_off` is `-1` (or `n_floats`), so exactly one thread writes
the LEAK sentinel `0x4C45414B` into a field another stage owns. All four fire on all 43 cases, on both
arms, naming case, field, index and both bit patterns — and the control is self-checking: a probe that
misses an injection exits 4 with `verdict=PROBE-IS-DEAD` rather than printing a clean stage line.

```
OWNERSHIP stage=1 group=1_gateup case=0 field=raw idx=1536 before=0x7FC00000 after=0x4C45414B    ×43 cases
OWNERSHIP stage=2 group=2_down  case=0 field=act idx=1024 before=0xBC3A0000 after=0x4C45414B    ×43 cases
OWNERSHIP stage=3 group=3_prob  case=0 field=raw idx=1536 before=0x3D760000 after=0x4C45414B    ×43 cases
OWNERSHIP stage=4 group=4_out   case=0 field=raw idx=3583 before=0x3C930000 after=0x4C45414B    ×43 cases
OWNERSHIPCONTROL stage=N group=… expected_cases=43 fired=43 verdict=probe-caught-it
```

Stage 2's injection is the `980fac7` defect class reproduced on demand — a thread computing column `-1`
carrying its poison default into stage 1's `act` field — and stage 4's is the same leak at the top edge
of the record (`idx=3583`, the last `raw` word). The expected case count comes from the case list and the
fired count is counted, never inferred. Full logs: `evidence/control-ownership-{device,host}-stage{1..4}
-2026-10-26.log`. The honest run's own output is unchanged by all of this: the host probe's `work/host_out.f32`
and `work/host_report.tsv` are `cmp`-identical to the pre-change files.

## Census (correctness-only arithmetic, no time)

```
CENSUSDEVICE cu_from_kfd=40 simd_count=80 simd_per_cu=2 max_waves_per_simd=16 lds_kb=64
CENSUSSLICER empty_f32_valu=0 eight_madd_f32_valu=17 sixteen_madd_f32_valu=33 eight_mul=4 eight_add=5 want=empty:0,eight>=8,sixteen>=16,sixteen>eight verdict=ok
OCCUPANCY kernel=se_canary_empty blocks_per_cu=16 threads=128 wavefronts_per_cu=64 canary=1
OCCUPANCY kernel=se_canary_full_lds blocks_per_cu=1 threads=128 wavefronts_per_cu=4 canary=1
OCCUPANCY kernel=se_k_gateup blocks_per_cu=16 threads=128 wavefronts_per_cu=64 canary=0
OCCUPANCY kernel=se_k_down blocks_per_cu=16 threads=128 wavefronts_per_cu=64 canary=0
OCCUPANCY kernel=se_k_prob blocks_per_cu=16 threads=128 wavefronts_per_cu=64 canary=0
OCCUPANCY kernel=se_k_out blocks_per_cu=16 threads=128 wavefronts_per_cu=64 canary=0
LAUNCHBILL stages=4 gateup,down,prob,out grouped_launches=4 per_case_launches=4*n_cases stage1_blocks_per_case=4 stage2_blocks_per_case=16 stage3_blocks_per_case=17
LAUNCHBILL merge_denied=gateup->down reason=every down row needs all 512 act values, which stage 1 produces across 4 distinct blocks; deleting that boundary buys no launch, it buys a grid-wide barrier.
```
The slicer is witnessed by two stubs compiled in the same TU (empty 0 < eight 17 < sixteen 33) and the
occupancy canary fires (`se_canary_full_lds` at 1 block/CU against the empty stub's 16), so neither
table can come back empty and green. `stage3_blocks_per_case=17` is the same number that makes the
device read `sg` 17 times.

## Three-arm selftest totals (both binaries, same suite)

```
SELFTEST arm=0 checks=26 equals=20 differs=6 changed=0 want=0 failures=0
SELFTEST arm=1 checks=26 equals=20 differs=6 changed=6 want=6 failures=6
SELFTEST arm=2 checks=26 equals=20 differs=6 changed=20 want=20 failures=20
SELFTEST3 arms_run=3 checks=26 equals=20 differs=6 honest_failures=0
```
Honest arm clean; the always-equal stub must break exactly the 6 differing checks; the crossed arm must
break exactly the 20 equal ones. `--arm=N` alone refuses to pass, for all three arms, on both binaries.
The two digest-blindness cases (`digest-xorsig-is-blind-to-x2-rescale`,
`digest-xorsig-is-blind-to-a-duplicated-pair`) are kept and pass: they are the `vsum` lesson turned into
a test, and they state what the xor/sum digest does **not** see.

## Injections observed firing (device and host arms, 11/11 each)

`cast_after, cols_half, down_half, f32_act, layout_pair_swap, layout_row_inter, layout_up_first,
perchan_gate, row_shift1, tail_drop, wide_misread` — each must differ from the honest case's own words
for the same block and row; the count is derived from the case list, never typed. Plus the controls
that must be **refused**: `--q8-rule 512`, `--claim-candidate up_first`, and the host-only single-arm
runs.

## NOT ESTABLISHED

* **`NOT ESTABLISHED:` the act band licenses a *device* libm as well as a host one.** The probe's two
  legs are glibc `expf` (the C leg, built by `build.sh`) and numpy's `exp` (the oracle leg). The device
  arm is held to the same band by the same field check — and passes it, 3 ≤ 4 — but its own ROCm libm was
  never the subject of a probe leg, so “the device transcendental is within 2 ULP of the anchor” is a
  licensed borrowing from the C leg, not a measured device fact.
* **`NOT ESTABLISHED:` the act probe's population is the live decode domain.** It is exhaustive over the
  dot pairs this fixture carries (gate hull −4.48011827 .. 3.05410409, 22016 pairs), not over the f32
  continuum: a gate dot outside that hull in real traffic is outside the measurement, and the probe says
  so on its own line rather than pretending to be the bf16 grid probe’s exhaustive-by-bit-pattern case.
* **`NOT ESTABLISHED:` write ownership in per-case launch mode.** The probe prints `attributable=no
  mode=per-case` there and the driver refuses it as evidence; the per-case kernels are unchanged, so the
  per-case numbers stay what they were, but their writes are not attributed to stages by anything.
* **`NOT ESTABLISHED:` the device's `sg` re-read count (17× per case) is intended.** The counter is
  right; whether the stage should be 17 blocks was not decided here.
* **`NOT ESTABLISHED:` any MTP-block behaviour.** `WIDTH mtp_tensors_in_file=0` — the published GGUF
  carries no MTP block, so the `mtp0` fixture is authority-only bf16 and proves nothing about a real
  MTP weight.
* **`NOT ESTABLISHED:` anything about performance.** No timing was taken; the launch bill and the slot
  census are arithmetic on shapes and the ISA, not a cost model.
