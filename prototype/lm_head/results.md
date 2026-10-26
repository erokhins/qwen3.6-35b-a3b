# prototype/lm_head — lane report, 2026-10-26 (worker `halogen`, lane `lm_head`)

No timing appears in this file. The `WALL`/`UTIL_*` lines in `evidence/` are lock and occupancy records
`AGENTS.md` requires around a device run, not measurements; no `ms`, `tok/s` or `GB/s` figure is
anywhere here, and nothing in this file is a bandwidth or latency claim.

## 1. Status, and what was re-verified to write this file

| claim | line | state |
|---|---|---|
| host + device gate | `RESULT PASS lm_head host+device checks=3012 findings=0` | green, `evidence/lm-head-device-clean-2026-10-26.log` (and `evidence/lm-head-device-2026-10-26.log`, the same verdict) |
| host-only gate, re-run while writing this file (2026-10-26, this session) | `RESULT PASS lm_head host-only checks=1513 findings=0`, `CHECKS ran=1513 failed=0` | green, rc=0 |
| judge self-controls, re-run | `RESULT PASS lm_head host selftest checks=31 findings=0` | green, rc=0 |
| oracle arm | `RESULT PASS oracle arm filled=1884/27488 words slab=557568 B hidden=68` | green — and the **1884 of 27488** is the whole of §3's second cluster |
| floor probe | `RESULT FAIL lm_head floor probe findings=2` | **RED ON PURPOSE**, re-run rc=1; see §6 — it is the open band question, and it is the user's to close |
| two controls | `RESULT FAIL lm_head (refused) checks=3010 findings=1` (`evidence/lm-head-control-band-unpin-red-2026-10-26.log`), `RESULT FAIL lm_head host+device checks=3012 findings=8` (`evidence/lm-head-control-seq-tamper-red-2026-10-26.log`) | **LEFT RED**: a control that walks past its own refusal is a control that judges nothing |

`gate.py`'s own docstring points here (`gate.py §"WHAT EACH DIGEST PROVABLY SEES (results.md carries
this table; one of them is provably blind)"`), which is the reason this file exists: the digest
semantics were only in the code comment, and one of those digests is provably blind to a whole error
class.

## 2. Commands, exit codes, and who may run them

| command | rc | what it is | device |
|---|---|---|---|
| `./gate` (wrapper; sole acquirer of the repo's stream lock) | 0 | full host + device gate → `checks=3012 findings=0` | yes, short correctness-only run |
| `... .venv/bin/python gate.py --check-only` | 0 | oracle + host twin + every judgement, no device | no |
| `... .venv/bin/python gate.py --selftest` | 0 | 31 controls: the instruments that must be able to fail | no |
| `... .venv/bin/python floor_probe.py` | **1** | the band's floor probe; 2 findings, §6 | no |
| `... .venv/bin/python three_arm.py` | 0 | host twin vs hipcc host path vs numpy oracle | no |
| `./gate` with the lock already held | 1 | `RESULT FAIL not-run lock-held`, `work/w-stale-demo.log` | not started |
| `./gate` with a dead holder's lockdir | 1 | `RESULT FAIL not-run lock-stale`, `evidence/lm-head-gate-stale-refused-2026-10-26.log` | not started |

`gate.py` contains zero references to the lock tool (checked with `grep -c gpu_lock gate.py` → `0`,
again while writing this file) and `./gate` verifies at run time that it is the only acquirer — the
lock file records `LOCKREF gate.py references to the lock tool must be 0: 0`. That is not ceremony: a
driver that takes the lock and is called by a caller already holding it parks forever with no CPU, no
GPU, no message and no exit (`pitfalls/50-harness-and-shell.md §"Self-locking gate drivers deadlock
silently"`). The backoff arm is witnessed refusing rather than queueing
(`evidence/lm-head-gate-backoff-refused-2026-10-26.log`).

## 3. The first device run, and the arc `findings=40 → 2 → 0`

The lane had never touched the iGPU before this run: `evidence/band-derivation-2026-10-26.md`
§"commands, exit codes, and what was NOT run" records *"No device run has ever happened in this lane"*.
The first execution printed the population and the arm census that everything below is about:

```
RUN arm=host    cases=192 skipped=0  items=1186 out_floats=27488
RUN arm=device  cases=178 skipped=14 items=738  planned=178  out_floats=27488
POPULATION tables=8 cases=192 injections=77 hidden=68 (real=32 synth=36) ids=11 out_floats=27488 slab_bytes=557568
```

178 of 192 cases are `on_device`; the 14 skips are not incidental — they are the `deep` batched/per-row
families (cases 62–75), which the plan declares host-only, and `CHECK name=device arm ran exactly the
on_device cases ok=1 n=178 planned=178 skipped=14` plus a per-case `host-only case was not written by
this arm` check (14 of them, `non-zero words=0`) keep that claim from meaning "the arm quietly skipped
something else". `RESULT FAIL lm_head (refused) checks=3010 findings=40` is where the first run ended.

**Cluster 1 — `hipMemset` destroying a sentinel (38 of the 40).** The device arm's own counters for the
first traced field disagreed with the gate's independent recount on 38 of the 178 device cases, and only
on that field:

```
CHECK name=device case=62 honest-bf16-lmhead-deep-batched read-set == recount ok=0 n=11 mismatch=[('lo', 0, 4294967295)] trace=1
CHECK name=device case=89 inj-bf16-lmhead-row-stride-head-dim read-set == recount ok=0 n=11 mismatch=[('lo', 0, 512)] trace=1
CHECK name=device case=112 inj-q8-lmhead-row-stride-head-dim read-set == recount ok=0 n=11 mismatch=[('lo', 0, 272)] trace=1
```

Distribution of the disagreement over the 38 cases: `('lo', 0, 4294967295)` ×20 (the device said 0 where
nothing was traced and the recount kept the sentinel), `4096` ×6, `2176` ×5, `512` ×2, `272` ×2, plus
`8192`, `67456` and `126976` once each — 38 in total, i.e. *every* device case, whatever it read.

Root cause, and the reason it was a harness bug and not a kernel bug: `LmhTrace::min_off` is an
`atomicMin` reduction whose identity element is `lmh_trace_init`'s `0xFFFFFFFF` sentinel
(`lm_head_kernels.h §"What is bit-exact here, and what is not"` block, field comment `min_off … init
0xFFFFFFFF`). The device copy of the trace buffer was `hipMemset` to 0 after the host array was filled,
so the min started at 0 and no unsigned offset is ever below 0: **every device case reported `lo=0`
whatever it addressed**. The value counters (`bytes/loads/xorb/vsum/bsum`) agreed across both arms the
whole time, which is what localised it to the reduction's initialiser rather than to the address walk.
The fix is one initialiser for both paths — the device buffer is initialised by copying the host array
`lmh_trace_init` filled (`lm_head.hip §"The trace buffer is initialised by COPYING the host array"`,
whose comment carries the same measurement). Nothing was re-banded and no `lo` expectation was edited to
meet the device: the device was made to start from the plan's sentinel, and the same 38 checks then
passed with the recount unchanged.

**Cluster 2 — the judge demanding bands the oracle never earned (the last 2).** After cluster 1 the run
was `RESULT FAIL lm_head host+device checks=3012 findings=2` (`work/w-device-run2.log`):

```
CHECK name=device batching amortises the weight read by exactly bcount ok=0 n=10 group 62: batched bytes 0 != geometry 131072; group 67: batched bytes 0 != geometry 131072
CHECK name=three-arm: neither arm leaves the derived band against the torch oracle ok=0 n=2440 outside={'host': 0, 'device': 382}
```

Both were the same mistake on the judge's side: the `on_device` filter was *counting every exclusion as
an observation*. Groups 62 and 67 are host-only families, so their device `bytes` were legitimately 0 —
the roll-up demanded `bcount×` of a run that never happened; and `three_way` asked for a band at every
word of the 27488-word output when the oracle derives expectations for **1884** of them
(`RESULT PASS oracle arm filled=1884/27488 words`), so 382 device words had no band to be inside and
were scored as leaving one. The fix does not delete the exclusions, it *counts and prints them*, which is
what makes the green meaningful:

```
AMORTISE tag=host    families_judged=10 families_host_only_excluded=none    batched_dense_cases=49 of 49 (on_device filter)
AMORTISE tag=device  families_judged=8  families_host_only_excluded=[62, 67] batched_dense_cases=47 of 49 (on_device filter)
THREEWAY oracle words={'host': 1220, 'device': 836} outside_band={'host': 0, 'device': 0} device_skipped_host_only cases=2 words=384 outside_there=382
```

`outside_there=382` is still printed: the excluded words are named and counted, not hidden. A counter
that reads 0 because nothing was ever asked of it is vacuous (`pitfalls/40-gates-and-admissions.md`),
which is also why the 14 host-only cases keep their own per-case checks in the device log.

The last thing the first run refused on was real and stayed a refusal until it was understood:
`FAIL case 84 word 1496 has no band: a re-association cannot be bounded` — a BAND word with no derived
band is a judge that cannot judge, and `gate.py` still refuses rather than defaulting a band in.

## 4. The five digests, what each one provably sees, and the one that is provably blind

Every `BAND`/`FIRES`/`IMMUNE` verdict in this lane is made from five digests the driver prints per case,
and `gate.py` recomputes each one from the slab bytes rather than believing the driver. Two of them are
*values* and three are *bit images*, and they were chosen to fail on different errors — the table is the
one `gate.py §"WHAT EACH DIGEST PROVABLY SEES"` promises here:

| digest | what it is | what it provably detects | what it provably **cannot** see |
|---|---|---|---|
| `oxor` | XOR of the f32 **bit images** this case wrote | any change that does not cancel across the case | any uniform XOR-cancelling change (two identical wrong writes cancel) |
| `osum` | real-valued sum of those words | a value change of the wrong magnitude | a **permutation** of the same words |
| `xorb` | XOR of the **decoded weight elements** it loaded | a wrong decode, a wrong row | **a uniform power-of-two rescale** — measured: `d=0.5` and `d=1.0` both give `0xFF400000` (`tmp/worker-lm-head/probe10_digest.cpp`, recorded in `gate.py §"WHAT EACH DIGEST PROVABLY SEES"`) |
| `vsum` | **integer** sum of those same bit images | that rescale — `vsum` is why the value verdict is carried here and not by `xorb` | a swap of two elements whose integer images sum the same |
| `bsum` | sum of the **raw storage byte values** | layout, stride, wrong-table reads | anything a decode change does to the values |

The blind cell is load-bearing, so it is witnessed twice: by the probe above, and by
`--control scale-q8` (a uniform rescale of a Q8_0 table), which now **exits 1** with a named refusal
after moving `vsum` on 31 of 31 non-exempt traced cases (`evidence/three-arm-host-and-host-claims-2026-10-26.md
§"what had to change in the judge for these claims to be checkable (nothing was loosened)"` item 5).
Before that fix the control exited 0 while printing 103 red inner checks — a control a driver walks
past, which is the same as no control.

The digests are only comparable if their **text form** is fixed, and it was not: `oxor`/`xorb` were
printed decimal and parsed hex, and `201392804` and `0x0C0102A4` are one value that both parse to
something plausible. The rule now lives in one place and is scraped, not mirrored
(`evidence/digest-canonical-form-2026-10-26.md §"The rule (one canonical form per field, declared
once)"`): `hex` = `0x` + 8 upper-case digits for the two bit images, `real` = `%.9g` for `osum`, `dec`
(no sign, no prefix, no leading zero) for everything else including `vsum` and `bsum`; producer and
consumer each validate it independently, count discipline re-witnessed as
`parsed=192 printed=192 fields=33 forms=hex:2,dec:30,real:1`, and both radix controls exit 1
(`RESULT FAIL lm_head (refused) checks=8 findings=1` for each).

### the metric policy for the three-arm agreement, and the 23× that makes it true

Load-bearing: **`relL2`**, **`relL2_bandnorm = sqrt(mean((|a-b|/band)^2))`** and **`ndiff`** over bit
images. Reported, never load-bearing: **`maxabs`**. Printed, never load-bearing: **`cos`**. The line
carries the policy itself (`THREEARMAGG … load_bearing=relL2_bandnorm+ndiff maxabs_load_bearing=0
cos_load_bearing=0`), and two measurements make it a fact rather than a preference
(`evidence/three-arm-host-and-host-claims-2026-10-26.md §"the metric policy, and why"`):

* **`cos` cannot see 1263 differing words.** Against the oracle, `all_honest` reads `n=1884
  bit_identical=0 ndiff=1263 maxabs=2.86102e-06 relL2=2.44258e-10 relL2_bandnorm=0.0001627
  worst_word=0.001646 cos=1` — `cos` prints `1` to nine digits while **67 % of the words differ in
  their bit image**. A cosine gate would pass that population without ever looking at `ndiff`.
* **`maxabs` orders the population the wrong way, by 23×.** `THREEARMRANK pair=host_twin_vs_oracle
  relL2_worst_group=table2_enc1 maxabs_worst_group=all_honest ranking_agrees=0`: `table7_enc1` carries
  the largest absolute deviation (`2.86102e-06`) while `table2_enc1` — a **Q8_0** table, `maxabs =
  1.2666e-07`, 23× *smaller* in absolute terms — is the worst group on the band-normalised primary
  (`0.0002346` vs `7.179e-05`). Triaging by `maxabs` spends the day on the wrong tensor family, which
  in this lane is the difference between chasing a synthetic table and chasing Q8_0.
* Honest negative, recorded so the argument does not overreach: `maxabs_blind_words=0` in all 48 groups
  of *this* population — no word here hides a >0.1 % relative deviation below 1 % of its group's worst.
  The blindness case rests on the two observations above and on the sibling precedent the file copies
  in with its numbers (the `shared_expert` row-map pair, where the *wrong* candidate was nearer on
  `maxabs`: `0.961426` vs honest `0.96875`).

## 5. The FMA-contraction prediction: **REFUTED**

The prediction, stated in `evidence/band-derivation-2026-10-26.md §"the FINDING (why the probe exits 1,
and why nothing was loosened)"` item 2: the header's "FMA contraction is therefore neutral" argument
rests on the product being exact, which is true for bf16 × bf16 and for a Q8_0 *decode*, but a Q8_0
**dot term** is `h_k × v_k` — a 19-bit decoded value times an 8-bit hidden element, 27 bits, rounded
once per term. So the host twin (g++ `-O2`, which does not contract) and the device binary (hipcc/clang,
which contracts multiply-add by default on AMD targets) *could* differ by one rounding per term exactly
where `three_way` demands bit-exact, zero tolerance.

The device run says they do not. From `evidence/lm-head-device-clean-2026-10-26.log`:

```
THREEWAYFAM enc=Q8_0 assoc=SEQ  path=dense  cases=59 words_compared=728 words_differing=0 cases_diverging=0
THREEWAYFAM enc=Q8_0 assoc=SEQ  path=gather cases=12 words_compared=144 words_differing=0 cases_diverging=0
THREEWAYFAM enc=Q8_0 assoc=REV  path=dense  cases=8  words_compared=128 words_differing=0 cases_diverging=0
THREEWAYFAM enc=Q8_0 assoc=SPLIT8 path=dense cases=1 words_compared=16  words_differing=0 cases_diverging=0
THREEWAYFAM families=8 words_compared=2192 words_differing=0
THREEWAY seq_cases=160 reassociated_cases=18 reassociated_bit_identical=18 seq_diverging_cases=0
CHECK name=three-arm: the host twin and the device agree bit-for-bit on every SEQ case ok=1 n=160 0 diverging cases
```

**Q8_0 SEQ: 59 + 12 = 71 cases, 728 + 144 = 872 words, zero differing** — and zero across all eight
(enc × assoc × path) families, 2192 words compared. The 18 re-associated cases came back bit-identical
too, so the band judgement was available and did not have to be used
(`NOTE name=re-associations-were-also-bit-identical 18 of 18 REV/SPLIT8 cases had ZERO words apart`).

The contraction is not hypothetical — it is in the shipped code object, and it is the only one:
`/opt/rocm/lib/llvm/bin/llvm-objdump -d prototype/lm_head/work/lm_head-hip-amdgcn-amd-amdhsa-gfx1151.out`
(re-run while writing this file) emits **1 × `v_fma_f32`**, 8 × `v_mul_f32`, 3 × `v_add_f32`; the two
host binaries contain **no FMA at all** (`objdump -d host_probe`: 9 `mulss`, 5 `addss`, 0 FMA-family; the
hipcc host path: 17 `mulss`, 15 `addss`, 0 FMA-family). So the device really did fuse a multiply into an
add that the host twin kept apart, and not one word moved. Output is byte-identical across the
contraction policies: `sha256(work/w-out-contract-on.f32) = sha256(work/w-out-contract-off.f32) =
sha256(work/out.f32) = 8c8bb078fd17c67bd2a868c668e1ce9742d34cf86a0034965fcab861e5346429` (`8c8bb078…6429`).

What the refutation is and is not: it is an empirical result over 872 Q8_0 SEQ words and one code object.
It does **not** repair the proof hole underneath it — the γ index question in §6 is the same premise
(`products are exact`) and it is still open. A prediction being false for this data is not the same as
the argument for it being valid; the header's sentence should still be corrected, and that correction is
now entangled with a band, which is the user's call.

## 6. The band, and the red things that stay red

The band every `BAND` word is judged by is derived, never typed
(`evidence/band-derivation-2026-10-26.md §"the band, stated"`):

```
band(word) = gamma_K * fsum_k |h_k * w_k|,   gamma_K = K*u/(1-K*u),  u = 2**-24,  K = h_used = 2048
```

`floor_probe.py` earns it — re-run while writing this file, rc **1**, same two findings as the committed
`work/floor-probe.log`: 66 cases, 1280 words, five legal re-associations, worst deviation `0.001646 ×
band` (`case 16 word 162`, reverse order), **607× of headroom**, smallest band `1.73e-05` against a
`1.86e-09` ULP (`band/ulp >= 5690`), `dev_nonzero=1154` of 1280 so the band bounds a real spread, and a
drop-the-last-term direction refused on 653 of 1280 live words. Nothing was loosened to get any of that,
and the probe refuses an all-zero arm population by name.

**The open question, kept open, and with the user.** Two findings, both Q8_0:

```
FINDING table 2 (enc=1): 7715 of 245760 dot terms h_k*w_k are NOT exact in binary32 (weight 18 bits x
  hidden 8 bits = up to 26). The DECODE is exact, the term is not, so the bound needs gamma_{2K-1}
  (index 4095, gamma=0.00024414061) where the oracle carries gamma_K (index 2048) …
FINDING table 3 (enc=1): 3466 of 204800 dot terms … (same form, deviation 2.98e-08, band 1.73e-05)
RESULT FAIL lm_head floor probe findings=2
```

So `gamma_K` is right for the bf16 families and **one rounding short for Q8_0**, where the form that
covers both is `gamma_{2K-1} = 2.4414061e-04` against the `gamma_K = 1.22085215e-04` the oracle carries
now. Measured deviations sit 443× below even the narrower band, so **no verdict in this lane is wrong
because of it today** — but the proof is not complete, and closing it means widening Q8_0's γ, which is a
band change. Bands are the user's decision in this repo (`DECISIONS.md` governs them; `AGENTS.md`: *never
loosen a gate to make it pass*), so: **γ_K vs γ_{2K-1} for Q8_0 is UNRESOLVED and with the user.** This
run touched no band, no factor, and nothing in `tools/ulp_policy.py`.

Still red on purpose, with their logs committed:

* `evidence/lm-head-control-band-unpin-red-2026-10-26.log` — unpinning a band makes the gate refuse
  (`RESULT FAIL lm_head (refused) checks=3010 findings=1`).
* `evidence/lm-head-control-seq-tamper-red-2026-10-26.log` — tampering with a SEQ case, which is held to
  zero tolerance, produces 8 findings (`checks=3012 findings=8`).
* `evidence/lm-head-gate-backoff-refused-2026-10-26.log` and `…-gate-stale-refused-…` — the wrapper
  refusing to queue onto a live holder, and refusing a stale lockdir, both `RESULT FAIL not-run`.
* the floor probe itself, §6, until the γ index is adjudicated.

## 7. The two claims this lane exists for: disjointness and amortisation

Both are about **reads**, both are witnessed on both arms, and neither needed a timer
(`evidence/three-arm-host-and-host-claims-2026-10-26.md §"disjointness — both colours witnessed"` and
§"amortisation — no timer anywhere"; the device-side lines are in
`evidence/lm-head-device-clean-2026-10-26.log`):

```
DISJOINT tag=device witness untied_cases=175 tied_cases=17 tied_peer_nonzero=17 tied_cross_max=0 untied_cross_arms=4 arms_with_cross>0=4
DISJOINT tag=device real_tables=5 overlaps=0 lmhead_files=['model-00026-of-00026.safetensors'] embd_files=['model-00001-of-00026.safetensors'] same_file=0
CHECK name=device the cross-table-read witness moves `cross` on every untied arm ok=1 n=4 arms=[(97, 4, 4), (120, 4, 4), (145, 4, 4), (169, 4, 4)]
```

* **Disjointness.** `lm_head.weight` and `embed_tokens.weight` come from *different shard files*
  (`same_file=0`) — this checkpoint is untied, and that is the claim — their five real slab ranges do not
  overlap (`real_tables=5 overlaps=0`), 175 untied cases read no foreign range, and the instruments are
  **witnessed moving in both directions**: `peer_hits > 0` on all 17 tied cases (never vacuous),
  `cross = 4` on each of the four arms whose mode sanctions a foreign read, and `cross = 0` on the tied
  arm of the same mode (the must-not-move arm). Three of those checks were **incapable of passing**
  before this round and their predicates are now what the geometry says — see
  §"what had to change in the judge for these claims to be checkable (nothing was loosened)", items 1–3.
* **Amortisation.** Batched `bytes` equal the row-set geometry and do not move when `bcount` goes 1→2→4
  or 4→8 for the same row set (five shapes, `flat`); the per-row arms sum to exactly `bcount ×` the
  batched case in every complete family (ratios 4.00 and 8.00, `families_judged=10` host / 8 device
  after §3's exclusion counting); and `BATCH_RELOOPS` — a batched kernel that secretly re-loops — is
  caught **on bytes alone**: 4 cases with words bit-identical to their twin, `xorb` and `vsum`
  unchanged, `bytes` and `loads` exactly `bcount×`. The immunity half (`bcount=1` is invisible by
  construction and must be classed IMMUNE) lives in the driver roster, so the two halves do not overlap.

## 8. NOT ESTABLISHED by this lane

* **Any timing, bandwidth or latency claim of any kind.** The device ran to be judged, not measured;
  `WALL`/`UTIL` lines are lock records. Rate claims belong to the main agent's exclusive window.
* **The 14 host-only cases** (62–75, the `deep` batched/per-row families) have no device claim: the
  device arm ran 178 of 192 and says so per case.
* **γ_K is not a proven bound for Q8_0** (§6, and `evidence/band-derivation-2026-10-26.md
  §"NOT ESTABLISHED by this file"`). Established instead: `gamma_{2K-1}` bounds the same families by the
  same argument, and the observed deviations are 443× below the narrower band.
* **The FMA refutation is empirical over this population and this code object**, not a proof for other
  data, other widths or a future LLVM. It also does not retire the header sentence that motivated it.
* **The `-ffp-contract=off` build line is not in a committed log.** The A/B rests on the two output
  images in `work/` (identical sha256, §5) plus the ISA census; if someone wants that A/B re-runnable,
  the flag belongs in `build.sh` and its log, not in a worker's memory.
* **Q6_K is not supported by these kernels**: the artifact stores `output.weight` as Q6_K
  (`ARTIFACTWIDTH q6_k_tensors=4 … Q6_K kernel support deferred`), so this lane reads it through the
  audited CPU decoder and makes no Q6_K device claim.
* **Nothing about the engine.** The slab here is a synthetic 557568 B fixture; no residency claim, no
  staging claim, no `lm_head` batching claim for `engine/`, and nothing about MTP or the draft block.
* The tied cases are a **control**, not a claim about this checkpoint, which is untied (§7).

## 9. Where the artefacts are

Committed under this directory: `results.md`, `gate.py`, `gate` (wrapper), `floor_probe.py`,
`oracle_arm.py`, `three_arm.py`, `lm_head.hip`, `lm_head_kernels.h`, `lm_head_plan.h`, `host_probe.cpp`,
`build.sh`, and everything in `evidence/`.

Not committed, and **do not `git add` this directory wholesale**:

* The two built drivers, `prototype/lm_head/lm_head` and `prototype/lm_head/host_probe` — already
  gitignored **by exact path** (`.gitignore` §"Built binaries: exact path, never the directory", which
  names the lm_head stale-binary trap). Do not loosen it to a directory rule.
* `prototype/lm_head/work/` is **not** gitignored in this lane (unlike `moe_gather` and
  `shared_expert`): three control logs in it are deliberately tracked, and the rest is untracked
  regenerable output — `out.f32`, `host_out.f32`, `expect.f32`, `band.f64`, `hidden.f32` (~1 MB of float
  images), `report.tsv`, `plan.tsv`, my re-verification logs `w-*.log`, the contraction pair
  `w-out-contract-{on,off}.f32`, and the build intermediates `lm_head-hip-amdgcn-amd-amdhsa-gfx1151.{o,hipi,out}`,`lm_head-host-x86_64-unknown-linux-gnu.hipi`, `lm_head.hip-hip-amdgcn-amd-amdhsa.hipfb`
  (the two 2.6 MB `.hipi` files and the `.out` are already ignored globally by `*.hipi` / `*.out`).
  If any of it is wanted as evidence, add it by exact path.
* Older scratch for this lane is in `tmp/worker-lm-head/` (gitignored `tmp/`), including
  `probe10_digest.cpp` — the source of the measured `xorb` blindness in §4.

