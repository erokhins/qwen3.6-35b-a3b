# evidence/band-derivation-2026-10-26.md — the band `prototype/lm_head` judges BAND words by, derived
# from the pinned rows and from geometry, and the floor probe that earns it.

Written 2026-10-26 for the fifth attempt at this lane (the box clock read `2026-10-25 22:11`; the
session's working date is the one the last three commits carry). CPU only: no iGPU, no `gpu_lock`,
no `halogen`, and **no timing anywhere in this file** — no `ms`, no `tok/s`, no `GB/s`, no warm loop
(`AGENTS.md` §"GPU execution is serialized", `pitfalls/00-measurement.md`).

## commands, exit codes, and what was NOT run

| command | rc | what it is |
|---|---|---|
| `/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python floor_probe.py` | **1** | the probe this file is about; rc=1 is one FINDING class, quoted below, reported on two families |
| `... python floor_probe.py --max-cases 3` | 0 | the same code on a 3-case slice (used to catch a shape bug, see §method step 2) |
| `... python gate.py --check-only` | 0 | `CHECKS ran=1513 failed=0` — oracle + host twin + all judgements, no device |
| `... python gate.py --selftest` | 0 | `CHECKS ran=31 failed=0` — 8 controls, 2 driver rosters |
| `./host_probe --selftest` | 0 | `DRIVER SELFTEST RESULT checks=66 failures=0` |
| `./build.sh` | 0 | compile only (hipcc + g++), "no device was touched and no timing taken" |

System `python3` has no numpy, so every python arm here runs under the sibling project's venv
interpreter, which is also what `gate.py` pins as `PY`. **No device run has ever happened in this
lane**; nothing in this file is a device claim.

Full probe output: `work/floor-probe.log` (66 `BANDPROBE` records + 5 `FLOORSUM` + 1 `FLOORAGG`
+ 2 `FINDING`).

## the band, stated

For every BAND-classed word (honest cases only — injections are judged against their twin, not
against a band):

```
band(word)  =  gamma_K * fsum_k |h_k * w_k|
gamma_K     =  K*u / (1 - K*u),      u = 2**-24  (binary32 unit roundoff)
K           =  h_used                (terms summed; 2048 here, = cfg h, read from the plan)
```

`oracle_arm.py` computes and prints it that way (`BANDFORMULA u=2^-24 gamma_K=K*u/(1-K*u)
band=gamma_K*fsum(|h_k||w_k|) K=h_used terms=h_used`), and `gate.py` refuses a word whose band is
not `> 0`. What was missing before this run — and is the reason `BAND` was `NOT ESTABLISHED` — was
any evidence that the form is right for this data and that the number is earned rather than asserted.
`floor_probe.py` supplies five things:

1. **The population is the pinned one.** `FLOORPOP plan_records=201 tables=8 cases=192 hidden=68
   pinned_rows=32 synth_rows=36 h=2048 head_dim=256 arms=seq,rev,split8,tree32,tree64
   matches_build=1`. Every column of `work/plan.tsv` outside the expect column is re-derived from a
   fresh `gate.build()` and compared; the hidden vectors used are indices `[0,32)` of `hidden.f32`,
   i.e. the rows of `data/dumps/2026-10-26/layer39.combined_out.bin` (`[32,2048]` bf16), never the
   synthetic shapes. Refuses if the plan on disk is not the population being probed.
2. **The geometry is measured, not claimed.** Each dot term `h_k * w_k` is compared against the
   binary32 grid from the artifact's own bytes (decoded through `tools/gguf_reader.py`, reached via
   `gate.Data.row_values`), and the significant-bit width of both factors is printed per family.
   (The first version of this check multiplied a `[rows,h]` by a `[bcount,h]` array and broadcast
   them into each other; the full 66-case run caught it — with `--max-cases 3` every case happened to
   have `nrows == bcount` and the probe passed while measuring the wrong pairs.)
3. **The data is recomputed.** `sigma = fsum|h_k w_k|` and `f32(fsum(terms))` are recomputed here and
   required to be **bit-identical** to `work/band.f64` and `work/expect.f32`.
   `FLOORAGG ... anchor_recompute_bad=0 band_recompute_bad=0` — so the band on disk *is*
   `gamma_K * sum|h_k||w_k|` from these bytes, for all 1280 probed words.
4. **The floor.** Five associations of the *same* terms (ascending, descending, 8 lanes combined in
   index order, 32-block pairwise, 64-block pairwise) run in f32. Their deviation from the
   correctly-rounded sum must stay inside the band, must be non-zero somewhere, and the band must sit
   above the f32 grid at the answer. A probe whose arms never move is refused by name — that is the
   dead-branch comparator of `pitfalls/30-numerics.md`.
5. **The two sabotage directions.** Dropping the last summed term must exceed the band wherever that
   term is live; and the print names the words where a uniform `x2` rescale of a family's weights is
   invisible (the maxabs-blindness class — see `three-arm-host-and-host-claims-2026-10-26.md`).

## what the data said (from `work/floor-probe.log`)

| table | enc | words | term pairs | inexact terms | w bits × h bits | γ index used | γ index required | max arm deviation | band_min | band/ulp_min | drop-live | drop rejected | rescale-blind |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | bf16 | 896 | 1,835,008 | 0 | 8 × 8 = 16 | 2048 | 2047 | 7.4505806e-08 | 5.25111254e-05 | 5690 | 896 | 515 | 11 |
| 1 | bf16 | 100 | 204,800 | 0 | 8 × 8 = 16 | 2048 | 2047 | 1.67638063e-08 | 1.73424475e-05 | 9311 | 100 | 38 | 0 |
| 2 | Q8_0 | 120 | 245,760 | **7715** | **18 × 8 = 26** | 2048 | **4095** | 1.2665987e-07 | 5.60775343e-05 | 10330 | 120 | 63 | 7 |
| 3 | Q8_0 | 100 | 204,800 | **3466** | **18 × 8 = 26** | 2048 | **4095** | 2.98023224e-08 | 1.73348368e-05 | 9307 | 100 | 38 | 0 |
| 4 | bf16 (tied) | 64 | 131,072 | 0 | 8 × 8 = 16 | 2048 | 2047 | 3.35276127e-08 | 5.60970388e-05 | 10330 | 64 | 37 | 4 |

`FLOORAGG probed_cases=66 printed_bandprobe=66 words=1280 arms=5 dev_nonzero=1154 dev_zero=126
worst_ratio=0.001646 headroom_x=607.6 worst_at=case 16 word 162 arm rev band_over_ulp_min=5690
anchor_recompute_bad=0 band_recompute_bad=0`

Read plainly:

* **The band holds, with 607× to spare.** Across 1280 words and five legal re-associations the worst
  deviation was `0.001646 × band` (`case 16 word 162`, reverse order). Nobody loosened a threshold to
  get that: the probe refuses at `ratio > 1.0` and at `ratio == 0.0` (an all-zero probe is as useless
  as a check that prints nothing).
* **The band is far above the representable floor.** The smallest band is `1.73e-05` against
  `ulp = 1.86e-09` at that answer — `band/ulp >= 5690`. So a BAND verdict is asking for something an
  f32 value can actually be: an absolute "max 1 ULP vs the oracle" gate could not be met (this is
  `pitfalls/30-numerics.md` §"A max-ULP band cannot see a whole-tensor 1-ULP error", and the reason
  `lm_head_kernels.h` splits the contract into bit-exact-for-SEQ / band-judged-for-REV-and-SPLIT8).
* **1154 of 1280 words actually moved** under a re-association, so the band is bounding a real
  spread and not an empty set. 126 words never moved (their partial sums stay exact in f32); those are
  carried by the bit-exact classes, not by the band.
* **The drop-last-term direction is refused by the band where it matters**: 653 of 1280 live words
  exceeded their band, so the band is tight enough to refuse a term-sized error. It is *not* tight
  enough on the remaining words (where the last term is smaller than the band) — which is why the
  must-catch evidence for this lane is the FIRES injections, not the band.
* **`gamma` is right for bf16 and one rounding short for Q8_0.** See the finding below: the derived
  requirement is `gamma_{2K-1} = 2.4414061e-04` for Q8_0, where the oracle carries
  `gamma_K = 1.22085215e-04`.

## the FINDING (why the probe exits 1, and why nothing was loosened)

```
FINDING table 2 (enc=1): 7715 of 245760 dot terms h_k*w_k are NOT exact in binary32 (weight 18 bits x
  hidden 8 bits = up to 26). The DECODE is exact, the term is not, so the bound needs gamma_{2K-1}
  (index 4095, gamma=0.00024414061) where the oracle carries gamma_K (index 2048);
  lm_head_kernels.h's 'Products are exact' block extends the exact-decode argument to the dot, which
  this measurement refutes. Observed deviation 1.27e-07 against a band of 5.61e-05, so this is a gap
  in the proof and not a wrong number; closing it changes a band, which is not this file's call to make
FINDING table 3 (enc=1): 3466 of 204800 dot terms ... (same form, deviation 2.98e-08, band 1.73e-05)
RESULT FAIL lm_head floor probe findings=2
```

The header's argument (`lm_head_kernels.h` §"What is bit-exact here, and what is not") is correct
about the two things it names: bf16 × bf16 = 16 bits, and the Q8_0 **decode** `d * q` = 11 + 8 = 19
bits. What it does not say is that a Q8_0 **dot term** is `h_k * v_k` — that exact 19-bit decoded
value times an 8-bit hidden element — which needs 27 bits, is rounded once per term, and therefore
adds one rounding per term to the sum's error. Consequences, in order of importance:

1. `gamma_K * sum|x|` is the right form for the bf16 families and **one rounding short** for Q8_0; the
   form that covers both is `gamma_{2K-1} * sum|x|` (`gamma` doubles, `1.22085215e-04 -> 2.4414061e-04`).
   The measured deviations sit 443× below the *current* band, so no verdict in this lane is currently
   wrong because of it.
2. The same sentence is the reason the header gives for **"FMA contraction is therefore neutral"**
   (`fma(x,y,acc) == RN(RN(x*y)+acc)` when `x*y` is exact). For the Q8_0 dot that premise fails, so
   the host twin (g++ `-O2`, which does not contract) and the device binary (hipcc/clang, which
   contracts multiply-add by default on AMD targets) can legitimately differ by one rounding per term
   on Q8_0 SEQ cases — and `three_way` demands **bit-exact, zero tolerance** there. That is a hazard
   for the device run this lane has never done, not for anything measured today: the host twin and the
   hipcc-compiled host path agreed bit-for-bit on all 192 cases (see
   `three-arm-host-and-host-claims-2026-10-26.md`), which is evidence that g++ and hipcc *here* both
   kept the multiply and the add apart, but it is not a guarantee for the iGPU codegen.
3. Fixing it is a band change (widening Q8_0's γ), and bands are a user/main-agent decision in this
   repo (`DECISIONS.md` governs widths and bands; `AGENTS.md`: "Never loosen a gate to make it pass").
   Nothing in `gate.py` was touched for this finding, `gate.py --check-only` is green on its own
   account, and the probe stays red until someone adjudicates the form.

## NOT ESTABLISHED by this file

* that `gamma_K` (as the oracle computes it today) is a bound for the Q8_0 families — see the finding.
  Established instead: `gamma_{2K-1}` bounds them by the same argument, and the measured deviations
  are 443× below even the narrower band.
* anything about the device. Every number here is CPU arithmetic over `work/slab.bin`,
  `work/hidden.f32` and `work/ids.i32`.
* that the band is *tight*: it is 607× wider than the worst of five re-associations. A tighter band
  would need the observed spread as an input, which is exactly the "band from what the build produces"
  move this repo forbids.
* that the 126 words no association ever moved are correct — they are bit-identical to the oracle and
  belong to the EXACT/bit-identity classes, not to a band claim.

## ADDED THE SAME DAY, AFTER THIS FILE WAS WRITTEN: the γ index was adjudicated

This file is a dated record and its measurements stand, but its open question is closed and a reader who
stops here would re-litigate it. The user was given both options with the numbers and chose **A**:
`gamma_{K-1}` for a dot-product family whose products are measurably exactly representable in binary32,
`gamma_{2K-1}` for one whose products are measurably not — the index derived per family from a measured
count, never typed. So:

* the first bullet of §"NOT ESTABLISHED by this file" is retired: the oracle no longer carries
  `gamma_K` for anything, and the Q8_0 families are priced `gamma_4095 = 2.4414061e-04` while the six
  exact families are priced `gamma_2047 = 1.22025596e-04`;
* item 2 of the finding (the FMA-contraction hazard) is unchanged in kind: it is still an empirical
  observation, now recorded as one in `lm_head_kernels.h` and in `results.md` §5;
* item 3 happened the way it said it had to — the adjudication is the user's, and `floor_probe.py` stayed
  red until it was made.

The ratification, the measurement that selects the index, the witnesses, and the two injection arms the
widening reclassified are in `band-form-gamma-index-2026-10-26.md`; the red lines are in
`lm-head-band-form-witnesses-2026-10-26.log`. `results.md` §6 is the current summary.
