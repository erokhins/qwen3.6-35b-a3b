# HANDOFF.md — prototype/lm_head, the γ-index correction (complete)

**Status: done.** CPU only throughout: no iGPU, no `gpu_lock` (`grep -c gpu_lock` = 0 on `gate.py`,
`oracle_arm.py`, `floor_probe.py`), no device run, and **no timing figure in any file this task touched**.

## What was ratified and is now implemented

User, 2026-10-26, choice **A** (verbatim: *"let's do A"*): `gamma_{2K-1}` for a dot-product family whose
products are measurably NOT exactly representable in binary32, `gamma_{K-1}` for a family whose products
are exact — the index derived per family from a measured count, never fitted, never typed.

One transcription note, not a decision: the brief quoted `gamma_2047 = 1.22085215e-04`, which is
`gamma_2048` (index = K, the superseded form). At the indices the rule now selects: `gamma_2047 =
1.22025596e-04` (exact families), `gamma_4095 = 2.4414061e-04` (inexact families).

| file | what it now does |
|---|---|
| `gate.py` | `U32`, `gamma_of_index(n)`, `gamma_index(K, products_inexact)` (the one rule; refuses a `None` measurement), `band_measure()` (census over **every** banded word, not a sample), `verify_band_form()` (4 named checks: element-wise equality against `band.f64`, coverage + ulp floor, anti-drift wide-on-exact, must-reject narrow-on-inexact), `tamper_band_file()` (3 tamper modes), controls `band-unpin` and `band-perturb`, an AST check that no `gamma_index` call passes a literal, and `derive_class`'s HALF_ACCUM now asks the same measured band instead of a third local index |
| `oracle_arm.py` | independent spelling of the rule; measures per family, then prices, in a deliberate second pass; prints `BANDCENSUS` / `BANDFAMILY` / `FORCEDFORM`; `--gamma-form` is a witness-only override that announces itself on every run |
| `floor_probe.py` | imports `G.gamma_index` so the judge and the instrument cannot drift, re-measures over the pinned rows, reads the on-disk index back out of `band.f64`, and refuses mispricing in **both** directions |
| `lm_head_kernels.h` | the bit-exactness block separates the exact **dequant** product from the measured, sometimes-inexact **dot term**, states the band rule, and records FMA-contraction neutrality as an empirical observation with its evidence |

## Final verdicts (all re-run last, on the licensed band)

```
RESULT PASS lm_head floor probe cases=66 words=1280 arms=5 worst_ratio=0.0009657 band_over_ulp_min=5687 \
      band_index_per_table=0:2047,1:2047,2:4095,3:4095,4:2047 all_words_priced_licensed=1     (was: FAIL findings=2)
RESULT PASS lm_head host-only checks=1517 findings=0                                          (was: checks=1513)
RESULT PASS lm_head host selftest checks=44 findings=0                                        (was: checks=31)
RESULT PASS lm_head three-arm host agreement groups=48 field_rows=13 compiled_pair_bit_identical=1 \
      oracle_inside_band=1                                                                    (unchanged line; worst_word 0.001646 -> 0.0009657)
FLOORAGG probed_cases=66 printed_bandprobe=66 words=1280 arms=5 dev_nonzero=1154 dev_zero=126 \
      worst_ratio=0.0009657 headroom_x=1035 worst_at=case 62 word 622 arm seq band_over_ulp_min=5687 \
      anchor_recompute_bad=0 band_recompute_bad=0
```

`v.check` call sites in `gate.py`: 63 → **68**.

## Witnesses, all re-run with the final code, all restored after

`evidence/lm-head-band-form-witnesses-2026-10-26.log` (nine blocks, each with its command line, its
sabotage marker and its red lines; full logs regenerable in `work/bandfix/`, distiller
`work/bandfix/mk_witness_log.py`, which refuses to write if a witness printed no marker or went neither
red nor refused).

* wide-forced producer → gate `findings=4`, floor probe `findings=4`, **and a bf16 must-catch injection
  stops firing** — the wide-on-exact band loses discriminating power, so the anti-drift check is not
  bookkeeping;
* narrow-forced producer → gate `findings=5`, and the two original FINDING lines reappear verbatim;
* band-unpin (380 words), one-word-halved, one-word-doubled → `findings=5 / 2 / 2`; a hand-edit of a
  single word in either direction is caught;
* band-perturb, 4×band on a word of the family whose band doubled → `findings=10`, including
  `BAND inside the derived band ok=0 … worst=4.000 x band outside=1` — the widened band still refuses;
* `no-fire` (pre-existing) now fails on a named coverage check (`unbanded=16`) instead of crashing the
  census — this was the one real bug the change introduced and it is witnessed.

## Reported, not buried

`case=101 inj-q8-lmhead-half-accum` and `case=149 inj-q8-embd-half-accum` reclassified `FIRES → INSIDE`
(1.03× → 0.514×, 1.25× → 0.625×) because their families' bands doubled. The other four `HALF_ACCUM` arms
still fire by 14.1×, 9.1×, 12.2× and 6.2×, so the mode keeps a firing witness; a full 192-case class diff
showed these two lines and no other change.

## What is left for the main agent

1. **`DECISIONS.md`** — the index entry for this adjudication is the main agent's to write; this lane does
   not own that file.
2. **The device arm must be re-run in a clean window.** Every committed device log for this lane predates
   the band change; `results.md` §1 now marks the `checks=3012` line stale, and the Q8_0 SEQ bit-exactness
   claim rests on the empirical no-contraction observation rather than exact products. If the device
   re-run diverges on a SEQ case, the fix is a build-level contraction policy, not a band.
3. **Commit.** Nothing is staged. The topic commit wants: `gate.py`, `oracle_arm.py`, `floor_probe.py`,
   `lm_head_kernels.h`, `results.md`, `evidence/band-form-gamma-index-2026-10-26.md`,
   `evidence/lm-head-band-form-witnesses-2026-10-26.log`, and the pointer appended to
   `evidence/band-derivation-2026-10-26.md`. Note that re-running the instruments also rewrote tracked
   regenerable artefacts (`work/band.f64`, seven `work/control-*.log`s); whether those belong in the
   commit is the main agent's call. `devices/evidence/halogen-restart.log` is modified but is **not**
   this lane's change.
4. `results.md` §2 and §9 were updated to match; §9 still says not to `git add prototype/lm_head` wholesale
   — the binaries and `work/` stay out.
