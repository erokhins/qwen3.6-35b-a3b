# evidence/band-form-gamma-index-2026-10-26.md — the band index is selected by measurement: what was
# ratified, what it changed, and the witnesses that keep it from drifting

Written 2026-10-26 by the implementation worker, on the user's instruction. CPU only: no iGPU, no
`gpu_lock` (`grep -c gpu_lock` is 0 on `gate.py`, `oracle_arm.py`, `floor_probe.py`), no device run, and
**no `ms`, `tok/s` or `GB/s` figure anywhere in this file** — nothing here is a rate or latency claim.

## The decision, and whose it is

**Ratified by the user, 2026-10-26**, in answer to the question `results.md` §6 left open and
`evidence/band-derivation-2026-10-26.md` §"the FINDING" named: the user was given both options with the
measurements and chose **A**, verbatim instruction *"let's do A"*. A is:

> adopt Higham's `gamma_{2K-1}` as the band form for dot-product families whose products are measurably
> NOT exactly representable in binary32, and keep `gamma_{K-1}` for families whose products are exact.

This is a **band change**, and the widening half of it (Q8_0's `gamma` doubles) is in the direction
`AGENTS.md` forbids taking to make something pass. It is legitimate because it is *derived*: the index
is a function of a measured count, it moves in both directions, and `DECISIONS.md`'s index entry for it
is the main agent's to write (this file records the ratification; it does not write `DECISIONS.md`,
which is outside this worker's allowed paths).

One transcription correction, not a decision. The instruction quoted `gamma_2047 = 1.22085215e-04`.
That number is `gamma_2048` — index = K, which is what the oracle carried before this change. The value
at the index the ratified rule now selects is:

```
gamma_n        = n*u/(1 - n*u),   u = 2**-24 = 5.9604644775390625e-08
gamma_2047     = 1.22025596e-04   (index K-1, products exact — what bf16 families get now)
gamma_2048     = 1.22085215e-04   (index K,   what every family carried before)
gamma_4095     = 2.4414061e-04    (index 2K-1, products inexact — what the Q8_0 families get now)
```

The indices are what was ratified; the fourth significant digit moves nothing, which is the same
argument `AGENTS.md` makes about the 0.4 GiB correction to the residency figures.

## Where the rule lives, and why it cannot be typed

| file | what it holds |
|---|---|
| `gate.py` `U32`, `gamma_of_index(n)`, `gamma_index(K, products_inexact)` | the ONE rule: `(K-1) if products_inexact == 0 else (2K-1)`. No literal index appears anywhere. |
| `gate.py` `band_measure(d, tag)` | measures `products_inexact` per family, over **every** word the oracle bands (not a sample), and each word's `sigma = fsum_k |h_k w_k|`; prints `BANDCENSUS` + one `BANDFAMILY` line per family |
| `oracle_arm.py` `gamma_of_index`, `gamma_index` | the oracle's own independent spelling (this file shares no code with `gate.py`; that is what makes a match an agreement). It measures, then prices — the family index cannot be known until every word has been measured, so the band is deliberately written in a second pass |
| `floor_probe.py` | imports `G.gamma_index` (so the rule cannot drift between the instrument and the judge) but spells `gamma_of_index` again, measures the same count over the pinned rows, reads the on-disk index **back out of `band.f64`**, and refuses a disagreement in either direction |
| `gate.py` `verify_band_form` | re-derives every band from the bytes and requires bit equality with `work/band.f64` |
| `gate.py` `derive_class`, `LMH_MODE_HALF_ACCUM` | now asks the injection against `forms[table]["gamma"]` — the same measured band the file carries. It used to compute a third index (`gamma_K`) locally, which was two rulers over one quantity |

`--selftest` carries a named check that the rule cannot be typed:
`CHECK name=every gamma_index call selects its index from measured data, never a literal ok=1 n=0
offenders=[]` — an AST scan that refuses a `gamma_index(K, <literal>)` call, or one indexed by `K`
itself.

## What the measurement said, before and after

From `work/bandfix/final2-gate-checkonly.log` (`BANDFAMILY tag=lane`), and `tag=oracle` / `tag=probe`
agree with it line for line:

| table | kind | K | products_inexact | term pairs | index before | index now | gamma now |
|---|---|---|---|---|---|---|---|
| 0 | real bf16 lm_head | 2048 | 0 | 1,998,848 | 2048 | **2047** | 1.22025596e-04 |
| 1 | real bf16 embd | 2048 | 0 | 368,640 | 2048 | **2047** | 1.22025596e-04 |
| 2 | real Q8_0 lm_head | 2048 | **7,727** | 409,600 | 2048 | **4095** | 2.4414061e-04 |
| 3 | real Q8_0 embd | 2048 | **3,474** | 368,640 | 2048 | **4095** | 2.4414061e-04 |
| 4 | tied copy bf16 | 2048 | 0 | 131,072 | 2048 | **2047** | 1.22025596e-04 |
| 5 | tied role B | 2048 | 0 | 163,840 | 2048 | **2047** | 1.22025596e-04 |
| 6 | synth bf16 | 2048 | 0 | 212,992 | 2048 | **2047** | 1.22025596e-04 |
| 7 | synth Q8_0 | 2048 | **0** | 204,800 | 2048 | **2047** | 1.22025596e-04 |

`BANDCENSUS tag=lane honest_cases=115 words_banded=1884 term_pairs=3858432 products_inexact=11201
families=8 rule=gamma_index(K,products_inexact)=(K-1 if inexact==0 else 2K-1) u=2^-24`

Table 7 is the family that proves the rule is a measurement and not a label: it **is** Q8_0, and it is
still priced `gamma_{K-1}`, because the synthesized table's decoded values happen to make every product
exact. A "Q8_0 ⇒ wide" rule would have priced it wrongly and nothing would have noticed.

The pinned-subset counts (`floor_probe.py`, `FLOORSUM ... products_inexact`) are 7,715 and 3,466 of the
probed words; the census counts 7,727 and 3,474 over all banded words. Every probed word matches
(`pinned_subset_agrees=1` on all five real families).

## The verdicts, verbatim

| instrument | before | after |
|---|---|---|
| `floor_probe.py` | `RESULT FAIL lm_head floor probe findings=2` | `RESULT PASS lm_head floor probe cases=66 words=1280 arms=5 worst_ratio=0.0009657 band_over_ulp_min=5687 band_index_per_table=0:2047,1:2047,2:4095,3:4095,4:2047 all_words_priced_licensed=1` |
| `gate.py --check-only` | `RESULT PASS lm_head host-only checks=1513 findings=0` | `RESULT PASS lm_head host-only checks=1517 findings=0` |
| `gate.py --selftest` | `RESULT PASS lm_head host selftest checks=31 findings=0` | `RESULT PASS lm_head host selftest checks=44 findings=0` |
| `three_arm.py` | `RESULT PASS lm_head three-arm host agreement groups=48 field_rows=13 compiled_pair_bit_identical=1 oracle_inside_band=1` | identical line; `worst_word` 0.001646 → 0.0009657 (`all_honest`), 0.001646 → 0.000823 (`table2_enc1`), 0.001187 → 0.0005935 (`table3_enc1`) |

`FLOORAGG` before: `probed_cases=66 printed_bandprobe=66 words=1280 arms=5 dev_nonzero=1154 dev_zero=126
worst_ratio=0.001646 headroom_x=607.6 worst_at=case 16 word 162 arm rev band_over_ulp_min=5690
anchor_recompute_bad=0 band_recompute_bad=0`

`FLOORAGG` after: `probed_cases=66 printed_bandprobe=66 words=1280 arms=5 dev_nonzero=1154 dev_zero=126
worst_ratio=0.0009657 headroom_x=1035 worst_at=case 62 word 622 arm seq band_over_ulp_min=5687
anchor_recompute_bad=0 band_recompute_bad=0`

So the worst legal re-association sits at 0.00097 of its band — **1035x of headroom**, and the HARD STOP
threshold (a measured ratio above 0.2, i.e. under 5x) is not remotely approached. `dev_nonzero` is
unchanged at 1154/1280, `band_recompute_bad=0` and `anchor_recompute_bad=0` still hold, and
`band_over_ulp_min` moved 5690 → 5687 (the bf16 families' band is a hair narrower at index 2047).
`drop_last_rejected` per family: 515/896, 38/100, **42/120** (was 63), **26/100** (was 38), 37/64 — the
drop-a-term direction still refuses on every family, including the two that doubled.

## The witnesses — every one of them was seen firing, then restored

All nine were re-run with the final code; the distilled record is committed as
`lm-head-band-form-witnesses-2026-10-26.log` (each full log is ~200 KB and regenerable under
`work/bandfix/` — the distiller is `work/bandfix/mk_witness_log.py`, and it refuses to write if any
witness printed no sabotage marker or went neither red nor refused).

| # | what was sabotaged | named check(s) that went RED | restored |
|---|---|---|---|
| W1 | `--gamma-form wide`: the **producer** prices every family `gamma_{2K-1}`, including the six whose `products_inexact == 0` | gate: `CHECK name=the band file equals gamma_index(products_inexact) * fsum(|h_k w_k|) recomputed from these bytes ok=0 n=380 verified=380 expected=1884 mismatches=6`; `CHECK name=no family whose dot products are all exact is priced with the gamma_{2K-1} index ok=0 n=6 exact_family_mispriced=[(0, 0, 0, 1.6268e-04, 8.1311e-05), …]`; **and two more that matter more than the bookkeeping**: `host case=125 inj-bf16-embd-half-accum FIRED beyond the band ok=0 … worst=0.61 x band` and `host every injection that must fire fired ok=0 … did not fire: inj-bf16-embd-half-accum` — pricing an exact family wide costs a must-catch injection its firing → `RESULT FAIL lm_head host-only checks=1517 findings=4`. On that same `band.f64`, `floor_probe.py` printed `RESULT FAIL lm_head floor probe findings=4`, three of them naming tables 0, 1 and 4: *"every one of 1835008 dot terms IS exactly representable … yet 896 of its 896 probed words are priced wider than that … pricing an exact family wide is a band chosen by hand"* | yes |
| W2 | `--gamma-form narrow`: the producer prices every family `gamma_{K-1}`, i.e. the pre-adjudication state for the inexact families | gate: `CHECK name=no family with a measurably inexact dot product is priced with the narrow gamma index ok=0 n=6` plus the equality check and the three `INSIDE`-class checks for `case=101`/`case=149` (worst back to 1.03x and 1.25x, i.e. the class flips of the previous section are exactly the price of the widening) → `RESULT FAIL lm_head host-only checks=1517 findings=5`. On that band, `floor_probe.py`: the two original FINDING lines **reappear** — `FINDING table 2 (enc=1): 7715 of 245760 dot terms h_k*w_k are NOT exact in binary32 … which licenses gamma_4095 … yet 120 of its 120 probed words are priced with a different index (120 at gamma_{K-1} = one rounding per term short …)`, same for table 3 → `RESULT FAIL lm_head floor probe findings=3` | yes |
| W3 | `--control band-unpin --tamper-band unpin-narrow`: every word's band rewritten on disk to `gamma_{K-1} * sigma` — 380 of 1884 words moved (`BANDTAMPER mode=unpin-narrow … words_changed=380 of 1884 … first_word=160 new_band=8.13262139e-05 licensed_band=0.000162712022`) | `RESULT FAIL lm_head host-only checks=1517 findings=5`, first red = the equality check, plus the narrow-on-inexact check and the three class checks | yes |
| W4 | `--tamper-band word-narrow`: **one** word's band halved (`words_changed=1 of 1884`, word 0, `8.13108825e-05 → 4.06554412e-05`) | `RESULT FAIL lm_head host-only checks=1517 findings=2`: the equality check (`verified=1883 expected=1884`, `mismatches=1`) and the mispricing check | yes |
| W5 | `--tamper-band word-wide`: **one** word's band doubled (word 0, `8.13108825e-05 → 1.62621765e-04`) — the *widening* direction, which is the forbidden one | `RESULT FAIL lm_head host-only checks=1517 findings=2`, same two named checks | yes |
| W6 | `--control band-perturb`: one output word of `case=16 honest-q8-lmhead-dense-batched` (**table 2 — the family whose band doubled**) moved by 4x its own band: `band=0.000162712022 perturbation=0.000650848088 old=-0.0532329381 new=-0.0525820889` | `CHECK name=host case=16 honest-q8-lmhead-dense-batched BAND inside the derived band ok=0 n=16 worst=4.000 x band outside=1 first=(160, -0.052582088857889175, -0.05323289334774017, 0.00016271202190668756)`, and nine downstream reds (the two digests of that case, the two EXACT-vs-twin checks that read word 160, the `INSIDE` guards) → `RESULT FAIL lm_head host-only checks=1517 findings=10`. **The doubled band still rejects a term-sized error** | yes |
| W7 | `--control no-fire` (pre-existing): an injection neutralised after the census ran, so 16 words are claimed honest but were never banded. Before this change the run refused on a NaN band; the census must not crash where the old code refused | the coverage check now names it: `CHECK name=every banded word was priced, none is un-banded, and no band is below one ulp of its own answer ok=0 n=1900 sigmas=1884 words=1900 unbanded=16 first_unbanded=[(190, 27456), …]` → `RESULT FAIL lm_head host-only checks=1518 findings=3` | yes |

W1 and W2 are the two directions the anti-drift requirement asks for: a family moves to the wide index
only when a measurement puts a product off the binary32 grid, and a family that measures exact is refused
the wide index — and W1 shows the refusal is not ceremonial, because the wide-on-exact band loses a
must-catch injection.

## The two class changes the widening caused — reported, not routed around

`pitfalls/30-numerics.md`: a widened band that costs a gate its discriminating power is *reported*. Two
`HALF_ACCUM` injection arms moved class, and no verdict went red:

```
before                                                            after
case=101 inj-q8-lmhead-half-accum FIRES   worst=1.03 x   ->   case=101 … INSIDE  worst=0.514 x
case=149 inj-q8-embd-half-accum   FIRES   worst=1.25 x   ->   case=149 … INSIDE  worst=0.625 x
```

They fire against a band that was half the size the proof supports, so their "firing" was measuring the
band's error, not the injection's. The other four `HALF_ACCUM` arms still fire by 14.1x, 9.1x, 12.2x and
6.2x, so the mode still has a firing witness in the plan (`ALTREACH` holds) and the must-catch set is
intact. A per-case diff of all 192 derived classes shows these two lines and no other change:

```
102c102  < case=101 derived=FIRES   ---   > case=101 derived=INSIDE
150c150  < case=149 derived=FIRES   ---   > case=149 derived=INSIDE
```

## NOT ESTABLISHED by this change

* **Nothing about the iGPU.** The device arm was not run here (CPU only). Every committed device log in
  this directory — including `lm-head-device-clean-2026-10-26.log` with `checks=3012 findings=0` — was
  produced under the OLD band, and so is stale for `BAND`-classed device words and for the SEQ
  bit-exactness claim. The device gate has to be re-run in the main agent's window before the lane
  report can carry a device verdict again.
* **The SEQ zero-tolerance claim is now resting on the empirical observation, not on exactness.**
  `lm_head_kernels.h` says so explicitly. If a device re-run finds a host/device SEQ divergence on a
  Q8_0 case, the answer is a build-level contraction policy and NOT a band: SEQ is bit-exact by
  contract, and widening a `BAND` buys it nothing.
* that the band is *tight*. It is ~1000x wider than the worst of five legal re-associations. A tighter
  band would need the observed spread as an input, which is the "band from what the build produces" move
  `AGENTS.md` forbids.
* that `gamma_{2K-1}` is the *sharpest* true bound. It is the one Higham's lemma gives for K terms with
  one pre-rounding each; a probabilistic or cancellation-aware bound would be smaller and is not claimed
  and not needed.
* anything outside this lane. No other prototype's band, and nothing in `tools/ulp_policy.py`, moved.
