# D-28 in `engine/block_device.py`: the compiled pair is judged in value space, not byte space

Run on this box 2026-10-27 against work dir `engine/work/p11-f3` (CPU witness: `device=off`,
`hip_calls=0+0`, `launches=0`, `no timing taken`). Every number below is printed by the run and
re-printed here; nothing in the derivation reads a device number, which is D-28's own rule ("No number
here may be read off the build it covers").

## What changed, and what did not

Two checks stopped requiring byte identity:

| old name | new name | verdict now rests on |
|---|---|---|
| `gather-compiled-pair-byte-identical-per-path` | `gather-compiled-pair-value-agreement-per-path` | policy A: per-field value band; `ref_bf16`: byte identity, unchanged |
| `gdn-compiled-pair-byte-identical` | `gdn-compiled-pair-value-agreement` | device-vs-host leg: per-field value band; the g++ twin pair: byte identity at zero tolerance, unchanged |

Left byte-exact by D-28 and untouched here: the host-twin pairs, the bf16 reference gather, the
router's `ids`/`ids_alt`/`wcast`, every M1 read-back digest, every pinned-oracle licence, and
`band_digest=27a68c7e82c9aa05`. The fields a documented transcendental does not reach — gather
`gate`/`up`, GDN `q_hat`/`k_hat`/`core_attn`/`normed`/`y` — get a derived bound of exactly **0**, from
the arm's own reach claim (`engine/device/m2_g_kernels.h::m2g_mode_field` and
`engine/device/m2_d_kernels.h::m2d_mode_fields`, echoed by each arm's `HDR … mode_fields=` line), so
"must stay exact" is a bound and not
a sentence.

## The three device-free sources

1. **`mr_exp` documented allowance × the arm's own host-path finite difference.**
   `prototype/moe_router/moe_router_kernels.h::mr_exp` says the two `exp` implementations are
   "allowed to differ by an ULP or two"; the run reads that sentence at run time
   (`vb_documented_allowance`), refuses if the wording moves, and uses 2 ULP. The amplification is
   measured by re-running the arm's *host* body with `exp` bumped ±2 ULP at every documented site
   (gather `exp_ulp_table`, population 4096 = `MG_K*MG_I`; GDN `decay_ulp_mask`, population 32 heads),
   under sign patterns `all_up`, `all_down` and, for the GDN, `split_odd_up`+`split_even_down` so a
   mixed-sign assignment is priced too. `BOUNDFROM-SITE … additivity=single-sign-legs-only` records
   that the mixed leg is *summed*, not multiplied: the legs are not additive, so `min` over sign
   patterns is the conservative read.
2. **`licensor_band_triangle`** — the licensor's published width for the same field
   (`speccheck/evidence/gdn-band-2026-10-26.json`, self-described `derived_from="floor probe (f64
   anchor) x the triangle inequality"`), doubled: `rel-err = 2·mean_rel`,
   `relL2 = 2·sqrt(2·one_minus_cos)`. Same shape as prior 1 in
   `specification/evidence/blockdevice-wpre-scope-2026-10-26.md` (`min(10, 9, 22)`). It exists only
   where the path under test is licensed: `scan_out`/`state` are ACCEPTED at this pass, so they carry
   this source; policy A's gather rows are UNLICENSED `policy-a-width-band-absent`, so the gather
   prints that as an unavailable reason rather than borrowing a number.
3. **`fp64_floor_probe`** — the same arithmetic in float64 from operands the record already carries:
   `s = silu(gate)*up` (`engine/device/m2_g_kernels.h::m2g_item1`) and
   `routed = Σ_slot raw[slot]·wcast[slot]` (`engine/device/m2_g_kernels.h::m2g_item4`). The probe
   measures |fp64 − host|; it bounds |device − host| only under the documented premise (D-19 §7) that
   both arms round the same terms to the same order of error, so the prior is **2×** the measured
   distance and the premise is printed inside the source label. `raw`'s down_proj dot has no floor
   probe — the record does not carry its operands — and says so instead of inventing one.

The strictest available prior binds (`min`), as in `WPRESCOPE`. Where exactly one prior exists the
line prints `single_source=yes`, and `d28-bounds-come-from-documented-sources` admits only three
shapes of that: zero-tolerance cells, `gather.raw` (D-14, admitted on the strength of the grown
must-catch set), and count cells whose other two families each print a reason for being absent.

## The derived bounds (policy A and the GDN device leg)

`derived` below is per field, per currency — `rel-err` = mean of |a−b| / max(|b|, 1e-3·max|b|) (the
licensor's own anchor denominator, `speccheck/gdn_ref.py::aggregates`), `relL2` = ‖a−b‖/‖b‖,
`ndiff_frac` = differing f32 words / words.

| pair | field | rel-err | relL2 | ndiff_frac | binding source |
|---|---|---|---|---|---|
| `g_policy_a_hip_device_vs_hip_hostpath` | `s` | 3.61992e-08 | 9.74303e-08 | 0.368164 | floor probe ×2, host fd |
| | `raw` | 6.26231e-07 | 2.00838e-07 | 0.855469 | host fd (single source, D-14) |
| | `routed` | 3.24921e-07 | 1.30428e-07 | 0.858887 | floor probe ×2, host fd |
| `d_hip_device_vs_d_hip_hostpath` | `scan_out` | 1.46299e-07 | 1.91771e-07 | 0.669678 | host fd, licensor band |
| | `state` | 6.17885e-08 | 1.74267e-07 | 0.695086 | host fd, licensor band |

Every other field of both pairs: `derived=0` in all three currencies (`teeth=zero-tolerance`).
`ndiff_frac` is deliberately the weakest currency — the value currencies are where the teeth are, and
each count line prints `teeth=fraction-forbidden=…` so a reader can see how much of the field a count
bound actually forbids (a derived count of ≥1.0 is refused outright by `d28-count-bound-vacuous`).

## The witness is separate, and in a CPU witness it says so

`BOUNDWITNESS` is printed per field per currency, never folded into a PASS. On this box the device
record does not exist, so the device-vs-host pair prints

```
BOUNDWITNESS pair=g_policy_a_hip_device_vs_hip_hostpath field=all metric=all measured=not-claimed derived=see-above margin=none inside=not-claimed for_leg=g_policy_a_hip_device_vs_hip_hostpath note=no device record exists in a CPU witness, so the device-vs-host leg is not claimed here; the command that takes it is engine/block_device.py --work engine/work/<work> --device
```

and the pair that *does* exist here — the g++ host twin against the hipcc host path — is measured for
real: 19 cells per pair, all `measured=0 inside=yes`. `d28-value-band-witness-inside-and-edge-legs-reject`
fails if any measured cell is outside its derived bound, and also if any rejection leg is missing.

The command that takes the device leg is exactly

```
/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python engine/block_device.py --work engine/work/m2dev --device
```

under the standing device rules (`tools/gpu_lock.sh` held, halogen down, zero active workers, main
agent in the foreground). Its required prints are the 36 `BOUNDFROM` lines (already derived, from the
same sources), then 36 `BOUNDWITNESS` cells with `measured=` numbers instead of `not-claimed`, plus
the two `REFUTED-CAUSE … teeth=` lines and the `DREPRO`/`GREPRO` diagnostics labelled
`role=diagnostic gate=no`.

## The rejection legs have a real edge

`BOUNDEDGE` fabricates, in memory only, a copy of the A side sized *from the bound* — one element
moved by `t·‖b‖` for `relL2`, one element at the denominator floor moved by `t·n·denom` for `rel-err`,
`floor(t·n)` differing words for `ndiff_frac` — then tightens (inside) or widens (outside) until the
leg is unambiguously on its own side, and prints where it landed:

```
BOUNDEDGE pair=g_policy_a_hip_device_vs_hip_hostpath metric=rel-err field=s leg=at-limit bound=3.61992e-08 measured=3.61878e-08 sized_at=1_of_bound inside=True verdict=INSIDE note=FABRICATION, in-memory only, …
BOUNDEDGE pair=g_policy_a_hip_device_vs_hip_hostpath metric=rel-err field=s leg=one-past-limit bound=3.61992e-08 measured=3.84714e-08 sized_at=1.062_of_bound inside=False verdict=OUTSIDE note=FABRICATION, in-memory only, …
```

Twelve legs print — at-limit and one-past for each of the three currencies on each pair's reached
field — and all twelve split `INSIDE` / `OUTSIDE` correctly;
`d28-value-band-witness-inside-and-edge-legs-reject` requires both verdicts per metric.

## The must-catch set is derived from geometry, not from the build

`MUSTCATCH-DERIVE` prints the geometry it used (`MG_K=8`, `MG_I=512`, `MG_H=2048`, gate|up laid out
along the output dim, each field's own span boundary; `GDN_H_V=32`, `GDN_D_K=GDN_D_V=128`,
state span 524,288 words) and the population it derived — 3 for the gather, 5 for the GDN, 8 printed
`MUSTCATCH … reddens=RED`, zero `CLEAN`, and the check reconciles printed against derived
(`must_catch_printed == must_catch_derived`, and a zero population reddens the run):

| name | defect | first currency that reddens |
|---|---|---|
| `gate_up_halves_swapped` | exchange the gate and up halves (512 words each) | `gate/rel-err 2.31 > 0` |
| `wrong_expert_id_slot_swapped` | exchange expert slots 0 and 1 of `s`/`raw`/`up` | `gate/rel-err 13.9 > 0` |
| `span_off_by_one` (gather) | rotate `raw` by one word | `raw/rel-err 4.1 > 6.26e-07` |
| `span_off_by_one` (GDN) | rotate `state` by one word | `state/rel-err 1.55 > 6.18e-08` |
| `state_two_heads_permuted` | swap head 0 and head 1 of the state | `state/rel-err 0.149 > 6.18e-08` |
| `state_one_head_zeroed` | zero head 2 of the state | `state/rel-err 0.0182 > 6.18e-08` |
| `carried_state_dropped` | zero the whole state (record-space skip-carry) | `state/rel-err 0.571 > 6.18e-08` |
| `decay_sign_inverted` | `state[h] *= exp(2·g_decay[h])`, i.e. `exp(−g)` where §04b.1 step 1 says `exp(+g)`, `g_decay` read from the slab span | `state/rel-err 0.0968 > 6.18e-08` |

Every one of them reddens by five to seven orders of magnitude in the value currencies, which is the
point D-28 makes: a real defect is not within 1 ULP of anything, so a band tight enough to catch it
costs nothing at the rounding scale.

## The refuted causes: one still refuted by the band, one that the band cannot refute

```
REFUTED-CAUSE cause=silu_exp_ulp1 pair=g_policy_a_hip_device_vs_hip_hostpath teeth=yes outside=routed/rel-err 5.65e-07 vs 3.25e-07;routed/relL2 1.69e-07 vs 1.3e-07 verdict=STILL-REFUTED refutation_carried_by=band:rel-err,relL2
REFUTED-CAUSE cause=decay_ulp1 pair=d_hip_device_vs_d_hip_hostpath teeth=none outside=none verdict=NOW-INSIDE-BAND refutation_carried_by=byte-count-ruler(leg-not-printed-in-this-run(mode=cpu)); required print wherever a real device record exists: DMAGMATCH/GMAGMATCH verdict=RULED-OUT cause=decay_ulp1, taken by engine/block_device.py --work … --device note=THE BAND IS TOO WIDE …
```

That second line is the honest result and it is stated rather than smoothed. `decay_ulp1` is +1 ULP on
`exp(g_decay)` for **all 32 heads**; the derived GDN bound is sized at the documented worst case, which
is ±2 ULP on all 32 heads. A bound that admits the larger member of the same family necessarily admits
the smaller one in value space — the only thing that ever separated them was the byte-count *ratio*
against the observed pair (264,704 B injected against 52,976 B observed, and 15,426 B against 3,744 B
on policy A), and the observed pair is device data. No device-free bound can encode it. So:

* the band keeps the teeth it can keep — `silu_exp_ulp1` is still outside it in `rel-err`/`relL2`;
* `decay_ulp1` stays a live control that keeps failing in the ruler that refuted it:
  `DMAGMATCH`/`GMAGMATCH verdict=RULED-OUT`, printed in every device window and reddening
  `gather-divergence-causes-priced`/`gdn-divergence-causes-priced` if a cause ever matches. The
  required print for that verdict is printed as an obligation on the line above, and
  `d28-refuted-causes-still-refuted-and-say-what-carries-it` refuses to pass if a refuted cause is
  inside the band without naming the control that carries it.

If the reader's judgement is that D-28 §3 must be satisfied *by the value band itself*, then the band
cannot be derived without the device and the derivation must be re-scoped — that is a user call, not
an agent one, and this file is the place the tension is recorded.

## The diagnostics stay printed, and they no longer vote

`GREPRO`, `DREPRO` and `DREPRO-WIN` each carry `role=diagnostic gate=no` at their print sites. In a
CPU witness not one of them prints (device legs; the run says so on two `DVOID … required-print=`
lines), so `d28-diagnostic-legs-are-labelled-and-do-not-decide` is written structurally: it parses
this file's own AST, finds every `print` whose first literal begins with one of the three leg names,
and requires the label there — `DREPRO(labelled),DREPRO-WIN(labelled),GREPRO(labelled)` — and it also
asserts that the two pair CHECK lines cite their `BOUNDWITNESS` cells and cite no
`vs_device_record_ndiff_bytes`.

## The escalation clause is wired as a rule and demonstrated

`d28_mechanism_offences` reads the run's own diagnostics for a value band that passed beside a
mechanism that is not the documented `exp` last bit: a `DREPRO`/`GREPRO` leg with `unexplained != 0` or
no `EXACT-REPRODUCTION` leg where one is required, a `DMAGMATCH`/`GMAGMATCH verdict=CANDIDATE` whose
cause is not an `exp`/`decay` cause, and a device pass with no reproduction leg at all.
`d28_qualification_decision` then returns `ESCALATE-USER(…)` in place of `QUALIFIED`.
`d28-escalation-clause-refuses-qualification` demonstrates it on a fabricated log (a residual-left
`DREPRO` and a `DMAGMATCH … cause=state_xpose verdict=CANDIDATE`): 2 refusals from the fabrications,
0 from this run's real diagnostics, and the verdict word computed here is `QUALIFIED`.

## Gate lines from the run that produced these numbers

```
BLOCKDEVICE verdict=QUALIFIED block=layer0 subsystem=shared_expert+router+gdn_recurrent rows=6 gdn_rows=7 parsed=17 licensed=9 accepted=9 rejected=0 unlicensed=8 checks=67 red=0 controls=31 fired=31 band_digest=27a68c7e82c9aa05 note=a row with no licence is never counted as accepted, and PASS is unreachable while any row is unlicensed
RESULT mode=cpu checks=67 findings=0 controls=31 fired=31 rows=6 gdn_rows=7 arms=policy_a device=off staged=not-attempted hip_calls=0+0 launches=0 bytes_to_device=0 bytes_from_device=0 counters_unparsed=0 band_digest=27a68c7e82c9aa05 no timing taken
```

`checks` grew from 60 to 67 — the 7 new D-28 checks — with `verdict=QUALIFIED red=0 controls=31
fired=31 findings=0 counters_unparsed=0` unchanged from the pre-D-28 baseline on this same box, and
`band_digest=27a68c7e82c9aa05` byte-identical. `QUALIFIED` (not `PASS`) is the standing word while the
8 policy-A rows are unlicensed, and it is the word `d28_qualification_decision` computes: the
escalation clause can only demote it, never promote anything.

Beside it: `engine/block_gate.py`
`checks=17 red=0 controls=11 fired=11`, `engine/gate.py` `RESULT PASS checks=27 findings=0`,
`python3 tools/doc_ref_check.py --strict` findings 0, and the device-shaped rehearsal
`REHEARSAL verdict=PASS no device act occurred, no timing taken` confined to `--work m2dev-rehearsal`.
