# block_device.py — the two device-only divergences, attributed as far as a CPU session can

Provenance: produced 2026-10-27 by the pass-7 worker (CPU-only; no device act, `hip_calls=0+0`,
`launches=0`, `bytes_to_device=0`, `bytes_from_device=0`, no `gpu_lock`, and no device timing of any kind
appears in this note or in the logs beside it — the only rate-shaped tokens in those logs are `out_elems=`
and `claims=`). The numbers below are byte counts, bit-image steps, and host RSS/wall.

## 0. The two rows this is about, verbatim from the crashed window

```text
devices/evidence/device-window-2026-10-27-g.md:129-130
  g_policy_a_hip_device_vs_hip_hostpath  gather_record.f32  122,880 B  ndiff_bytes=3,744   (3.05%)
  d_hip_device_vs_d_hip_hostpath         gdn_record.f32   2,187,264 B  ndiff_bytes=52,976  (2.42%)
and, beside them:
  g_ref_bf16_hip_device_vs_hip_hostpath  gather_record.f32  122,880 B  ndiff_bytes=0
```

Window `g` died in `control_gather_read_back` (`KeyError: '_bytes'`, `block_device.py:2055`) before it
could print anything finer, so the two rows were totals with no attribution. Nothing here bands them,
shrinks a claimed exactness, or touches `band_digest=27a68c7e82c9aa05`.

## 1. What was built to attribute them

`engine/block_device.py` now prints, for every compiled pair it has both sides of:

- `GATTRIB` / `DATTRIB` — one line per field, `ndiff_words`, `ndiff_bytes`, `max_bit_step`,
  `median_bit_step`, then an accounting line whose `sum_of_fields` must equal the pair's own
  `ndiff_bytes` (`pitfalls/40-gate §the-table-that-accounts-for-its-cells`).
  `check field-attribution-accounts-to-the-pair` holds every printed table to that, and it has been
  witnessed non-zero: the first version of the splitter mis-read the record layout's
  `(start, length)` pairs as `(start, end)` and the accounting line printed
  `accounting=MISMATCH` on 9 tables before the mistake was fixed.
- `DHEAD` — the GDN `state` field split by head (32 × 16,384 words), because one `exp()` per head is
  applied to a whole head-block, so "k of the 32 decays differ" is only visible as "k head-blocks moved".
- `GINJECT` / `DINJECT` — the same per-field table for a **cause injected inside the arm** (an arm mode,
  not a second Python implementation that could drift from the thing under test), and `GCAUSE`/`DCAUSE`
  one-line summaries. `GMAGMATCH`/`DMAGMATCH` compare them with the device pair when a device window
  supplies it; in a CPU run the device legs print as `GVOID`/`DVOID ... status=not-exercised`.
- `GEXPSCOPE` / `DEXPSCOPE` — how faithful the **host** side of the pair is at the transcendental sites,
  glibc `expf` (the function the host body links) against `Decimal.exp()` at 60 digits rounded once to
  f32 (the second route `DECISIONS.md` D-14 used).

## 2. Measured injected magnitudes (host path of the hipcc arm, one input set)

### gather, `gather_record.f32` (122,880 B)

| width | cause | total ndiff_bytes | fields (ndiff_bytes, max_bit_step) | ÷ device 3,744 |
|---|---|---|---|---|
| ref_bf16 | `silu_exp_ulp1` | **0** | none | — |
| ref_bf16 | `cast_after` | 968 | routed 968, 2.43e9 | — |
| ref_bf16 | `acc_bf16` | 0 | none | — |
| ref_bf16 | `rank_accum` | 1,284 | routed 1,284, 2.46e9 | — |
| ref_bf16 | `byte_shift_1` | 54,924 | gate 7,642 up 7,637 s 6,302 raw 29,641 routed 3,702 | — |
| policy_a | `silu_exp_ulp1` | **15,426** | s 975 (3), raw 12,743 (14,336), routed 1,708 (1,536); gate/up 0 | **4.12** |
| policy_a | `cast_after` | **4,982** | routed 4,982 (1.43e7) | **0.75** |
| policy_a | `acc_bf16` | **5,402** | routed 5,402 (3.37e9) | **0.69** |
| policy_a | `rank_accum` | **1,254** | routed 1,254 (640) | **2.99** |
| policy_a | `byte_shift_1` | 110,212 | all five fields | 29.4 |

The `ref_bf16` row for `silu_exp_ulp1` reading **0** while `policy_a` reads 15,426, on the same kernel
and the same injected mode, is the load-bearing cell: at ref width every `proj_out` weight is exactly 0
(`0.0 * anything = 0`), so a silu last-bit cannot reach the output there — and the same asymmetry is
what the device shows (`ref` byte-exact, `policy_a` 3,744). The counter reads both 0 and non-zero in
this run, so it is not vacuous.

### GDN, `gdn_record.f32` (2,187,264 B)

| cause | total | fields (ndiff_bytes, max_bit_step) | ÷ device 52,976 |
|---|---|---|---|
| `decay_ulp1` | **264,704** | scan_out 2,446 (768), state 262,258 (12,288); q_hat/k_hat/core_attn/normed/y **0** | **5.00** |
| `decay_head_off1` | 1,533,482 | scan_out 11,873 core_attn 3,970 state 1,511,231 normed 3,723 y 2,685 | 28.9 |
| `state_xpose` | 1,308,457 | scan_out 15,858 core_attn 7,689 state 1,273,446 normed 7,707 y 3,757 | 24.7 |

`DHEAD` for `decay_ulp1`: **26 of 32 heads** moved, ~10.7 KB each (6 heads unchanged, where the 1-ULP
bump does not change the rounded `exp`). The honest host-vs-host pair: all 32 heads 0, `accounting=ok`.

### 2b. The arms' own reach tables were wrong twice, and the injection controls caught it

Every arm publishes, in its `HDR` line, which record fields the mode it is running can reach
(`m2g_mode_field`, `m2d_mode_fields`). Pricing a cause is only meaningful if that claim is true, so
`GINJECT`/`DINJECT` now compare the claim with what the mode actually moved: `GREACH`/`DREACH` lines, and
the two `…-divergence-causes-priced` controls go quiet-with-reason when they disagree. Rules, one per lane:
**moving a field the claim does not name is always an offence**; **claiming a field the mode did not move**
is an offence only when the cause moves nothing on any width — at ref width `s`/`raw` are rounded to bf16
and a last-bit `exp` is swallowed whole, which is not an inert mode but the very asymmetry this note is
about, and the line prints it as `claimed_but_unmoved=raw,routed,s`.

Two claims were wrong and are corrected against the measurements above:

- `m2g_mode_field(MG_MODE_BYTE_SHIFT_1)` claimed `gate,up,s,raw`; it also reaches `routed` through `raw`.
  Before the correction the CPU witness printed
  `GREACH path=ref_bf16 cause=byte_shift_1 … beyond_claim=routed` and the control went quiet with
  `findings=1` — the rule is witnessed biting, not decorative.
- `m2d_mode_fields` returned the five state-stage fields for every non-`correct` mode, including the decay
  one-ULP-up mode, which measures exactly `scan_out,state` (264,704 B). An over-claimed reach is how a
  partly-inert injection would pass as a priced cause.

Both corrections tighten a claim, never relax one: no band, floor or factor was authored, and
`band_digest=27a68c7e82c9aa05` is unchanged.

### 2c. The route-disagreement counter was itself wrong, and by a factor of n

`GEXPSCOPE`/`DEXPSCOPE` print how many of the transcendental sites in *this row's own inputs* glibc's `expf`
disagrees with `Decimal.exp()` at 60 digits rounded once to f32. An earlier revision built the reference
route from `struct.unpack(...)`, which returns a **tuple**, so the array came back `(n,1)`, broadcast against
the host route's `(n,)`, and the printed "count" was a fraction of an n×n grid:
`GEXPSCOPE sites=4096 … disagree=13236276 fraction=3231.512695` — a "disagreement rate" of 3,231× the
population, which is impossible and printed anyway because nothing held the count to its denominator.
Fixed (`[0]` on the unpack) and now guarded twice: the two routes' shapes must equal the population's, and a
count larger than the population refuses the run with `exp-route-count-exceeds-population`. The correct
numbers, from the log beside this note:

```text
GEXPSCOPE sites=4096 … disagree=1 fraction=0.000244 max_bit_step=1 path=policy_a
GEXPSCOPE sites=4096 … disagree=2 fraction=0.000488 max_bit_step=1 path=ref_bf16
DEXPSCOPE sites=32   … disagree=0 fraction=0.000000 max_bit_step=0
```

Each of the three glibc disagreements is exactly one bit-image step, so the host side of both pairs is within
1 step of a correctly-rounded answer at every transcendental site it uses — the number §3 rests on. This is
also `pitfalls/40` §a-check-that-can-print-nothing-cannot-fail applied to a *rate*: a printed count must be
inside the denominator it is printed with, and the denominator must be printed with it.

## 3. Mechanism claim, per field, and what is ruled out

**gather, 3,744 B.** Two candidate families fit the *size*, and they are separable by field:
- *transcendental last-bit in silu*: reproduces the field **shape** `s,raw,routed` with `gate=up=0`
  exactly as the kernel geometry requires, but needs ~4× the bytes the device moved;
- *combine-stage difference* (`cast_after` 4,982 / `acc_bf16` 5,402 / `rank_accum` 1,254): all three are
  within a factor of 3 of 3,744 and are **`routed`-only**. Two of them are inside a factor of 1.4.
  Note that all three leave `s`/`raw` untouched, which the device total cannot distinguish from the
  transcendental family.

So the honest statement is: **no injected cause reproduces the device's 3,744 B together with its field
split, because the field split has never been printed.** The two families make *different* predictions —
`s,raw,routed` with `max_bit_step` in the low thousands, versus `routed` only — and one window run
distinguishes them. `silu_exp_ulp1` is *not* ruled out (4.1× over is consistent with "the device's libm
disagrees on a quarter of the sites glibc-and-correct disagrees on ×1.03"), but it is not established.

**GDN, 52,976 B.** The transcendental family is the only one of the three that leaves
`q_hat/k_hat/core_attn/normed/y` at zero, which the geometry requires (those five fields contain no
decay), and its per-head ruler says the device's 52,976 B ≈ **4.9 head-blocks**, i.e. ~5 of 32 decays
differ, where a uniform 1-ULP bump moves 26. So the same cause, at ~19% of its sites, fits exactly.
The two index defects are ruled out **for this pair by the two host paths only insofar as** the honest
pair is byte-exact; they remain live as *device-only* index/stride defects and are the next thing to
check if `DHEAD` on the device shows a scattered-but-large or contiguous pattern.

**The transcendental hypothesis is quantitatively strained, and here is the number that strains it:**
on the inputs these rows actually use, glibc `expf` disagrees with a correctly-rounded route on
**1 of 4,096** silu sites (policy_a), **2 of 4,096** (ref_bf16) and **0 of 32** GDN decay sites. For the
device to be a pure last-bit libm story, `ocml` would have to differ from glibc on ~5× more of these
inputs than glibc differs from correct — possible, but it must be *said*, not assumed, and it is exactly
the kind of number that has been quoted from a different workload before (`pitfalls/30-numerics.md`
§"the same number quoted from two places that measure different things").

**What is NOT the cause, measurably:** a wrong span/rank/offset — the cheapest device-only index defect
(`byte_shift_1`, one-byte `rank_off`) moves **110,212 B = 29× the observed 3,744** and moves all five
fields with huge bit steps. The device's 3,744 cannot be that defect. Same for `decay_head_off1` and
`state_xpose` (25-29× too large, and they move fields the device pair does not).

## 4. Is there a document that already declares this scope? — No, and here is the closest one

- `DECISIONS.md` D-16 (user, 2026-10-26) records, for `prototype/moe_gather`, that
  "the host twin's `s/raw` are glibc `expf`, the device's are `libdevice.expf` (libdevice-vs-host-glibc
  `expf` disagreement ... 16584 of 44950 in one block, **counted and never banded**)". That is the same
  *kind* of difference in the same *fields* — but it is a declaration about the prototype lane's own
  pair and cross-checks, D-16's cross-arm whole-buffer row explicitly says it "changes no band", and
  D-26's outranking clause says a quantized oracle may only **add** checks, never loosen a band. It is
  not a licence for the M2 device pair's `routed` field, and `routed` is not named in it at all.
- `DECISIONS.md` D-14 accepted **one** `ssm_a` `exp(A_log)` case, on two routes agreeing within 1 ULP,
  and says explicitly it is not a general 1-ULP allowance. The GDN decay here is a different `exp` at a
  different site; D-14's 1-ULP predicate would in fact *not* pass the GDN record (max bit step 12,288
  on `state`, because one decay last-bit propagates through 12,288 state words).
- `prototype/gdn/gdn_kernels.h:61` records that the decay uses plain `expf`, "**not** `__expf` —
  `__expf` is the fast intrinsic ... it would be a new band". The engine arm follows it. There is no
  document that declares a device-vs-host-libm scope for `engine/device/m2_d_*`.
- `pitfalls/30-numerics.md` §"the ±1-ULP band" states the acceptance is ±1 **value**-ULP, not ±1 bit
  step, and that bit-image comparison is a *diagnostic*; and §a-divergence-count-that-changes-between-
  identical-runs is a race gives the three-rung test (constant count, adjacent images, attributed third
  side) that must be green before any scope word is used.

**Conclusion, stated as the constraint it is:** an agent may not declare these two divergences
"documented scope", because no document covers this pair-and-field, and because relaxing an exactness
claim is on the list reserved to the user (`DECISIONS.md` D-18, and the reserved list in `AGENTS.md`
§GOAL). The `PAIR ... verdict=PASS` lines stay as they are, the two pair checks stay RED on a device
window, and the attribution now prints enough to let the user decide in one look.

## 5. What the next device window has to print (nothing here can settle it)

`engine/HANDOFF.md` §PASS 7 has the exact command. The lines it must produce, and what each decides:

1. `GATTRIB pair=g_policy_a_hip_device_vs_hip_hostpath ... field=...` ×5 + accounting — if
   `fields_with_diff = s,raw,routed` with `max_bit_step` ≤ ~4·10³, the transcendental family is the
   mechanism and the user is being asked to license it; if `fields_with_diff = routed` only, the
   mechanism is in the combine stage and it is a **kernel defect**, not a scope.
2. `DATTRIB pair=d_hip_device_vs_d_hip_hostpath ...` + `DHEAD pair=d_hip_device_vs_d_hip_hostpath ...`
   — `heads_with_diff≈5` with the other five fields at 0 is the ~5-of-32-decay story; any nonzero
   `q_hat`/`k_hat` immediately reopens the index-defect class.
3. `GMAGMATCH`/`DMAGMATCH ... matched=<cause>|none` — the arm's own verdict against the factor-two rule.
4. The same command run **twice**, with the `GATTRIB`/`DATTRIB`/`DHEAD` lines diffed: a constant count
   plus adjacent bit images is rounding; a moving count is uninitialised scratch or a race and may
   never be banded (`pitfalls/00-measurement.md`).
5. `GINJECT`/`DINJECT` in that window will re-price the causes on the device window's own input set; the
   device-path injection leg and the two "constant across identical runs" legs are printed as
   `DVOID ... status=not-exercised` in a CPU run and are the window's to clear.

## 6. Logs

- `specification/evidence/blockdevice-cpu-2026-10-27-p7.log` — the CPU witness that printed every number
  above. `RESULT mode=cpu checks=53 findings=0 controls=28 fired=28 ... band_digest=27a68c7e82c9aa05`
  with zero silent controls; host wall 8.0 s, RSS 544 MB.
- `specification/evidence/blockgate-cpu-2026-10-27-p7.log` — `engine/block_gate.py`, unchanged by this
  pass and still green: `checks=17 findings=0 controls=11 fired=11 ... band_digest=27a68c7e82c9aa05`.
- `devices/evidence/device-window-2026-10-27-g.md` — the crashed window (the totals being attributed).
