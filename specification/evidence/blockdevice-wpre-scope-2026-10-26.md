# `wpre` across two compilations — measurement, attribution, and the derived scope (2026-10-26)

Provenance: first device execution of the router arm, window `e2`
(`specification/evidence/blockdevice-device-2026-10-26-e2.log`), which printed
`PAIR pair=rt_hip_device_vs_rt_hip_hostpath … ndiff_bytes=237
claimed_ndiff=ids=0/8,ids_alt=0/8,wcast=0/8,wpre=5/8 verdict=CLAIMED-DIFFER` and reddened
`rt-device-host-claimed-fields-identical`. This file establishes **what that difference is** and **what
limit is admissible for it**. No band, licence, or `data/gdn-band-2026-10-26.json` byte is touched here;
the band digest throughout is
`27a68c7e82c9aa0507b9bcd677adc0d5d1b0bc865dd1e86bf495ed90f47392ee`.

## 1. Inputs, and the rule that made them available

Window `e2`'s three router records were copied out of `engine/work/pass2/` before that directory was
reused, and are the only device material this analysis used. **No device act was performed for any
number below** — no launch, no `hipcc --device` run, no `gpu_lock` (`engine/work/e2-device-records/`):

```
9430f861b126153ae8bb01e32b56b94665985a6b6849ed5850ede2d4e262bb97  rt_out_device.bin
63dccf2caf93be1f17fc0a9a4e1c443b1a1c2f249826df382199d976523c0c36  rt_out_host.bin     (g++ twin)
63dccf2caf93be1f17fc0a9a4e1c443b1a1c2f249826df382199d976523c0c36  rt_out_hip.bin      (hipcc host path)
```

`rt_out_host.bin` and `rt_out_hip.bin` are byte-identical, which is the accepted
`compiled-pair-byte-identical` fact restated: one compilation, two host front-ends. So
"device-vs-host" here means **one translation unit compiled for `gfx1151` against the same translation
unit compiled for the host** — the compiler/codegen axis, not a math-library-port axis.

## 2. Measurement (`engine/work/replay_e2_pair.py`, verbatim in `blockdevice-wpre-scope-replay-2026-10-26.log`)

Whole record: 237 of 3232 bytes differ. By field (`ndiff` = elements differing, worst step in fp32 ULP
under the licensor's ruler, `engine/block_gate.py::ulp_diff`):

| field | n | ndiff | max ULP | reading |
|---|---|---|---|---|
| `logits` (R1 → bf16) | 256 | 0 | 0 | R1's output, bit-identical |
| `logits_f32` (R1 raw) | 256 | 0 | 0 | R1 itself, bit-identical |
| `probs` (R2) | 256 | 232 | 3 | the `exp()`/dot route moved |
| `wpre` (R4, fp32 pre-cast) | 8 | **5** | **2** | the field under dispute |
| `wcast` (R5, bf16 cast) | 8 | 0 | 0 | the cast absorbs the fp32 spread |
| `ids`, `ids_alt` | 8 | 0 | 0 | selection unchanged |
| `wleak` | 8 | 0 | 0 | both arms round R1 to bf16 before renormalising |

`wpre` per-slot ULP distance, slots in `ids` order
(`ids = 79,211,101,135,196,179,106,40`): `1,1,1,0,0,0,2,1`. Relative differences on the moving slots:
`9.85e-08, 8.53e-08, 8.96e-08, 1.42e-07, 9.13e-08` — one or two steps of binary32 spacing at
`w ≈ 0.09–0.15`, i.e. `2^-24…2^-23` of the value.

**Causality, not correlation.** For every slot `k` whose `wpre` moved, the numerator addend `probs[ids[k]]`
moved too (5 of 5); for every slot whose `wpre` did not move, it did not. `wpre` is
`p_k / SUM_K p_j` with each `p_i = exp(logit_i) / SUM_E exp(logit_e)` (`prototype/moe_router/
moe_router_kernels.h` §R2, §R4), so an `exp()` difference of one step is the only thing that can move a
`wpre` slot while `logits` and `logits_f32` sit still. Both routes are inside the licensor's own
`weights_vs_pinned_f32_ulp = 5` against the pinned pre-cast weights (device 4 ULP, host twin 2 ULP), so
the drift is inside the width the licensor already licensed for this field on a single route.

**Attribution verdict: CONSISTENT-WITH-R5-EXP-SCOPE, not a codegen defect.** §R5
(`prototype/moe_router/moe_router_kernels.h §R5`) states an independent fp32 route reproduces the pinned
pre-cast weights "to single-digit ULP and NOT bit-exactly"; §mr_exp names the mechanism ("`expf` on both
arms … agree to an ULP or two — if they ever disagree by more, a math-library difference is hiding in a
route difference"). Every prediction of that story holds on these bytes, including the two predictions
that could have falsified it cheaply (`logits_f32` identical, `wcast` identical). The four conditions are
each a separate answer in the driver's `RTATTRIB` line and are jointly checked by
`rt-device-host-difference-is-the-documented-exp-scope`, so the story is answerable and not a sentence:
if a future build moves `logits_f32`, or moves a `wpre` slot whose `probs` addend is unchanged, or moves
`wcast`, that check reddens and the work belongs in `engine/device/m2_rt_kernels.h`, not in the checker.

## 3. The scope, derived before it was compared with the measurement

The claim on `wpre` must not be deleted (D-18 reserves that relaxation to the user) and must not be
"whatever this build produced". Three **independent** priors were computed first, from artifacts and
documents, none of them from the 2 ULP above; the strictest binds. `engine/block_device.py::rt_wpre_scope`
recomputes them on every run and refuses if the wording it cites has moved.

1. **Licensor's own band, by the triangle inequality: 10 ULP.**
   `weights_vs_pinned_f32_ulp = 5` in
   `prototype/moe_router/evidence/moe-router-host-2026-10-26.json §bands` (floor probe; the same number
   `block_gate.prototype_limits()` feeds the licensed rows), re-read at run time. If route A (host) and
   route D (device) are each within 5 ULP of the pinned pre-cast arm, they are within 10 of each other —
   exactly, not approximately, because ULP distance is the ordered integer image of the floats and so is a
   metric. The premise is measured, not assumed: `rt-wpre-premise-measured-against-the-pin` requires both
   routes inside 5 of the pin (4 and 2 here) and reddens if either drifts past its own licence.
2. **§R5's own words: 9 ULP.** "single-digit ULP" is read out of the header text at run time
   (`rt_words`), and the most that phrase licenses is the largest one-digit number.
3. **Error propagation from §mr_exp's documented exp scope: 22 ULP.** With `d` the relative size of one
   `exp()` step (`2 ULP/call × 2^-23`, the worst reading of "an ULP or two"), `p_i = exp_i / SUM_E exp`
   and `w_k = p_k / SUM_K p` are all same-sign sums, so each sum's relative deviation is the addends'
   worst, and R4 costs numerator + denominator + two division roundings:
   `rho_w = 4d + 3·2^-23 = 1.311e-06`; at the smallest binary32 spacing (`2^-24`) that is ≤ 22 ULP.

`min(10, 9, 22) = 9`, printed with its sources as
`WPRESCOPE … scope_ulp=9 binding=documented §R5 … prior=10 … prior=9 … prior=22`. Measured 2 ULP lands
inside all three; the priors are 4.5×–11× the measurement, so the bound was not fitted to it. The bound
is **stricter than the prototype lane's own cross-arm allowance** for this same field
(`prototype/moe_router/gate.py::compare_arms` uses `4 × weights_vs_pinned_f32_ulp = 20`), which is the
direction a re-derivation is allowed to move in. The artifact's grip on the bound is witnessed, not
claimed: the triangle prior is exactly twice `weights_vs_pinned_f32_ulp` as the artifact reads it (5 → 10),
and re-deriving with that number set to 1 in memory makes the prior 2
(`rt-wpre-scope-derived-from-artifacts`) — a bound typed into a driver would not move. The per-slot ruler
here is also forced to agree with `engine/block_gate.py::ulp_diff` on the same data
(`rt-ulp-ruler-matches-the-licensor`), so there is one ULP ruler in the repo.

## 4. The bound is seen to reject

`CONTROL name=wpre-scope-rejects-a-bigger-perturbation` (10th control; `findings=0 fired=10/10 in the CPU
witness `engine/work/blockdevice-cpu-2026-10-26-p4.log`) runs six legs, every perturbation sized from the
limit rather than from the spread: the honest pair INSIDE; the lane's own named mistake
`mode=softmax_bf16`, executed on the arm, OUTSIDE at 53,110 ULP; every slot moved by exactly 9 ULP
INSIDE, by 10 ULP OUTSIDE, by 64 ULP (an exponent bit at this magnitude) OUTSIDE; and a single-bit xor on
each zero-tolerance field reporting `wcast=1/8, ids=1/8, ids_alt=1/8`, so scoping `wpre` cannot have
swallowed the three fields licensed at exactness. The same legs replayed against the preserved device
record: `at-limit` INSIDE, `one-past-limit` OUTSIDE, `far-past-limit` OUTSIDE
(`blockdevice-wpre-scope-replay-2026-10-26.log`).

## 5. What was proven where, and what still needs a window

* **Proven on the CPU witness** (no device): the scope derivation and its sensitivity, the ULP ruler's
  agreement with `block_gate.ulp_diff`, the premise against the pinned pre-cast weights, all six rejection
  legs, the tie control on the `host_cpubody` path (`fired=RED legs=host_cpubody:2 claims=host_cpubody`).
* **Proven by replay of window `e2`'s preserved records** (no device): the device-vs-host field census,
  the causal `probs`→`wpre` chain, the licensed trio at zero, `wpre` = 2 ULP inside the derived 9, and the
  rejection legs against the device record.
* **Not provable without a window**: that the *live* device arm reproduces those numbers on a fresh build
  (`WPRESCOPE`/`RTATTRIB`/`PAIR` in device mode), that the tie control's two device legs redden
  `ids_alt` and move `ids`+weights on device, that the bound control's device leg (the real
  `mode=softmax_bf16` run on the device arm) lands OUTSIDE, and that the device legs satisfy
  `device-path-claims-executed-on-device`. The corrected command block is `engine/HANDOFF.md` §pass 4.
* The shared-expert rows remain `UNLICENSED` and were not touched; `STAGETALLY unlicensed=3` and
  `verdict=QUALIFIED` are the expected shape until policy-A bands exist.
