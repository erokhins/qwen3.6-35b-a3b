# `ssm_a` at one f32 ULP: how far does a recomputed `A = -exp(A_log)` compound?

Written 2026-10-26. CPU only, stdlib only, no device, no timings (`AGENTS.md` §"GPU execution is
serialized"; this file contains no `ms` figure and takes no measurement of anything that runs).

Instrument: `specification/evidence/ssm_a_ulp_bound.py` (12 selftest arms, all PASS; run it — the
numbers below are its output, not a transcription of a belief).

## Q. The question, and why it is open

`DECISIONS.md` D-14 (ratified by the user 2026-10-26) accepts **one** 1-f32-ulp deviation: the
published file holds `blk.1.ssm_a[21] = -0.9096216559410095` (`0xbf68dcf7`) where both
correctly-rounded routes of `-exp(A_log)` give `-0.9096215963363647` (`0xbf68dcf6`) — an absolute
deviation of `5.960464477539063e-08`, which is exactly one f32 ulp at that magnitude (witnessed by
the `D-14 witness` selftest arm: `bits(file) == bits(route) + 1`).

D-14's accepted reasoning was that the engine **reads** `ssm_a` and never evaluates the
transcendental, so the last bit of a function we do not compute cannot be a fidelity property of
our kernels. One caveat in that decision is unquantified and is the whole subject of this file:

> if a path ever **recomputes** `ssm_a` instead of reading it, `A` becomes `A + 1 ulp` — and `A` is
> the per-token decay rate of every GDN head, multiplied once per token, for up to 262,144 tokens.

That is the compounding question. The first attempt at this lane (2026-10-26) was killed by its own
harness before delivering anything; see §"How not to compute this" at the end.

## The recurrence, as this repo specifies it

Nothing here is imported from another project. The authoritative form is
`specification/04-gdn-linear-attention.md`, quoted.

§04.5, the gate:

> ```
> beta_t[h] = sigmoid( in_proj_b(x_t)[h] )                       in the activation dtype
> g_t[h]    = -exp(A_log[h]) * softplus( in_proj_a(x_t)[h] + dt_bias[h] )
> softplus(u) = log(1 + exp(u))
> ```

with its own fact table stating `g_t[h] <= 0 always` — "it is a log-decay … The per-step multiplier
on the state is `exp(g_t[h])` in `(0, 1]`" — and, for the tensor under discussion, "what the file
stores: `blk.N.ssm_a` is already `-exp(A_log)`, computed by the producer at conversion time … It is
a **recomputation**, so its 1-ulp history is governed by `DECISIONS.md` D-14 and
`specification/ASSUMPTIONS.md` A-13 — never call it bit-exact with the checkpoint".

§04.6, the recurrent update, "in the reference's exact order", all of it F32 (`mamba_ssm_dtype:
float32`, §04.4):

> ```
> 1  S      = S * exp(g_t[h])                                  # decay first
> 2  kv_mem[v] = sum_k S[k, v] * k_t[k]                        # what the state already predicts
> 3  delta[v] = (v_t[v] - kv_mem[v]) * beta_t[h]               # the correction, scaled by beta
> 4  S[k, v] = S[k, v] + k_t[k] * delta[v]                     # delta-rule write
> 5  out_t[v] = sum_k S[k, v] * q_t[k]                         # read AFTER the write
> ```

and the same section's outer-product form, which is what fixes the ordering:
`S_t = exp(g_t) * S_{t-1} * (I - beta_t k_t k_t^T) + beta_t v_t k_t^T`.

Two facts from §04.4 make the rest trivial and are load-bearing here:

> `k = k * rsqrt( sum(k*k, dim=-1) + 1e-6 )` — l2norm with eps **inside** the rsqrt, so
> `||k||_2 <= 1`; and `beta = sigmoid(b)`, so `beta in (0,1)`.

## The derivation

Write `A = -exp(A_log) < 0` (this is what the file stores) and `dt_t = softplus(a_t + dt_bias) >= 0`,
so `g_t = A dt_t` and the per-step propagator is `r_t = exp(g_t) in (0,1]`. Then §04.6 is

```
S_t = r_t * S_{t-1} (I - beta_t k_t k_t^T)  +  beta_t v_t k_t^T ,      out_t = S_t^T q_t
       \_________________ A enters only here __________________/         (write is undecayed)
```

1. **The delta-rule write is non-expansive.** `I - beta k k^T` is symmetric with eigenvalues `1`
   (multiplicity `d_k - 1`) and `1 - beta ||k||^2 in [0, 1)`, so its spectral norm is `1`. Nothing
   in steps 2–4 can amplify anything, for any `v_t`, `k_t`, `beta_t`.
2. **Therefore every route from `A` to the state goes through the scalar `r_t` and nowhere else.**
   Perturbing `A -> A + delta` changes step `t`'s propagator from `r_t` to `r_t exp(delta dt_t)`;
   the step's own write carries no `r_t` (it happens after the decay), so the injection at step `t`
   is exactly `r_t (exp(delta dt_t) - 1) S_{t-1}`. Two corollaries worth saying out loud: the
   perturbation's sign does not matter (`|exp(x)-1|` is largest for `x > 0`, and the bound uses that
   worst case), and **at `N = 1` the state is `beta v k^T` and does not depend on `A` at all** — a
   selftest arm (`arm N=1 order`) asserts the simulated error is exactly `0.0` there while the bound
   stays strictly positive, so the arm cannot pass by accident.
3. **Unroll.** With `s = max_t ||S_t||_2`, `alpha = |A| dt` (constant per step), `r = exp(-alpha)`,
   `eps = exp(|delta| dt) - 1 = expm1(rho * alpha)` and `a = r (1 + 2 eps)`:

   ```
   B_rel :  ||dS_N|| / s  <=  r * eps * sum_{k=0}^{N-1} a^k
   B_abs :  ||dS_N|| / V  <=  r * eps * (sum_{k=0}^{N-1} a^k) (sum_{m=0}^{N-1} r^m)
   ```

   where `V = max_t ||v_t||_2` and the second line uses the state's own envelope
   `s <= V sum_{m=0}^{N-1} r^m`, from the same unrolling with `||k|| <= 1`, `beta <= 1`. The
   `1 + 2 eps` on `a` is the Gronwall closure (the injection multiplies the *perturbed* state); it
   changes `B` by a factor `1 + O(rho)` and it is what makes the first line a theorem rather than a
   linearization. `a < 1` whenever `rho < 1/2`, i.e. always for a real ulp.
4. **Only three numbers matter:** `N`, and the pair

   ```
   alpha = |A| * dt          (dimensionless per-step exponent; the per-step GATE is exp(-alpha))
   rho   = ulp(A)/|A|        (the relative size of the perturbation: 2^-24 <= rho <= 2^-23)
   ```

   `A` enters through `rho` **only** — never through its magnitude. That is why the bound is stated
   in the gate's own currency and read off the measured gate population.
5. **Shape in `alpha`** (the leading `r` is not cosmetic):

   | regime | `B_rel` | reads as |
   |---|---|---|
   | `alpha << 1/N` | `~ rho * alpha * N` | no decay yet, and the state is only `alpha*N` of the way built |
   | `1/N << alpha << 1` | `~ rho (1 - alpha)` | **the plateau, `= rho`, independent of `N`** |
   | `alpha >~ 1` | `~ rho * alpha * exp(-alpha)` | the state itself is `exp(-alpha) V`: a dead head |

   So the perturbation **cannot compound with length**. Its supremum over the whole gate range is
   `rho`, attained near `alpha ~ 1/N`, and it decays at both ends. `B_abs` grows as the gate opens
   (`~ 1/alpha`) only because the state it is a fraction of has grown by the same factor.

## The bound, in one line

```
B(N, alpha) = exp(-alpha) * expm1(rho*alpha) * sum_{k=0}^{N-1} [ exp(-alpha) (1 + 2 expm1(rho*alpha)) ]^k
bound       : ||dS_N|| / max_t ||S_t||_2  <=  B(N, alpha) ,   rho = ulp(A)/|A| in [2^-24, 2^-23]
```

and since `out = S^T q` with `||q||_2 = 1/sqrt(d_k) = 1/sqrt(128)` (§04.4), the block output's own
relative error is the same `||dS||/||S||` absent cancellation in the read.

Supersampled check: a 16-point/decade sweep and a 200-point/decade sweep over `alpha in [1e-8, 1e8]`
give the same supremum to 5 decimals (`sup B_rel / rho = 0.998755` vs `0.998779` at `N = 4096`,
`0.999973` at `N = 262144`). The plateau is broad; the grid is not what makes it look flat.

## Numbers at N = 4,096 / 32,768 / 262,144

`N = 262144` is `text_config.max_position_embeddings`; the three widths are the ones a GDN state
actually lives through. Figures are at the binding `rho = 1.19002e-07` (the max over the model's 960
heads), state-relative currency unless labelled `B_abs`, which is in units of `V = max_t ||v_t||_2`.

| | N = 4,096 | N = 32,768 | N = 262,144 |
|---|---|---|---|
| `sup_alpha B_rel` | **1.18854e-07** | **1.18979e-07** | **1.18999e-07** |
| …as a multiple of `rho` | 0.9988 | 0.9998 | 0.99997 |
| …at `alpha` | 2.371e-03 | 3.652e-04 | 4.870e-05 |
| `sup_alpha B_abs` (units of `V`) | 1.98444e-04 | 1.58669e-03 | 1.27037e-02 |
| `B_abs` for heads with `s/V <= 10` | 1.02937e-06 | 1.02937e-06 | 1.02937e-06 |
| measured anchor (960 heads): `B_rel` min / med / max | 1.7403e-45 / 7.6287e-08 / 1.18663e-07 | same / 7.71348e-08 / 1.18663e-07 | same / 7.71919e-08 / 1.18663e-07 |
| measured anchor: `B_abs` min / med / max | 1.7403e-45 / 1.21981e-06 / 1.88313e-04 | … / 1.21981e-06 / 1.27588e-03 | … / 1.21981e-06 / 1.65821e-03 |

Read the `sup B_abs` row honestly: it is attained at `alpha ~ 1/N`, where the state's own envelope is
`1.5e5 V`. It is a statement about a gigantic state, not about a big error — the error is `rho` **of
that state**. For any head whose state stays within `10 V` the absolute figure is `1.03e-06`.

Worked example — the D-14 element itself, `blk.1.ssm_a[21]`, `A = -0.9096216559410095`
(`0xbf68dcf7`), `ulp(A) = 5.96046448e-08`, `rho = 6.55269e-08`; its `blk.1.ssm_dt.bias[21] =
0.341796875`, so at the `a = 0` anchor `dt = 0.878578`, `alpha = 0.799174`, gate `= 0.4497`:

| N | `B_rel` | `B_abs` | per-step gate perturbation `|exp(delta dt)-1|` |
|---|---|---|---|
| 4,096 | 4.27942e-08 | 7.77652e-08 | 5.23673441e-08 |
| 32,768 | 4.27942e-08 | 7.77652e-08 | (same — it is per-step) |
| 262,144 | 4.27942e-08 | 7.77652e-08 | (same) |

`0.653 * rho`, flat in `N`, `330x` under the floor below.

## The gate / `dt` distribution — from data, on CPU

Tensors read, with the element counts the script asserts non-zero:

| tensor | type | shape (GGUF `ne`) | tensors | elements |
|---|---|---|---|---|
| `blk.N.ssm_a`, `N` in the 30 `linear_attention` layers | F32 | `[32]` | 30 | 960 |
| `blk.N.ssm_dt.bias`, same layers | F32 | `[32]` | 30 | 960 |

Geometry from `checkpoints/Qwen3.6-35B-A3B.config.json`: 40 layers, 30 `linear_attention`,
`h_v = 32`, `d_k = d_v = 128`, `max_position_embeddings = 262144`. Read through
`tools/gguf_reader.py` under `tools/audit_published_quant.py`'s selftest token, which refuses the
data section until that selftest's marker is current.

| population | n | min | median | max |
|---|---|---|---|---|
| `|A|` (= `exp(A_log)`) | 960 | 0.0186041 | 0.754841 | 105.241 |
| `ulp(A)`, absolute f32 | 960 | 1.86265e-09 | 5.96046e-08 | 7.62939e-06 |
| `rho = ulp(A)/|A|` | 960 | 5.96832e-08 | 8.48989e-08 | 1.19002e-07 |
| `dt_bias` | 960 | -7.71875 | -3.09375 | 15.5625 |
| `dt` at the `a = 0` anchor, `softplus(dt_bias)` | 960 | 4.44317e-04 | 0.0443342 | 15.5625 |
| `alpha = |A| dt` at the anchor | 960 | 6.3995e-05 | 0.0670955 | 91.5772 |
| per-step gate `exp(-alpha)` at the anchor | 960 | 1.69248e-40 | 0.935106 | 0.999936 |
| per-step gate perturbation `|exp(delta dt)-1|` at the anchor | 960 | 6.62084e-12 | 5.45912e-09 | 1.02825e-05 |

`rho` lands inside `[2^-24, 2^-23] = [5.96046448e-08, 1.1920929e-07]` for all 960 heads, as it must;
the script checks that and says so.

**What is and is not measured.** `dt_bias` is stored, and the audit's 1.4a pass holds it at zero
tolerance against the bf16 original widened to F32 (`specification/10-reference-dumps-and-gates.md`
§10.2). `a = in_proj_a(x_t)[h]` is an **activation**, and no GDN activation is pinned: counted, not
asserted — 104 `.bin` files in `data/dumps/2026-10-26/`, of which **0** match
`ssm|gdn|gdt|conv|linear_attn|state`; the pinned block is the MoE one
(`specification/10-reference-dumps-and-gates.md` §10.1). So `dt = softplus(a + dt_bias)` is
**unobserved**, and the `a = 0` row above is one slice of the `(token, head)` grid, not a sample of
the running distribution. The verdict is therefore not carried by that slice; see the widening
below.

HF-side cross-check, on the shards that are actually on disk (`models/qwen36-src/`, four files:
layers 0, 1, 38, 39 and `mtp.*`): 96 `dt_bias` and 96 `A_log` elements, layers `[0, 1, 38]`, all
BF16 `[32]` in the shard. Under `specification/04-gdn-linear-attention.md` §04.9's tiled→grouped
permutation `j -> (j % 16)*2 + j // 16`: **96/96** `dt_bias` bit-exact after bf16→f32 widening, and
**96/96** `ssm_a` within 1 ulp of the double-precision route `-exp(A_log)`. Both counters are
non-zero, and the permutation is thereby witnessed rather than assumed.

### Widening over the unknown range, rather than trusting the observed one

`alpha` was swept in closed form over `1e-6 .. 1e8` (225 points, 16/decade) at the binding `rho`.
`#{alpha : B_rel > floor} = 0` at all three widths — the bound is under the floor **everywhere on the
range**, at gates from `0.999999` down to `e^-1e8`. That range of `alpha` is `dt` up to `1e8/|A|`,
i.e. `a + dt_bias` of order `10^6` at the smallest `|A|` in the file (`0.0186041`), against a largest
stored `dt_bias` of `15.5625`. No per-step gate the architecture can present, however far outside the
stored bias, brings the perturbation up to the floor.

### A data-derived ceiling on the unobserved `a`, for the 3 layers whose weights are on disk

This prices the gate range over **all inputs**, not a sample. `x` is the Form-A `input_layernorm`
output (`specification/03-block-glue-and-normalization.md` §03.1: `(1+w) * x_hat` with
`x_hat = x*rsqrt(mean(x^2) + 1e-6)`, cast back to bf16), so `||x_hat||_2 = sqrt(d)` exactly and

```
(1)  ||x||_2 <= sqrt(2048) * max_i |1 + w_i| * (1 + 2^-9)   [the bf16 cast's rounding half-width]
(2)  |a_h|   <= ||W_a[h,:]||_2 * ||x||_2                     [in_proj_a has no bias — shard header]
(3)  dt_h    <= softplus(|a_h| + dt_bias_h) ,  alpha_h = |A_h| * dt_h
```

Priced for layers `[0, 1, 38]` = 96 of 960 heads (the other shards are not on disk):

| quantity | n | min | median | max |
|---|---|---|---|---|
| `||W_a[h,:]||_2` (exact, from the bf16 row) | 96 | 4.35144e-07 | 0.966271 | 1.63865 |
| `||x||_2` ceiling | 96 | 60.31 | 61.0185 | 81.4761 |
| `|a_h|` ceiling | 96 | 2.62436e-05 | 64.8018 | 105.237 |
| `dt` ceiling | 96 | 0.363265 | 62.3979 | 100.8 |
| `alpha` ceiling | 96 | 0.954594 | 20.1578 | 1216.84 |

Worst over all inputs those 96 heads can present: `B_rel = 5.98231e-08`, `B_abs = 9.72683e-08`, at
every one of the three widths. Their `alpha` ceiling (1217) sits far inside the scanned range.

## Floors, and the verdict

Four floors, each with the figure and its source in this repo:

| floor | value | source |
|---|---|---|
| pooled `1-cos` background of a **legal** change at 16,384 lanes | **1.41e-05** | `pitfalls/30-numerics.md` §"Byte-identity at short widths is not byte-identity at long context" (copied in 2026-10-26): "differed at 16,384 lanes: 1−cos 1.41e-05, argmax SAME, top-5 SAME, all 16,385 greedy tokens equal" |
| one bf16 ULP of the residual stream the block writes into | 2^-8 = 3.90625e-03 | format-derived from `pitfalls/30-numerics.md` §"Format changes" ("bf16's 1+8+7"); the round-to-nearest half-width is 2^-9 = 1.953125e-03 |
| flush-to-zero edge the repo already treats as a range census, not a band | ~5.96e-08 | `pitfalls/30-numerics.md` §"Format changes" — numerically the same size as one f32 ulp at `|A| in [0.5,1)`, i.e. the D-14 deviation itself |
| one f32 ULP of the state itself (the scan is F32, §04.4) | 2^-24 = 5.96046e-08 | format-derived, same section (F32 = 1+8+23); the resolution ceiling of any hypothetical bit-exact F32 state gate |

Operative figure: `sup` over the scanned gate range and all three widths of `B_rel` =
**1.18999e-07** — `1.000 * rho`, `118.5x` under the primary floor, `32,826x` under one bf16 ULP, and
`1.996x` one f32 ULP.

NOT RESOLVABLE

A 1-f32-ulp change in `A`, compounded through §04.6 over every head in the model and every
`N in {4096, 32768, 262144}`, perturbs the recurrent state by at most `1.19e-07` of its own
magnitude. No gate this repo specifies could resolve it — which is D-14's claim, now quantified past
the single element it was made about.

The one instrument that could care is a **bit-exact** gate on the F32 state: at `1.996` state ULPs
the bound is of order two ULPs, and a bound of two ULPs neither proves nor denies that a bit flips.
No such gate is specified — `specification/10-reference-dumps-and-gates.md` §10.2/§10.3 give the GDN
scan cos + mean-rel aggregates plus a floor-probe cancellation band, not bit-exactness — and the
selftest's simulated-dynamics arm saw the F32 scan round the 1-ulp gate change away entirely, which
is the same message from the other side.

## What the instrument is

`specification/evidence/ssm_a_ulp_bound.py`, 12 arms, all PASS (run
`python3 specification/evidence/ssm_a_ulp_bound.py --selftest` for the arms alone; the full run adds
the data sections and prints `VERDICT: NOT RESOLVABLE`):

| arm | what it asserts | verdict |
|---|---|---|
| `zero` | `delta = 0` gives `B == 0.0` **exactly** (`repr(0.0)`), not "below epsilon" | PASS |
| `one-ulp` | the real figure, `B = 5.472722e-08` at `alpha = 0.35`, bracketed by `rho/2 < B < 4 rho` | PASS |
| `thousand-ulp` | negative control: `1000 ulp` gives `B = 5.473383e-05`, ratio **1000.12x** — monotonicity in `delta` is the property under test | PASS |
| `monotone` | `B` strictly increasing in `delta` across 8 decades (`1e-3 .. 1e4` ulp) | PASS |
| `sign` | `delta = ±1 ulp` is covered by one figure (the `|expm1|` form) | PASS |
| closed-vs-explicit | the closed form vs the O(N) summation, 203 `(head, alpha, N)` pairs, `N <= 1024` only, max rel err `3.79e-14` | PASS |
| tripwire | the O(N) summation **refuses** to run at `N = 1025` (the guard must be able to fire) | PASS |
| bound-vs-simulated | 8 runs of a 2x2 §04.6 state, `A` vs `A + delta`, `N = 1024`: true error never exceeds the bound, 6 of 8 non-zero, max `true/bound = 0.387` | PASS |
| `N=1 order` | the write is undecayed, so the true error at `N = 1` is exactly `0.0` while the bound is `1.17e-03` | PASS |
| ulp witness | bit-walk ulp == exponent-derived ulp on 13 values incl. exact powers of two | PASS |
| trap witness | `math.nextafter` returns the **double** ulp `1.110e-16`, eight orders off the f32 ulp `5.960e-08` | PASS |
| D-14 witness | `|file - route| == ulp(route)` and `bits(file) == bits(route) + 1` | PASS |

Populations the run evaluates, printed and asserted non-zero: 60 tensors / 1920 elements / 960 heads,
225-point `alpha` grid, 203 closed-vs-explicit cross-check pairs, 8 simulated runs, 96 input-envelope
rows, 192 HF elements.

## SCOPE / NOT ESTABLISHED

`SCOPE:` — for the recurrence as this repo specifies it (§04.5 + §04.6, F32 scan, `||k||_2 <= 1` and
`beta in (0,1)` from §04.4), and for a **constant** per-step exponent `alpha`:

- `B(N, alpha) = r*eps*sum_{k<N}[r(1+2eps)]^k`, `r = exp(-alpha)`, `eps = expm1(rho*alpha)`, bounds
  `||dS_N|| / max_t ||S_t||_2`; `B <= 1.000 * rho` for **every** `alpha` and **every** `N` —
  N-independent — with `rho = ulp(A)/|A|` measured `5.96832e-08 / 8.48989e-08 / 1.19002e-07`
  (min/median/max) over all 960 `(layer, V head)` pairs;
- the absolute currency is `B_abs <= 106752 * rho` in units of `V = max_t ||v_t||_2`, attained only
  at `alpha ~ 4.87e-06` where the state's own envelope is `1.48e5 V`; for a head whose state stays
  within `10 V` it is `1.02937e-06`;
- the injection is exactly the per-step gate perturbation `|exp(delta*dt) - 1|`, measured max
  `1.02825e-05` at the `a = 0` anchor — one relative ulp of `A` times the step's own exponent;
- the delta-rule write `I - beta k k^T` contributes **no** amplification, so no arrangement of
  `v_t, k_t, beta_t` can make `A` matter more than the gate does;
- at `N = 1` the state is independent of `A` (the write is undecayed), which the `N=1 order` arm
  pins against a simulation, so the ordering claim is falsifiable and not just quoted;
- under the floor for every `alpha` in `1e-6 .. 1e8`, at all three widths, at the binding `rho`.

`NOT ESTABLISHED:`

- **the true per-step `dt` is unobserved.** `a = in_proj_a(x_t)` is an activation; no GDN activation
  is pinned (104 dump files, MoE block only). The `a = 0` slice is an anchor, not the running
  distribution. The verdict is carried by the `alpha` sweep, not by the measured bias — a `dt`
  outside the swept envelope would have to come from an activation the model was never trained to
  produce, and it was not checked there.
- the input-wise `alpha` ceiling is priced for **3 of 30** linear layers (`[0, 1, 38]`): shards 3–24
  and 26 are not on disk, so 27 layers' `in_proj_a` and `input_layernorm` weights were never read.
  For those layers the gate range is the widened one, not a measured one.
- **`alpha` is constant inside the `N`-step window** in the closed form. A varying trajectory is
  covered by evaluating the same expression at the worse end (`min alpha` for the decay, `max` for
  the injection); no per-token `dt` sequence was observed, so this is a modelling assumption.
- the bound is on the **state**. The read `out_t = S_t^T q_t` (§04.6 step 5) can cancel, turning a
  state-relative error into a larger output-relative one; that is the cancellation-band question of
  `specification/10-reference-dumps-and-gates.md` §10.3 step 3 and is not settled here.
- **nothing past the scan is priced**: the gated norm §04.7, `out_proj` §04.8, the residual add, the
  30-layer stack, the logits row. A per-layer amplification factor of up to `118x` would still leave
  the result under the floor; no such factor was measured and none is assumed.
- the HF-side witness covers 192 elements (3 layers) because 4 of 26 shards are on disk; the other
  layers' `dt_bias`/`A_log` exactness is cited from `quantization/evidence/audit-exactness-2026-10-26.json`,
  not re-derived here.
- **this authorizes no recompute.** The file's `ssm_a` is still read. D-14's 1-ulp predicate is
  route-derived and stays, and its must-catch set (±2 ulp, sign flip, element permutation, wrong
  `A_log` read) is unchanged by anything here — a bound this far under the floor is not a licence to
  widen a gate (`AGENTS.md` §"Standing rules": never loosen a gate to make it pass).

## How not to compute this

The first lane died on its own harness, and the scratch survives at `tmp/ulp_check.py`,
`tmp/scratch_bound.py` and `tmp/scratch_bound2.py`. Two lessons, both now enforced in code:

1. **The closed form is the bound.** The dead lane computed the closed form *and* an O(N) explicit
   summation, and called both from inside a ~1,281-point `dt` sweep x 32 heads — ~1e10 interpreted
   iterations, ~48 minutes, killed with nothing delivered. Here the O(N) summation exists only as a
   cross-check, is asserted at `N <= 1024`, runs on 203 pairs, and the refusal to exceed that is
   itself a selftest arm (`tripwire`). The bound itself is closed form only — no loop in this file
   iterates over tokens, and no timing is taken or printed anywhere in it.
2. **`math.nextafter` on a Python float gives the DOUBLE ulp.** `tmp/scratch_bound.py` used it and
   was eight orders of magnitude wrong (`1.110e-16` instead of `5.960e-08`); `tmp/scratch_bound2.py`
   used the f32 bit-walk and was right. This file uses the bit-walk, walks **away from zero** (the
   direction error is visible at exact powers of two, and the `ulp witness` arm covers those), and
   keeps the double-ulp trap as a positive arm rather than a comment.
