# Quantization policy — D-01 (experts at 4-bit, and what to do about everything else)

Status: **experts at 4-bit is authorized** (user, 2026-10-26: "experts q4"). **Attention width is
still the user's call** ("attention — don't know"); this file gives the numbers for it, and §
"the attention question" is the recommendation. int4/int8 work requires the user's explicit words
(`AGENTS.md` standing rules) — the decision is recorded in `DECISIONS.md` D-01, not inferred here.

## Why quantization is not optional here

`devices/measured-facts.md` §"Residency above the pool" (measured 2026-10-26) closed the one escape
hatch that existed: **the device-visible pool is box-wide (≈55.5–56.5 GiB) and host-pinning spends it
rather than raising it** — `hipHostMalloc` of 34/36/38/40 GiB fails in 0.0 s, and two *separate
processes* asking 20 GiB each could not coexist (`tools/pin_pair.sh`). What is left for one process
depends on who else is resident: **≈32 GiB while halogen holds its ~23.5 GiB**, ≈55 GiB if the engine
owns the box (`DECISIONS.md` D-11).

So bf16 (65.72 GiB) is impossible in either budget, and **Q8-throughout (34.97 GiB) is impossible
co-resident but possible dedicated** — which does not change the plan, because (a) we do not control
co-residency, and (b) 4-bit experts are the faster choice regardless: 2.331 vs 2.811 GiB/token, ~95 vs
~79 tok/s at 237 GB/s. The same measurement says the *location* of the bytes does not matter: pinned
host memory reads at 215.8 GB/s against 198.6 GB/s for device memory in the same window, i.e. a
bandwidth-bound engine loses nothing by reading pinned host pages.

## Family census (derived, validated)

`python3 tools/quant_scenarios.py --census`, shapes from `checkpoints/Qwen3.6-35B-A3B.config.json`,
cross-checked against the shard index's own `total_size` (delta 1.86 %, which is the vision tower):

| family | bf16 GiB | share | read per token? |
|---|---|---|---|
| routed experts (40 layers + MTP block) | **61.500** | 91.84 % | 8/256 of it |
| GDN linear attention (30 layers) | 1.416 | 2.11 % | all of it |
| token embeddings | 0.947 | 1.41 % | one row (~4 KiB) |
| `lm_head` (untied) | 0.947 | 1.41 % | all of it |
| full attention (10 layers + MTP) | 0.559 | 0.83 % | all of it |
| shared experts (41) | 0.240 | 0.36 % | all of it |
| norms and misc | 0.071 | 0.11 % | all of it |
| routers `mlp.gate` (41) | 0.040 | 0.06 % | all of it |
| **text total** | **65.720** | | **5.19 GiB/token at bf16** |

Two things fall out of this and matter more than the headline. First, experts are 92 % of the bytes
but only ~36 % of a bf16 decode's per-token read, because only 8 of 256 fire per token. Second, the
**non-expert walk (attention + GDN + `lm_head` + shared) is 3.2 GiB/token and is read in full every
token** — which is why "quantize the experts and leave the rest alone" buys far less than 92 %.

## The policies, priced

`python3 tools/quant_scenarios.py --read 237 210 190`. Budget is the measured device-visible pool: the
**≈32 GiB co-resident** figure is the design budget (`DECISIONS.md` D-11); ~55 GiB is what the engine
gets if the box is its. tok/s are **ceilings** (bytes ÷ bandwidth). The 0.8B engine reached ~60 % of
its own ceiling, so multiply by ~0.6 before quoting a target. 190 GB/s is included because that is what
this box actually delivered in the 2026-10-26 window, not the ledger's 237.

| policy | resident GiB | GiB/token | ceiling @237 | @210 | @190 |
|---|---|---|---|---|---|
| bf16 | 65.72 ✗ | 5.194 | 42 | 38 | 34 |
| **A — DECIDED** experts Q4_K, everything else Q8_0 | **19.59** | 2.331 | **95** | 84 | 76 |
| B experts Q4_K, everything else bf16 | 21.52 | 3.813 | 58 | 51 | 46 |
| C experts MXFP4, rest Q8_0 | 18.63 | 2.301 | 96 | 85 | 77 |
| D experts IQ4_XS, rest Q8_0 | 18.63 | 2.301 | 96 | 85 | 77 |
| E experts Q4_K, **attention Q4_K too** | 18.81 | 1.663 | **133** | 118 | 106 |
| F Q8_0 throughout | 34.97 ✗ co-resident, ✓ if dedicated | 2.811 | 79 | 70 | 63 |
| G (not taken, D-10) experts Q4_K, attention Q8_0, `lm_head` Q6_K | 19.36 | 2.216 | 100 | 88 | 80 |

## The attention question (**DECIDED 2026-10-26: Q8_0** — the reasoning that got us there)

Three defensible answers, all with the experts at Q4_K:

| choice | resident | GiB/token | ceiling @210 | quality evidence |
|---|---|---|---|---|
| attention/GDN **bf16** | 21.52 | 3.813 | 51 tok/s | zero added error, but throws away a third of the rate |
| attention/GDN **Q8_0** (policy A/G) | 19.4–19.6 | 2.22–2.33 | 84–88 tok/s | Q8_0 measured +0.0004 ppl at 7 B (`pitfalls/30`); the classes that resist quantization are exactly the ones Q8 covers |
| attention/GDN **Q4_K** (policy E) | 18.81 | 1.663 | 118 tok/s | +40 % rate, and it is what llama.cpp's own `Q4_K_M` does by default — but unsloth's per-tensor sweep on **this model family** found `attn_*` the *most* sensitive class on a hybrid arch and `ssm_out` (GDN `out_proj`) the worst single tensor |

**Decision (D-01a, user): attention and GDN at Q8_0 — policy A.** The reasoning that supported it, kept
because it is what makes policy E reversible:

1. The saving from Q4 attention is 2.5 GiB of residency we do not need (both fit) and ~40 % of ceiling
   — real, but the same 40 % is available from a *safer* place: `lm_head` Q6_K and `lm_head`/embed
   layout are cheap to revisit, and the engine will be at ~60 % of ceiling for other reasons.
2. The risk is not evenly distributed. `embed`/`lm_head` sit under the argmax that decides every
   token, and GDN `out_proj`/`in_proj_*` feed a recurrent state whose error accumulates across the
   sequence — a per-token error there is not a per-token error in a feed-forward block.
3. It is reversible in one direction only: a green bf16 gate set plus a Q8 attention path can be
   downgraded to Q4 later and measured, whereas a shipping engine built on Q4 attention has to be
   re-argued from scratch if quality says no.

If speed later demands it, policy E is +40 % and the gates (below) will say quickly whether it is real —
and per `AGENTS.md`'s numeric-format rule, adopting it requires only that the existing gates stay green
(or that a widened band still catches every geometry-derived must-catch injection).

## Independent confirmation of the attention decision

A tuned public quant of **this exact model** was parsed from its own header
(`tools/gguf_header_types.py`, evidence in
`quantization/evidence/q4km-published-width-map-2026-10-26.md`): unsloth's
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` puts **every attention and GDN matrix at Q8_0** (full-attn q/k/v/out,
GDN `attn_qkv`, `attn_gate`, `ssm_out`), the router and all GDN control tensors at **F32**, experts
gate/up at **Q4_K**, shared expert at Q8_0, embeddings Q8_0 — i.e. it lands on the same width assignment
D-01a reached from KLD evidence, from a producer with a calibration rig and no stake in our reasoning.
The two policies differ in exactly two places: that file spends **more** on expert `down_proj`
(Q5_K, with Q6_K on three layers) and **less** on `lm_head` (Q6_K).

That leaves one open refinement worth putting to the user later, not decided here: **gate/up Q4_K +
down Q5_K** is what a producer chose to protect, it costs about 1.3 GiB of residency and roughly
0.07 GiB/token over the current policy-A arithmetic (`quant_scenarios.py` has not been re-run for that
variant, so the per-token figure is a note, not a policy).

## Which 4-bit format (D-08)

Research notes are in `quantization/evidence/`. Short version, all of it measured by someone else on
this model family or in llama.cpp's source:

- **Q4_K (4.5 bpw)** is the default recommendation and what llama.cpp ships for MoE. llama.cpp's own
  tensor policy (`evidence/llama-cpp-tensor-policy-2026-10-26.md`) is: never quantize the router,
  norms, conv1d or anything <2-D; force the latent/gate categories to ≥ Q8_0; `output` → Q6_K;
  everything else (including experts) → Q4_K. That is a coherent, published policy and it matches
  policy A/G above.
- **MXFP4 (4.25 bpw)** has a shipping `MXFP4_MOE` type in llama.cpp (experts MXFP4, everything else
  Q8_0 — i.e. policy A/C with a different 4-bit format) and a crowd claim of lower perplexity than
  Q4_K_M. Unsloth's per-tensor sweep for **Qwen3.5-35B-A3B** disagrees on this family: MXFP4_MOE at
  18.17 GB has 99.9 % KLD 0.7789 vs Q4_K_M at 18.49 GB with 0.5478, and their words are "It's better
  to use Q4_K than MXFP4 when choosing between them" (they retired MXFP4 from their mixed quants).
  They also note MXFP4 is 4.25 bpw against Q4_K's 4.5, so it is *smaller and worse* here.
- **IQ4_XS (4.25 bpw)** is the size-down lever: 16.4 GB at KLD 0.8067 (AesSedai) — better KLD-per-byte
  than MXFP4 — but i-quants cost decode speed, ~5 % in unsloth's own table (`tg128`: q4_k 90.38 vs
  iq3_xxs 85.68), and a codebook dequant is exactly the kind of inner-loop cost we would have to
  measure here, not assume.
- **Cross-check of our own arithmetic**: policy A/G at 19.4–19.6 GiB = 20.8–21.0 GB vs the published
  18.49 GB (unsloth Q4_K_M) and 19.77 GB (bartowski Q4_K_M). Our number is 6–13 % high, which is the
  expected direction: those quants are "dynamic" and put some expert tensors *below* Q4_K, and
  llama.cpp's Q4_K_M also leaves attention at Q4_K (policy E), which our A/G do not. Treat the
  agreement as ±10 % and re-derive when the real weights are on disk.

**DECIDED (D-08, 2026-10-26): Q4_K for the routed experts.** Read the standard Q4_K block layout
**verbatim** — 256-weight super-block, 8 sub-blocks of 32 with a 6-bit scale *and* a 6-bit min under fp16
super-block scales — so our reader is byte-compatible with published quants (unsloth, bartowski and
lmstudio-community all ship Qwen3.6-35B-A3B in Q4_K variants). Dropping the min term to simplify the
kernel would be a *different format* and would break that compatibility, so it is not an optimization of
this decision; it would need its own decision and its own quality evidence. Not MXFP4, not IQ4_XS: keep
IQ4_XS as the reserved size-down option and re-test it if we ever need the last GiB.

## How a quantized path gets gated (no gate gets loosened)

The oracle stays bf16. `speccheck/` computes the reference from pinned HF dumps; the quantized kernel
is judged against that reference, never against another quantization. What that buys, and what it
costs:

1. **New bands, sized by a floor probe, per family** (`pitfalls/30`, `pitfalls/40`). A 4-bit expert
   output is not within a bf16 ULP of anything, so the band must come from a probe over the pinned
   dumps — cos/meanrel aggregates for the long dot, plus decision-level agreement (argmax and
   top-5 on the pinned rows) as the metric that actually has to hold.
2. **Must-catch sets derived from geometry**, not taste: a band that a 4-bit path passes cannot be
   accepted until the same band is shown to catch injected damage in exactly the tensors that carry
   it (wrong sub-block scale, wrong expert index, a swapped gate/up half, a router that picked
   different experts). See `pitfalls/40`.
3. **Router, norms, conv1d, `A_log`, `dt_bias` stay exact** in the first cut — they are 0.11 GiB and
   every one of them is a scale or a control, where a 1-ULP change is not a rounding difference.
4. **The engine-level check is decision agreement over a corpus, at 4k context**, with the accepted
   chaos floor of bf16 (`pitfalls/30`) as the background against which quantization error has to
   stand out. If it cannot be made to stand out, that is the finding, and it is reported, not
   routed around.

## Open items this file hands back

- **D-01a is DECIDED (2026-10-26): attention/GDN at Q8_0**, so **policy A** — experts Q4_K, attention +
  GDN + shared expert Q8_0, router/norms exact — is the shipping shape: 19.59 GiB resident, 2.331
  GiB/token, ceiling 95 / 84 / 76 tok/s at 237 / 210 / 190 GB/s. Policy E (attention Q4_K too,
  1.663 GiB/token, ~133 tok/s) stays a funded experiment behind a KLD study, not a default.
- **D-10 is DECIDED (2026-10-26): `lm_head` and embeddings at Q8_0.** Policy G (Q6_K, +4 % of ceiling)
  is not taken, so policy A's 19.59 GiB / 2.331 GiB/token is final as a width assignment.
- **D-09 is DECIDED (2026-10-26): the MTP draft block is uniform** with the stack it drafts for. The
  price of widening it later is in `DECISIONS.md` D-09 (~2–3 % of a step), and acceptance — not
  bandwidth — is the quantity that would justify doing so.
- **D-08 is DECIDED (2026-10-26): experts at Q4_K** (`DECISIONS.md` D-08). `quantization/evidence/mxfp4-vs-q4k-2026-10-26.md`
  carries why (MXFP4 is E2M1 + one power-of-two E8M0 scale per 32; native only on MI350/Blackwell class;
  llama.cpp repacks it to Q8_1 on RDNA3.5; 0.96 GiB + ~1.3 % of ceiling against a ~42 % worse KLD). What
  stays open is only the **dequant-cost** measurement — `prototype/moe_gemv` timing both inner loops —
  which is the one finding that could reopen the format question.
- The per-token figures above assume **greedy single-token decode at short context** (KV ignored,
  which is right at 128 lanes and ~+0.04 GiB at 4k). Re-run with the real KV geometry once the engine
  exists, and remember the MTP path changes the arithmetic: a drafted token reads experts too.
- Nothing in this file is a measurement of *this engine's* dequant speed. That is a
  `prototype/moe_gemv` probe (`tools/bw_probe_hip.cpp` shape + a dequant in the inner loop), and until
  it runs, the ceiling column is an upper bound on a kernel we have not written.
