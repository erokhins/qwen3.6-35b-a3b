# 07 — The decode step

One token in, one token out, against a populated state bundle. This is the loop the whole engine
exists to serve, and the pass where the router becomes the per-token serial point. Differences from
prefill are called out explicitly, because most decode bugs are a prefill path accidentally taken (or
the reverse).

## §07.1 Inputs

| input | shape | note |
|---|---|---|
| `input_ids` | `[B, 1]` | the id emitted by the previous step (or the first continuation token) |
| position | scalar per row: `past_seen_tokens` | absolute, not reset by any window (§02.3) |
| state bundle | `specification/08-state-management.md` | recurrent `S`, conv window, KV — all read **and** written by this step |

## §07.2 The ordered op list for one token

Identical to `specification/06-prefill-pass.md` §06.2 steps 3–33 with `T = 1`, plus the four
differences below. Only those four are spelled out here; everything else keeps its prefill definition
byte for byte, which is the point — the two paths must be indistinguishable in their arithmetic.

| step | prefill | decode |
|---|---|---|
| rope | tables for `0..T-1` | `cos/sin` for the single position `past_seen_tokens` (§05.4) |
| conv (§04.3) | causal conv over the whole window, keep last `T` | **update path**: concatenate the stored window with the new column, write back the last `state_len` columns, convolve with no padding, keep the last output |
| GDN scan (§04b §04b.1) | chunked, `C = 64`, `initial_state = 0` | **recurrent** (§04.6), `initial_state = S`, 1 step, new `S` written |
| KV (§05.7) | `T` columns appended after rope | 1 column appended after rope, then attention reads `past_seen_tokens + 1` keys |
| logits crop | the rows being scored | exactly 1 row per request |

Everything that does **not** change is worth stating because each is a place an engine can drift:
the conv still applies SiLU to all 8192 channels after the conv; `z` still bypasses the conv; the
recurrent scan still runs in F32 and still reads the **post-update** state (§04.6 step 5); the K/V
still enter the cache already rotated; the MoE block still gets `h_mlp`, still adds the shared expert
with a scalar sigmoid gate, and still does not add the residual.

## §07.3 The router is the per-token serial point

The router (`specification/11-moe-block.md` §11.2) gates every expert read: 256 logits, softmax in
F32, top-8, renormalise, cast. Its output is the only thing that decides which 8 of 256 experts' 48
MiB are read per MoE block, and there are 41 blocks per token.

The reference's selection is host-bound by construction `[REF: Qwen3_5MoeExperts.forward —
one_hot(num_experts+1).permute(...) → greater(...).nonzero() → for expert_idx in expert_hit:
torch.where(expert_mask[expert_idx]) ...]`, and each of `nonzero` and `torch.where` is a
device→host synchronisation:

| sync site in the reference | count per MoE block |
|---|---|
| `expert_hit = (...).nonzero()` | 1 |
| `torch.where(expert_mask[expert_idx])` | one per hit expert = 8 |
| **per token, 41 MoE blocks** | **≥ 369 host-visible routing values** |

That count is read off the reference's code, not measured, and it is the mechanism behind
`AGENTS.md` prediction 1. Two consequences the spec pins, both checkable without a timer:

1. **The block's outputs must be computable with zero host-visible routing values**
   (`specification/11-moe-block.md` §11.7). The top-`K` ids and weights stay on device; the expert
   gather is indexed by them. Any implementation that needs an expert list on the host has taken the
   reference's shape, not its math.
2. **One grouped launch per MoE block** — all selected experts of one layer in one launch, 41
   independent launches per token rather than 328 dependent reads (`AGENTS.md` prediction 2,
   `specification/11-moe-block.md` §11.5). Which launch shape actually wins is an engine measurement
   owned by the main agent; what this file owns is that the *result* is the sum over the selected
   experts in ascending expert-id order.

## §07.4 Per-token byte budget (weights only)

`python3 specification/derive.py --per-token`. No KV or state read is included — that grows with
context and is `specification/08-state-management.md`'s accounting, not this table's.

| component | bf16 | policy A | published file |
|---|---|---|---|
| routed experts gate\|up, 8 of 256 | 1.2812 | 0.3604 | 0.3604 |
| routed experts down, 8 of 256 | 0.6406 | 0.1802 | 0.2227 |
| routers, all 256 logits × 41 | 0.0400 | 0.0801 | 0.0801 |
| shared experts + gate × 41 | 0.2404 | 0.1277 | 0.1277 |
| full attention, 10 + draft | 0.5586 | 0.2968 | 0.2968 |
| GDN projections, 30 layers | 1.8842 | 1.0010 | 1.0010 |
| GDN control | 0.0000 | 0.0000 | 0.0000 |
| `lm_head`, read whole | 0.9473 | 0.5032 | 0.3885 |
| embeddings, one row | 0.0000 | 0.0000 | 0.0000 |
| **total** | **5.5923** | **2.5493** | **2.4771** |

| reading | consequence |
|---|---|
| the three non-expert families that are read **in full** every token (GDN projections, `lm_head`, shared experts) are 3.06 GiB of the policy-A column | "quantize the experts and leave the rest alone" buys far less than the 92 % expert share suggests (`quantization/README.md` §"Family census") |
| the published column is cheaper per token than policy A while being *larger* resident | its `lm_head` is Q6_K (−0.115 GiB/token) and its expert `down_proj` is Q5_K/Q6_K (+0.043) — the two families where the shipped file departs from D-01/D-10 (`specification/01-weights-and-shapes.md` §01.5) |
| the router is 0.0801 GiB/token under policy A — F32 storage | 3.4 % of a token's bytes, spent deciding the other 21 %; it stays exact (`quantization/README.md` §"How a quantized path gets gated") |
| an MTP draft step adds a second MoE block, a second attention layer and a second `lm_head` read | the drafter's own arithmetic is `specification/09-mtp-draft-layer.md` §09.6 |

## §07.5 Stop test and loop exit

```
next_id = argmax(logits)                                   # 03.4, greedy
if next_id in STOP_IDS: stop, and do not emit next_id       # 02.4 — the set is registered, not decided
append next_id; position += 1; loop
```

| rule | statement |
|---|---|
| one argmax per step | over all `V = 248320` logits, in the logits' own dtype |
| `STOP_IDS` | §02.4: the two committed witnesses (config 248044, artifact tokenizer 248046) are both recorded and neither is promoted; A-04 owns the question |
| position bookkeeping | the position handed to the next step is the number of tokens already in the state bundle, not the number of tokens in the window |
| no sampling | temperature, top-k, top-p and beam search are out of scope; the artifact's `general.sampling.*` keys are inputs to nothing here (§02.5) |
| interaction with MTP | a stop id appearing in a *draft* sequence does not terminate the request; verification decides (`specification/09-mtp-draft-layer.md` §09.5) |
| state on stop | the state bundle is left where the last accepted token left it; there is no partial-step rollback in a plain decode loop (`specification/08-state-management.md` §08.4 covers when rollback is needed) |

## §07.6 What a decode step must reproduce to be considered correct

Per step, `speccheck/` pins and compares: `h_in`, the mixer output `m`, `h_mlp`, the MoE output, the
layer output (§03.3's rule about which tensor a hook must record), the new `S`, the new conv window,
the appended KV column, the router's `ids` and `w` (both — an ids-only comparison cannot see a
weight-permutation error), and the final logits row. Bands come from a floor probe over the pinned
rows, never from another model (`pitfalls/30-numerics.md`, `pitfalls/40-gates-and-admissions.md`), and
the long-dot families (expert GEMV, `lm_head` GEMV) are judged on cos/mean-rel plus decision-level
agreement, not on max-ULP.
