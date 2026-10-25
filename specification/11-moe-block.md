# 11 — The MoE block: router, top-k, grouped experts, shared expert

This is the subsystem this project exists to land (`AGENTS.md` §SCOPE). Everything the block computes
is defined here; every kernel, `speccheck/` file and gate that touches routing is derived from this
file. Symbols come from `specification/00-notation-and-hyperparameters.md` §00.2, tensor names and
bytes from `specification/01-weights-and-shapes.md` §01.3.

The block is present in **every** one of the 40 layers plus the MTP draft block — 41 MoE blocks, no
dense layer, no `mlp_only_layers` key in this config.

## §11.1 Position in the layer, and the block's only input

```
y = DecoderLayer(x):  h = RMSNorm1p(x, w_in)         -> token mixer      -> x + mixer(h)
                      g = RMSNorm1p(x + mixer, w_post)                    <- the block's input
                      out = g + MoE(g)                                    -> y
```

`MoE(·)` takes exactly one tensor, the `post_attention_layernorm` output `g`, shaped `[B, T, d]`. It
flattens to `[B*T, d]`, and every sub-op below is a per-row operation on that flat view. The residual
add is the decoder layer's, **outside** the block (`specification/03-block-glue-and-normalization.md`
§03.3): the block returns routed-sum + shared-sum, it does not add `g`.

Order inside the block, as the reference writes it: shared expert first, then router, then the routed
experts, then the shared expert's sigmoid gate, then the sum
`[REF: Qwen3_5MoeSparseMoeBlock.forward]`. The shared-expert MLP does not depend on the router, so
the two branches are independent; only the final add joins them. Do not read that statement order as
a data dependency.

## §11.2 Router — `mlp.gate.weight` to routing weights, with the cast order

`mlp.gate.weight` / `blk.N.ffn_gate_inp.weight`, `[E, d] = [256, 2048]`, BF16 in the checkpoint, F32
in the published file. No bias. Per row of `g`:

| # | operation | precision |
|---|---|---|
| R1 | `logits[l] = sum_{j<d} g[j] * W[l][j]`, for `l = 0..255` | the activation dtype (BF16) — this is an ordinary `d`-wide dot product, and it is the **only** place the router's own width enters |
| R2 | `probs = softmax(logits, dim=-1, dtype=F32)` over all 256 | **F32**, upcast from the BF16 logits — not softmax-of-BF16 |
| R3 | `(w[0..7], id[0..7]) = topk(probs, 8, dim=-1)` | selection, no arithmetic |
| R4 | `w[k] = w[k] / sum_{k<8} w[k]` | **F32** — the renormalisation happens on the F32 softmax values |
| R5 | `w = cast_to_dtype_of(logits)(w)` | **the cast is to the LOGITS' dtype (BF16), and it happens BEFORE the multiply** |

R2–R4 are `[REF: Qwen3_5MoeTopKRouter.forward]`, quoted so this file is self-contained:

```
router_logits  = F.linear(hidden_states, self.weight)
router_probs   = F.softmax(router_logits, dtype=torch.float, dim=-1)
router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
router_top_value = router_top_value.to(router_logits.dtype)
```

**The cast order is authoritative as written.** The renormalised weights are cast back to the logits'
dtype (BF16 here) *and then* multiply the expert output. The tempting rewrite — keep the weights in
F32, multiply in F32, cast the product afterwards — is a different number, not an equivalent one: it
keeps 8 mantissa bits of routing weight that the reference threw away, so it changes every one of the
8 scaled outputs, and on a near-tie row it can change which 8 experts `topk` picked as well. A
`speccheck/` oracle reproduces R5. An engine that wants the extra precision is a numerics change and
must show it against the R5 oracle with a floor-probe-sized band and the geometry-derived must-catch
set (`pitfalls/30-numerics.md`, `pitfalls/40-gates-and-admissions.md`); it is not adopted silently.

`AGENTS.md` lists this as prediction 4 to falsify, and it stays a prediction until the dump says
which side the pinned rows actually separate — `specification/ASSUMPTIONS.md` A-05.

Three further router facts:

- **No `norm_topk_prob` flag exists in this model's router class.** The renormalisation at R4 is
  unconditional, so there is no "renormalise or not" switch to set; a config that omits the field and
  a config that sets it to true behave identically here.
- **The router is read whole every token**: all 256 × 2048 values, 41 layers, 0.0400 GiB/token at
  bf16 (`python3 specification/derive.py --per-token`). It is 0.06 % of the weights and gates 92 % of
  them.
- **The router stays exact under every decided width policy** (`quantization/README.md` §"How a
  quantized path gets gated" item 3): a wrong router row does not degrade quality, it selects the
  wrong experts and still prints fluent text.

## §11.3 Selection semantics that a reimplementation must pin

| question | answer | why it is not free |
|---|---|---|
| order of the 8 outputs | `topk` returns them in **descending probability** order | R5's vector and the `id` vector must stay index-aligned |
| ties in `probs` | `topk`'s tie-break is not specified by this spec | a near-tie row can swap an expert with no arithmetic error anywhere; `specification/ASSUMPTIONS.md` A-03 owns it, and the pinned dump must contain a row whose 8th/9th margins are under one BF16 ULP of the 8th value before any gate claims to cover it |
| accumulation order of the 8 contributions | **ascending expert id**, not descending weight (§11.5) | a kernel that accumulates in top-k rank order is legal but a different number (`pitfalls/30-numerics.md` §association) |
| duplicate ids | impossible from `topk` on distinct positions; a masked row is dropped instead | see the `E+1` note below |
| masked/padding rows | the reference builds the one-hot over `E + 1` classes and skips class `E`, so a padding token's routing is discarded rather than folded into expert 255 | only reachable with an `attention_mask`; unreachable in the single-request text path this spec covers |
| routing weights of the shared expert | **not** involved — the shared expert has its own scalar gate (§11.6) | folding the shared expert into the top-k sum is a different model |

## §11.4 The expert body — one SwiGLU, three matrices, `I_e = 512`

Per layer the routed experts are two fused tensors
(`[REF: Qwen3_5MoeExperts.__init__]`):

```
gate_up_proj : [E, 2*I_e, d] = [256, 1024, 2048]
down_proj    : [E, d,   I_e] = [256, 2048,  512]
```

and per selected expert `e` of a token's row `x` (`[REF: Qwen3_5MoeExperts.forward]`):

```
gate, up = linear(x, gate_up_proj[e]).chunk(2, dim=-1)      # both [I_e]
s        = silu(gate) * up                                    # [I_e]
o        = linear(s, down_proj[e])                            # [d]
```

- **The split is along the output axis, not the input axis.** `chunk(2, dim=-1)` cuts the `2*I_e`
  output rows of the expert's matrix in half: `gate` is rows `0..511`, `up` is rows `512..1023` of
  `gate_up_proj[e]`. In the fused GGUF spelling that boundary is `ne[1] = 1024` split at 512 with
  **gate first**. A SwiGLU kernel written for two separate `[I_e, d]` matrices has to be checked for a
  stride/offset swap against this, not re-derived (`AGENTS.md` prediction 3).
- Split and fused are both legal storage: the published file stores `ffn_gate_exps` + `ffn_up_exps`
  (40 + 40 Q4_K tensors, no `ffn_gate_up_exps` at all), and `--fuse-gate-up-exps` (default off) emits
  the fused `ffn_gate_up_exps` by concatenating gate then up along that same axis. **The reader must
  accept both** (`specification/01-weights-and-shapes.md` §01.6 item 5).
- `silu(z) = z / (1 + exp(-z))` from `hidden_act: silu`; the product `silu(gate) * up` is in the
  activation dtype.
- There are no biases anywhere in this path (`attention_bias: false`, every `nn.Linear(..., bias=False)`).
- One expert is `3 * I_e * d = 3,145,728` elements = **6.0 MiB BF16**; all 256 experts of one layer
  are 1.50 GiB. `python3 specification/derive.py --check` asserts both fused shapes against real
  safetensors headers.

## §11.5 Combining the K=8 routed outputs

```
routed[t] = sum over the token's 8 selected experts e, in ascending e:
                expert_out(e, x_t) * w[t, rank_of_e]
```

`w[t, rank_of_e]` is the R5 weight at the rank `e` occupies in the descending `topk` order — the
reference reaches it through a gather over the (token, rank) pairs, so a kernel that indexes `w` by
expert id instead of by rank is wrong by a permutation. The sum is accumulated in the activation
dtype; the reference adds each expert's scaled output into a `[T, d]` buffer in **ascending expert-id
order**, because it loops over hit-experts (whose indices come back sorted) and gathers their tokens.

For a single-token decode that is 8 dependent rounds of (6 MiB read, 2048-wide write). Across 41 MoE
blocks that is 328 dependent expert reads per drafted-or-not token and 48 MiB of expert bytes per MoE
layer — 1.2812 + 0.6406 = **1.9218 GiB of expert reads per token** at bf16, 0.5406 GiB under policy A
(all from `python3 specification/derive.py --per-token`). `AGENTS.md` prediction 2's fix — one grouped
launch per layer over all selected experts, 41 independent launches rather than 328 dependent reads —
is an engine decision; this file only pins that the *math* is order-independent across experts except
for the accumulation named above.

## §11.6 The shared expert, and exactly how it is combined

Every MoE block carries one always-on dense SwiGLU with `I_s = 512` plus its own scalar gate:

| HF name | GGUF name | shape | n |
|---|---|---|---|
| `layers.N.mlp.shared_expert.gate_proj.weight` | `blk.N.ffn_gate_shexp.weight` | [512, 2048] | 40 |
| `layers.N.mlp.shared_expert.up_proj.weight` | `blk.N.ffn_up_shexp.weight` | [512, 2048] | 40 |
| `layers.N.mlp.shared_expert.down_proj.weight` | `blk.N.ffn_down_shexp.weight` | [2048, 512] | 40 |
| `layers.N.mlp.shared_expert_gate.weight` | `blk.N.ffn_gate_inp_shexp.weight` | [1, 2048] | 40 |

`shared_expert` is the same three-matrix SwiGLU as §11.4 with `I_e -> I_s`, on the same input `g`.
`shared_expert_gate` is a `[1, d]` Linear with no bias producing **one scalar per token**, and the
combination is `[REF: Qwen3_5MoeSparseMoeBlock.forward]`:

```
shared = silu_mlp(g)                                   # [T, d]
shared = sigmoid(linear(g, shared_expert_gate)) * shared
out    = routed + shared
```

| fact | value | consequence |
|---|---|---|
| gate range | `(0, 1)` — sigmoid, not tanh, not a softmax over experts | the shared expert can never be sign-flipped off |
| gate shape | `[T, 1]`, broadcast over `d` | one scalar per token, not per channel |
| gate precision | computed from a BF16 `[1, d]` matmul, sigmoid in the activation dtype | widening the matmul is the same class of change as §11.2 R5 |
| combine | plain add, no scaling constant, no second norm | `out` is the block's return value; the residual add is the layer's |
| bytes/token (41 blocks) | 0.2404 GiB bf16, 0.1277 GiB policy A | read in full every token, unlike the routed stack |

`shared_expert_gate.weight` is a **router-shaped control tensor** even though llama.cpp's tensor policy
puts `ffn_gate_inp_shexp` next to the norms: the published file stores it F32, like `ffn_gate_inp` and
the norms. Keep it in the exact family (`specification/10-reference-dumps-and-gates.md` §10).

## §11.7 The router as the serial point — a fact about the math, not the kernel

The reference selects experts with `one_hot` → `nonzero` → `torch.where` inside a `no_grad` block,
then loops over the resulting host-side index list
`[REF: Qwen3_5MoeExperts.forward]`. Each of those is a device→host synchronisation, so the naive form
makes the router a per-token host round-trip before any expert byte is read — which is `AGENTS.md`
prediction 1's mechanism, and why "count host syncs per decode step before optimising any expert
kernel" comes first. Nothing in §11.2–§11.6 requires a host value: R1–R5, the gather of `w` by rank
and the accumulation over `e` are all expressible with the top-`K` indices and weights staying on
device. That is a spec property, so a gate can assert it: **the block's outputs must be computable
with zero host-visible routing values.**

## §11.8 What a quantized path may and may not change

| may change (declared widths) | may not change |
|---|---|
| expert `gate_up`/`down` storage width (Q4_K per D-08; the shipped file has down at Q5_K/Q6_K) | the R1–R5 precision classes, the R5 cast position, the top-`K` set, the accumulation order |
| shared expert width (Q8_0) | the sigmoid gate's scalar-per-token semantics |
| block order, gather strategy, launch shape | any band: bands for a 4-bit expert output come from a floor probe over pinned dumps, and a band is only admissible once it still catches every geometry-derived injection — wrong sub-block scale, wrong expert index, swapped gate/up half, a router that picked different experts (`pitfalls/40-gates-and-admissions.md`) |

The oracle stays bf16 (`quantization/README.md` §"How a quantized path gets gated"): a quantized
expert is judged against the bf16 reference, never against another quantization, and never against
what the shipped file happens to produce.
