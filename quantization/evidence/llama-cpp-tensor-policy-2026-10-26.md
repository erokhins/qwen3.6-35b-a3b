# llama.cpp's tensor-by-tensor quantization policy (external evidence)

**Provenance.** Read on 2026-10-26 from a fresh clone of `ggml-org/llama.cpp` at tag **b11869**
(commit `d327bf1`, "jinja : fix select and selectattr argument handling"), file
`src/llama-quant.cpp`, functions `tensor_allows_quantization()` (≈line 294) and
`get_quant_type_use()` (≈line 500), plus the `quantize --help` type table from llama.cpp's own
docs/discussion 2094. This is an *external* source: the rules below are what a popular engine does,
copied here so `quantization/README.md` can argue from them without opening that repo (rule:
`pitfalls/60`). Re-read before relying on them in code; upstream moves weekly.

## What it refuses to quantize at all

`tensor_allows_quantization()`: anything with <2 dimensions (so 1-D scales, `A_log`, `dt_bias`, all
norms); every `*_norm.weight`; the **expert gating tensor** (`ffn_gate_inp.weight`, which is this
family's `mlp.gate.weight`); the i32 token→expert routing table on architectures that have one;
`altup`/`laurel` (4×4 things); `per_layer_model_proj`; positional embeddings and token types;
**Mamba/Kimi conv1d weights** (`ssm_conv1d`, `shortconv.conv.weight` — the analogue of this family's
`linear_attn.conv1d.weight`); MiniMax's tiny indexer projections; and architecture-specific small
sets. Also: `output.weight` unless `quantize_output_tensor` is set.

Read that list against this model: the tensors it protects are exactly the router, the norms, the GDN
conv, `A_log`/`dt_bias` and the norms of both attention kinds — 0.11 GiB of the 65.72 GiB text path.
Cost of copying the rule: nothing. Confidence gained: no scale or control lives in a 4-bit block.

## What it forces upward

- Categories `FFN_LATENT` and `ATTENTION_GATE`: if the chosen type is below Q8_0, **it is raised to
  Q8_0**.
- `OUTPUT` (the `lm_head`/output table, and token embeddings when tied): default **Q6_K**
  (Q5_K under the very aggressive IQ1/IQ2 types; **Q8_0 for the `MXFP4_MOE` type and for Falcon or
  when `nx % qk_k != 0`**).
- For 8-expert MoE models, `attn_v` and `attn_k` are bumped to **Q8_0** with the comment "trades just
  ~128MB" — the same reasoning that applies here, where all attention tensors together are 1.97 GiB
  and Q8_0 costs 1.05 GiB of it.
- `attn_v` also gets Q6_K inside `Q4_K_M`/`Q5_K_M` for a subset of layers (`use_more_bits()`), and
  Q5_K for the first four layers of `Q4_K_S`.

## The named types it ships

`quantize --help` calls **Q4_K_M (*recommended*)**, Q5_K_S and Q5_K_M the recommended levels, and
labels the legacy Q4_0/Q4_1/Q5_0/Q5_1 "prefer using Q4_K_M" etc. Its published quality ladder at 7 B
(ppl increase vs unquantized): Q4_0 +0.2499, Q4_K_S +0.1149, **Q4_K_M +0.0535**, Q5_K_M +0.0142,
Q6_K +0.0044, Q8_0 +0.0004.

`LLAMA_FTYPE_MOSTLY_MXFP4_MOE` is the type that matches D-01's shape most directly: **3-D (routed
expert) tensors → MXFP4, every other tensor → Q8_0**, with `output` explicitly forced to Q8_0. That
is policy A in `quantization/README.md` with MXFP4 in place of Q4_K — which is why the format
question (D-08) had to be decided on measured quality rather than on the fact that a 4-bit-experts
type exists.

## Where this leaves our decision

llama.cpp's policy and the unsloth per-tensor sweep (see the sibling note) agree on the shape: **put
the 4 bits in the routed experts, keep attention/latent/control tensors wide, and spend the extra on
`output`**. Where they differ from a naive reading of "Q4_K_M" is that llama.cpp's own default leaves
attention at Q4_K for a model with 256 experts (the Q8_0 bump is gated on `n_expert == 8`), which is
policy **E**, the +40 % option — the one `quantization/README.md` recommends funding as an experiment
instead of shipping first.
