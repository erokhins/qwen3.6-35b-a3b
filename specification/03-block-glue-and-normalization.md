# 03 — Block glue: the two RMSNorm forms, the layer skeleton, the logits head

Everything between token mixers: which norm is which form, what order the residual stream is
updated in, and how the final hidden state becomes logits. Symbols from
`specification/00-notation-and-hyperparameters.md` §00.2; `eps_rms = 1e-6` `[CFG:
text_config.rms_norm_eps]`.

## §03.1 There are two RMSNorm forms in this model, and they are not interchangeable

**Form A — zero-centred, `x_hat * (1 + w)`.** Used by every norm in the residual stream and by the
per-head QK norms. `[REF: Qwen3_5MoeRMSNorm]`:

```
_norm(x)  = x * rsqrt( mean(x * x, dim=-1) + eps_rms )
out       = _norm(x.float())                       # F32 core
out       = out * (1.0 + w.float())                # the +1 is the form, and the weight is F32
return    cast_to_dtype_of(x)(out)                 # the cast back is the last operation
```

**Form B — plain, gated, `w * x_hat` then a SiLU gate.** Used only by `linear_attn.norm`, and it is
specified in `specification/04-gdn-linear-attention.md` §04.7 with its variance-over-`d_v` and its
intermediate cast. It has **no `+1`**.

| | Form A | Form B |
|---|---|---|
| weight applied | `(1 + w) * x_hat` | `w * x_hat`, after a cast back to the input dtype |
| gate | none | `* silu(z.float())` |
| variance axis | the tensor's last axis (2048 for the stream, 256 for QK, 2048 for the MTP fusion norms) | `d_v = 128`, per V head |
| tensors | `input_layernorm`, `post_attention_layernorm`, `model.language_model.norm`, `self_attn.q_norm`, `self_attn.k_norm`, and the MTP block's five norms | `linear_attn.norm` only |
| stored value near | 0 (so that `1 + w ≈ 1`) | 1 |
| converter `+1` | yes — the GGUF holds `w + 1` in F32 | **no** — the GGUF holds `w` as stored |

Two independent witnesses that the split is real, both from real bytes
(`python3 specification/derive.py --values model.language_model.layers.0.input_layernorm.weight
model.language_model.layers.0.linear_attn.norm.weight`):

| tensor | n | min | max | mean | reading |
|---|---|---|---|---|---|
| `layers.0.input_layernorm.weight` | 2048 | −0.0800781 | 0.330078 | 0.0311106 | stored near 0 ⇒ must be read as `1 + w` |
| `layers.0.linear_attn.norm.weight` | 128 | 0.523438 | 0.957031 | 0.884155 | stored near 1 ⇒ plain `w` |

and the converter agrees from the other side: it adds 1 to names ending `norm.weight` **and explicitly
excludes** `linear_attn.norm.weight` (`specification/01-weights-and-shapes.md` §01.6 item 3).

**The trap this creates for an engine reading the GGUF.** The published file's norms already contain
the `+1` (in F32), so a loader that reads them and then applies `1 + w` again computes `2 + w` — a
silent 2× on every norm in the stream that still produces fluent text. The rule is single-sourced:
**the form is a property of the artifact you are reading, not of the model.** From the checkpoint:
add 1. From the published file: do not. Adding 1 in BF16 (rather than in F32, as the converter does)
loses bits the file keeps, which is why the exactness pass compares these tensors at zero tolerance
(`PLAN.md` §Sub-steps 1.4a).

## §03.2 The decoder layer, in order

Both layer types share the skeleton; only the mixer differs. `mixer_N` is the GDN block
(`specification/04-gdn-linear-attention.md`) when `layer_types[N] == "linear_attention"` and the full
attention block (`specification/05-gated-full-attention.md`) when it is `"full_attention"`.

```
1  h_in   = RMSNormA(x, w_input_layernorm)            # Form A, [CFG] eps 1e-6
2  m      = mixer_N(h_in)                              # no residual inside the mixer
3  x1     = x + m                                      # activation dtype add
4  h_mlp  = RMSNormA(x1, w_post_attention_layernorm)   # Form A
5  f      = MoE(h_mlp)                                 # specification/11-moe-block.md
6  y      = x1 + f
return y
```

| fact | detail |
|---|---|
| pre-norm, both times | there is no post-mixer norm, and no norm on the residual |
| the two norms are different tensors | `input_layernorm` and `post_attention_layernorm`, 40 of each — a loader that aliases them trains nothing and reads the wrong scale |
| no layer scales, no per-layer temperature, no hyperconnections | the Gated-Residual/hyperconnection family is Flash-Next-only and absent from this checkpoint (`DECISIONS.md` D-04) |
| `is_causal` and masking | only the mixer sees a mask; §05.3 and `specification/04-gdn-linear-attention.md` §04.2's padding rule |
| residual add precision | plain activation-dtype add. Hoisting or reassociating it is a numerics change (`pitfalls/30-numerics.md` §"A hoisted scalar constant is a different number") |
| tuple unpacking | the reference tolerates the MoE returning `(out, aux)`; with `output_router_logits: false` there is no second value, and this spec defines the block as returning one tensor (§11.1) |

`[REF: Qwen3_5MoeDecoderLayer.forward]` — `residual = hidden_states; hidden_states =
self.input_layernorm(hidden_states); ...; hidden_states = residual + hidden_states;` then the same
pair around `self.mlp(self.post_attention_layernorm(...))`.

## §03.3 What is handed to a mixer, and what a hook must record

The mixer's input is the **normed** tensor `h_in` (step 1), and the mixer's output is the pre-add
tensor `m` (step 2). Symmetrically for the MLP: `h_mlp` in, `f` out. `speccheck/` dumps must pin
`h_in`, `m`, `h_mlp`, `f` and both residuals — and the rule from `pitfalls/30-numerics.md` applies
verbatim: a hook that records the layer's **output** when the consumer wants the **input** reproduces
real reference tensors, passes every self-check, and is the wrong quantity. For the MoE block that
means the shared-expert MLP's activation is `down_proj`'s **input**, not the block's output
(`specification/11-moe-block.md` §11.6).

## §03.4 Final norm and the logits head

```
x_final   = RMSNormA(x_(L-1), model.language_model.norm.weight)      # Form A, [2048]
logits    = lm_head(x_final)                                          # [V, d] @ [d] = [V]
```

| fact | value | consequence |
|---|---|---|
| the head is **untied** | `lm_head.weight` [248320, 2048], a separate tensor from `embed_tokens.weight` (`tie_word_embeddings: false`, both at top level and in `text_config`) | 508,559,360 elements, 0.9473 GiB BF16, **read whole for every token** — a fifth of a bf16 decode's per-token bytes (`python3 specification/derive.py --per-token`) |
| the head has no bias | `nn.Linear(V, d, bias=False)` | nothing to allocate |
| the final norm is Form A | `model.language_model.norm.weight` → GGUF `output_norm.weight` (F32, `+1` baked in) | §03.1's artifact rule applies |
| only one row is needed in decode | logits are required at the positions whose tokens are being scored; the reference crops with `inp_out_ids` before the head | a decode step that computes `T` rows of logits pays `T × 0.95 GiB` of reads for nothing |
| greedy decoding | `argmax` over the 248,320 logits, in the logits' own dtype | the comparison that decides every token; its band is decision-level, never a ULP bound (`pitfalls/30-numerics.md`) |
| sampling beyond greedy | out of scope | `specification/README.md` §Scope |

## §03.5 Precision classes owned by this file

| op | precision |
|---|---|
| Form A core `_norm` | F32, from `x.float()` |
| Form A weight multiply | F32 (`w.float()`), i.e. `(x_hat * (1 + w))` is evaluated in F32 and cast once, last |
| both residual adds | activation dtype |
| final norm, logits matmul | activation dtype (the head's own storage width is a policy question, not a precision class — `quantization/README.md`) |

A `speccheck/` implementation that computes Form A's weight multiply in BF16 has changed the model,
even though the reference's own comment calls out exactly this difference against Llama's form
("`Llama does x.to(float16) * w` whilst Qwen3_5Moe is `(x * w).to(float16)`") — that sentence is the
reason the cast position is written into this spec instead of left to an implementation note.
