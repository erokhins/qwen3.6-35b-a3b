# 05 — Gated full attention

The token mixer of the 10 `full_attention` layers — `N ∈ {3, 7, 11, 15, 19, 23, 27, 31, 35, 39}`,
i.e. `(N + 1) % full_attention_interval == 0` with `full_attention_interval: 4` `[CFG]`. Which
`blk.N` carry `attn_q` rather than `attn_qkv` in the published file confirms the same 10 indices
(`specification/01-weights-and-shapes.md` §01.4). The MTP draft block is a full-attention layer too,
so this file's math applies to it as well (`specification/09-mtp-draft-layer.md`).

Geometry: `n_q = 16`, `n_kv = 2` (GQA group 8), `d_head = 256`, `d = 2048`, `attn_output_gate: true`,
no biases anywhere (`attention_bias: false`), `attention_dropout: 0.0`.

## §05.1 Tensors, and why `attn_q` has twice the rows it "should"

| HF name (under `layers.N.self_attn.`) | GGUF name | HF shape | GGUF `ne` | dtype (ckpt) |
|---|---|---|---|---|
| `q_proj.weight` | `blk.N.attn_q.weight` | [8192, 2048] | [2048, 8192] | BF16 |
| `k_proj.weight` | `blk.N.attn_k.weight` | [512, 2048] | [2048, 512] | BF16 |
| `v_proj.weight` | `blk.N.attn_v.weight` | [512, 2048] | [2048, 512] | BF16 |
| `o_proj.weight` | `blk.N.attn_output.weight` | [2048, 4096] | [4096, 2048] | BF16 |
| `q_norm.weight` | `blk.N.attn_q_norm.weight` | [256] | [256] | BF16 |
| `k_norm.weight` | `blk.N.attn_k_norm.weight` | [256] | [256] | BF16 |

`q_proj` has `n_q * d_head * 2 = 8192` output rows `[REF: Qwen3_5MoeAttention.__init__ —
nn.Linear(hidden_size, num_attention_heads * head_dim * 2)]`: **one query head and one gate head per
query head.** Sizing this tensor at `n_q * d_head` reads half of it and raises nothing. The 6 other
tensors are the obvious shapes; there is no separate gate tensor — the gate is inside `attn_q`.

The per-head layout inside those 8192 rows is **interleaved, not two halves** `[REF:
torch.chunk(self.q_proj(x).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)]`. View the output
vector as `[n_q, 2 * d_head]` and cut the *last* axis in two, so for head `a`:

```
q[:, a, 0:256]     -> query head a
q[:, a, 256:512]   -> output gate head a
```

In row indices of the HF `[8192, 2048]` tensor, query head `a` is rows `512a .. 512a + 255` and gate
head `a` is rows `512a + 256 .. 512a + 511`. In the GGUF spelling the same axis is `ne[1]`, so the
gate rows are `ne[1] ∈ [512a + 256, 512a + 512)`. A reader who takes the upper half of the tensor as
the gate (`rows 4096..8191`) gets head 8's query as head 0's gate — silent, and fluent.

## §05.2 Projections, then per-head QK norm, then rope — in that order

```
qg     = q_proj(x)                                  # [B, T, 8192]   activation dtype
q, gate = view(qg, [B, T, n_q, 2*d_head]) chunked on the last axis   # [B,T,n_q,d_head] each
gate   = reshape(gate, [B, T, n_q*d_head])          # [B, T, 4096]
q      = q_norm(q.view(B, T, n_q, d_head)).transpose(1, 2)     # [B, n_q, T, d_head]
k      = k_norm(k_proj(x).view(B, T, n_kv, d_head)).transpose(1, 2)
v      = v_proj(x).view(B, T, n_kv, d_head).transpose(1, 2)
q, k   = rope(q, k, cos, sin)                        # §05.4
```

| fact | detail |
|---|---|
| `q_norm`, `k_norm` are the **zero-centred** RMSNorm of §03.1 — `x_hat * (1 + w)` | they end in `norm.weight`, so the converter stores them as `w + 1` in F32 (`specification/01-weights-and-shapes.md` §01.6 item 3) |
| their width is `d_head = 256`, applied per head | one variance per (token, head) over the whole 256-wide head, **not** over the rotary slice and not over 4096 |
| the norm is applied **before** rope | norm-then-rotate; rotating-then-norming is a different number because rope mixes pairs |
| the gate is not normed | `gate` comes straight out of the projection and is used only in §05.5 |
| `v_proj` is the same shape as `k_proj` | no separate value width: `n_kv * d_head = 512` |
| rope sees q and k after the transpose to `[B, heads, T, d_head]` | `cos`/`sin` broadcast with `unsqueeze(1)` |

## §05.3 The attention itself

```
groups = n_q / n_kv = 8
K, V   = repeat each KV head groups times along the head axis      # [B, n_q, T, d_head]
scores = (q @ K^T) * scale                                          # scale = d_head ** -0.5 = 1/16
scores = scores + causal_mask                                       # additive, 0 / -inf
p      = softmax(scores, dim=-1, dtype=F32) . cast_to(query dtype)  # F32 softmax, then downcast
attn   = (p @ V) . transpose(1, 2) . contiguous                     # [B, T, n_q, d_head]
```

| fact | value | note |
|---|---|---|
| `scale` | `1/16` exactly (`256 ** -0.5`) | not `1/sqrt(d)` of the hidden size; the reference sets `self.scaling = self.head_dim**-0.5` `[REF]` |
| softmax precision | computed in **F32 over the full key axis**, then cast back to the query dtype **before** the PV product | the downcast position matters as much as the upcast (§05.6) |
| mask | causal; a left-to-right mask over the *cached* key length during decode | additive form, so a padding mask and a causal mask are the same tensor addition |
| GQA expansion | `n_rep = 8`, each KV head serving heads `8g .. 8g+7` | grouped, and this is the one place "grouped" is the correct word: GQA grouping is HF's, unrelated to the GDN V-head order |
| dropout | none (inference) | `attention_dropout: 0.0` |
| KV written to cache | K and V **before** the attention, after rope, at `[B, n_kv, T, d_head]` | §05.7 and `specification/08-state-management.md` |

## §05.4 RoPE — theta, the 0.25 partial width, and the interleaved mrope sections

```
inv_freq[i] = 1 / theta ** (i / R)          for i = 0 .. R/2 - 1,  R = int(d_head * 0.25) = 64
freq[t, i]  = pos_t * inv_freq[i]
cos, sin    = cat(f, f, dim=-1)             # width R = 64, the 32 frequencies written twice
```

with `theta = rope_theta = 1e7` `[CFG]`. The rotation replaces the **first `R = 64`** dims of each
head, pairing `i` with `i + 32`:

```
for i in 0 .. 31:
    out[i]      = x[i]      * cos[i]      - x[i + 32] * sin[i]
    out[i + 32] = x[i + 32] * cos[i + 32] + x[i]      * sin[i + 32]
out[64 : 256]   = x[64 : 256]                                  # untouched
```

Because `cos`/`sin` are the 32 values duplicated, `cos[i] == cos[i + 32]` and the pair is rotated by
one frequency — the ordinary half-style rotation, restricted to the first quarter of the head. All
witnessed on this checkpoint, not inferred: a CPU probe of the checkpoint's own rotary module with
this committed config returns `cos` of width 64 for `head_dim` 256, `cos == cat(c, c)`, the pair
`(7, 39)` agreeing with the two lines above to bf16 rounding, all rotated indices inside 0..63, and
perturbing input dim 100 moving only dim 100 (command and output in
`specification/ASSUMPTIONS.md` A-14's witness).

| fact | detail |
|---|---|
| `R = 64` is the product, not a stored field | `int(d_head * partial_rotary_factor)`; the producer writes the same number as `qwen35moe` rope dimension count, so an engine reading either source gets 64 |
| 32 frequency pairs | `arange(0, R, 2) / R` in F32, `1.0` down to `1.6548e-07` for this theta |
| `cos`/`sin` computed in F32, cast to the hidden dtype on return | the rope table is therefore BF16 in a bf16 build; that cast is part of the form |
| `partial_rotary_factor` appears twice | once at `text_config.partial_rotary_factor`, once inside `text_config.rope_parameters`; both 0.25 `[CFG]`. The rotary module reads the one inside `rope_parameters` (defaulting to 1.0 when absent — the reason 3.6 was chosen over 3.5, `README.md` §Target) |
| `rope_type: default` | no YaRN, no linear scaling; extension beyond 262144 is out of scope (`DECISIONS.md` D-04) |

**mrope for text-only is inert, and that is a claim worth being explicit about.**
`mrope_interleaved: true` with `mrope_section: [11, 11, 10]` `[CFG]` selects, for each of the 32
frequency columns, which of the three position axes supplies `pos`: column `j` takes axis T unless
`j % 3 == 1 and j < 33` (axis H) or `j % 3 == 2 and j < 30` (axis W) — `11 + 11 + 10 = 32` `[REF:
recomposition_frequencies]`. The text model builds `position_ids` as one row of consecutive integers
broadcast over all four axes, so **T, H and W carry the same integer and the section assignment
changes nothing**: for text-only prefill and decode the result is exactly the ordinary 1-D rotation
above. The sections only start to matter if a vision/grid path is added, which is out of scope
(`DECISIONS.md` D-04). The same probe above witnesses the inertness: `cos` equals
`cat(cos(pos * inv_freq), cos(pos * inv_freq))` built from a single position value.

## §05.5 The output gate — where it sits and what supplies it

The gate is supplied by the second half of each head of `attn_q` (§05.1), and it is applied **after**
the attention and its head-concat, **before** `o_proj` `[REF: Qwen3_5MoeAttention.forward]`:

```
attn = attn.transpose(1, 2).reshape(B, T, n_q * d_head)      # [B, T, 4096], head-major
y    = attn * sigmoid(gate)                                   # [B, T, 4096], activation dtype
out  = o_proj(y)                                              # [B, T, 2048]
```

| fact | detail |
|---|---|
| position in the order | `attn → reshape → multiply by sigmoid(gate) → o_proj`. Not before the attention, not after `o_proj` |
| the sigmoid is computed on the raw gate rows | no norm, no scale, no bias on those rows; range `(0, 1)` |
| element alignment | gate element `(a, i)` scales attention element `(a, i)` — head `a`, in-head index `i`. Both vectors are head-major `[n_q * d_head]`, so the alignment is positional and needs no permutation |
| precision | `sigmoid` in the activation dtype; widening it is the same class of numerics change as the router's cast order (§11.2) |
| independence | the gate is a per-element multiplicative mask on the head outputs; it does **not** normalise across heads and is not a softmax |

## §05.6 Exit, and the precision classes to pin

`o_proj` is `[d, n_q * d_head] = [2048, 4096]`, no bias; its output is what the decoder layer adds to
the residual (§03.3).

| op | precision the reference uses |
|---|---|
| q/k/v projections, `o_proj`, the gate multiply | activation dtype (BF16) |
| `q_norm`, `k_norm` | F32 core, weight multiply in F32, cast back last — §03.1's `(1 + w)` form |
| rope arithmetic | F32 tables, cast to activation dtype before use |
| QK scores | matmul in the activation dtype, scaled, mask added |
| softmax | **F32**, then cast to the query dtype before PV |
| PV, reshape, gate multiply | activation dtype |

## §05.7 KV inventory

Per layer per token: `2 (K, V) × n_kv × d_head × 2 B = 2 × 2 × 256 × 2 = 2048 B`. Over the 10
full-attention layers that is 20,480 B = **20.0 KiB per token**; with the draft block's own KV layer,
22,528 B = 22.0 KiB — `python3 specification/derive.py --state` prints both, and 80.00 MiB at 4096
lanes. `README.md`'s "10 KiB/token" row disagrees; `specification/ASSUMPTIONS.md` A-08 carries both
numbers and the arithmetic that adjudicates them.
