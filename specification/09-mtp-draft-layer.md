# 09 — The MTP draft layer (NextN block)

`mtp_num_hidden_layers: 1` `[CFG]`, and the card says the model was "trained with multi-steps". MTP is
in scope by `DECISIONS.md` D-05, sequenced after the dense path is attributable, and its width is
decided uniform with the stack it drafts for (`DECISIONS.md` D-09).

**The honest starting point: the checkpoint's own reference implementation does not implement this
block.** `Qwen3_5MoePreTrainedModel` and `Qwen3_5MoeForConditionalGeneration` list
`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` `[REF]`, so `transformers` loads the checkpoint and
**throws all 19 `mtp.*` tensors away**. Every form statement in this file therefore comes from the one
shipped implementation that does run them — llama.cpp at the pinned commit
`d327bf1f0780f87b8c01b367b61853b0c291d7c4`, `src/models/qwen35moe.cpp`, `graph_mtp` — and each place
where that leaves a genuine gap is an `[ASSUMPTION]`, listed in §09.7 and registered in
`specification/ASSUMPTIONS.md`. Nothing here was inferred from the checkpoint alone.

## §09.1 What the block is

Structurally: **one full-attention decoder layer with an MoE FFN, plus four NextN tensors.** It has no
GDN block (`derive.py --check` asserts "the draft block has NO GDN block"), its own router, its own
256-expert stack, its own shared expert, and its own KV layer.

| quantity | value | witness |
|---|---|---|
| tensors | 19 | `python3 specification/derive.py --mtp` (real safetensors headers, shards 25 and 26) |
| elements | 844,640,768 | same |
| bf16 bytes | 1,611.024 MiB | same |
| of which the two fused expert tensors | 805,306,368 elts = **95.34 %** | `mtp.layers.0.mlp.experts.gate_up_proj` [256,1024,2048] + `…down_proj` [256,2048,512] |
| in the published GGUF | **0 tensors** — layer indices 0..39, no name containing `mtp` | `tools/gguf_header_types.py` on the shipping artifact |
| GGUF layer index when converted | `blk.40.*` | `specification/01-weights-and-shapes.md` §01.7 |

`PLAN.md`'s fetch table originally priced this block at 41,943,040 elts / 79.9 MiB, wrong by 20.14×,
because it omitted the block's own MoE stack; the corrected row is in `PLAN.md` itself and the
discrepancy is `specification/ASSUMPTIONS.md` A-10.

## §09.2 The input fusion — the four NextN tensors and their halves

| HF name | GGUF name | shape | role |
|---|---|---|---|
| `mtp.pre_fc_norm_embedding.weight` | `blk.40.nextn.enorm.weight` | [2048] | Form A norm on the **token embedding** |
| `mtp.pre_fc_norm_hidden.weight` | `blk.40.nextn.hnorm.weight` | [2048] | Form A norm on the **target hidden state** |
| `mtp.fc.weight` | `blk.40.nextn.eh_proj.weight` | [2048, 4096] | fuses the two halves into the draft layer's input |
| `mtp.norm.weight` | `blk.40.nextn.shared_head_norm.weight` | [2048] | the draft's own final norm, before the shared `lm_head` |

```
e = embed_tokens[candidate_token]                    # [d]     — trunk table, see §09.7 A-19
h = target hidden state at the drafted-from position # [d]     — see §09.7 A-18
c = concat( RMSNormA(e, enorm), RMSNormA(h, hnorm) ) # [2d] = [4096], embedding half FIRST
cur = eh_proj @ c                                    # [d]
```

| fact | statement |
|---|---|
| concat order | **embedding half first, hidden half second** `[REF: ggml_concat(e_norm, h_norm, dim=0)]`. In `mtp.fc.weight` [2048, 4096] the input axis is columns, so columns `0..2047` weight the embedding half and `2048..4095` the hidden half. Swapping them is silent and produces fluent garbage |
| the two norms are Form A | both end in `norm.weight`, so the GGUF stores them `+1` in F32 (`specification/01-weights-and-shapes.md` §01.6 item 3) |
| no dedicated embedding table | `mtp_use_dedicated_embeddings: false` `[CFG]`, and the loader creates `nextn.embed_tokens` as *not required* `[REF]`; the drafter gathers from `token_embd.weight` |
| no dedicated head | the loader creates `nextn.shared_head_head` as not required and the checkpoint has no such tensor, so the drafter multiplies by the trunk `output.weight` — the same 0.9473 GiB table, read a second time per drafted token |
| the draft does have its own final norm | `mtp.norm.weight` is present, so `shared_head_norm` is used and the trunk's `output_norm` is **not** |

## §09.3 The draft layer's op order

`[REF: llama_model_qwen35moe::graph_mtp]`, which is the same body as a trunk full-attention layer
(`specification/05-gated-full-attention.md`) with the fusion in front and the shared head at the back:

```
 1  e_norm = RMSNormA(embed_tokens[candidate], enorm)
 2  h_norm = RMSNormA(h, hnorm)
 3  cur    = eh_proj @ concat(e_norm, h_norm)
 4  resA   = cur                                             # residual for the mixer
 5  h_in   = RMSNormA(cur, blk.40.attn_norm)
 6  qg     = q_proj(h_in)                                     # [8192] — doubled, §05.1
 7  q, gate = per-head chunk of qg;  q = q_norm(q); k = k_norm(k_proj(h_in)); v = v_proj(h_in)
 8  q, k   = rope(q, k, cos, sin)                             # same tables, this layer's positions
 9  append k, v to the DRAFT layer's own KV
10  attn   = causal_attention(q, draft KV, scale = 1/16)
11  m      = o_proj(attn * sigmoid(gate))
12  x1     = m + resA
13  h_mlp  = RMSNormA(x1, blk.40.post_attention_norm)
14  routed = MoE_routed(h_mlp)                                # its own router + 256 experts,  spec 11
15  shared = silu_mlp(h_mlp) * sigmoid(shared_expert_gate(h_mlp))
16  y      = x1 + routed + shared
17  hn     = RMSNormA(y, mtp.norm)                            # shared_head_norm
18  logits = output.weight @ hn[crop]                         # the trunk lm_head
```

| fact | statement |
|---|---|
| its attention is the trunk's §05 in full | doubled `attn_q` with the per-head interleaved gate, QK norms, rope with the same `theta`/`R = 64`/sections, `scale = 1/16`, F32 softmax |
| its MoE is the trunk's §11 in full | including the R1–R5 precision classes and the R5 cast position, the ascending-expert-id accumulation, and the scalar sigmoid shared gate |
| its KV is its own | one extra `n_kv × d_head` layer slot; it never reads or writes a trunk layer's KV (§09.4) |
| `is_recr` is false for it | the loader flags MTP layers non-recurrent explicitly, and derives the recurrent mask from `(i+1) % 4` only for `i < n_layer` `[REF]`, so a draft layer can never be mistaken for a GDN layer |
| the crop is before the head | same rule as §03.4; a draft step that scores `k` candidate positions pays `k × 0.9473 GiB` of head reads |

## §09.4 What state the draft block touches

| state | draft block | trunk |
|---|---|---|
| KV | 1 layer, `[B, 2, T, 256]`, 2048 B per token | 10 layers |
| recurrent `S` | none — no GDN in the block | 30 layers, 60.00 MiB |
| conv window | none | 30 layers |

So the drafter's marginal state is small and disposable, which is what makes truncating a rejected
draft branch cheap (`specification/08-state-management.md` §08.4). The trunk's three state kinds are
advanced **only** by accepted tokens.

## §09.5 Draft, verify, accept

```
loop:
  d_1..d_k      = draft block run k times (see A-19 for the multi-step recursion)
  t_1..t_{k+1}  = target model over the accepted prefix + d_1..d_k, one argmax per position
  accept the longest prefix with d_i == t_i for all i in it; the target's own argmax at the first
  mismatch is emitted as the bonus token
  trunk state advances by the number of accepted tokens plus the bonus token
  draft KV for the rejected positions is truncated
```

| rule | status |
|---|---|
| the acceptance test above is **greedy exact-match plus a bonus token** | `[ASSUMPTION A-17]` — the graph defines the drafter; the accept loop is engine-level in the reference, so the rule is this spec's definition, and its falsifying test is the strong one below |
| **a spec+verify loop must emit exactly the token sequence of a plain greedy run** (`specification/07-decode-step.md` §07.5) at the same temperature-free setting | the gate that makes acceptance checkable without a corpus: any divergence is a bug in acceptance, state rollback or the drafter, never "within band" |
| a stop id inside a draft sequence does not end the request | verification decides; a drafted stop id that the target does not confirm was never generated |
| rejected tokens never reach the trunk state | §08.4's rewind rule, all three kinds together where they exist |
| acceptance *rate* is not a spec quantity | it is a measurement, owned by the main agent, and it is the quantity that would justify widening the draft block (`DECISIONS.md` D-09) |

## §09.6 What a draft step costs

Per drafted token, in the same units as §07.4 and from the same instrument
(`python3 specification/derive.py --per-token`, whose MTP column adds a 41st MoE block and an 11th
attention layer):

| component | bf16 | policy A | note |
|---|---|---|---|
| draft MoE: 8 of 256 experts | 0.0469 | 0.0132 | one block, not 41 — 48 MiB of expert bytes at bf16 |
| draft router | 0.00098 | 0.00195 | 256 logits × 2048 |
| draft shared expert + gate | 0.00586 | 0.00312 | |
| draft attention (q/k/v/o + q/k norms) | 0.0557 | 0.0296 | `specification/01-weights-and-shapes.md` §01.7's four attention tensors |
| draft NextN fusion (`eh_proj` + 4 norms) | 0.0156 | 0.0083 | `mtp.fc` dominates it |
| `lm_head`, read again | 0.9473 | 0.5032 | the dominant cost of a draft step, and the reason drafting `k` tokens in one batched pass matters |

The last row is the reason the drafter's logits are computed for all `k` candidates in one pass rather
than `k` sequential passes: the head is untied and read whole every time
(`specification/03-block-glue-and-normalization.md` §03.4, `AGENTS.md` prediction 5).

## §09.7 Open questions this file refuses to paper over

| id | question | what is known | registered as |
|---|---|---|---|
| A-01 | what is the authoritative form of this block at all | HF-ref ignores `mtp.*`; llama.cpp's `graph_mtp` is the only shipped implementation read here, at a pinned commit | `specification/ASSUMPTIONS.md` A-01 |
| A-18 | which tensor is `h`, the "target hidden state" fed to `hnorm` | the graph takes it as a graph input named `mtp_h_input`; the block carries its own `mtp.norm`, which is consistent with the tap being the last trunk hidden state **before** the final norm | A-18, with the falsifying test |
| A-19 | how a second and later draft step is seeded (its token input and its hidden input), given `mtp_num_hidden_layers: 1` and "trained with multi-steps" | the loader asserts `n_layer_nextn == 1`; recursion over one block is a runtime decision, and the two candidate forms differ in whether the second step reads the trunk hidden state again or the draft block's own output | A-19 |
| A-20 | whether `STOP_IDS` participates in drafting at all | §09.5 rules out ending a request on a drafted stop id; whether the drafter may *propose* one is undecided | A-20 |
