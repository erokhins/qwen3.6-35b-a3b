# 06 — The prefill pass

The whole-request forward pass as an ordered op list: `T` tokens in, one next-token distribution out,
and a fully populated state bundle for the decode loop that follows. Per-op math lives in the files
named at each step; this file owns only the **order**, the **shapes at the boundaries**, and **what
state is written and when**.

## §06.1 Inputs and the state they imply

| input | shape | note |
|---|---|---|
| `input_ids` | `[B, T]` | §02.1 |
| `position_ids` | `[B, T]` or absent → `arange(T)` (with `past_seen_tokens = 0` for a fresh request) | §02.3 |
| `attention_mask` | `[B, T]` or absent | only `B > 1` makes it non-trivial |
| prior state | none for a fresh request; a continuation starts from `specification/08-state-management.md`'s bundle | |

Before layer 0, the state bundle for this request is:

| state | count | shape per unit | dtype | total |
|---|---|---|---|---|
| GDN recurrent `S` | 30 (one per linear layer) | `[h_v, d_k, d_v] = [32, 128, 128]` | F32 | 60.00 MiB |
| GDN conv window | 30 | `[conv_dim, k_conv] = [8192, 4]` (or `k_conv - 1 = 3`, A-09) | activation dtype | 1.875 / 1.406 MiB |
| KV | 10 (one per full-attention layer) | `[B, n_kv, T, d_head]` per layer, both K and V | activation dtype | 2048 B per layer per token |
| draft block KV (only when an MTP step runs) | 1 | same | activation dtype | 2048 B per token |

All zero-initialised on a fresh request (`specification/08-state-management.md` §08.1). Numbers from
`python3 specification/derive.py --state`.

## §06.2 The ordered op list

```
 1  x = embed_tokens[input_ids]                                  # [B, T, 2048]           §02.2
 2  build the rope tables cos, sin for positions 0..T-1          # [B, T, 64] F32 -> cast  §05.4
 3  for N = 0 .. 39:
 4      h_in  = RMSNormA(x, w_in[N])                                                      §03.1
 5      if layer_types[N] == linear_attention:                                            §04
 6          mixed_qkv = in_proj_qkv(h_in)                        # [B,T,8192]             §04.2
 7          z        = reshape(in_proj_z(h_in), [B,T,32,128])                             §04.2
 8          a, b     = in_proj_a(h_in), in_proj_b(h_in)          # [B,T,32] each          §04.2
 9          mixed_qkv = causal_conv_over_window(mixed_qkv, conv1d.weight, window)
 10         write the conv window for this layer                   <- STATE WRITE 1        §04.3
 11         q,k,v    = split(mixed_qkv, [2048, 2048, 4096]) -> [B,T,h,128]                §04.2
 12         beta     = sigmoid(b);  g = ssm_a * softplus(a + dt_bias)                     §04.5
 13         q,k      = l2norm(q), l2norm(k);  q = q / sqrt(128);  expand k,q to 32 heads  §04.4
 14         out, S   = chunked_delta_scan(q,k,v,beta,g, initial_state=0, C=64)             §04b
 15         write S for this layer                             <- STATE WRITE 2            §04.6
 16         out      = RMSNormGated(out, z, ssm_norm)                                      §04.7
 17         m        = out_proj(out)                                                        §04.8
 18     else:                                                                             §05
 19         qg       = q_proj(h_in)                                   # [B,T,8192]         §05.1
 20         q, gate  = per-head chunk of qg                            # [B,T,16,256] each  §05.1
 21         q        = q_norm(q); k = k_norm(k_proj(h_in)); v = v_proj(h_in)               §05.2
 22         q, k     = rope(q, k, cos, sin)                                                 §05.4
 23         append k, v to this layer's KV                     <- STATE WRITE 3             §05.7
 24         attn     = causal_attention(q, KV, scale=1/16)                                  §05.3
 25         m        = o_proj(attn * sigmoid(gate))                                         §05.5
 26     x1   = x + m                                                                        §03.2
 27     h_mlp = RMSNormA(x1, w_post[N])                                                     §03.1
 28     (w, ids) = router(h_mlp)            # softmax F32 over 256, top-8, renorm, cast     §11.2
 29     routed = grouped_experts(h_mlp, ids, w)   # one grouped launch per layer, §11.4-5
 30     shared = silu_mlp(h_mlp) * sigmoid(shared_expert_gate(h_mlp))                       §11.6
 31     x      = x1 + routed + shared                                                       §03.2
 32 x_final = RMSNormA(x, norm.weight)                                                      §03.4
 33 logits  = lm_head(x_final[crop])                                 # crop = rows scored   §03.4
```

Steps 29–31 are the MoE block; `specification/11-moe-block.md` §11.1 states the invariant that
`shared` is independent of the router and that the block does not add the residual.

## §06.3 Order facts that are load-bearing, and why

| fact | why it is a spec statement |
|---|---|
| the rope tables are built **once** per request, before layer 0 | the same `cos`/`sin` serve all 10 full-attention layers and the draft block; rebuilding per layer is a recompute, rebuilding per token is a much bigger one |
| the conv window is written from the conv **inputs**, before the split into q/k/v | writing it after the split loses the q/k channels and corrupts the next chunk's window (§04.3 consequence 3) |
| the conv write precedes the scan | the scan is what consumes the conv output; the window must hold the last `k_conv` `mixed_qkv` columns of this request when the decode loop starts |
| the KV write is after rope and before the attention | the cached K must already be rotated, or a decode step reading it applies rope twice (`specification/05-gated-full-attention.md` §05.2) |
| the recurrent state is written **after** the scan, not per chunk | intermediate chunk states are not a thing the decode loop may resume from; only the final one is state |
| the router runs on `h_mlp`, i.e. after the mixer's residual add | routing on the pre-mixer hidden state is a different model |
| the crop at step 33 happens **before** the head | the head is the largest matmul in the pass; scoring rows nobody reads costs 0.9473 GiB of reads each (`specification/03-block-glue-and-normalization.md` §03.4) |
| prefill runs the **chunked** scan even for the last chunk alone | the dispatch table is `specification/04b-gdn-chunked-form.md` §04b.1; a `T == 1` request that arrives with no state still takes the chunked path |

## §06.4 What "prefill" costs, in the units this repo owns

Bytes and counts only — this repo takes no timings, and the roofline columns live in `README.md` §
"Memory and roofline arithmetic" and `quantization/README.md` §"The policies, priced".

| quantity | value | source |
|---|---|---|
| weight bytes touched, all 40 layers, every family resident | same set as one decode step: 5.5923 GiB bf16 / 2.5493 GiB policy A | `python3 specification/derive.py --per-token` (that tool is per-token by construction; prefill reads the same weights once and amortises them over `T` tokens) |
| expert bytes per token | 48 MiB per MoE layer × 41 blocks = 1.9218 GiB at bf16 | `--per-token` |
| router work | one 256-logit softmax + top-8 **per token per MoE block** (41 per token) | §11.2 |
| dependent expert reads per token | 8 per block; 328 across the 41 blocks if issued naively, 41 grouped launches if not | §11.5 |
| GDN prefill work per linear layer | `ceil(T/64)` chunk matmuls plus a 64-step triangular solve | §04b |
| state written per request | §06.1's table; KV grows linearly with `T` | `--state` |

## §06.5 What must be pinned for this file to be checkable

`speccheck/` must reproduce, from this file plus the files it names: the layer order and the two mixer
forms, the three state-write points, the crop position, and the invariant that no routing value is
ever host-visible (`specification/11-moe-block.md` §11.7). The dumps that make that checkable, and
the bands the checks are judged with, are `specification/10-reference-dumps-and-gates.md` §10.
