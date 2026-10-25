# 04 — GDN linear attention (Gated DeltaNet + short conv), recurrent form

The token mixer of the 30 `linear_attention` layers — layers where `layer_types[N] ==
"linear_attention"`, i.e. every `N` in 0..39 with `(N+1) % 4 != 0`. The chunked (prefill) form of
§04.4–§04.6 is a separate file, `specification/04b-gdn-chunked-form.md`, because the two forms must be
shown equal by a gate and that claim deserves its own page.

`[REF]` quotes are from the checkpoint's own reference implementation, class
`Qwen3_5MoeGatedDeltaNet` and the two scan functions it calls, transcribed here so this directory is
self-contained (`pitfalls/60-docs-and-citations.md` §"The self-containment rule").

## §04.1 What the block owns

| HF name (under `layers.N.linear_attn.`) | GGUF name | HF shape | dtype (ckpt) | role |
|---|---|---|---|---|
| `in_proj_qkv.weight` | `blk.N.attn_qkv.weight` | [8192, 2048] | BF16 | q, k, v in one projection |
| `in_proj_z.weight` | `blk.N.attn_gate.weight` | [4096, 2048] | BF16 | the output gate |
| `in_proj_a.weight` | `blk.N.ssm_alpha.weight` | [32, 2048] | BF16 | discretisation input (→ `g`) |
| `in_proj_b.weight` | `blk.N.ssm_beta.weight` | [32, 2048] | BF16 | write strength (→ `beta`) |
| `conv1d.weight` | `blk.N.ssm_conv1d.weight` | [8192, 1, 4] | BF16 | depthwise short conv, no bias |
| `A_log` | `blk.N.ssm_a` | [32] | BF16 | per-V-head decay parameter (§04.5) |
| `dt_bias` | `blk.N.ssm_dt.bias` | [32] | BF16 | per-V-head discretisation bias |
| `norm.weight` | `blk.N.ssm_norm.weight` | [128] | BF16 | the gated RMSNorm's weight |
| `out_proj.weight` | `blk.N.ssm_out.weight` | [2048, 4096] | BF16 | mixes V heads into the residual |

9 tensors per linear layer, 270 over the 30 layers; all shapes witnessed in real safetensors headers
by `python3 specification/derive.py --check`. Geometry: `h_k = 16`, `h_v = 32`, `d_k = d_v = 128`,
`key_dim = 2048`, `value_dim = 4096`, `conv_dim = 8192`, `k_conv = 4`, `h_v / h_k = 2`.

## §04.2 Projections and the packed qkv layout

For the block's input `x` (`[B, T, d]`, the `input_layernorm` output — §03.3):

```
mixed_qkv = in_proj_qkv(x)                # [B, T, 8192]
z         = in_proj_z(x).view(B, T, h_v, d_v)
b         = in_proj_b(x)                  # [B, T, h_v]
a         = in_proj_a(x)                  # [B, T, h_v]
```

`mixed_qkv`'s 8192 output rows are laid out as **three contiguous blocks**, not interleaved by head
`[REF: torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)]`:

| slice of `mixed_qkv` | rows | meaning |
|---|---|---|
| `[..., 0:2048]` | `d_k * h_k` | q, as `h_k` heads of width `d_k` |
| `[..., 2048:4096]` | `d_k * h_k` | k, same head structure |
| `[..., 4096:8192]` | `d_v * h_v` | v, as `h_v` heads of width `d_v` |

Then `q, k -> [B, T, h_k, d_k]` and `v -> [B, T, h_v, d_v]`, in the activation dtype. `in_proj_a` and
`in_proj_b` are separate `[h_v, d]` matrices in this checkpoint — there is no fused `in_proj_ba`, and
there is no `in_proj_qkvz` here either (the converter has a branch for that older packing; this
checkpoint does not use it).

## §04.3 The short conv, and that SiLU is on the conv output only

`conv1d` is depthwise over the full `conv_dim = 8192` channels with `kernel_size = k_conv = 4`,
`groups = conv_dim`, `bias=False`, applied to `mixed_qkv` along time, causal: left-pad with
`k_conv - 1 = 3` zeros, convolve, keep the first `T` outputs
`[REF: causal_conv1d_fn — F.conv1d(x, weight, padding=k_conv-1, groups=conv_dim)[:, :, :seq_len]]`,
then apply the activation, which is `hidden_act = silu`.

```
conv_out[c, t] = silu( sum_{s=0..3} W[c, s] * mixed_qkv[c, t - 3 + s] )
```

Three consequences that are easy to get wrong:

1. SiLU is applied **after** the conv and to **all 8192 channels** — q, k and v alike. `z` does not go
   through the conv at all.
2. `W` is stored `[conv_dim, 1, k_conv]` in the checkpoint and squeezed to `[8192, 4]` before writing
   the GGUF, whose `ne` is `[4, 8192]` (`specification/01-weights-and-shapes.md` §01.6 item 4).
3. The conv state is a window of previous `mixed_qkv` **inputs**, not of conv outputs. `derive.py
   --state` prints both the `k_conv = 4`-column window that the reference's cache keeps and the
   `k_conv - 1 = 3`-column window that is actually sufficient; which one the engine stores is an
   engine choice, registered as `specification/ASSUMPTIONS.md` A-09 with its 2× cost.

For a single cached decode step the reference takes the update path: concatenate the stored window
with the new column, write the last `state_len` columns back, convolve with no padding, keep the last
output `[REF: causal_conv1d_update]`. That is the same arithmetic as the equation above.

## §04.4 Entering the scan, and the head expansion

```
beta = sigmoid(b)                                   # [B, T, h_v]
g    = -exp(A_log.float()) * softplus(a.float() + dt_bias)     # [B, T, h_v]   (§04.5)
if h_v / h_k > 1:  q, k = repeat_interleave(q, 2, dim=2), repeat_interleave(k, 2, dim=2)
```

then, all in the `mamba_ssm_dtype` (F32) `[REF: torch_recurrent_gated_delta_rule]`:

```
q, k, v, beta, g : cast to F32, transposed to [B, heads, T, ...]
q = q * rsqrt( sum(q*q, dim=-1) + 1e-6 )            # l2norm, eps 1e-6, per head vector
k = k * rsqrt( sum(k*k, dim=-1) + 1e-6 )
q = q / sqrt(d_k)                                   # always, independent of any attention scale
```

| fact | value | why it matters |
|---|---|---|
| l2norm eps | `1e-6`, added **inside** the rsqrt of a sum of squares (not a mean) | it is not the RMSNorm eps; the two happen to share the value 1e-6 |
| q scaling | `1 / sqrt(d_k) = 1 / sqrt(128)` applied to q, once, before the scan | there is no separate attention-logit scale in a delta-rule block |
| l2norm scope | q and k only; `v`, `beta`, `g` are not normalised | |
| head expansion | `h_v / h_k = 2`, `repeat_interleave` ⇒ **grouped**: V head `j` uses K head `j // 2` | the GGUF stores V heads tiled, where the pairing is `j % 16`; §04.9 |
| scan dtype | F32 for the state, the decay, the norms and the output of the scan; the result is cast back to the activation dtype on return | `mamba_ssm_dtype: float32` is the scan dtype, not the storage dtype |

## §04.5 The decay and the write strength

```
beta_t[h] = sigmoid( in_proj_b(x_t)[h] )                       in the activation dtype
g_t[h]    = -exp(A_log[h]) * softplus( in_proj_a(x_t)[h] + dt_bias[h] )
softplus(u) = log(1 + exp(u))
```

| fact | detail |
|---|---|
| `A_log[h]` is a **log**, not a decay | the value witness: layer 0's `A_log` spans −3.98…4.28 (`derive.py --values`), so it must be `log|A|` and the decay is `-exp(A_log)` |
| `dt_bias[h]` spans both signs | layer 0: −5.28…15.56, so it is trained, not the `ones` init ⇒ `softplus` must be evaluated on the **sum** `a + dt_bias`, not on `a` |
| `g_t[h] <= 0` always | it is a log-decay; the chunked form asserts it. The per-step multiplier on the state is `exp(g_t[h])` in `(0, 1]` |
| both casts are explicit | `A_log.float()` and `a.float()` — computing `softplus` in BF16 and then upcasting is a different number |
| what the file stores | `blk.N.ssm_a` is already `-exp(A_log)`, computed by the producer at conversion time (`specification/01-weights-and-shapes.md` §01.6 item 6). It is a **recomputation**, so its 1-ulp history is governed by `DECISIONS.md` D-14 and `specification/ASSUMPTIONS.md` A-13 — never call it bit-exact with the checkpoint |
| `A_log`/`dt_bias` are per **V** head | length `h_v = 32`, and in the GGUF they are in the tiled order (§04.9) |

## §04.6 The recurrent update, in the reference's exact order

State per layer: `S` of shape `[h_v, d_k, d_v]`, F32, indexed `S[k, v]`. Initial state is zero. Per
token `t`, per V head `h` (all of it in F32) `[REF: torch_recurrent_gated_delta_rule]`:

```
1  S      = S * exp(g_t[h])                                  # decay first
2  kv_mem[v] = sum_k S[k, v] * k_t[k]                        # what the state already predicts
3  delta[v] = (v_t[v] - kv_mem[v]) * beta_t[h]               # the correction, scaled by beta
4  S[k, v] = S[k, v] + k_t[k] * delta[v]                     # delta-rule write
5  out_t[v] = sum_k S[k, v] * q_t[k]                         # read AFTER the write
```

The order of steps 1–5 is part of the definition. Two things in it are load-bearing:

- **the output reads the post-update state**, so a token's own key/value are visible to its own output;
- decay precedes the write, so step 4's contribution is undecayed for this step and decays from the
  next one onward.

Equivalence to the outer-product form `S_t = exp(g_t) * S_{t-1} * (I - beta_t k_t k_t^T) + beta_t
v_t k_t^T` — with the read `out_t = S_t^T q_t` — is what `specification/04b-gdn-chunked-form.md`
proves and what a gate must check; the state is stored as `[d_k, d_v]` per head, **not** `[d_v, d_k]`,
and reading it transposed raises nothing.

## §04.7 The gated output norm — a different RMSNorm form from §03.1

Per token and V head, over the `d_v = 128` axis, with weight `w = ssm_norm.weight` (`[128]`) and gate
`z` from §04.2 (reshaped `[B, T, h_v, d_v]`) `[REF: Qwen3_5MoeRMSNormGated.forward]`:

```
dtype0    = out.dtype
var       = mean( out.float()^2, dim=-1 )
out       = out.float() * rsqrt( var + eps_rms )       # Norm before gate
out       = w * cast_to_dtype0(out)                    # weight applied in the INPUT dtype
out       = out * silu( z.float() )                    # gate in F32
return    cast_to_dtype0(out)
```

| fact | detail |
|---|---|
| variance is over `d_v = 128` | one variance per (token, V head), never over the 4096-wide flat vector |
| the weight is **plain** `w * x_hat`, not `(1 + w) * x_hat` | the value witness: layer 0's `linear_attn.norm.weight` sits at 0.52…0.96 (mean 0.884), i.e. stored near 1 (`derive.py --values`), and the converter's `+1` explicitly excludes `linear_attn.norm.weight`. This is the opposite of §03.1's block norms, which store `w` near 0 and are read as `1 + w` |
| `eps` is `rms_norm_eps` = 1e-6 | the module is constructed with `layer_norm_epsilon = config.rms_norm_eps` |
| `z` is not normalised | it is a gate: `silu` then multiply |
| the intermediate cast back to the input dtype | written as the reference writes it. `w * x_hat` in F32 without the intermediate cast is a different product tree (`pitfalls/30-numerics.md` §"A hoisted scalar constant is a different number") |

## §04.8 Output projection and exit

```
y = out_proj( normed.reshape(B, T, h_v * d_v) )       # [B, T, 4096] -> [B, T, 2048]
```

`out_proj` is `[d, value_dim] = [2048, 4096]`, no bias, and its columns follow the same V-head order
as everything else (§04.9). `y` is what the decoder layer adds to the residual (§03.3). Nothing else
leaves the block: the state and the conv window are state, not outputs.

## §04.9 Head order: HF grouped, GGUF tiled, and what the pairing becomes

The reference expands K heads with `repeat_interleave(2, dim=2)`, which is **grouped** alignment: V
head `j` pairs with K head `j // (h_v / h_k) = j // 2`. HF's stored weights use that same grouped
order for the V rows of `in_proj_qkv`, all of `in_proj_z`, the rows of `in_proj_a`/`in_proj_b`,
`A_log`, `dt_bias`, the V channels of `conv1d` and the columns of `out_proj`.

The published file does **not**: its converter permutes those V axes from grouped to tiled, so GGUF V
head `j` holds HF V head `(j % h_k) * (h_v / h_k) + (j // h_k)`, i.e. with 16 K and 32 V heads,
`j -> (j % 16) * 2 + j // 16`. The point of the permutation is that in tiled order the K↔V pairing
becomes `j % h_k`, so a broadcast replaces an interleaved repeat. Both facts are in
`specification/01-weights-and-shapes.md` §01.6 item 1 with the tensor-by-tensor list; the reason it is
repeated here is that **the pairing and the head order must come from the same assumption** — mixing
GGUF's tiled V order with the grouped pairing reads the wrong K head for half the V heads and raises
nothing. Registered as `specification/ASSUMPTIONS.md` A-12, with the falsifying test.

## §04.10 State inventory

| state | shape per layer | dtype | per request (30 layers) |
|---|---|---|---|
| recurrent `S` | `[h_v, d_k, d_v] = [32, 128, 128]` | F32 | 62,914,560 B = 60.00 MiB |
| conv window | `[conv_dim, k_conv] = [8192, 4]` (or `k_conv-1 = 3`) | the activation dtype in the reference's cache | 3.750 MiB at 4 taps, 2.812 MiB at 3 in F32; 1.875 / 1.406 MiB in BF16 |

`python3 specification/derive.py --state` prints all four conv rows and the per-lane totals; the dtype
question is A-09. Zeroing rule: both states start at zero for a new sequence, and a recurrent state is
never reused across sequences (§08.2).
