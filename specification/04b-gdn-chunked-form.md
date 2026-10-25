# 04b — GDN chunked form (the prefill scan)

The chunked rewrite of `specification/04-gdn-linear-attention.md` §04.4–§04.6, which is what the
reference runs when more than one token is being scanned. Everything about the block that is *not* the
scan — projections §04.2, the short conv §04.3, the decay and write strength §04.5, the gated norm
§04.7, `out_proj` §04.8, head order §04.9, state inventory §04.10 — lives in file 04 and is not
restated here.

`[REF]` quotes are from the checkpoint's own reference implementation, function
`torch_chunk_gated_delta_rule` (the fallback the checkpoint ships with, used whenever a fused
kernel is not installed).

## §04b.1 Which form runs when

| situation | conv path | scan path |
|---|---|---|
| prefill (no cached state, `T > 1`) | causal conv over the whole window, keep the last `T` columns | **chunked** (§this file) |
| decode with cache, `T == 1` | conv update against the stored window | **recurrent** (file 04 §04.6) |
| first step after a prefill, `T == 1` and a state exists | conv update | recurrent |
| `T > 1` with a state present (chunked continuation, beam/rollback) | conv over the window | chunked, with `initial_state` supplied |

The reference dispatches on `use_precomputed_states and seq_len == 1` `[REF:
Qwen3_5MoeGatedDeltaNet.forward]`. This is a spec fact because the two forms must agree: a request
whose prefill used the chunked form and whose decode uses the recurrent form continues from a state
that the recurrent form never computed. `specification/06-prefill-pass.md` and
`specification/07-decode-step.md` inherit the table; the equality itself is §04b.6.

## §04b.2 Setup, chunking, and padding

Inputs are `q, k, v, beta, g` as file 04 §04.4 leaves them, i.e. cast to F32 and transposed to
`[B, heads, T, ...]`, with q already l2-normalised and scaled by `1 / sqrt(d_k)` and k already
l2-normalised. `h_v` heads throughout, `q` and `k` already expanded from `h_k` by the grouped repeat.

| step | rule |
|---|---|
| chunk width | `C = 64` — the function's default, not a config field `[REF]`. Registering it here is what makes "prefill was run at width 64" a checkable claim rather than a coincidence |
| padding | pad `T` up to a multiple of `C`. `q`, `k`, `v` are zero-padded; `beta` and `g` are **zero**-padded, so a pad column writes nothing (`beta = 0`) and does not decay the state (`exp(0) = 1`) `[REF: F.pad(x, (0, pad_size))]` |
| chunk count | `ceil(T / C)`; padding exists to make it exact and is stripped from the output |
| per-head work | every matrix below is per (batch, V head); `S` is `[d_k, d_v]` as in file 04 §04.6 |

## §04b.3 The intra-chunk quantities

For one chunk, with `t`, `s` indexing positions inside the chunk and all products over the head's
feature axis:

```
cum_decay[t]        = sum_{u <= t} g[u]                       cumulative log-decay within the chunk
pairwise[t, s]      = exp(cum_decay[t] - cum_decay[s])        for s <= t,   0 for s > t
k_beta[k, t]        = k[k, t] * beta[t]                       the "learning rate" is applied to K
v_beta[v, t]        = v[v, t] * beta[t]
ut_system[t, s]     = (k_beta[t] . k[s]) * pairwise[t, s]     strictly-lower part matters; diag is left as-is
intra_attn[t, s]    = (q[t] . k[s]) * pairwise[t, s]
dec_k_beta[k, t]    = k_beta[k, t] * exp(cum_decay[t])
```

`pairwise` is built by subtracting the two broadcast cumulative vectors and then masking the strictly
upper triangle to `-inf` **before** the `exp`, so the above-diagonal entries come out exactly `0`
rather than as the reciprocal of a possibly huge number `[REF: masked_fill(strictly_upper_mask, -inf)`
then `.exp()`]. Any implementation that masks after `exp`, or that computes `exp` of a positive
difference, is not this form.

## §04b.4 The UT solve, and the two legal code paths

```
new_values     = solve_triangular(ut_system, v_beta,      upper=False, unitriangular=True)
k_cumdecay     = solve_triangular(ut_system, dec_k_beta,  upper=False, unitriangular=True)
```

`unitriangular=True` means the system solved is `I + strictly_lower(ut_system)`, i.e. the delta-rule
corrections are collapsed into a few matmuls instead of being applied one token at a time `[REF:
"Gated delta attention uses a UT transform to condense several delta rule updates into a few
matmuls"]`.

The reference carries a **second** implementation of the same solve for export targets that lack a
triangular solver: it keeps `-ut_system.tril(-1)`, forward-substitutes row by row, adds the identity,
and left-multiplies. Both branches are legal and they are **not the same number** — one is a
factorisation, the other a row-ordered accumulation. This spec pins the first as authoritative for
`speccheck/` and registers the divergence as `specification/ASSUMPTIONS.md` A-16, with the note that a
gate comparing engine output to a dump must say which branch produced the dump.

Then, once rather than per chunk:

```
q[t]        *= exp(cum_decay[t])                          decay from chunk start to t
k[t]        *= exp(cum_decay[C-1] - cum_decay[t])         decay from t to chunk end
chunk_decay  = exp(cum_decay[C-1])                        the whole-chunk decay, applied once per chunk
```

## §04b.5 The sequential scan over chunks

The only genuinely sequential loop; chunks are independent inside, dependent across.

```
S = initial_state, or zeros [d_k, d_v] per head
for chunk i in order:
    v_new          = new_values[i]  - k_cumdecay[i] @ S          # subtract what S already predicts
    out[i]         = q[i] @ S  +  intra_attn[i] @ v_new          # inter + intra
    S              = S * chunk_decay[i]  +  k[i]^T @ v_new       # decay once, then write
```

| correspondence with the recurrent form | chunked quantity |
|---|---|
| `delta = (v - kv_mem) * beta`, solved jointly | `v_new = new_values - k_cumdecay @ S` |
| `out[t] = sum_k S[k, t] q[k]` after the write | `out[i] = q[i] @ S + intra_attn[i] @ v_new` |
| `S = exp(g) S + k ⊗ delta` per token | `S = chunk_decay * S + k^T @ v_new` per chunk |

The final state is returned only when requested, and the output's padding is dropped before the result
is cast back to the activation dtype and reshaped to `[B, T, h_v, d_v]` — from there file 04 §04.7
takes over.

## §04b.6 The two forms must be shown equal, and by what

`S_t` from §04b.5 after chunk `i` must equal `S_t` from file 04 §04.6 after the same number of tokens,
and so must the outputs. That equivalence is the gate, not an assumption — but the *metric* has
rules, and they come from `pitfalls/30-numerics.md`:

| comparison | metric | why |
|---|---|---|
| state `S` after `T` tokens, chunked vs recurrent | cos + mean-rel over the whole `[h_v, d_k, d_v]` block, **plus** a per-head maximum | a max-ULP bound cannot see a whole-tensor 1-ULP error, and `ndiff` on a chained dot is 29–47 % by itself and therefore measures nothing |
| `out` before the gated norm | cos + mean-rel, plus decision-level agreement after §04.7 + `out_proj` | the long-dot case needs a cancellation band sized by a floor probe, not a taste band |
| the prefill/decode handoff | recurrent decode of the tokens that the chunked prefill just consumed must reproduce the state to the same band | this is the only check that catches a padding or `chunk_decay` error that is invisible within one chunk |
| both forms vs a pinned HF dump | the band comes from a floor probe over the pinned rows; **no band is copied from another project or another model** | `AGENTS.md` §"Traps that do NOT transfer" |

Must-catch set for this file, derived from the geometry above rather than listed by hand, and each
member must fire on exactly the tensors named and on nothing else: a wrong `chunk_decay` (off by one
position in `cum_decay`), a missing `-inf` mask in `pairwise`, the unitriangular diagonal dropped,
`k_beta` applied to `v` instead of `k`, `q` decayed twice, and the state stored transposed.

## §04b.7 What the chunk width is and is not

`C = 64` changes the association of the sums, so it changes the last bits of every output; it does not
change the mathematics. It is therefore **not** a tuning knob this spec can hand to an engine without a
measurement: changing `C` is a numerics change that owes a floor probe and a still-green must-catch
set, exactly as a 16/32-bit format change would (`pitfalls/30-numerics.md` §"Format changes"). Two
widths in the same run — e.g. prefill at 64 and a state-comparison harness at 32 — is how a
self-consistent-but-wrong state stays hidden, so a dump's manifest must record the width that produced
it (`specification/10-reference-dumps-and-gates.md` §10).
