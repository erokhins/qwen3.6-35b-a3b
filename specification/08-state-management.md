# 08 — State management

Three kinds of state, their exact shapes and bytes, who writes them, who reads them, and what a
request's total budget is. Capacity is a hard constraint in this project (`AGENTS.md` §SCOPE: 65.72
GiB of bf16 weights against a box-wide pool of ≈55.5–56.5 GiB, ≈32 GiB for one process while halogen
is resident, `devices/measured-facts.md` §"Residency above the pool"), so state is sized here rather
than discovered later.

Every number in this file is `python3 specification/derive.py --state` output, which derives it from
`checkpoints/Qwen3.6-35B-A3B.config.json` geometry.

## §08.1 The three state kinds

| kind | units | shape per unit | dtype | bytes | written by | read by |
|---|---|---|---|---|---|---|
| GDN recurrent `S` | 30 (one per `linear_attention` layer) | `[h_v, d_k, d_v] = [32, 128, 128]` | **F32** (`mamba_ssm_dtype`) | 2,097,152 per layer → 62,914,560 = **60.00 MiB** | the scan (§04.6 / §04b.5) | the next step's scan |
| GDN conv window | 30 | `[conv_dim, w] = [8192, w]`, `w = k_conv = 4` in the reference's cache, `w = k_conv - 1 = 3` suffices | activation dtype in the reference; F32 is the other legal choice (A-09) | 4 taps: 3.750 MiB F32 / 1.875 MiB BF16 · 3 taps: 2.812 MiB F32 / **1.406 MiB BF16** | the conv update (§04.3) | the next step's conv |
| KV | 10 trunk full-attention layers (+1 draft layer when MTP runs) | `[B, n_kv, T, d_head] = [B, 2, T, 256]`, K and V | activation dtype | **2048 B per layer per token** → 20,480 B = 20.0 KiB/token trunk, 22.0 KiB/token with the draft layer | rope'd K and V before the attention (§05.2) | every subsequent step's attention |

| rule | statement |
|---|---|
| zero-init | all three start at zero for a fresh request; the conv window is **not** "undefined for the first `k_conv - 1` positions", it is zeros, which is what the causal left-padding means (§04.3) |
| never shared across requests | a recurrent state carries the whole prefix; reusing one across requests is a context leak, not an optimisation |
| state is not output | the mixer's only output is `m` (§03.3); `S` and the window leave the block only into the bundle |
| dtype of `S` is not negotiable | the scan is defined in F32 (§04.4). Storing `S` in BF16 halves the biggest state and is a numerics change owed a floor probe, not a resident-size tuning knob |
| dtype of the conv window is | the reference stores it in the activation dtype (`zeros_like` of the conv input, which is BF16); the F32 variant is 2× and is a declared engine choice, registered as A-09 with its cost |
| KV dtype follows the artifact | BF16 in a bf16 build; under policy A the *weights* are Q8_0 while KV stays an activation-dtype buffer — KV quantization is not covered by any decided width |

## §08.2 Per-request budget

`python3 specification/derive.py --state`, conv at the sufficient minimal window (`k_conv - 1 = 3`,
F32), KV over the 10 trunk layers:

| lanes (context tokens) | KV | recurrent | conv window | total |
|---|---|---|---|---|
| 1 | 0.02 MiB | 60.00 MiB | 2.812 MiB | **62.83 MiB** |
| 128 | 2.50 MiB | 60.00 MiB | 2.812 MiB | **65.31 MiB** |
| 1,024 | 20.00 MiB | 60.00 MiB | 2.812 MiB | **82.81 MiB** |
| 4,096 | 80.00 MiB | 60.00 MiB | 2.812 MiB | **142.81 MiB** |
| 32,768 | 640.00 MiB | 60.00 MiB | 2.812 MiB | **702.81 MiB** |

| fact | value |
|---|---|
| draft-block KV | +2048 B per token; a 4096-lane request with an active MTP block is **150.81 MiB** |
| requests per 1 GiB of state pool at 4096 lanes | **7.2** with the 3-tap window, **7.1** with the reference's 4-tap window |
| fixed cost of a request | 62.83 MiB regardless of length — the recurrent state is 95 % of a short request's budget |
| where the state pool sits relative to the weights | policy A's weights are 19.59 GiB resident (`quantization/README.md` §"The policies, priced"); a 1 GiB state pool is 5 % of that, and the whole box has ≈32 GiB for one process while halogen is resident |
| this model vs the 0.8B's state | 60 MiB recurrent (30 layers × 32 V heads × 128×128 F32) and 20 KiB/token KV — the sibling project's 18.0 MiB and 12 KiB/token are facts about a different checkpoint and are **not** a baseline to reuse (`AGENTS.md` §"Traps that do NOT transfer") |

`README.md` §"Memory and roofline arithmetic" states "GDN 30 layers × 32 V heads × 128×128 F32 ≈ 60
MiB" (which this table reproduces) and "KV is 10 KiB/token over 10 full-attn layers" (which it does
not: 2 × 2 × 256 × 2 B × 10 = 20,480 B). Both numbers are kept where they are and the delta is
registered as A-08 with the arithmetic that adjudicates it.

## §08.3 Lifecycle

| event | recurrent `S` | conv window | KV |
|---|---|---|---|
| request created | zeros | zeros | empty |
| prefill (§06) | written once per linear layer, after the chunked scan | written once per linear layer, from the conv **inputs** | `T` columns appended per full-attention layer |
| decode step (§07) | read, advanced by one token, written | read, shifted by one column, written | 1 column appended per full-attention layer |
| token rejected by verification (§09.5) | must be rewound to the state before the rejected token | same | its column dropped |
| request finished | released | released | released |
| preemption / swap | all three travel together, per layer, or the request cannot be resumed | | |

| rule | why it is a statement and not an implementation detail |
|---|---|
| per-layer identity | state is indexed by *layer index*, and the 30 linear layers are not contiguous (3, 7, 11, … are the full-attention ones). A bundle keyed by "recurrent slot 0..29" must record the mapping to `layer_types` or a swap silently mis-homes a layer's state, which raises nothing and reads fluently |
| KV is the only unbounded one | `S` and the window are `O(1)` per request; KV is `O(T)` and is what a 32k plan spends (`PLAN.md` §Step 2) |
| the draft block's KV is separate | the MTP layer has its own KV slot and does not write the trunk's (`specification/09-mtp-draft-layer.md` §09.4) |
| the conv window and `S` are coupled | both advance exactly one token per accepted token; a rollback that rewinds one and not the other produces a state that is self-consistent per tensor and wrong as a pair — the two-self-consistent-layouts failure shape one level up (`pitfalls/40-gates-and-admissions.md`) |
| a resumed continuation appends, it does not restart | positions continue from `past_seen_tokens` (§02.3) and the rope tables are rebuilt for the new positions only |

## §08.4 Rollback, and who may rewind state

The plain decode loop has no partial steps: one token is generated, appended and accepted together
(§07.5), so nothing to rewind. Rollback exists for exactly one consumer — speculative decoding, where
the draft block proposes `k` tokens that the target model then accepts or rejects
(`specification/09-mtp-draft-layer.md` §09.5). The rule that makes that safe:

| rule | statement |
|---|---|
| accepted prefix only | trunk state advances by the number of **accepted** tokens and no more |
| all three kinds move together | per accepted token: `S`, the conv window and the KV columns for every layer, or none of them |
| draft state is cheap to discard | the draft block's KV for rejected positions is truncated; the draft block has no recurrent state at all (it is a full-attention layer, `specification/01-weights-and-shapes.md` §01.7) |
| no rollback across requests | a finished request's bundle is released, not parked |
| verification recomputes from a checkpoint, not from a guess | a rejected branch must resume from a state that was actually produced by the target model at that prefix, not from a state extrapolated backwards |

## §08.5 What `speccheck/` must pin here

For a pinned prompt at a pinned lane count: `S` and the conv window of at least one linear layer
**before and after** one decode step (a state that does not change is a bug no output-level metric can
see), the appended KV column of one full-attention layer, and the lane count that produced them. The
conv window is the tensor most likely to be pinned wrong: it is the conv **input** window, not the conv
**output**, and pinning the output reproduces a real-looking tensor that is the wrong quantity
(`pitfalls/30-numerics.md` §"A hook that reads `output` can pin the answer under the name of an
intermediate").
