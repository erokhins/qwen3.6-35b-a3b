# 00 — Notation and hyperparameters

Every constant the forward pass uses, the exact config field it is read from, and — for anything not
stored — the arithmetic that produces it. `[CFG]` = read directly out of
`checkpoints/Qwen3.6-35B-A3B.config.json`; `[DERIVED]` = computed here from `[CFG]` fields, with the
formula shown; `[REF]` = not in the config, taken from the checkpoint's own reference implementation
and quoted where it is used; `[ASSUMPTION A-xx]` = registered in `specification/ASSUMPTIONS.md`.

Model: `Qwen/Qwen3.6-35B-A3B`, `architectures: ["Qwen3_5MoeForConditionalGeneration"]`,
`model_type: qwen3_5_moe`. All language-tower fields live under `text_config`
(`model_type: qwen3_5_moe_text`); the field column below names them as
`text_config.<field>` and top-level ones as `<field>`.

## §00.1 Symbol conventions

| symbol | means |
|---|---|
| `d` | hidden size of the residual stream |
| `H` | same as `d`; used when a matrix's input axis is meant |
| `N` | decoder-layer index, 0..39 |
| `n_q`, `n_kv` | full-attention query / key-value head counts |
| `d_head` | full-attention head width (Q, K and V all equal it here) |
| `h_k`, `h_v` | GDN key / value head counts |
| `d_k`, `d_v` | GDN key / value head widths |
| `E`, `K` | routed-expert count, experts selected per token |
| `I_e`, `I_s` | routed-expert and shared-expert intermediate widths |
| `V` | vocabulary size |
| `R` | rotary width actually rotated inside one head (§00.3) |
| `x_hat` | `x * rsqrt(mean(x^2, axis=-1) + eps)`, the RMSNorm core (§03.1) |
| `S_t` | GDN recurrent state after token t, `[h_v, d_k, d_v]` per head |

Shapes are written row-major, slowest axis first (PyTorch/HF order), throughout. GGUF's reversed
`ne` convention appears only in `specification/01-weights-and-shapes.md`, labelled as such.
`i:j` slices are half-open. Element indices are 0-based. "F32"/"BF16" name storage or compute
precision, never a library type.

## §00.2 Hyperparameters read from the config

| symbol | value | config field | tag |
|---|---|---|---|
| `d` | 2048 | `text_config.hidden_size` | [CFG] |
| `L` | 40 | `text_config.num_hidden_layers` | [CFG] |
| layer schedule | 30 `linear_attention`, 10 `full_attention` | `text_config.layer_types` (40 entries) + `text_config.full_attention_interval: 4` | [CFG] |
| `d_head` | 256 | `text_config.head_dim` | [CFG] |
| `n_q` | 16 | `text_config.num_attention_heads` | [CFG] |
| `n_kv` | 2 | `text_config.num_key_value_heads` | [CFG] |
| output gate on | true | `text_config.attn_output_gate` | [CFG] |
| attention biases | none | `text_config.attention_bias: false` | [CFG] |
| `theta` | 10,000,000 | `text_config.rope_parameters.rope_theta` | [CFG] |
| partial rotary factor | 0.25 | `text_config.rope_parameters.partial_rotary_factor` (also duplicated at `text_config.partial_rotary_factor`) | [CFG] |
| rope type | `default` | `text_config.rope_parameters.rope_type` | [CFG] |
| mrope interleaved | true | `text_config.rope_parameters.mrope_interleaved` | [CFG] |
| mrope sections | `[11, 11, 10]` | `text_config.rope_parameters.mrope_section` | [CFG] |
| `h_k` | 16 | `text_config.linear_num_key_heads` | [CFG] |
| `h_v` | 32 | `text_config.linear_num_value_heads` | [CFG] |
| `d_k` | 128 | `text_config.linear_key_head_dim` | [CFG] |
| `d_v` | 128 | `text_config.linear_value_head_dim` | [CFG] |
| `k_conv` | 4 | `text_config.linear_conv_kernel_dim` | [CFG] |
| GDN scan dtype | float32 | `text_config.mamba_ssm_dtype` | [CFG] |
| `E` | 256 | `text_config.num_experts` | [CFG] |
| `K` | 8 | `text_config.num_experts_per_tok` | [CFG] |
| `I_e` | 512 | `text_config.moe_intermediate_size` | [CFG] |
| `I_s` | 512 | `text_config.shared_expert_intermediate_size` | [CFG] |
| activation | `silu` (`x / (1 + exp(-x))`) | `text_config.hidden_act` | [CFG] |
| `eps_rms` | 1e-6 | `text_config.rms_norm_eps` | [CFG] |
| `V` | 248320 | `text_config.vocab_size` | [CFG] |
| untied head | `lm_head.weight` exists separately | `text_config.tie_word_embeddings: false` and top-level `tie_word_embeddings: false` | [CFG] |
| MTP blocks | 1 | `text_config.mtp_num_hidden_layers` | [CFG] |
| dedicated MTP embeddings | false → the drafter reuses `embed_tokens` | `text_config.mtp_use_dedicated_embeddings` | [CFG] |
| native context | 262144 | `text_config.max_position_embeddings` | [CFG] |
| storage dtype | bfloat16 | `text_config.dtype` | [CFG] |
| eos / bos (model config) | 248044 / 248044 | `text_config.eos_token_id`, `text_config.bos_token_id` | [CFG] |
| eos (tokenizer metadata) | 248046 | GGUF `tokenizer.ggml.eos_token_id`; `bos_token_id` 248044, `padding_token_id` 248055, `add_bos_token` false | [WITNESS] |
| GDN l2norm eps | 1e-6 | not a config field — hard-coded in the reference's `l2norm(x, dim=-1, eps=1e-6)` | [REF] |
| chunk width (prefill GDN) | 64 | not a config field — default `chunk_size` of the reference's chunked scan | [REF] |
| attention scale | `d_head ** -0.5` = 1/16 | not a config field — `Qwen3_5MoeAttention.__init__` sets `self.scaling = self.head_dim**-0.5`; `attention_dropout: 0.0` so inference has no dropout | [REF] |

**The stop rule is not settled by this table.** `text_config.eos_token_id` is 248044 while the
published artifact's own tokenizer metadata says `tokenizer.ggml.eos_token_id = 248046` (and the
0.8B project's rule was "248046 stops"). Both witnesses are recorded; neither is promoted to "the"
stop set, because the file that would adjudicate it — this checkpoint's `generation_config.json` — is
not on disk. `specification/ASSUMPTIONS.md` A-04 states the risk and the falsifying test. Token ids
are named by number only, never printed literally (`AGENTS.md` §"Never print chat-template special
tokens").

## §00.3 Derived geometry — what is computed, not stored

Every row is arithmetic over §00.2; `python3 specification/derive.py --check` asserts 30 of these
against real safetensors headers and exits non-zero on drift.

| quantity | formula | value |
|---|---|---|
| full-attention Q rows | `2 * n_q * d_head` (the factor 2 is the output gate) | 8192 |
| full-attention K/V rows | `n_kv * d_head` | 512 |
| full-attention O rows | `n_q * d_head` | 4096 |
| GQA group size | `n_q / n_kv` | 8 |
| rotary width `R` | `int(d_head * partial_rotary_factor)` = `int(256 * 0.25)` | 64 |
| rotary frequency pairs | `R / 2` (= entries of `inv_freq`, `arange(0, R, 2) / R`) | 32 |
| mrope column assignment | column `j` of the 32 takes axis T unless `j % 3 == 1 and j < 33` (axis H) or `j % 3 == 2 and j < 30` (axis W) — `11 + 11 + 10 = 32` | §05.4 |
| GDN `key_dim` | `d_k * h_k` | 2048 |
| GDN `value_dim` | `d_v * h_v` | 4096 |
| GDN `conv_dim` | `2 * key_dim + value_dim` | 8192 |
| GDN V-per-K expansion | `h_v / h_k` | 2 |
| GDN state per layer | `h_v * d_k * d_v * 4 B` (F32) | 2,097,152 B |
| GDN state per request | 30 layers × 2 MiB | 62,914,560 B = 60.00 MiB |
| KV bytes per token per layer | `2 (K,V) * n_kv * d_head * 2 B` | 2048 B |
| KV per token, trunk | 10 full-attn layers × 2048 B | 20,480 B = 20.0 KiB |
| KV per token, with the draft block | 11 layers × 2048 B | 22,528 B = 22.0 KiB |
| expert matrix sizes | `gate_up [E, 2*I_e, d]`, `down [E, d, I_e]` | `[256,1024,2048]`, `[256,2048,512]` |
| one expert | `3 * I_e * d` elements × 2 B | 3,145,728 elts = 6.0 MiB BF16 |
| one layer's routed experts | `E * 3 * I_e * d * 2 B` | 1.50 GiB BF16 |
| router | `[E, d]` | `[256, 2048]` |
| shared expert | three `[I_s, d]` / `[d, I_s]` matrices + a `[1, d]` gate | §11.6 |
| logits head | `[V, d]`, untied | 508,559,360 elts = 0.9473 GiB BF16 |

`tools/gguf_header_types.py` prints the same geometry as the producer wrote it:
`qwen35moe.expert_count 256`, `expert_used_count 8`, `expert_feed_forward_length 512`,
`expert_shared_feed_forward_length 512`, `full_attention_interval 4`, `value_length 256`,
`embedding_length 2048`, `attention.layer_norm_rms_epsilon 9.999999974752427e-07`. That last value is
`1e-6` rounded to F32, i.e. the GGUF carries `eps_rms` as stored F32, not as the decimal literal —
quote it as `1e-6` and do not read a second constant into it.

## §00.4 Precision classes

These are the classes a `speccheck/` implementation must reproduce, not suggestions. Each is quoted
from the checkpoint's own reference implementation at the point of use, in the file that owns the op.

| class | what is computed in F32 regardless of storage | where it is established |
|---|---|---|
| router softmax | `softmax(logits, dtype=float32)` over all `E` logits, and the top-`K` renormalisation, before the cast back | `specification/11-moe-block.md` §11.2 |
| routing-weight multiply | the renormalised top-`K` weights are cast to the **logits' dtype** before multiplying the expert output | `specification/11-moe-block.md` §11.2 |
| RMSNorm core | `x_hat` is computed on `x.float()`; the weight multiply is F32 and the cast back is last | `specification/03-block-glue-and-normalization.md` §03.1 |
| GDN scan | the whole recurrent/chunked scan runs in the `mamba_ssm_dtype` (F32), including q/k l2norm and the `1/sqrt(d_k)` scaling | `specification/04-gdn-linear-attention.md` §04.4 |
| GDN decay | `g = -exp(A_log.float()) * softplus(a.float() + dt_bias)` — both casts are explicit | `specification/04-gdn-linear-attention.md` §04.5 |
| GDN gated norm | variance in F32, weight applied after a cast back to the input dtype, then the SiLU gate in F32 | `specification/04-gdn-linear-attention.md` §04.7 |
| attention softmax | `softmax(..., dim=-1, dtype=float32)` then cast back to the query dtype before the PV product | `specification/05-gated-full-attention.md` §05.3 |
| rope | `inv_freq`, `cos`, `sin` computed in F32, cast to the hidden dtype on return | `specification/05-gated-full-attention.md` §05.4 |
| everything else | matmuls and elementwise products in the activation dtype (BF16 in the bf16 build) | §03.3 |

`mamba_ssm_dtype: float32` is the **scan** dtype, not the storage dtype: `A_log`, `dt_bias`,
`in_proj_a/b` and `conv1d` are all BF16 in the checkpoint (`derive.py --check` asserts it), and the
published file widens the control tensors to F32. Conflating the two is `specification/ASSUMPTIONS.md`
A-06.

## §00.5 What the config says and this spec ignores

Out of scope by `DECISIONS.md` D-04 and `specification/README.md` §Scope, listed so that "not
mentioned" is never mistaken for "not present":

| field | value | why it is not used |
|---|---|---|
| `image_token_id`, `video_token_id`, `vision_start_token_id`, `vision_end_token_id` | 248056, 248057, 248053, 248054 | vision tower is out; these ids only fire on grid positions |
| `vision_config.*` | 27-block tower, 893,142,496 B | out of scope; every one of its 333 names is in the index |
| `text_config.output_router_logits` | false | no load-balancing loss in inference |
| `text_config.router_aux_loss_coef` | 0.001 | training-only, and unused while the flag above is false |
| `text_config.attention_dropout` | 0.0 | inference only |
| `text_config.initializer_range` | 0.02 | initialisation, not forward |
| `text_config.use_cache` | true | the state model is `specification/08-state-management.md`, not a switch |
| `text_config.pad_token_id` | null | no batching/padding pass in scope; the published tokenizer says 248055 |
| `transformers_version` | 4.57.1 | the installed reference used on 2026-10-26 was 5.19.0; version drift on *form* is A-02 |

Two fields that a reader may expect and that **do not exist in this checkpoint at all**:
`text_config.intermediate_size` (no dense-MLP escape hatch — every layer is sparse) and
`text_config.norm_topk_prob` (the router in this model's class renormalises unconditionally, so there
is no flag to set; see `specification/11-moe-block.md` §11.2).

## §00.6 Per-token and per-request summary

Weight bytes read per token and per-request state, both as the committed instrument prints them —
`python3 specification/derive.py --per-token` and `--state`:

| column | bf16 | policy A | published file |
|---|---|---|---|
| weight bytes / token (weights only, no KV/state) | 5.5923 GiB | 2.5493 GiB | 2.4771 GiB |

| state | bytes |
|---|---|
| GDN recurrent, 30 layers, F32 | 62,914,560 B = 60.00 MiB per request |
| GDN conv window, `k_conv-1 = 3` taps, stored in the activation dtype | 2.812 MiB per request |
| KV at 4096 lanes, trunk | 80.00 MiB; + 2.0 KiB/token of draft-block KV when MTP is active |
| one request at 4096 lanes, all three | 142.81 MiB (150.81 MiB with the draft block's KV) |

The bf16 column differs from `quantization/README.md`'s 5.194 GiB/token and the KV figures differ from
`README.md`'s "10 KiB/token" for the reasons recorded in `specification/ASSUMPTIONS.md` A-07 and A-08;
this table quotes the instrument, those documents keep their own numbers, and neither was edited from
this task.
