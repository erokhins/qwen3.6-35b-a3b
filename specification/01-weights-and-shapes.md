# 01 — Weights and shapes: the complete text-path inventory

Every row is one HF tensor name (or name template), its GGUF name, its shape in both conventions,
its stored dtype in each artifact, its count and its bytes. Nothing here is estimated: the checkpoint
side is derived from `checkpoints/Qwen3.6-35B-A3B.config.json` geometry and checked name-for-name
against `checkpoints/Qwen3.6-35B-A3B.safetensors.index.json` and against the real safetensors headers
of the four shards under `models/qwen36-src/`; the GGUF side is read out of the published file's own
tensor-info records and priced by block geometry re-derived from the pinned `ggml-common.h`, and the
whole table is accepted only after `sum(row_bytes) + data_start == file size`.

Tag meanings are in `specification/README.md` §Tag conventions. This file adds no measurement and no
timing.

## §01.1 Instruments — the commands that reproduce every number below

```sh
python3 tools/quant_scenarios.py --census                       # family GiB, index delta 1.86 %
python3 specification/derive.py --families                      # 712 names identical, delta 0 B
python3 specification/derive.py --header --filter language_model.layers.0.   # 18 tensors, real header
python3 specification/derive.py --header --filter layers.39.                 # 15 tensors, real header
python3 specification/derive.py --mtp                           # 19 tensors, 844,640,768 elts
python3 tools/gguf_width_table.py models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --check-header
python3 tools/gguf_header_types.py models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

The published-file rows in §01.4 came from a header-only pass over
`models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (733/733 records, 0 unmatched names,
`sum(row_bytes) 22123538944 + data_start 10990048 = 22134528992 = file size`). The two committed
instruments for that are `tools/gguf_width_table.py` (size identity + type census) and
`tools/gguf_header_types.py` (per-family width map); both price widths from
`engines/llama.cpp/ggml/src/ggml-common.h` struct sizes via `tools/ggml_block_geom.py`, never from a
hand-typed bpw table — the last hand-typed table put Q5_K at 5.5625 bpw where the struct is
176 B / 256 = 5.5, which overpriced this model's table by 74.00 MiB.

## §01.2 Checkpoint-side totals

`python3 specification/derive.py --families` (2026-10-26): **712 text tensors derived from config
geometry, and the name set is IDENTICAL to the 712 text names in the committed index** — nothing
invented, nothing missing. Element and byte totals follow.

| family | tensors | elements | bf16 GiB |
|---|---|---|---|
| routed experts (40 layers + MTP block) | 82 | 33,017,561,088 | 61.5000 |
| GDN projections (`in_proj_qkv/z/a/b`, `conv1d`, `out_proj`) | 180 | 1,011,548,160 | 1.8842 |
| token embeddings | 1 | 508,559,360 | 0.9473 |
| `lm_head` (untied) | 1 | 508,559,360 | 0.9473 |
| full attention q/k/v/o (10 layers) | 40 | 272,629,760 | 0.5078 |
| shared experts + shared-expert gate (41 blocks) | 164 | 129,058,816 | 0.2404 |
| MTP block attention q/k/v/o | 4 | 27,262,976 | 0.0508 |
| routers `mlp.gate` (40 + MTP) | 41 | 21,495,808 | 0.0400 |
| `mtp.fc` (the embedding\|hidden fusion) | 1 | 8,388,608 | 0.0156 |
| norms (all RMSNorm weights, incl. q/k and MTP) | 108 | 181,760 | 0.0003 |
| GDN control (`A_log`, `dt_bias`, `linear_attn.norm`) | 90 | 5,760 | 0.0000 |
| **text total** | **712** | **35,505,251,456** | **66.1337** |

Reconciliation: index `metadata.total_size` = 71,903,645,408 B = 66.9655 GiB; the vision tower
(333 tensors, every name present in the shards held) is 893,142,496 B = 0.8318 GiB;
**index − vision = 71,010,502,912 B = 66.1337 GiB = derived text × 2 B, delta 0 B (0 elements)**.
`tools/quant_scenarios.py --census` reports 65.720 GiB and a 1.86 % delta for the same population;
the difference is that its `gdn_linear_attn` group omits `linear_attn.out_proj` and its
`full_attn`/`norms_misc` groups are cut differently. `specification/ASSUMPTIONS.md` A-07 carries both
numbers and the command that adjudicates them; this file quotes the 0-B-delta census.

## §01.3 Tensor-by-tensor inventory (checkpoint side)

`n` is the number of tensors of that template in the 712. HF shape is row-major, slowest axis first.
GGUF `ne` is the file's own order (fastest axis first) — `ne` is HF's shape reversed, so a reader who
reads `[512,2048,256]` as HF `[512,2048,256]` has transposed the expert stack.
Dtype column is the **checkpoint**; §01.4/§01.5 give the file's.

| HF name (under `model.language_model.` unless stated) | GGUF name | HF shape | GGUF `ne` | ckpt dtype | n |
|---|---|---|---|---|---|
| `embed_tokens.weight` | `token_embd.weight` | [248320, 2048] | [2048, 248320] | BF16 | 1 |
| `norm.weight` | `output_norm.weight` | [2048] | [2048] | BF16 | 1 |
| `layers.N.input_layernorm.weight` | `blk.N.attn_norm.weight` | [2048] | [2048] | BF16 | 40 |
| `layers.N.post_attention_layernorm.weight` | `blk.N.post_attention_norm.weight` | [2048] | [2048] | BF16 | 40 |
| `layers.N.linear_attn.in_proj_qkv.weight` | `blk.N.attn_qkv.weight` | [8192, 2048] | [2048, 8192] | BF16 | 30 |
| `layers.N.linear_attn.in_proj_z.weight` | `blk.N.attn_gate.weight` | [4096, 2048] | [2048, 4096] | BF16 | 30 |
| `layers.N.linear_attn.in_proj_a.weight` | `blk.N.ssm_alpha.weight` | [32, 2048] | [2048, 32] | BF16 | 30 |
| `layers.N.linear_attn.in_proj_b.weight` | `blk.N.ssm_beta.weight` | [32, 2048] | [2048, 32] | BF16 | 30 |
| `layers.N.linear_attn.conv1d.weight` | `blk.N.ssm_conv1d.weight` | [8192, 1, 4] | [4, 8192] | BF16 | 30 |
| `layers.N.linear_attn.out_proj.weight` | `blk.N.ssm_out.weight` | [2048, 4096] | [4096, 2048] | BF16 | 30 |
| `layers.N.linear_attn.A_log` | `blk.N.ssm_a` (no `.weight`) | [32] | [32] | BF16 | 30 |
| `layers.N.linear_attn.dt_bias` | `blk.N.ssm_dt.bias` | [32] | [32] | BF16 | 30 |
| `layers.N.linear_attn.norm.weight` | `blk.N.ssm_norm.weight` | [128] | [128] | BF16 | 30 |
| `layers.N.self_attn.q_proj.weight` | `blk.N.attn_q.weight` | [8192, 2048] | [2048, 8192] | BF16 | 10 |
| `layers.N.self_attn.k_proj.weight` | `blk.N.attn_k.weight` | [512, 2048] | [2048, 512] | BF16 | 10 |
| `layers.N.self_attn.v_proj.weight` | `blk.N.attn_v.weight` | [512, 2048] | [2048, 512] | BF16 | 10 |
| `layers.N.self_attn.o_proj.weight` | `blk.N.attn_output.weight` | [2048, 4096] | [4096, 2048] | BF16 | 10 |
| `layers.N.self_attn.q_norm.weight` | `blk.N.attn_q_norm.weight` | [256] | [256] | BF16 | 10 |
| `layers.N.self_attn.k_norm.weight` | `blk.N.attn_k_norm.weight` | [256] | [256] | BF16 | 10 |
| `layers.N.mlp.gate.weight` | `blk.N.ffn_gate_inp.weight` | [256, 2048] | [2048, 256] | BF16 | 40 |
| `layers.N.mlp.experts.gate_up_proj` | `blk.N.ffn_gate_exps.weight` + `blk.N.ffn_up_exps.weight` (split), or `blk.N.ffn_gate_up_exps.weight` (fused) | [256, 1024, 2048] → 2× [256, 512, 2048] | [2048, 512, 256] each | BF16 | 40 |
| `layers.N.mlp.experts.down_proj` | `blk.N.ffn_down_exps.weight` | [256, 2048, 512] | [512, 2048, 256] | BF16 | 40 |
| `layers.N.mlp.shared_expert.gate_proj.weight` | `blk.N.ffn_gate_shexp.weight` | [512, 2048] | [2048, 512] | BF16 | 40 |
| `layers.N.mlp.shared_expert.up_proj.weight` | `blk.N.ffn_up_shexp.weight` | [512, 2048] | [2048, 512] | BF16 | 40 |
| `layers.N.mlp.shared_expert.down_proj.weight` | `blk.N.ffn_down_shexp.weight` | [2048, 512] | [512, 2048] | BF16 | 40 |
| `layers.N.mlp.shared_expert_gate.weight` | `blk.N.ffn_gate_inp_shexp.weight` | [1, 2048] | [2048] | BF16 | 40 |
| `lm_head.weight` (top level, not under `model.`) | `output.weight` | [248320, 2048] | [2048, 248320] | BF16 | 1 |

`N` runs over 0..39; `linear_attn.*` exists only where `layer_types[N] == "linear_attention"` (30 of
them) and `self_attn.*` only where it is `"full_attention"` (10: **3, 7, 11, 15, 19, 23, 27, 31, 35,
39** — `(N+1) % full_attention_interval == 0`, witnessed both in the config's `layer_types` and in
which `blk.N` carry `attn_q` vs `attn_qkv` in the published file). Every layer carries the full MoE
set: there is no dense-MLP layer and no `mlp_only_layers` key in this config.

Per-block totals: a linear layer is 18 tensors, a full-attention layer 15, the MTP block 19 (§01.7).

## §01.4 The published GGUF as it stands on disk

733 tensor-info records, 0 names unclassified, and the row bytes reconcile against the file's own
length. `ne` is as stored. These are the bytes a reader of the shipping artifact actually reads.

| GGUF family | n | stored type | GGUF `ne` | elements | bytes |
|---|---|---|---|---|---|
| `token_embd.weight` | 1 | Q8_0 | [2048, 248320] | 508,559,360 | 540,344,320 |
| `output.weight` | 1 | Q6_K | [2048, 248320] | 508,559,360 | 417,177,600 |
| `output_norm.weight` | 1 | F32 | [2048] | 2,048 | 8,192 |
| `blk.N.attn_norm.weight` | 40 | F32 | [2048] | 81,920 | 327,680 |
| `blk.N.post_attention_norm.weight` | 40 | F32 | [2048] | 81,920 | 327,680 |
| `blk.N.ffn_gate_inp.weight` | 40 | F32 | [2048, 256] | 20,971,520 | 83,886,080 |
| `blk.N.ffn_gate_inp_shexp.weight` | 40 | F32 | [2048] | 81,920 | 327,680 |
| `blk.N.ffn_gate_exps.weight` | 40 | Q4_K | [2048, 512, 256] | 10,737,418,240 | 6,039,797,760 |
| `blk.N.ffn_up_exps.weight` | 40 | Q4_K | [2048, 512, 256] | 10,737,418,240 | 6,039,797,760 |
| `blk.N.ffn_down_exps.weight` | 37 Q5_K + 3 Q6_K | Q5_K/Q6_K | [512, 2048, 256] | 10,737,418,240 | 7,488,929,792 |
| `blk.N.ffn_gate_shexp.weight` | 40 | Q8_0 | [2048, 512] | 41,943,040 | 44,564,480 |
| `blk.N.ffn_up_shexp.weight` | 40 | Q8_0 | [2048, 512] | 41,943,040 | 44,564,480 |
| `blk.N.ffn_down_shexp.weight` | 40 | Q8_0 | [512, 2048] | 41,943,040 | 44,564,480 |
| `blk.N.attn_q.weight` (10 full-attn) | 10 | Q8_0 | [2048, 8192] | 167,772,160 | 178,257,920 |
| `blk.N.attn_k.weight` | 10 | Q8_0 | [2048, 512] | 10,485,760 | 11,141,120 |
| `blk.N.attn_v.weight` | 10 | Q8_0 | [2048, 512] | 10,485,760 | 11,141,120 |
| `blk.N.attn_output.weight` | 10 | Q8_0 | [4096, 2048] | 83,886,080 | 89,128,960 |
| `blk.N.attn_q_norm.weight` / `attn_k_norm.weight` | 10 + 10 | F32 | [256] | 2,560 + 2,560 | 10,240 + 10,240 |
| `blk.N.attn_qkv.weight` (30 GDN) | 30 | Q8_0 | [2048, 8192] | 503,316,480 | 534,773,760 |
| `blk.N.attn_gate.weight` (GDN z) | 30 | Q8_0 | [2048, 4096] | 251,658,240 | 267,386,880 |
| `blk.N.ssm_a` | 30 | F32 | [32] | 960 | 3,840 |
| `blk.N.ssm_alpha.weight` / `ssm_beta.weight` | 30 + 30 | F32 | [2048, 32] | 1,966,080 ×2 | 7,864,320 ×2 |
| `blk.N.ssm_dt.bias` | 30 | F32 | [32] | 960 | 3,840 |
| `blk.N.ssm_conv1d.weight` | 30 | F32 | [4, 8192] | 983,040 | 3,932,160 |
| `blk.N.ssm_norm.weight` | 30 | F32 | [128] | 3,840 | 15,360 |
| `blk.N.ssm_out.weight` | 30 | Q8_0 | [4096, 2048] | 251,658,240 | 267,386,880 |
| **total** | **733** | | | | **22,123,538,944** |

Type census (same parse, `tools/gguf_width_table.py --check-header`): Q4_K 80, Q5_K 37, Q6_K 4,
Q8_0 251, F32 361 — 733. The three Q6_K `ffn_down_exps` are layers **34, 38, 39**; the fourth Q6_K is
`output.weight`. Q4_K counts 80 = 40 layers × (gate + up): **the published file stores the experts
SPLIT, not fused**, and there is no `ffn_gate_up_exps` tensor in it.
`general.file_type = 15` (mostly-Q4_K) with `general.quantization_version = 2` names the width of
only those 80 tensors out of 733; every norm, router, shared-expert gate and GDN-control tensor is
F32, and `lm_head`/expert `down_proj` are not Q4_K either.

## §01.5 Width assignment: our policy vs the file we ship

Policy A is the decided assignment (`quantization/README.md` §"The policies, priced",
`DECISIONS.md` D-01/D-01a/D-08/D-09/D-10). The shipped published artifact differs in exactly two
families. A reader must know which column their bytes came from.

| family | policy A (our preference) | published UD-Q4_K_M | match |
|---|---|---|---|
| routed experts gate/up | Q4_K | Q4_K ×80 | ✓ |
| routed experts down | Q4_K | **Q5_K ×37, Q6_K ×3** | ✗ published wider |
| shared expert (3 matrices) | Q8_0 | Q8_0 ×120 | ✓ |
| full attention q/k/v/output | Q8_0 | Q8_0 ×40 | ✓ |
| GDN `attn_qkv`, `attn_gate`, `ssm_out` | Q8_0 | Q8_0 ×90 | ✓ |
| GDN control (`ssm_a`, `ssm_alpha/beta`, `ssm_dt`, `ssm_norm`, `ssm_conv1d`) | exact | **F32 ×150** | ✓ (wider than "exact") |
| router `ffn_gate_inp` | exact | **F32 ×40** | ✓ (wider) |
| norms | exact | F32 ×101 | ✓ (wider) |
| embeddings | Q8_0 (D-10) | Q8_0 | ✓ |
| `lm_head` (`output`) | Q8_0 (D-10) | **Q6_K** | ✗ published narrower |
| MTP draft block | uniform with the trunk (D-09) | **absent — 0 tensors** | ✗ see §01.7 |

## §01.6 Seven storage facts that read silently wrong if assumed

Each is a converter decision, not an HF fact, and each fails silently — the wrong reading produces
plausible text, not an error.

1. **GDN V heads are stored tiled, not in HF's grouped order** `[REF: llama.cpp conversion/qwen.py,
   _LinearAttentionVReorderBase._reorder_v_heads]`. HF stores V heads grouped by their K head
   `[G0_v0, G0_v1, G1_v0, G1_v1, …]`; the converter reshapes the V rows to
   `[n_k, n_v/n_k, head_v_dim]`, swaps the first two axes and flattens, giving
   `[G0_v0, G1_v0, …, G15_v0, G0_v1, …]`. With 16 K and 32 V heads, GGUF V-head `j` holds HF V-head
   `(j % 16) * 2 + (j // 16)`. Applied to: the V third of `attn_qkv` rows, all of `attn_gate` rows,
   `ssm_alpha`/`ssm_beta` rows, `ssm_a`, `ssm_dt.bias`, the V channels of `ssm_conv1d`, and the
   columns of `ssm_out`. A reader assuming grouped order reads the wrong heads and raises nothing.
   `specification/ASSUMPTIONS.md` A-12.
2. **`attn_q` is `[2*n_q*d_head, H] = [8192, 2048]`** because `attn_output_gate: true`
   `[REF: Qwen3_5MoeAttention.__init__]`. Sizing it at `n_q*d_head` reads half a tensor. The gate half
   is interleaved **inside each head** (`specification/05-gated-full-attention.md` §05.1), not appended as a second half of the tensor.
3. **Norms are stored as `HF + 1`, computed in F32** `[REF: conversion/qwen.py modify_tensors:
   `data_torch = data_torch + 1` for names ending `norm.weight`]. Adding 1 in BF16 loses bits the file
   keeps, and the exception matters: `linear_attn.norm.weight` is *excluded* from the +1, which is the
   witness that the two RMSNorm forms differ (`specification/03-block-glue-and-normalization.md` §03.1, `python3 specification/derive.py --values
   model.language_model.layers.0.input_layernorm.weight
   model.language_model.layers.0.linear_attn.norm.weight`).
4. **`conv1d` is squeezed before writing**: HF `[8192, 1, 4]` → `[8192, 4]` → GGUF `ne [4, 8192]`. A
   reader that reconstitutes rank 3 as `[1, 4, 8192]` reads the taps as channels.
5. **Expert gate/up fusion is opt-in and OFF by default** (`--fuse-gate-up-exps`, default `False`,
   `engines/llama.cpp/conversion/base.py`). HF stores one fused `[256, 1024, 2048]` tensor; the
   converter splits it on dim −2 into `ffn_gate_exps`/`ffn_up_exps`, and with the flag on emits
   `ffn_gate_up_exps` by concatenating gate then up along that same axis. So gate|up lies along
   **`ne[1]`, gate first** — and a reader must accept both spellings, as llama.cpp's loader does.
6. **`A_log` is not stored as `A_log`** — the file holds `-exp(A_log)` under the name `blk.N.ssm_a`
   `[REF: conversion/qwen.py: `data_torch = -torch.exp(data_torch)`]`. `specification/10-reference-dumps-and-gates.md` §10 and A-13 carry why this is a
   recomputation with a 1-ulp history, not a copy.
7. **`dt_bias` changes name class**: HF `.dt_bias` → `dt_proj.bias` → GGUF `blk.N.ssm_dt.bias`, so a
   loader that only looks for `.weight` suffixes will miss it (it is one of the 361 F32 tensors).

## §01.7 The MTP draft block — 19 tensors, present only in the HF shards

`python3 specification/derive.py --mtp` reads the real headers of
`models/qwen36-src/model-00025-of-00026.safetensors` and `model-00026-of-00026.safetensors`:
**19 tensors, 844,640,768 elements = 1,611.024 MiB BF16**, names and shapes identical to the
config-derived expectation. Of that, 805,306,368 elements (**95.34 %**) are the block's own two fused
expert tensors — `mtp.layers.0.mlp.experts.gate_up_proj` [256, 1024, 2048] and `…down_proj`
[256, 2048, 512]. The block is a full-attention decoder layer plus MoE (no GDN) plus the NextN fusion
tensors; its op order, its tap points and the fact that it reuses the trunk `output_norm`/`output`
are in `specification/09-mtp-draft-layer.md`.

The published GGUF contains **zero** draft-block tensors (layer indices 0..39, no name containing
`mtp`; `tools/gguf_header_types.py` prints that line). So the shipping artifact alone cannot serve
D-05, and a reader who wants the drafter reads the 19 tensors above. In GGUF naming they become
`blk.40.*` — `mtp.fc` → `blk.40.nextn.eh_proj.weight`, `mtp.pre_fc_norm_embedding` →
`blk.40.nextn.enorm.weight`, `mtp.pre_fc_norm_hidden` → `blk.40.nextn.hnorm.weight`,
`mtp.norm` → `blk.40.nextn.shared_head_norm.weight`, and `mtp.layers.0.*` follows the §01.3 table at
`N = 40`.

## §01.8 What is deliberately not here

The vision tower (`model.visual.*`, 333 tensors, 893,142,496 B) is out of scope
(`DECISIONS.md` D-04, `specification/README.md` §Scope), and so are the three Flash-Next-only
subsystems, which this checkpoint does not contain at all. Biases: `attention_bias: false` and every
Linear in the text path is `bias=False`, so **the only bias-shaped tensor in the text path is
`ssm_dt.bias`** (32 values per GDN layer) — there is no `attn_q.bias` to allocate.
