# qwen3.6-35b-a3b

The second engine model. Same box, same engine family, **one new subsystem: MoE**.

This directory is the home of a project whose target is `Qwen/Qwen3.6-35B-A3B`,
text-only path, on the Framework Desktop (Strix Halo, `gfx1151`). It exists as the
**MoE stepping stone** on the road to the real target, `Qwen/Qwen3.8-Flash-Next`
(`qwen4_exp`, 125B-A6B + 51B n-gram PLE + QSA + Gated Residual).

```
reference implementation   → ../qwen3.5-0.8b            (shipping, gated, 100 % of its gates green)
this project               → ../qwen3.6-35b-a3b         (you are here)
end goal                   → Qwen/Qwen3.8-Flash-Next    (its three lab-only subsystems killed ../qwen3.8-flash-next-old)
the failed predecessor     → ../qwen3.8-flash-next-old  (device facts and quantization analysis are still useful)
```

**Read `AGENTS.md` first, then `pitfalls/README.md`.** `reference-implementation.md` is the
reading list for which 0.8B code to reuse and how; every *fact* you need is already in this repo.

## Target — verified facts

Read from `config.json` + `model.safetensors.index.json` fetched from the Hub on 2026-10-26,
not from a blog post. Architecture `Qwen3_5MoeForConditionalGeneration`, `model_type: qwen3_5_moe`.

Language tower (`text_config`):

| field | value | vs the 0.8B engine |
|---|---|---|
| hidden_size | 2048 | 1024 |
| num_hidden_layers | 40 = 30 `linear_attention` + 10 `full_attention` (`full_attention_interval: 4`) | 24 = 18 + 6 |
| head_dim | 256 | same |
| attention heads | 16 Q / 2 KV (GQA), `attn_output_gate: true` | 8 Q / 2 KV |
| RoPE | `rope_theta 1e7`, `partial_rotary_factor: 0.25` (explicit), `mrope_interleaved: true`, sections `[11,11,10]` | same |
| GDN (Gated DeltaNet) | `linear_num_key_heads: 16`, `linear_num_value_heads: 32`, key/value head dim 128, `linear_conv_kernel_dim: 4`, `mamba_ssm_dtype: float32` | 16 V heads → **32** |
| **MoE** | **256 experts, 8 routed + 1 shared**, `moe_intermediate_size: 512`, `shared_expert_intermediate_size: 512`, every layer sparse | **new** |
| dense MLP | none — no `intermediate_size` escape hatch, no `mlp_only_layers` | the 0.8B is all-dense |
| embeddings | `vocab_size: 248320`, `tie_word_embeddings: **false**` → `lm_head.weight` is a separate 1.0 GiB read every token | 0.8B is tied |
| MTP | `mtp_num_hidden_layers: 1`, card says "trained with multi-steps" | exists in 0.8B, was descoped there |
| context | `max_position_embeddings: 262144` (YaRN to ~1.01M) | same native |
| token ids | `eos_token_id: 248044`, `bos_token_id: 248044` — **re-derive the stop rule from this checkpoint's own tokenizer/generation config, do not carry the 0.8B's `248046`-is-stop rule over** (refer to these ids by number, never literally, in any file or log) | 0.8B stops on 248046 |
| checkpoint | 26 safetensors shards, `total_size` 71,903,645,408 B = **66.97 GiB**, 1,045 tensors, plus a 27-block vision tower (out of scope) | 1.63 GiB |

Expert storage (from the index + the HF reference): one **fused tensor per layer**,
`model.language_model.layers.N.mlp.experts.gate_up_proj` `[256, 2·512, 2048]` and
`…​.experts.down_proj` `[256, 2048, 512]`, plus `mlp.gate.weight`, `mlp.shared_expert*`.
Per expert that is 3,145,728 elements = **6.0 MiB bf16**; per layer all 256 experts = 1.50 GiB;
40 layers plus the MTP block's own MoE layer = **61.5 GiB, i.e. 91.8 % of the text path**. The MTP
block matters for sizing, and its width is decided (uniform with the stack it drafts for,
`DECISIONS.md` D-09).

Which 35B-A3B: `Qwen3.6-35B-A3B` and `Qwen3.5-35B-A3B` are **geometrically identical**
(only `bos/pad_token_id`, `output_router_logits`, `mlp_only_layers`, `partial_rotary_factor`
differ). 3.6 is chosen because it states `partial_rotary_factor: 0.25` — the value the
existing engine's rope code implements — while 3.5 omits it and would default to full rotary.

## Memory and roofline arithmetic

Derived from `checkpoints/Qwen3.6-35B-A3B.config.json` shapes (element counts × bytes) and validated
against the index's own `total_size` to within 1.86 % — reproduce with
`python3 tools/quant_scenarios.py --census`. Measured box ceilings, recorded in-repo at
`devices/measured-facts.md`: **190–237 GB/s** warm read (window-dependent, see `pitfalls/00`), and a
**device-visible pool of ≈55.5–56.5 GiB for the whole box**, of which halogen's `flash_serve` is inside
— so one process gets **≈32 GiB while halogen is resident** and ≈55 GiB only if it owns the box, and
host-pinning spends the pool rather than raising it
(`devices/measured-facts.md` §"Residency above the pool", measured 2026-10-26; `DECISIONS.md` D-11).

| quantity | value | consequence |
|---|---|---|
| text weights, bf16 | **66.1337 GiB** with the MTP draft block, **64.5604 GiB** without it — routed experts **61.500** = 93.0 % (60.0000 in the artifact + 1.500 in the draft block's own 256-expert stack), GDN **1.885**, `lm_head` 0.947, embeddings 0.947, full attention 0.559, shared experts 0.240, routers 0.040, norms ~0.001, `mtp.fc` 0.016 (`quantization/evidence/bf16-census-adjudication-2026-10-26.md`; artifact-side cut reproducible with `tools/gguf_suffix_census.py`) | **above the whole pool, not just above what is free** — capacity is a hard constraint, not a tuning knob. This figure read 65.72 GiB until 2026-10-26; that came from a config model with four wrong terms and matched *neither* independent ruler. GDN was under-counted by 0.469 (the omitted `ssm_out`) and 0.0704 of phantoms sat in "norms and misc" |
| active bytes / token, bf16 | **5.61 GiB** (adjudicated 2026-10-26; the component split that used to be quoted here summed to 5.194 and is superseded — re-print it from the repaired `tools/quant_scenarios.py`, do not re-derive it by hand) | still near the model card's "3B activated", which counts *parameters*, not bytes |
| decode roofline, bf16 | **academic** — bf16 does not fit in either budget, so no rate is quoted for it while the table above is partly stale | see the staleness note under §policy table |
| **policy A — DECIDED** (experts Q4_K, attention/GDN Q8_0) | **19.7848 GiB resident, 2.5252 GiB/token with one draft pass (2.4653 target-only) → ~87 tok/s ceiling @237 GB/s, ~77 @210, ~70 @190** | the shipping shape: `DECISIONS.md` D-01 + D-01a (user, 2026-10-26). Supersedes 19.59 / 2.331 / ~95: the resident figure moved +0.99 % and the fit verdicts are unchanged, but **per-token rose 8.3 % and the rate prediction fell with it**. These are predictions against the measured copy rate, not measurements |
| policy G (same, `lm_head` Q6_K) | 19.36 GiB, 2.216 GiB/token → ~100 tok/s | **not taken** — D-10 decided `lm_head` Q8_0 |
| experts + attention both Q4_K (policy E) | 18.81 GiB, 1.663 GiB/token → ~133 tok/s | a funded experiment behind a KLD study, not a default |
| per-request state | GDN 30 layers × 32 V heads × 128×128 F32 ≈ **60 MiB** + conv (the 0.8B's is 18.0 MiB); KV is 10 KiB/token over 10 full-attn layers instead of 6, so ~1.7× the 0.8B's 12 KiB/token on the same accounting | concurrency budget ~2× tighter than the 0.8B's, inside a budget the weights already consume |
| prefill | compute- and launch-bound as before, but 40 layers × a router + 8-expert gather per token | the router is a new per-token serial point |

Ceilings are bytes ÷ bandwidth, and the 0.8B engine reached ~60 % of its own, so treat every tok/s
figure here as an upper bound on a kernel that does not exist yet. Full table (eight policies,
residency + per-token + three bandwidths): `tools/quant_scenarios.py`; the reasoning and the external
evidence behind the format choice: `quantization/README.md`.

**Quantization is therefore decided and load-bearing, not optional** (`DECISIONS.md` D-01 + D-01a: user,
2026-10-26 — routed experts at 4-bit, attention/GDN at Q8_0 ⇒ **policy A**). The consequence for
gate design is that new numerics bands have to be bought with a floor probe and geometry-derived
must-catch sets, against a bf16 oracle — no existing gate gets loosened to admit a quantized path.

## What this project inherits and what is new

| subsystem | source of truth | status here |
|---|---|---|
| Gated DeltaNet + short conv | `../qwen3.5-0.8b/specification/04-gdn-linear-attention.md`, `engine/kernels_gdn.cpp`, `gdn_rec_*.h` | **reuse the form**, new widths (32 V heads, 40 layers) |
| gated GQA attention | spec §05, `engine/kernels_attn.cpp` | **reuse the form**, 16 Q / 2 KV, same head_dim/rope |
| SwiGLU MLP | spec §03, `prototype/mlp_gemv`, `engine/kernels_mlp.cpp` | **reuse as the expert body** (an expert *is* a 3-matrix SwiGLU with I=512) |
| MTP draft layer | spec §09, `speccheck/mtp.py`, `engine/mtp_*.cpp` | returns as a real win here (M3 was descoped in the 0.8B) |
| **router + top-k + grouped experts + shared expert** | `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py:818-915` | **the new work of this project** |
| untied `lm_head` | `prototype/lm_head_gemv` | small change, real consequence (1.0 GiB/token read) |
| vision, QSA indexer, n-gram PLE, Gated Residual / hyperconnections | `transformers/models/qwen4_exp/*` | **not here** — Flash-Next-only; see `DECISIONS.md` D-04 |

The MoE block of the *end goal* is the same code as this model's: diffing
`qwen3_5_moe/modeling_qwen3_5_moe.py:818-915` against
`qwen4_exp/modeling_qwen4_exp.py:896-994` gives **0 differing lines** in `Experts` and in
`SparseMoeBlock`, 4 cosmetic lines in `MLP`, and one real flag in `TopKRouter`
(`norm_topk_prob`, default `True`, unset by Flash-Next's config ⇒ same behaviour today).
So kernels written and gated here port to Flash-Next unchanged; what does **not** port is the
hyperconnection wrapper around the block, the QSA indexer and the n-gram PLE — the three
things that actually killed the previous attempt at that model.

## Layout

Mirrors `../qwen3.5-0.8b` deliberately, so a reader can move between them without learning a
second structure:

```
README.md                  this file: target facts and sizing
PLAN.md                    the funded queue: step 1 = published quant + MTP + audit (APPROVED, NOT STARTED)
AGENTS.md                  rules, GPU policy, the self-containment rule
reference-implementation.md  reading list for the 0.8B engine's *code shape* (not a fact store)
DECISIONS.md               the decision log (decided: D-03 weight acquisition ⇒ published quant + MTP +
                           audit, D-01/D-01a/D-09/D-10 widths ⇒ policy A (our preference; the shipped
                           published file differs — see D-03), D-02 residency, D-04, D-05 MTP, D-06,
                           D-07, D-08 4-bit format = Q4_K, D-13 endpoint policy; open: D-11 pool
                           ownership)
pitfalls/                  the trap library, copied in so nothing cites another repo
devices/measured-facts.md  this box's measured ceilings, residency budget and the roofline arithmetic
devices/evidence/          raw output of the device probes behind those numbers
quantization/              D-01 policy: scenarios, formats, gate plan, and the external evidence
tools/                     gpu_lock.sh + device probes + census/scenario scripts (see tools/PROVENANCE.md)
checkpoints/               committed config.json + shard index: the source of every sizing claim
models/<modelname>/        downloaded checkpoints, gitignored
specification/             to be authored: math-only spec of the 3.6-35B-A3B text path
speccheck/                 to be authored: independent torch-CPU implementation + gates
engine/                    to be authored: the HIP engine
prototype/<op>/            one dir per op: standalone kernel + reference dump + gate
progress/<YYYY-MM-DD>/     dated journals
```

## Download (D-03 decided ⇒ this is `PLAN.md` step 1, and it has **not** been run)

The approved acquisition is **not** the whole checkpoint. It is the published shipping artifact, the
MTP block the published artifact lacks, and four shards to audit the first two against — ≈31 GiB, none
of which touches the iGPU:

```sh
# 1. the shipping artifact (20.7 GiB of weights; carries tokenizer, chat template and rope keys)
mkdir -p models && hf download unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    --local-dir models/unsloth-q4km

# 2. MTP + the sampled audit input (19 mtp.* tensors + layers 0, 1, 39)
mkdir -p models/qwen36-src && hf download Qwen/Qwen3.6-35B-A3B \
    model-00001-of-00026.safetensors model-00002-of-00026.safetensors \
    model-00025-of-00026.safetensors model-00026-of-00026.safetensors \
    --local-dir models/qwen36-src
```

Which shards, and why those layers: `tools/mtp_shard_cost.py` (`--mtp`, `--layers 0,1,39`). Step 1.0 —
writing `tools/audit_published_quant.py` (planned) — comes **before** either command, so the gate is
designed from the spec rather than from the data it judges.

The full checkpoint was **authorized for download by the user on 2026-10-27 (D-29)** as oracle material,
superseding D-03's deferral. It is not shipping material — D-27 still ships the published GGUF:

```sh
hf download Qwen/Qwen3.6-35B-A3B --local-dir models/qwen36-src   # 67 GiB — oracle material, D-29
```

Disk free on this box: check `df -h /home` (2.4 TiB available as of 2026-10-26). Do **not** download as
a side effect of another task; each of these is a decision with a wall cost.
