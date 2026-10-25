# What the published Q4_K_M for this model actually quantizes (header-parsed, 2026-10-26)

Question this answers: D-12's open half — if llama.cpp is our cross-check oracle, **which quantization
does the published artifact use?** Not answerable from quantize-tool prose (a K-quant is a *policy over
tensors*), so it was read out of the file's own tensor-info header with
`tools/gguf_header_types.py`, which fetches a byte range rather than 19 GiB:

```
python3 tools/gguf_header_types.py \
  "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
```

Raw output: `quantization/evidence/q4km-published-width-map-2026-10-26.txt`. Parsed 733 of 733
tensor-info records; `arch=qwen35moe`, `expert_count 256`, `expert_used_count 8`,
`expert_feed_forward_length 512`, `full_attention_interval 4`, `attention.value_length 256` — the
header agrees with our `checkpoints/` census on every geometry key.

## The map, and how it compares to our decided policy A

| family | published **UD-Q4_K_M** | stored GiB | our policy A (D-01/D-01a/D-09/D-10) | same? |
|---|---|---|---|---|
| routed experts gate/up | **Q4_K** ×80 | 11.250 | Q4_K | ✓ |
| routed experts down | **Q5_K** ×37, **Q6_K** ×3 | 7.047 | Q4_K | **✗ published is wider** |
| shared expert | Q8_0 ×120 | 0.125 | Q8_0 | ✓ |
| full attention q/k/v/output | Q8_0 ×10 each | 0.269 | Q8_0 | ✓ |
| linear-attn (GDN) `attn_qkv` | Q8_0 ×30 | 0.498 | Q8_0 | ✓ |
| linear-attn z-gate `attn_gate` | Q8_0 ×30 | 0.249 | Q8_0 | ✓ |
| GDN `ssm_out` | Q8_0 ×30 | 0.249 | Q8_0 | ✓ |
| GDN control (`ssm_a/alpha/beta/dt`, `conv1d`, `ssm_norm`) | **F32** ×180 | 0.019 | exact (bf16) | ✓ (published is wider) |
| router `ffn_gate_inp` | **F32** ×80 | 0.078 | exact (bf16) | ✓ (published is wider) |
| norms | F32 ×101 | 0.001 | exact | ✓ |
| embeddings | **Q8_0** | 0.503 | Q8_0 (D-10) | ✓ |
| `lm_head` (`output`) | **Q6_K** | 0.389 | Q8_0 (D-10) | **✗ published is narrower** |
| **total weights** | | **20.676 GiB** (22.20 GB dec.) | 19.59 GiB (arithmetic) | +1.09 GiB |

So the two policies differ in **exactly two places**: the published file spends **more** bits on expert
`down_proj` (Q5_K, with Q6_K on three layers) and **fewer** on `lm_head` (Q6_K). The down-proj choice
costs ~1.3 GiB over Q4_K and is the reason the published total is ~1 GiB above our arithmetic — the
lm_head choice saves ~0.13 GiB back.

**Three things this settles:**

1. **An independent producer landed on our attention/GDN decision.** unsloth's dynamic recipe puts
   every attention and GDN matrix at Q8_0 — the same place D-01a landed from KLD evidence and the
   `ssm_out` warning. That is worth having on the record: policy A is not an idiosyncratic choice, and
   the "Q8 attention is over-cautious" reading of llama.cpp's stock defaults does not survive contact
   with a tuned quant of this exact model.
2. **`down_proj` is the half of the expert that a producer chose to protect.** Unsloth's own per-tensor
   sensitivity work (see `quantization/evidence/unsloth-qwen3.5-35b-a3b-2026-10.md`) is why the
   dynamic recipes bump down-proj; our D-01 says "4-bit experts" but did **not** decide gate/up vs down
   separately. This is a cheap refinement to put in front of the user later: gate/up Q4_K + down Q5_K
   costs ~1.3 GiB and per-token bytes go up by ~0.45 GiB·(5.5625/4.5−1)/3 ≈ 0.07 GiB — I have not
   re-run `tools/quant_scenarios.py` for that variant, so the number is quoted as a note, not a policy.
3. **The published file carries no draft block.** Zero tensors contain `mtp` and layer indices run
   0..39, i.e. 40 layers and no extra block. Since D-05 (MTP) is in scope, **a published Q4_K_M cannot
   be the shipping artifact** — the MTP weights exist only in the HF checkpoint. That is a harder
   argument for the HF download than anything in D-03's original framing.

## Why the earlier "policy delta would swamp the comparison" worry mostly goes away

D-12's second catch assumed the reference would be a *published* Q4_K_M whose tensor policy differs
from ours broadly. It doesn't — the delta is down-proj and lm_head. And the delta disappears
altogether in the arrangement that matters most: **produce our policy-A GGUF with upstream
`llama-quantize --tensor-type-file` and let llama.cpp read *that*.** Same bytes on both sides, so
engine-vs-engine comparison is about our kernels only. What that arrangement still needs is the f16
source to quantize from — which is D-03 again, plus the MTP fact above.

## What else the file carries (it is runnable on its own)

Same header, `--keys` pass:

| key | value | why it matters here |
|---|---|---|
| `qwen35moe.rope.dimension_count` | **64** | = `partial_rotary_factor 0.25` × `head_dim 256`. The GGUF carries the partial-rope fact we would otherwise have to carry ourselves. |
| `qwen35moe.rope.dimension_sections` | I32 ×4 | **not vanilla rope**: the file declares rope dimension *sections* (MRoPE-style, inherited from the VL parent). A text-path rope implementation that assumes a plain 0..63 rotation will be wrong; read this key. |
| `qwen35moe.rope.freq_base` | 1e7 | matches the checkpoint's `rope_parameters.rope_theta` |
| `tokenizer.ggml.model` | gpt2 (BPE) | tokenizer ships in the file |
| `bos / eos / padding` | **248044 / 248046 / 248055** | the stop id is the same 248046 as the 0.8B, and `add_bos_token = False` |
| `tokenizer.chat_template` | present, vision-aware | ChatML parity comes from the file, not from an external template. It counts image/video, i.e. it was carried over from the VL checkpoint — treat its vision branches as dead code, not as capability. |

So the published file is sufficient to *run* the model: weights, tokenizer, template and rope parameters
are all inside it. What it is missing is exactly one thing — the draft block, below.

## The one thing it really lacks, and what it costs

`mtp.*` is absent, and `tools/mtp_shard_cost.py` prices the fix from the committed index: the 19
`mtp.*` tensors live in **2 shards of 26, ~5.2 GiB** (estimate = index `total_size` ÷ shard count; the
listing gives the real number at fetch time). So keeping D-05 costs ~5 GiB, not 67. The same tool
shows a **sampled provenance audit** — MTP plus layers 0, 1 and 39, enough to compare a published
quant's Q4_K blocks against the bf16 originals at the format's own rounding floor — costs **4 shards,
~10.3 GiB**. By contrast a witness spread over every layer (all routers, norms and GDN control
tensors) touches **17 shards ≈ 44 GiB**, which is a full download in disguise: that is the number
which says *sample* the audit, don't spread it.

## If we ship the published file rather than converting our own

Priced on the same roofline as policy A. Published bytes/token ≈ **2.26 GiB** (policy A 2.331): its
`lm_head` at Q6_K is 0.114 GiB/token *cheaper* and its expert `down_proj` at Q5_K is ~0.042
GiB/token dearer, so it is marginally **faster**, not slower — around 98 tok/s at 237 GB/s against 95.
Residency is worse (20.676 GiB vs 19.59) and residency is not the binding constraint. The costs are
not speed, they are: no way to measure quantization error (inherited from a producer's calibration),
no way to change a width decision (including the `lm_head` Q6_K that D-10 rejected on tail-risk
grounds), no witness against llama.cpp agreeing with itself, and no quantizer-writing capability
built at 35B — which is the step where that capability would have been cheap to build and is needed
for a fresh 125B checkpoint.

## Caveats on this evidence

- **One file, one producer.** This is unsloth's *UD* (dynamic) recipe. llama.cpp's *stock* `Q4_K_M` is
  different: from `src/llama-quant.cpp`, baseline Q4_K, `ffn_down`/`attn_v` bumped to Q6_K on
  `use_more_bits` layers (first 1/8, last 1/8, every third between), `FFN_LATENT`/`ATTENTION_GATE`
  forced to ≥ Q8_0, `output` → Q6_K unless `--output-tensor-type`, router/`ssm_conv1d`/norms never
  quantized. A `Q4_K_XL` or `_L` variant from a different producer is a different map — re-run the
  tool rather than re-citing this table.
- Byte totals use standard ggml block sizes (Q4_K 4.5, Q5_K 5.5625, Q6_K 6.5625, Q8_0 8.5 bpw) as
  encoded in `tools/gguf_header_types.py`'s `BPW` table; an unknown type would print a warning and
  count as 0 rather than be guessed. No warning fired here.
- Parsing this header cost two runs and one real bug: **a GGUF `BOOL` metadata value is one byte, not
  eight.** Reading it as 8 desyncs the metadata walk and the next string length comes back as
  ~8.8e18, which looks like a corrupt file rather than an off-by-seven. The tool says so in its
  source, and `pitfalls/50` carries the line.
