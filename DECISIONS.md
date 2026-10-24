# Decision log — open questions that are the user's, not the agent's

An agent that "answers" one of these by itself has spent the user's budget in the wrong place.
Status is one of `OPEN`, `DECIDED`, `SUPERSEDED`. A decision gets a paragraph of evidence and the
number or failure that forced it.

| id | question | status |
|---|---|---|
| D-01 | quantization to fit | **DECIDED** (experts 4-bit) — sub-question D-01a open |
| D-02 | residency above the pool | **DECIDED** by measurement: no |
| D-03 | download the 67 GiB now? | **OPEN** (one word needed) |
| D-04 | Flash-Next-only subsystems | **DECIDED**: out of scope |
| D-05 | MTP in scope? | **DECIDED**: yes |
| D-06 | which 35B-A3B checkpoint | **DECIDED**: `Qwen/Qwen3.6-35B-A3B` |
| D-07 | does MoE work transfer to Flash-Next | **DECIDED**: the block, not its surroundings |
| D-08 | which 4-bit format | **OPEN** (recommendation Q4_K) |
| D-09 | is the MTP expert block quantized the same way | **OPEN** |

---

## D-01 — Is quantization to fit in scope? — **DECIDED 2026-10-26: yes for the routed experts (4-bit); attention width is still the user's (D-01a)**

User's words: "quantization -- experts q4, attention -- don't know". So the routed experts go to 4
bits and everything else is an open sub-question rather than an inherited assumption. What made the
decision easy is that D-02 (below) closed the alternative: the ~32 GiB device-visible budget is a
per-process total and host-pinning does not raise it, so bf16 (65.72 GiB) and Q8-throughout
(34.97 GiB) are not "slow but honest" options — they are impossible. Evidence and the full policy
table: `quantization/README.md`; the measurement is `devices/measured-facts.md`
§"Residency above the pool".

What it commits this repo to: a dequant-capable expert kernel, and **new numerics bands bought with a
floor probe and geometry-derived must-catch sets** (`pitfalls/30`, `pitfalls/40`) — the bf16 oracle in
`speccheck/` does not move, and no existing gate gets loosened to admit a quantized path. Router,
norms, GDN conv, `A_log`/`dt_bias` stay exact: 0.11 GiB in total, and every one of them is a scale or
a control.

Still true and now moot for residency: the MoE **subsystem** can be developed and gated in bf16 on the
CPU oracle without any quantization (see `reference-implementation.md`), so the 4-bit kernel does not
have to be the first thing written.

### D-01a — attention / GDN width: bf16, Q8_0 or Q4_K — **OPEN (needs the user's word; int8 is theirs to authorize too)**

Three live choices, from `python3 tools/quant_scenarios.py`, all with the experts at Q4_K:

| choice | resident | GiB/token | ceiling @210 GB/s |
|---|---|---|---|
| attention/GDN bf16 | 21.52 GiB | 3.813 | 51 tok/s |
| **attention/GDN Q8_0** (policy A/G) | 19.4–19.6 GiB | 2.22–2.33 | **84–88 tok/s** |
| attention/GDN Q4_K (policy E) | 18.81 GiB | 1.663 | 118 tok/s |

Recommendation: **Q8_0 now** (policy G = experts Q4_K, attention/GDN/shared Q8_0, `lm_head` Q6_K), and
**fund policy E as a measured experiment** rather than shipping it. The +40 % in the last row is real;
so is that the published per-tensor evidence for this exact architecture family says `attn_*` is the
most sensitive class on a hybrid model and GDN `out_proj` the worst single tensor
(`quantization/evidence/unsloth-qwen3.5-35b-a3b-2026-10.md`), and llama.cpp's own policy forces the
latent/gate categories to ≥ Q8_0 and spends its spare bits on `output`
(`quantization/evidence/llama-cpp-tensor-policy-2026-10-26.md`). Residency and 40 % of ceiling are
cheap to give up once; a quality regression found after shipping is not.

## D-02 — Can a kernel read host-mapped memory above the pool? — **DECIDED 2026-10-26 by measurement: no**

Run blocking with the GPU lock held and zero subagents: `tools/hostmap_probe.cpp` (new here), raw
output in `devices/evidence/d02-hostmap-{probe,sweep,multi,ceiling}-2026-10-26.txt`. Headlines, every
line carrying an independent read-checksum so it cannot be an artifact of skipped work:

- pinned + mapped host memory reads at **195–225 GB/s from 1 GiB up to 32 GiB**, including 215.8 GB/s
  for 12 GiB pinned against 198.6 GB/s for 20 GiB of device memory in the same window. Location is
  not the problem.
- the limit is **total per process**: `hipMalloc` 20 GiB + `hipHostMalloc` 12 GiB is granted
  (32 GiB); `hipMalloc` 20 GiB + `hipHostMalloc` 24 GiB refuses the pinned half; and **two 20 GiB
  pins get one grant and one refusal**. A single 34 GiB pin fails in 0.0 s with MemAvailable at
  87 GiB, so it is an accounting limit, not memory pressure.
- `hipMalloc` 40 GiB still fails; `hipHostRegister` of 40 GiB answers `invalid argument`.
- an unregistered `malloc` pointer handed to a kernel **faults the queue**, and the checksum witness is
  what stopped the resulting "1,082,301 GB/s" from becoming a number.

Consequences: quantization is a prerequisite (D-01); a two-tier device + pinned layout is
bandwidth-viable but cannot exceed ~30 GiB of total device-visible bytes; and the Flash-Next endgame
(≈58 GiB even at 4-bit main weights) needs a bigger pool or real streaming, not a mapping trick.

## D-03 — Weight acquisition: download 67 GiB now, or after the spec is written? — **OPEN (needs one word: yes/no)**

`hf download Qwen/Qwen3.6-35B-A3B --local-dir models/Qwen3.6-35B-A3B` — ~30 min, 66.97 GiB across 26
shards (1,045 tensors) including the vision tower. Disk was 2.4 TiB free on 2026-10-26. Nothing
written so far needed the weights: the census came from `checkpoints/` metadata validated to 1.86 %
against the index `total_size`, and the residency decision came from a probe that allocates its own
buffers. What does need them: every pinned dump, every `speccheck/` gate, any per-tensor sensitivity
check of our own (the class of measurement unsloth published), and the dequant cost on real data.
Recommendation unchanged: say yes and let it run alongside the spec pass.

## D-04 — Flash-Next-only subsystems (Gated Residual/hyperconnections, QSA indexer, n-gram PLE) — **DECIDED 2026-10-26: out of scope**

User's words: "out of scope". None of them exists in 3.6-35B-A3B, so this project cannot exercise
them; they are also the three that produced the chance-level accept rate that stopped
`../qwen3.8-flash-next-old`. Revisit only on an explicit request — and per D-07, the MoE work here is
what makes that a smaller project later.

## D-05 — Is MTP part of the scope here? — **DECIDED 2026-10-26: yes, it is in scope**

User's words: "MTP part of the scope". It is a real part of the target: one MTP block whose own expert
block is another 1.500 GiB of the census, and whose draft changes which experts fire, so router and
MTP interact from day one. The 0.8B engine carries draft/accept/verify code and its measurement
discipline as precedent (`reference-implementation.md`).

Sequencing note that does not narrow the decision: land the MoE block and the dense path first so MTP
has something attributable to be measured against — accepting MTP and MoE-shaped changes in the same
window makes neither attributable. MTP is scheduled, not optional.

---

## D-06 — Which 35B-A3B checkpoint? — **DECIDED 2026-10-26: `Qwen/Qwen3.6-35B-A3B`**

`Qwen3.5-35B-A3B` and `Qwen3.6-35B-A3B` are geometrically identical (diff is `bos/pad_token_id`,
`output_router_logits`, `mlp_only_layers`, `partial_rotary_factor`). 3.6 states
`partial_rotary_factor: 0.25`, which is what the existing engine's rope implements; 3.5 omits it
and would silently mean full rotary. 3.6 is also the version with live Strix-Halo GGUF/MTP work
around it. Evidence: `README.md` §"Which 35B-A3B".

## D-07 — Does the MoE work transfer to Flash-Next? — **DECIDED 2026-10-26: yes, the block; no, its surroundings**

Diff of the HF reference classes: `qwen3_5_moe/modeling_qwen3_5_moe.py:818-915` vs
`qwen4_exp/modeling_qwen4_exp.py:896-994` → 0 differing lines in `Experts` and `SparseMoeBlock`,
4 cosmetic in `MLP`, one real flag in `TopKRouter` (`norm_topk_prob`, default `True`, unset in
Flash-Next's `config.json` ⇒ identical behaviour today). What does not transfer is the
`mlp_hyper_connection` wrapper (4-stream mix + per-token inject weights, ~0.63 GMAC/token at
hc_count 4 / hc_lowrank 320 over 48 layers, plus a 4× wider residual stream), the QSA indexer and
the PLE. Evidence: read of the two installed modeling files, 2026-10-26.

## D-08 — Which 4-bit format for the routed experts? — **OPEN (recommendation: Q4_K)**

Sourced in `quantization/README.md` and `quantization/evidence/`:

| format | bpw | why it is or is not first |
|---|---|---|
| **Q4_K** | 4.5 | llama.cpp's own `*recommended` default; dequant is a sub-block scale multiply, cheap to write well for `gfx1151`; the block layout (32-element sub-blocks, fp16 scales) is a kernel requirement, not just a size |
| MXFP4 | 4.25 | ships as `MXFP4_MOE` (experts MXFP4, everything else Q8_0 — the same shape as our policy A), and there is a perplexity-only crowd claim that it beats Q4_K_M; but on **this** architecture family the published KLD comparison is worse at equal size (0.7789 @ 18.17 GB vs 0.5478 @ 18.49 GB) and unsloth retired it from mixed quants: "better to use Q4_K than MXFP4 when choosing between them" |
| IQ4_XS | 4.25 | the size-down lever (~1 GiB below Q4_K, better KLD-per-byte than MXFP4), but codebook dequant costs decode speed (~5 % in the published table) and that cost must be measured here, not assumed |

Until a `prototype/moe_gemv` probe measures dequant cost on this GPU, every tok/s figure in this repo
remains a ceiling rather than a forecast — which is the honest reason the format choice is still open.

## D-09 — Is the MTP block's expert stack quantized like the main one? — **OPEN (cheap to answer, easy to forget)**

`mtp.layers.0.mlp.experts.*` is a full MoE layer worth 1.500 GiB of the census, read once per draft
token. Probably the same answer as D-01, but it should be decided rather than assumed: a draft head at
a different precision from the stack it drafts for is a systematic accept-rate penalty wearing the
costume of a draft-quality one.
