# Decision log — open questions that are the user's, not the agent's

An agent that "answers" one of these by itself has spent the user's budget in the wrong place.
Status is one of `OPEN`, `DECIDED`, `SUPERSEDED`. A decision gets a paragraph of evidence and the
number or failure that forced it.

---

## D-01 — Is quantization to fit in scope? — **OPEN (blocks any GPU-resident run)**

bf16 text weights are 64.6 GiB against a measured ~32–34 GiB device-visible pool. Q8_0 is
32.3 GiB, i.e. sitting *on* the ceiling with nothing left for KV/state; 4–6 bit is the only
GPU-resident regime (Q4_K_M ≈ 16–21 GiB, Q6_K ≈ 24–26 GiB). The 0.8B project explicitly banned
quantization and reserved int8/int4 work for an explicit user decision, and this repo inherits
that rule.

What it unlocks: any end-to-end tokens/s on device.
What it costs: new numerics bands (a quantized path cannot be gated against bf16 pins bit-for-bit,
so each band must be re-bought with a floor probe + must-catch set), a dequant design decision, and
a calibration/Q&A round.
Cheap partial answer already true: the MoE **subsystem** can be developed and gated in bf16 on
the CPU oracle without any quantization (see `reference-implementation.md`).

## D-02 — Residency probe: can a kernel read >34 GiB of host-mapped memory at ~237 GB/s? — **OPEN (one short device experiment)**

`pitfalls/10-device-memory.md` records that host-pinned reads run at GTT speed (measured on this
box), and that the ~32–34 GiB limit is a *total device-visible pool* per process which
`GPU_MAX_HEAP_SIZE` does not move. What has **not** been measured here is mapping a >34 GiB host
buffer (plain `malloc` + register) and reading it from a kernel. If that works, Q8 35B-A3B and a
large slice of Flash-Next become residency-tractable and D-01's pressure drops a lot; if it does
not, quantization is mandatory and we should buy the bands deliberately rather than reluctantly.

Cost: one blocking main-agent run of a probe derived from `tools/mem_mode_probe_hip.cpp`, a few
minutes. Must be run with zero subagents active.

## D-03 — Weight acquisition: download 67 GiB now, or after the spec is written? — **OPEN**

`hf download Qwen/Qwen3.6-35B-A3B --local-dir models/Qwen3.6-35B-A3B`. ~30 min, 67 GiB disk,
and it is a prerequisite for any pinned dump. Disk was 2.4 TiB free on 2026-10-26. Recommendation:
download in parallel with the spec pass, since nothing else can be gated without it.

## D-04 — Flash-Next-only subsystems (Gated Residual/hyperconnections, QSA indexer, n-gram PLE) stay out of scope? — **OPEN, recommended: keep out**

None of them exists in 3.6-35B-A3B, so this project cannot exercise them. They are also the three
that produced the chance-level accept rate that stopped `../qwen3.8-flash-next-old`. The question
is whether a *third* project is opened for them (with the Flash-Next checkpoint as its fixture
source) or whether this repo grows them later after MoE ships. Recommendation: keep out; decide
again when MoE is green.

## D-05 — Is MTP part of the first milestone here? — **OPEN**

3.6-35B-A3B ships a real MTP head and the 0.8B engine already has draft/accept/verify code and
weeks of MTP measurement discipline behind it (precedent, not a dependency: see
`reference-implementation.md`).
But MTP interacts with the router (a drafted token changes expert selection), so doing both at
once makes neither attributable. Recommendation: dense path green first, MTP as its own milestone.

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
