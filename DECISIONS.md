# Decision log — open questions that are the user's, not the agent's

An agent that "answers" one of these by itself has spent the user's budget in the wrong place.
Status is one of `OPEN`, `DECIDED`, `DECIDED (agent, YYYY-MM-DD)`, `SUPERSEDED`. A decision gets a
paragraph of evidence and the number or failure that forced it. **Since 2026-10-26 the user has
delegated engineering judgement to the agent (D-18); agent-made decisions are recorded in the same
detail and clearly labelled as agent-made, never as user ratification. See §Agent-delegated
decisions at the foot of this file, and `AGENTS.md §GOAL and decision protocol`.**

| id | question | status |
|---|---|---|
| D-01 | quantization to fit | **DECIDED** (experts 4-bit) |
| D-01a | attention/GDN width — **Q8_0** ⇒ policy A is the shipping shape | **DECIDED** |
| D-02 | residency above the pool | **DECIDED** by measurement: no — **corrected same day**: the pool is box-wide, not per-process |
| D-03 | weight acquisition (bf16 oracle vs published GGUF vs both) | **DECIDED** (2026-10-26, user): **published quant + MTP shards + sampled audit**, bf16 deferred — `PLAN.md` step 1, approved, **not started** |
| D-04 | Flash-Next-only subsystems | **DECIDED**: out of scope |
| D-05 | MTP in scope? | **DECIDED**: yes |
| D-06 | which 35B-A3B checkpoint | **DECIDED**: `Qwen/Qwen3.6-35B-A3B` |
| D-07 | does MoE work transfer to Flash-Next | **DECIDED**: the block, not its surroundings |
| D-08 | which 4-bit format | **DECIDED**: Q4_K — **note**: the shipped published file carries expert `down_proj` at Q5_K/Q6_K, see D-03 |
| D-09 | is the MTP expert block quantized the same way | **DECIDED**: uniform with the stack it drafts for |
| D-10 | `lm_head`/embed width | **DECIDED**: Q8_0 — **note**: the shipped published file carries `lm_head` at Q6_K, see D-03 |
| D-11 | does the engine own the pool, or share it with halogen | **PART-DECIDED by the user 2026-10-26**: the engine runs with **halogen stopped** (deployment budget ≈55 GiB); development stays co-resident (≈32 GiB); Q4 is **strategic**, not a consequence of local fit. The ≈55 GiB row still needs its measurement |
| D-12 | llama.cpp as an oracle (cross-check vs accuracy oracle) | **ADOPTED** as cross-check; accuracy oracle stays bf16 |
| D-13 | how agents and tests share the two model endpoints | **DECIDED** (2026-10-26, user): ≤4 workers = ≤3 `tensorfold` + ≤1 `halogen`, main on `tensorfold`, 4 of 5 slots; halogen down for every measurement. **Cap revised same day (D-15): ≤2 `tensorfold` workers, 3 of 5 slots** |
| D-14 | is a recomputed transcendental a bit-exactness obligation? (`ssm_a` = `-exp(A_log)`) | **RATIFIED by the user 2026-10-26** (proposed by the main agent): copied families keep zero tolerance; the one transcendental route keeps its route-derived 1-ulp predicate with must-catch grown. **Scoped to the 2 `ssm_a` cases — it is not a general 1-ulp allowance** |
| D-15 | which endpoint gets a worker | **DECIDED by the user 2026-10-26**: when only one worker is being started, it runs on `halogen`; with ≥2 workers, `halogen` takes the single most expensive job and the rest go to `tensorfold` |

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

### D-01a — attention / GDN width: bf16, Q8_0 or Q4_K — **DECIDED 2026-10-26: Q8_0** (user)

**The user's words: "attention/GDN in Q8_0, and experts in Q4_K".** With D-01 that fixes the shipping
shape as **policy A**: routed experts Q4_K, attention + GDN + shared expert Q8_0, norms and routers
exact. `lm_head`/embeddings are the one width knob left and are raised as **D-10** (default: Q8_0, as
this decision implies; Q6_K is a 4 %-of-ceiling alternative).

| choice | resident | GiB/token | ceiling 237 / 210 / 190 GB/s |
|---|---|---|---|
| attention/GDN bf16 (policy B) | 21.93 GiB | 4.2269 | 52 / 46 / 42 tok/s |
| **attention/GDN Q8_0 — DECIDED (policy A)** | **19.78 GiB** | **2.5252** | **87 / 77 / 70 tok/s** |
| attention/GDN Q4_K (policy E) | 18.88 GiB | 1.7397 | 127 / 112 / 102 tok/s |

*Re-printed verbatim from `tools/quant_scenarios.py` on 2026-10-26 (rows A / B / E), which derives every
width from `engines/llama.cpp/ggml/src/ggml-common.h` struct sizes at run time. The figures these replaced
— 21.52 / 3.813 / 51, 19.59 / 2.331 / 84, 18.81 / 1.663 / 118 — predate the bf16 census correction and were
stale rather than merely different; they were that −0.4138 GiB error's last holdouts in this file
(`quantization/evidence/bf16-census-adjudication-2026-10-26.md`).*

Why Q8_0 was the recommendation and why it is the right call at this stage: the published per-tensor
evidence for this exact architecture family says `attn_*` is the most sensitive class on a hybrid model
and GDN `out_proj` the worst single tensor
(`quantization/evidence/unsloth-qwen3.5-35b-a3b-2026-10.md`). The
40 % of ceiling that policy E leaves on the table is a funded experiment with a KLD study in front of
it, not a default — and per `pitfalls/30-numerics.md`, a wider numeric band must still catch every
injection in a geometry-derived must-catch set before it can be adopted.

> **[support withdrawn 2026-10-26]** An earlier version of this paragraph also cited llama.cpp as forcing
> the latent/gate categories to ≥ Q8_0 (`quantization/evidence/llama-cpp-tensor-policy-2026-10-26.md`).
> **That support is void.** At the pinned checkout (`engines/llama.cpp` @ `d327bf1`) the bump sits inside
> `if (arch == LLM_ARCH_MUSE_SPARK && (category == FFN_LATENT || category == ATTENTION_GATE))`
> (`src/llama-quant.cpp`), and the comment above it says widening the guard to other architectures "wants a
> perplexity comparison per architecture first". It says nothing about `QWEN35MOE`. The error entered
> because our evidence file stated the rule without its arch guard; that file now carries a dated
> corrigendum. This decision survives on the sensitivity sweep and on the artifact's own choice below — not
> on this.

**The strongest support, and it was missing from the original reasoning: the shipping artifact already does
this.** Reading the pinned GGUF's tensor records (`tools/gguf_suffix_census.py`, 2026-10-26), every
attention, GDN and shared-expert tensor in the file is **Q8_0** — `attn_qkv` / `attn_gate` / `ssm_out` (30
GDN layers each), `attn_q` / `attn_k` / `attn_v` / `attn_output` (10 full-attention layers each), and
`ffn_gate_shexp` / `ffn_up_shexp` / `ffn_down_shexp` (40 each) — while routers, every norm and the 1-D GDN
controls (`ssm_alpha`, `ssm_beta`, `ssm_conv1d`) are **F32**. So D-01a is not a requantization decision
about the artifact we run today; there is nothing to change. It governs what our own quantizer must emit,
and what we would choose if we ever built our own file — which makes it cheap to keep and consequential only
in the future tense.

**One correction to how the experts have been described here.** "Experts Q4_K" is true of the gate and up
projections only: `ffn_gate_exps` and `ffn_up_exps` are Q4_K across all 40 layers, but **`ffn_down_exps` is
Q5_K ×37 and Q6_K ×3** in the shipping file. Anyone writing or gating a dequant path must take the width
**per family, not per group** — "the experts are Q4_K" is two-thirds true and would silently price the wrong
kernel.

### Ratification 2026-10-26 — the user's words

> "im fine wiht Q8_0 + fine with the fact that unslosh already did a decigion for me by picking the optimal
> quantization for attention"

**D-01a stands at Q8_0**, re-confirmed with the void support withdrawn and the artifact's own choice placed
in front of the decision. The user's framing is worth preserving: the width was chosen by the party who
actually measured per-tensor sensitivity, so this repo inherits a measurement instead of making an
uninformed pick. The caveat stays inside the decision rather than outside it — "optimal" means optimal for
*their* release, and any future policy-E move still needs its own KLD study and its own must-catch set, not
a re-reading of this paragraph.

**What this decision makes mandatory:** the GDN and attention tensors (40 layers' worth, 1.975 GiB
bf16) need an 8-bit dequant path in the engine, not just the experts — so `specification/` must carry
both widths, and the numerics gates need an 8-bit oracle band as well as a 4-bit one.

## D-02 — Can a kernel read host-mapped memory above the pool? — **DECIDED 2026-10-26 by measurement: no**

Run blocking with the GPU lock held and zero subagents: `tools/hostmap_probe.cpp` (new here), raw
output in `devices/evidence/d02-hostmap-{probe,sweep,multi,ceiling}-2026-10-26.txt`. Headlines, every
line carrying an independent read-checksum so it cannot be an artifact of skipped work:

- pinned + mapped host memory reads at **195–225 GB/s from 1 GiB up to 32 GiB**, including 215.8 GB/s
  for 12 GiB pinned against 198.6 GB/s for 20 GiB of device memory in the same window. Location is
  not the problem.
- the limit is **total, not per allocation**: `hipMalloc` 20 GiB + `hipHostMalloc` 12 GiB is granted
  (32 GiB); `hipMalloc` 20 GiB + `hipHostMalloc` 24 GiB refuses the pinned half; a single 34 GiB pin
  fails in 0.0 s with MemAvailable at 87 GiB, so it is an accounting limit, not memory pressure.
  **Correction the same day:** an earlier line here called it "total per process" and cited two 20 GiB
  pins in one process. That reading is retired — two **separate processes** asking 20 GiB each also
  could not coexist (`tools/pin_pair.sh 20 2`: one grant, one refusal), and 20 GiB < 32 GiB, so the cap
  is not per-process. The pool is **≈55.5–56.5 GiB box-wide and halogen's 23.5 GiB is inside it**
  (23.5 + 32 = 55.5 granted; 23.5 + 34 = 57.5 refused; 23.5 + 20 + 20 = 63.5 refused). See
  `devices/measured-facts.md` §"Residency above the pool" and D-11.
- `hipMalloc` 40 GiB still fails; `hipHostRegister` of 40 GiB answers `invalid argument`.
- an unregistered `malloc` pointer handed to a kernel **faults the queue**, and the checksum witness is
  what stopped the resulting "1,082,301 GB/s" from becoming a number.

Consequences: quantization is a prerequisite (D-01) **at the co-resident budget**; a two-tier device +
pinned layout is bandwidth-viable but spends the pool rather than extending it; and the Flash-Next
endgame (≈58 GiB even at 4-bit main weights) needs a bigger pool or real streaming, not a mapping
trick. What the correction changes is only the size of the ceiling and who else pays it (D-11).

## D-03 — Weight acquisition: bf16 67 GiB, a published GGUF, both, or neither yet? — **DECIDED 2026-10-26 (user): published quant + MTP shards + sampled audit; the full bf16 download is deferred, not cancelled**

The user's approval, in their words: **"I approve the mtp+published+audit, but don't start it."** So the
shipping artifact is a **published quant** — `unsloth/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` — plus the MTP
block fetched separately (the published file provably has none), plus a **sampled provenance audit** of
that file against the checkpoint. ≈ **31 GiB** total, no iGPU time. The queue, per-sub-step pass
conditions and fetch list are in **`PLAN.md` step 1**; nothing there has run.

**This is not what this file recommended**, and the difference is the substance of the decision. The
reasoning below argued for bf16 as the *measuring instrument* first; the user chose to defer that and
buy the artifact instead. What that trade actually costs is itemised in `PLAN.md` §"Step 1 does not
buy" and, in one line each: quantization quality becomes unmeasurable (we inherit the producer's KLD
claim), **two width decisions are overridden by inheritance** — D-08's experts (their `down_proj` is
Q5_K) and D-10's `lm_head` (their Q6_K, which D-10 rejected on tail-risk grounds) — value coverage is
sampled rather than complete, and llama.cpp-plus-file is a closed loop with no third witness. Those
decisions are **not** re-opened as preferences: they stand as what we would choose if we converted our
own, and are annotated as amended in the table above so nobody reads policy A as the shipping bytes.
The deferral carries four named triggers, so the checkpoint is bought by a question rather than by
taste.

The analysis that produced the recommendation stays below, because it is what the triggers refer to.

The question asked back was the right one: **if we ship a quantized model, why download the bf16
checkpoint at all?** Three answers, in order of hardness.

**1. The bf16 checkpoint is the measuring instrument, not the thing we are not shipping.** Every
numerics band in this project is defined as a distance *from the bf16 reference dumps*. With only a
quantized file on disk, a gate can check nothing stronger than "our kernel agrees with somebody else's
Q4_K dequant" — and that check passes for a wrong scale convention, a wrong sub-block order, or an
off-by-one in the 6-bit scale unpack, which are exactly the bugs this repo's trap index is full of. The
D-08 closure written today (KLD + argmax agreement against pinned bf16 logits) is itself unrunnable
without it. `AGENTS.md`'s rule — never loosen a gate to admit a quantized path — presupposes a bf16
oracle.

**2. No published quant matches the policy we decided.** Policy A is experts Q4_K + attention/GDN/shared
Q8_0 + `lm_head`/embeddings Q8_0. Published `Q4_K_M` follows llama.cpp's per-tensor defaults (attention
sits lower, embed/output default below Q8_0 except in the `_L`/`_XL` variants), so gating numerics
against someone else's GGUF would be calibrating bands against a *different width assignment* than we
ship. We need our own quantizer regardless — and its input is the bf16 source.

**3. "Just the part we need" does not exist.** From the committed index: `total_size` 66.97 GiB over 1,045
tensors, of which the text path is **65.72 GiB**; the vision tower is 333 tensors but only ~1.25 GiB, so
excluding non-text saves **1.9 %** of the download. The routed experts alone are 61.5 GiB. There is no
partial-download route that meaningfully reduces it.

**What the download actually costs — RAM, not disk.** Disk is a non-issue (2.4 TiB free, 27 % used; 67 GiB
bf16 + ~20 GiB of our policy-A artifact is noise). The real constraint is the **CPU oracle run**: the
installed `transformers` `qwen3_5_moe` path needs ~66 GiB resident in host RAM, and measured
2026-10-26 while halogen is resident: MemAvailable **86 GiB**, halogen `VmRSS` **73.7 GiB**, swap 46.6 GiB
with 4.3 GiB used. So the oracle run *fits*, with roughly 20 GiB of slack — but it occupies nearly the
whole window, nothing else large may run alongside it, and if it spills the first thing squeezed is the
process serving this session. That is an argument for **sequencing** the download, not for skipping it.

**Recommended shape (staged, and it buys something earlier):**

1. **Now, ~18.5 GiB:** `unsloth/Qwen3.6-35B-A3B-GGUF` `UD-Q4_K_M` (published 3.6 quants exist — unsloth,
   bartowski, lmstudio-community). Purpose: a **perf ruler** (llama.cpp tokens/s on this box is the
   number our engine must beat) and a **real Q4_K data source** to write and test our block reader
   against. Check on arrival whether it carries the `mtp.*` tensors — if it does not, it cannot exercise
   the draft path, which is one more reason it cannot stand in for the source.
2. **Before the first numerics gate:** full `Qwen/Qwen3.6-35B-A3B` bf16, 26 shards / 66.97 GiB, into
   `models/Qwen3.6-35B-A3B`. Produces the pinned reference dumps (the oracle) and our own policy-A
   quantized artifact, and is kept as the oracle for every later band.

If the preference is to download exactly one thing, it is **bf16** — the GGUF is a convenience and the
oracle is not. *(This line was the recommendation; the user's answer was neither of the offered words —
see the decision at the head of this section: published + MTP + audit, with the oracle deferred.)*

**Constraint found later the same day, and it is a harder argument than the oracle one.** Header parse
of `unsloth/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (`tools/gguf_header_types.py`; evidence
`quantization/evidence/q4km-published-width-map-2026-10-26.md`) shows layer indices **0..39 and zero
`mtp` tensors** — a published quant of this model **carries no draft block**. D-05 (MTP) is in scope,
so the MTP weights can only come from the HF checkpoint: this is not about measurement preference, it
is that the artifact does not contain them. (`mtp.*` naming may differ per producer, so this is a
statement about the files checked, not a proof about all GGUFs — but the one we would have used
doesn't have it.)

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

## D-08 — Which 4-bit format for the routed experts? — **DECIDED 2026-10-26: Q4_K** (user)

**Decision, in the user's words: "D8 Q4_K".** Routed experts are quantized **Q4_K** (4.5 bpw, 256-weight
super-blocks of 8×32 with 6-bit scale *and* min per sub-block under fp16 super-block scales). MXFP4 and
IQ4_XS are not taken, and the reasoning is in
`quantization/evidence/mxfp4-vs-q4k-2026-10-26.md` and `quantization/evidence/unsloth-qwen3.5-35b-a3b-2026-10.md`.

**What is fixed, and what stays measurable.** This closes the *format* question: the engine must read the
Q4_K block layout, and a Q4_K writer is required (D-03). It does **not** close the *dequant-cost*
question — MXFP4's one local advantage (one power-of-two scale per block, so a block can accumulate in
integers and take a single scale at the end) is unmeasured, and if a `prototype/moe_gemv` probe ever shows
decode is dequant-bound rather than bandwidth-bound, that finding reopens the format question on evidence.
It also licenses no band loosening: the Q4_K numerics band is bought with a floor probe and
geometry-derived must-catch sets against the bf16 oracle (`pitfalls/30-numerics.md`).

Sourced in `quantization/README.md` and `quantization/evidence/`:

| format | bpw | why it is or is not first |
|---|---|---|
| **Q4_K** | 4.5 | llama.cpp's own `*recommended` default; dequant is a sub-block scale multiply, cheap to write well for `gfx1151`; the block layout (32-element sub-blocks, fp16 scales) is a kernel requirement, not just a size |
| MXFP4 | 4.25 | ships as `MXFP4_MOE` (experts MXFP4, everything else Q8_0 — the same shape as our policy A), and there is a perplexity-only crowd claim that it beats Q4_K_M; but on **this** architecture family the published KLD comparison is worse at equal size (0.7789 @ 18.17 GB vs 0.5478 @ 18.49 GB) and unsloth retired it from mixed quants: "better to use Q4_K than MXFP4 when choosing between them" |
| IQ4_XS | 4.25 | the size-down lever (~1 GiB below Q4_K, better KLD-per-byte than MXFP4), but codebook dequant costs decode speed (~5 % in the published table) and that cost must be measured here, not assumed |

Until a `prototype/moe_gemv` probe measures dequant cost on this GPU, every tok/s figure in this repo
remains a ceiling rather than a forecast — which is the honest reason the format choice is still open.

**Explained properly in `quantization/evidence/mxfp4-vs-q4k-2026-10-26.md`** (2026-10-26). On the quality
question specifically: **perplexity does not separate the two** (MXFP4_MOE 6.600 vs Q4_K_M 6.6053 — the
crowd claim that MXFP4 wins lives entirely inside that 0.005), while **KLD does**: 0.7789 vs 0.5478 at the
99.9th percentile and 0.0272 vs 0.0192 in the mean, i.e. ~42 % more shift away from the bf16 output
distribution for MXFP4, which is the quantity a decoder is actually judged on. Part of that gap is
*tensor policy* rather than *block format* (llama.cpp's MXFP4 recipe keeps more categories at Q8_0), and
what survives the confound is that unsloth retired MXFP4 inside their own mixed recipes. Direction safe,
magnitude theirs — which is why D-08 is recommended rather than decided, and why the closing measurement
is ours to run once the weights land: both formats against the pinned bf16 dumps on one corpus, KLD +
argmax agreement.

The same file carries the part that changes how to think about it: MXFP4 is E2M1 elements in blocks of 32
with one **E8M0** scale,
i.e. a *power-of-two* scale, which is what makes it fast **only where the hardware multiplies blocks
natively** (MI350/CDNA4, Blackwell). On gfx1151 there is no MX matrix instruction, and llama.cpp's own
RDNA3.5 path repacks MXFP4 into a Q8_1 shared-memory tile (`mmq-config-rdna3-5.cuh`), so here it is a
storage format like any other. Priced: policy C (experts MXFP4) is 18.63 GiB / 2.301 GiB/token vs
policy A's 19.59 / 2.331 — **0.96 GiB and ~1.3 % of ceiling**, against a published KLD that is *worse*
(0.7789 vs 0.5478). Its one genuine local advantage is kernel-side: one power-of-two scale per block
means a block can accumulate in integers and take one scale at the end, which Q4_K (non-power-of-two
scales plus a per-sub-block min) cannot do — so if decode ever proves dequant-bound, MXFP4 is worth a
re-look, and the decision instrument is `prototype/moe_gemv`, not another forum post.

## D-09 — Is the MTP block's expert stack quantized like the main one? — **DECIDED 2026-10-26: uniform** (user)

**What "the block's width" asks.** The checkpoint ships one extra decoder layer, `mtp.layers.0.*`, whose
job is to predict one token ahead of the real stack — that is all MTP is. It owns its **own MoE** (256
experts + router, **1.500 GiB bf16**, read once per draft token), its own `self_attn`, a norm, and the
token/hidden fusion projection. D-09 asks only: does that *draft* stack load the **same widths** as the
stack it drafts for (experts Q4_K, attention Q8_0, per D-01a) or wider ones? It is not about kernels —
the draft runs the same dequant and GEMV paths either way; it is about which weight files it reads.

**Why it is a real decision rather than a footnote of D-01.** Draft weights do not trade bytes against
bandwidth the way ordinary weights do; they trade against **acceptance**, and acceptance multiplies the
headline. A draft quantized more aggressively than its target sees a different logit surface, so its
top-1 disagrees with the target more often and each round commits fewer tokens at the same round cost.
The 0.8B learned exactly this shape with `lm_head`: bf16 → Q8 there cost ~1.5 % of acceptance and 28 %
of the gate's decision margin, and the resolution was to keep the sensitive piece wide. A draft at a
different precision from its target is a systematic accept-rate penalty wearing the costume of a
draft-quality one — which is why it must be decided, not inherited.

**Price of each option** (policy A base: 2.331 GiB/token; measured accept 0.678 → ~1.91 tokens/round):

| draft-block width | resident | bytes per draft token | vs policy A |
|---|---|---|---|
| Q4_K experts + Q8_0 attn — **uniform, DECIDED** | 19.59 GiB | 0.0234 GiB | baseline (~1.0 % of a step) |
| Q8_0 experts + Q8_0 attn | 20.17 GiB | 0.0467 GiB | +0.58 GiB resident, ~+2.0 % of a step |
| bf16 for the whole draft block | 20.67 GiB | 0.0834 GiB | +1.08 GiB resident, ~+2.6 % of a step |

**Decision (user, 2026-10-26): uniform.** The draft block loads the same widths as the stack it drafts
for — experts Q4_K, attention Q8_0, router/norms exact. Consequences: no extra weight resident (+0 GiB
against the bf16-draft alternative), one dequant path instead of two, and the acceptance question is
deferred to a measurement rather than pre-paid.

**What this decision licenses us to do later, explicitly:** if measured acceptance under-predicts the
0.8B's 0.678, widening the draft block is the **first** lever to price (2.0 % of a step for Q8 experts,
2.6 % for a whole-bf16 draft block) and it fits either residency budget (D-11). That is a reversal this
decision anticipates, not a contradiction of it; the test is one corpus accept-rate A/B, rendered through
the same chat template on both sides, with the flipped-token adjudication rules in
`pitfalls/30-numerics.md` (the must-catch discipline from the sibling project's MTP work: the flipped
token has to be the runner-up inside one bf16 ULP of the top logit before a single flip counts).

## D-10 — `lm_head` / embedding width — **DECIDED 2026-10-26: Q8_0** (user)

**Decision, in the user's words: "lm_head/embed: Q8_0".** That completes policy A exactly as
`tools/quant_scenarios.py` defines it — experts Q4_K, attention + GDN + shared expert Q8_0, `lm_head`
and embeddings Q8_0, router and norms exact:

- **resident 19.59 GiB** (fits the ~32 GiB co-resident budget with ~12 GiB left for state and KV);
- **2.331 GiB/token**, ceiling **95 / 84 / 76 tok/s** at 237 / 210 / 190 GB/s;
- the Q6_K alternative (policy G, 19.36 GiB, 2.216 GiB/token, ~100 tok/s) is **not taken**: 4 % of
  ceiling against the precedent that this family's `lm_head` is acceptance-sensitive (the 0.8B lost
  ~1.5 % acceptance bf16→Q8 there), and `lm_head` is the piece an MTP draft would want wide.

**What is now fixed** — the shipping width assignment is complete and no width question is open. What
remains open in quantization is only the 4-bit *format* (D-08: Q4_K vs MXFP4/IQ4_XS, worth ~1.3 % of
ceiling and decided by a `prototype/moe_gemv` dequant timing), and the residency question D-11.

## D-11 — Does the engine own the device-visible pool, or share it with halogen? — **OPEN (operational), and it changes what we may give up later**

> **2026-10-26, the blocker moved.** This section's obstacle was "halogen serves this session, so it
> cannot be stopped from here". The main session now runs on `tensorfold` (remote), so stopping halogen
> is reachable via `tools/halogen_gate.sh down` with the GPU lock held — which makes the ≈55 GiB row of
> the table below measurable at last. It is still the user's call which budget to design against; what
> changed is only that the number is obtainable. Queue it as a main-agent run when halogen is free, and
> record it here rather than in a commit message.

Raised by D-02's correction. The pool is ≈55.5–56.5 GiB **box-wide** and `flash_serve` (halogen, the
engine serving this very session) holds ~23.5 GiB of it, which is why a single process stopped at ~32
GiB:

| deployment | engine budget | what fits | consequence |
|---|---|---|---|
| co-resident with halogen (today) | ≈32 GiB | policy A 19.59, policy G 19.36 | 4-bit experts is **mandatory for fit** |
| engine owns the box | ≈55 GiB | Q8-throughout 34.97, Q6-expert variants | 4-bit becomes a **speed** choice: 2.331 vs 2.811 GiB/token, ~95 vs ~79 tok/s |

bf16 (65.72 GiB) fits neither. This does **not** change what we build first — policy A is the plan either
way and the Q4 expert kernels are the same work — but it changes what stays available for quality, and
it moves the Flash-Next endgame from "58 GiB against 32" to "58 against 55": still short, by 3 GiB
instead of 26.

**Needed:** an operational answer about who is resident when the engine runs (halogen cannot be stopped
from inside a session it serves), and ideally one allocation measurement from a root/service context.
Until then **the ~32 GiB co-resident budget is the design budget** and 55 GiB is headroom we have not
been granted.

### Decisions 2026-10-26 — the user's words

Asked whether the design budget stays co-resident, whether the engine may run with halogen stopped, and
about the Q4 focus:

> "I want Q4 for export, because in the future we will be doing the qwen3.8-flash-next and it won't fit in
> any other way, so that's why we are focusing on the Q4 quantization. And I propose to stop halogen when we
> will be running our engine"

**Decided.**

1. **Q4 is strategic, not a consequence of local capacity.** The binding reason is the *next* model —
   `Qwen/Qwen3.8-Flash-Next` will not fit this box at any other width. So the 4-bit expert path is the
   deliverable regardless of what the ≈55 GiB row eventually says, and "Q8-throughout becomes affordable at
   55 GiB" is **not** a fork in the plan. Anyone reading the freed budget as licence to re-open D-01 has
   misread this paragraph: the width decision is anchored to a model we cannot yet run, which is exactly why
   it was never ours to trade away for local comfort.
2. **The engine runs with halogen stopped.** Deployment therefore gets the box: ≈55 GiB, not ≈32.

**Two budgets, and every residency claim must now name which one it means.** This is the part the original
section could not see: D-15 puts a lone worker on halogen, and a halogen worker keeps the container resident,
so *development* on this box is co-resident almost by definition — kernels, gates and correctness checks all
run against ≈32 GiB while the engine itself is designed to load against ≈55. A residency figure quoted
without saying which budget it belongs to is now ambiguous, not merely imprecise.

**What this does to the quality menu.** Halogen-down at deployment makes the 55 GiB options real rather than
hypothetical — policy F (Q8-throughout, 35.16 GiB) and 6-bit expert variants are no longer blocked by fit,
only by numerics: each still needs its floor probe and its geometry-derived must-catch set before adoption
(`pitfalls/30-numerics.md`). They sit behind the Q4 work, not beside it.

**Still open, and it is a measurement rather than a decision.** The ≈55 GiB row is still an inference from
D-02's allocation probes, not an observation: nobody has loaded the 21 GiB artifact with halogen down and
read peak RSS and `process_vmem`. That is queued for the main agent's device window (with the tokens/s ruler,
the restart cost and W7's `UNMEASURED` list), and until it comes back this section's table stays *expected*,
not measured — including the 35.16 GiB figure a deployment decision now leans on.

**The numbers above the block are stale** (19.59 / 34.97 / 65.72 GiB, 2.331 GiB/token) and predate the census
adjudication; correct values are policy A **19.78 GiB**, policy F **35.16 GiB**, text bf16 **64.5604 GiB**
without the draft block and **66.1337 GiB** with it, policy A at **2.5252 GiB/token**. Re-print from
`tools/quant_scenarios.py` rather than transcribing either set by hand (`quantization/evidence/bf16-census-adjudication-2026-10-26.md`).

### Measured 2026-10-26 — the demand half of the row is no longer an inference

First device window (main agent, blocking, zero workers, halogen down;
`devices/evidence/device-window-2026-10-26.md`):

- **Engine footprint: ≈26.9 GiB** running the shipping artifact at 41/41 layers offloaded, single sequence,
  `n_ctx=26214` — `mem_info_gtt_used` 26.86–26.88 GiB over five runs, corroborated by host `MemAvailable`
  falling 27.33 GiB. RSS (1.47 GiB) and `mem_info_vram_used` (1.68 GiB, the 2 GiB dedicated BAR) both read
  low and both are wrong here; `devices/measured-facts.md` §"Measured engine footprint" says why.
- **That is 7.1 GiB above the modelled 19.78 GiB** of policy A. The gap is KV plus compute buffers, and it
  means every budget conclusion drawn from the model alone — including this section's "policy A 19.59/19.78
  fits co-resident with room" — is about **7 GiB optimistic**. 26.9 still fits ≈32 GiB, with ~5 GiB of margin
  rather than ~12.
- **Restart cost measured at last: stop 14 s, ready 17 s.** D-13's "batch, do not oscillate" was written on
  the assumption that a restart might be expensive. It is not, at this scale — so take one window per
  measurement campaign, not one per session.

**What is still not measured:** the ≈55.5–56.5 GiB ceiling remains D-02's allocation-probe result. The sysfs
totals are *not* that number — `gtt_total` = 112.0 GiB, `vram_total` = 2.0 GiB — and must not be used to
"correct" it. Per-process device memory is unavailable on this kernel (no KFD proc files), and the split of
the ≥6.24 GiB non-weight footprint waits for the engine's own allocator.

## D-12 — Can llama.cpp be the oracle? — **ADOPTED as the cross-check oracle; the accuracy oracle stays bf16** (user, 2026-10-26)

**The proposal, in the user's words:** "use the llama.cpp on the particular checkpoint to be the
oracle … on Q4 — essentially on the same quant." Gating our quantized path against llama.cpp running
the same quantization is cheaper than a bf16 reference and, if it worked, would also remove the reason
to download 67 GiB (D-03).

**First finding: llama.cpp on this box already supports this architecture.** The llama.cpp checkout
under the 0.8B project is a **HIP build** (`libggml-hip.so` present, `version: 0.7.0-dev`, commit
`797de63`) whose `src/llama-model.cpp` maps `LLM_ARCH_QWEN35MOE → llama_model_qwen35moe`. Recorded as
a fact about a *version*, not as a path dependency: per the self-containment rule we build our own copy
under `engines/` and pin the version in `tools/PROVENANCE.md` when we do.

**What llama.cpp genuinely can adjudicate — adopted:**

1. **Q4_K block-layout bit-parity.** Dequant is a deterministic operation (`nibble → value → × scale
   (+ min)`), and llama.cpp is an *independent* implementation of it, so requiring our reader to
   reproduce llama.cpp's dequantized output bit-for-bit on the same bytes is a real witness — not
   self-consistency. Disagreement localizes to exactly one of: sub-block ordering, the 6-bit scale/min
   unpack, super-block scale use, or the row mapping of the `[E, N, K]` expert tensors. It is CPU-only,
   needs no device and no bf16 weights, and therefore **becomes the first gate this repo can write** —
   it needs only the ~18.5 GiB published GGUF.
2. **Behavioural cross-check + perf ruler**: the same artifact in both engines, decision-level agreement
   (argmax, top-5) over a corpus, and llama.cpp's tokens/s on this box as the number we must beat.
3. **Policy A artifact production is upstream's problem.** `llama-quantize` supports
   `--tensor-type name=ggml_type`, `--tensor-type-file`, `--output-tensor-type` and
   `--token-embedding-type`, so "experts Q4_K, attention/GDN/shared Q8_0, `lm_head`/embeddings Q8_0" is
   expressible in the stock tool. That deletes "write a quantizer-writer" from our plan; the *reader*
   stays ours, and it is the half that has to be fast.

**What it cannot adjudicate is one thing — but it is the load-bearing one: quantization error itself.**
If both sides carry the same quantization, the error cancels. The pairing can establish "you agree with
llama.cpp" and can never establish "you agree with the model". Consequences: no numeric band can be
*sized* from it (`pitfalls/30` sizes bands against an exact reference), the D-08 closure written today
(KLD + argmax agreement against pinned bf16 logits) is unrunnable, and when our engine and llama.cpp
*disagree* there is nothing to say which side is wrong — the 0.8B's measured chaos floor (two legal
bf16 implementations of the *same* weights agreed only 0.8776 of the way on greedy continuation at
depth) is exactly the kind of noise that needs an anchor to be attributed.

**Second catch, measured narrower than expected.** "Essentially the same quant" is not policy A — but
the delta turned out to be two families, not a policy gulf: header-parsing the published
`UD-Q4_K_M` (evidence `quantization/evidence/q4km-published-width-map-2026-10-26.md`) shows it agrees
with policy A on everything except **expert `down_proj`** (Q5_K/Q6_K where we chose Q4_K) and **`lm_head`**
(Q6_K where we chose Q8_0). And the delta disappears entirely in the arrangement that matters most:
**produce our policy-A GGUF with upstream `llama-quantize` and let llama.cpp read *that*.** Same bytes
on both sides, so engine-vs-engine disagreement is about our kernels only. What that still needs is the
f16 source to quantize from — D-03 again, plus the missing-draft-block fact recorded there.

**Adopted shape (sequencing change, and it is a good one):**

- **llama.cpp = cross-check oracle** (layout bit-parity, behaviour, ruler) — available after ~18.5 GiB,
  and its first gate needs no device time at all.
- **bf16 HF = accuracy oracle, and transient**: download once, produce the pinned dumps and the policy-A
  artifact, then the 67 GiB may be deleted — the dumps (MiB) and the artifact are what we keep. So D-03's
  real cost is one RAM window during the dump run, not a permanent residency.
- If bf16 is skipped permanently, that is a legitimate choice with a name: we ship Q4 knowing it matches
  llama.cpp and not knowing its distance from the model, and every future format question becomes
  unadjudicable without a one-off bf16 rental. Needs the user's words to accept it:
  **"both, delete after"** or **"llama.cpp only"**.

## D-13 — How do agents and tests share the box's two model endpoints? — **DECIDED 2026-10-26 (user)**

User's words: *"4 concurrent children, one of which (most expensive) — halogen + main on tensorfold.
We won't use all 5 concurrent slots on tensorfold — only 4, for stability"*, and `halogen_gate.sh down`
*"may run unattended"*.

**What made it a decision and not a cleanup.** The inherited rule was "never more than 1 subagent at a
time", justified by *"exactly one model endpoint on this box, and every agent shares it"*. That premise
is now false in a way that cuts both ways: the main session moved to `tensorfold`
(`192.168.10.27:8888`, a **remote** box, `/health` → `max_batch_size: 5`), so a worker served by it
costs this iGPU **nothing** and the cap was blocking parallelism for no reason; while `halogen`
(`127.0.0.1:8731`) is a **local** container charging **22.62 GiB** to the device-visible pool
(`rocm-smi --showpids`, 2026-10-26) and decoding at **58.5 t/s** against `tensorfold`'s ~35 — so it is
both the fast endpoint *and* a device hazard, and it needs *more* discipline than it had, not less.

**The decided shape** (full text: `AGENTS.md` §"Two endpoints, three classes of work"):

| class | endpoint | cap | gate rights |
|---|---|---|---|
| main session | `tensorfold` | 1 | owns every timing |
| herd worker | `tensorfold` | ≤3 | correctness-only, ≤60 s, under `gpu_lock` |
| heavy worker | `halogen` | ≤1 ever | correctness-only; never in a timing window |
| measurement | — | — | main agent, zero workers, lock held, **halogen down** |

Three things are load-bearing and are *not* derivable from the table alone:

1. **4 of 5 slots is the margin, not an oversight.** `max_batch_size: 5` is not the cap; main + 3
   workers is. Nobody re-derives it from the health endpoint.
2. **Parallelism is for thinking, the device stays single-tenant.** A measurement still needs zero
   active workers, because a quiet worker can fire a compiler or a gate at any moment.
3. **A `halogen` worker is a device activity.** It holds `/tmp/pi_halogen.lock` for its whole life so
   the down-window can refuse; two of them is never correct, idle box or not.

**Still unmeasured, and deliberately not decided by inference** (both are main-agent runs once halogen
is free; protocol in `pitfalls/00-measurement.md` §"The probe that has to happen before `docker pause`
becomes policy"): the restart cost (`devices/evidence/halogen-restart.log` is empty, so batch one
down-window per session until it says otherwise), and whether `docker pause` isolates a measurement at
all. The pool figure D-11 wants comes out of the same window.

## D-14 — Is a recomputed transcendental a bit-exactness obligation? — **RATIFIED 2026-10-26 by the user** (proposed and implemented by the main agent)

**What triggered it.** `PLAN.md` step 1.4 demanded bit-exactness, zero tolerance, of the F32-stored
families against the bf16 authority widened to F32. The audit (`tools/audit_published_quant.py`,
evidence `quantization/evidence/audit-exactness-2026-10-26.json`) came back **26/27 bit-exact**, the
single outlier being `blk.1.ssm_a` element 21: the file holds `-0.9096216559410095` where `-exp(A_log)`
gives `-0.9096215963363647` under **two** independently computed correctly-rounded routes (a double `exp`
rounded to f32, and a 60-digit `Decimal` `exp`) which agree with each other and with the file on the
other 63 `A_log` elements across layers 0/1/39. Routers, every norm, `dt_bias`/`alpha`/`beta`/`conv1d`,
the V reorder and the fused gate/up split were all bit-exact. The worker **refused to invent a
tolerance** and left the verdict FAIL with the deviation attributed as a near miss — the behaviour this
repo's standing rules exist to produce, and the reason accepting the deviation costs nothing in trust.

**Why zero tolerance was the wrong predicate here, and not merely inconvenient.** `ssm_a` is not a
stored quantity. The HF reference module keeps `A_log` as the parameter and computes
`A = -exp(A_log.float())` at module init, so there is no authoritative bit pattern anywhere in the
checkpoint for `ssm_a` to match — the "authority" is whatever a runtime's `exp` returns. Demanding
bit-exactness therefore demands that unsloth's converter used the *same libm as we did*, which is an
unknowable property of an artifact we are auditing, not a property of it. And it is not an engine
fidelity obligation either: the engine reads `ssm_a` from the GGUF and never recomputes `exp`, so the
last bit of a transcendental we do not evaluate cannot be a fidelity property of our kernels. 1 f32 ulp
at |0.91| is ≈6e-8 relative on a decay-rate control — five orders of magnitude below the bf16 chaos
floor this repo already accepts as background (`pitfalls/30-numerics.md`), so no gate this project can
build would ever resolve it.

**The decided shape** (full text in `PLAN.md` §"Amendment to 1.4"): split the pass.

| | families | predicate |
|---|---|---|
| **1.4a copied** | routers, all norms, `A_log`, `dt_bias`, `alpha`, `beta`, `conv1d`, V reorder, gate/up split | **zero tolerance, unchanged** — any difference here is a copy bug |
| **1.4b one transcendental route** | `ssm_a` via `-exp(A_log)` only | the two reference routes must agree **bit-for-bit with each other**, and the file within **exactly 1 ulp**, ulp from `nextafter` on the value's own exponent |

Four clauses keep this from being the forbidden move ("never loosen a gate to make it pass",
`AGENTS.md`), which is a band sized from what the current build happens to produce: **(1)** the bound is
derived from the *route* — a tensor qualifies only because its documented conversion is a transcendental,
so `A_log` itself, which is copied, stays at zero tolerance; **(2)** there is no epsilon constant and no
general tolerance knob — a future mismatch on any copied family still fails; **(3)** the two-route
agreement is a **FAIL condition, never "within tolerance"**, so the gate cannot absorb a disagreement
between its own references as slack; **(4)** must-catch **grew** — ±2 ulp, sign flip, element permutation
and a wrong-`A_log` read on that same tensor must all still fire, witnessed non-vacuous in the direction
immediately adjacent to the slack, because a 1-ulp rule that has never been shown to reject 2 ulp is not
a 1-ulp rule.

**What stays visible.** The verdict becomes PASS-with-finding, but the original zero-tolerance FAIL line
and the per-element bit patterns stay in `audit-exactness-2026-10-26.json` beside it, the summary must
say "26 bit-exact + 1 within-route" and must not call that element bit-exact, and the pin
(`checkpoints/Qwen3.6-35B-A3B.gguf-pin.json`) keeps the finding — the sha256 already makes a silent
re-upload under the same name a hash mismatch, so nothing is hidden by the flip.

**What this is NOT.** It is not a numerics band and it touches none of the user's width decisions:
D-01/D-01a/D-08/D-09/D-10 are unmodified, no quantized path is admitted by it, and the bands that judge
the Q4_K experts and Q8_0 attention still come from the floor probe plus geometry-derived must-catch
sets. **If the user prefers the stricter reading**, the change is one flag: revert 1.4b, 1.4 reports FAIL
again as report-only, the artifact stays pinnable, and nothing else in the audit moves. Escalated for
ratification rather than assumed, because `AGENTS.md` makes "what counts as a band" the user's question
and this is the nearest thing to one that a structural audit can produce.

**Status 2026-10-26: implemented, ratification still open.** The split is in
`tools/audit_published_quant.py` as `1.4a`/`1.4b`; a fresh re-run of the gate reads `1.4 PASS-with-finding
(zero-tolerance view, kept for the record: FAIL on blk.1.ssm_a)`, with 1.4a at 25 cases / 0 outside, 1.4b
at 64 elements judged → 63 bit-exact + 1 within-route + 0 outside + 0 reference disagreements, and 14
route injections (±2 steps, sign flip, permutation, wrong-`A_log`-order) all firing while ±1 step does
not. The selftest caught a live bug in its own predicate on the way — an inverted bit mapping made
"within one step" mean "within 3.8", a bound that could not fail — which is the reason the spacing is
now witnessed against hand-worked binary neighbours rather than trusted. If the user prefers the flat
predicate, the flag above restores it and 1.4 returns to FAIL.

### Ratification 2026-10-26 — the user's words

> "So, I'm fine with continue using unsloth, it contains very small difference, it is fine to allow such
> small difference during the checks -- it is just one step difference"

**Ratified: the published artifact stays, and 1.4b's route predicate stands.** The reading recorded here —
which the user should correct if it is wrong — is that "such small difference" refers to *this* one-step
deviation on the derived-transcendental route, i.e. the 2 `ssm_a` cases that 1.4b judges. It is **not**
recorded as a general 1-ulp allowance for the checks at large: the 25 copied families keep zero tolerance,
the two-route agreement stays a FAIL condition, and no band anywhere else in this repo inherits a 1-ulp
slack from this sentence.

That reading is not left to good intentions. The predicate's population — exactly
`{blk.0.ssm_a, blk.1.ssm_a}`, 2 cases / 64 elements — is pinned inside
`tools/audit_published_quant.py` itself, so a tensor joining the transcendental category must change
that code and re-run the route injections, rather than be waved in by a doc that says "it is just one
step". **Status of that machine check:** the pin is being added under this decision — expected set
`{blk.0.ssm_a, blk.1.ssm_a}`, failing loudly if the set grows *or shrinks*, with four injections required
to fire (extra case, missing case, a zero-tolerance norm migrating in, clean run). Until it lands, the
narrow reading above is enforced by this paragraph alone and nothing else — which is exactly the gap the
check exists to close.

## D-15 — Which endpoint gets a worker? — **DECIDED 2026-10-26 by the user**

**The user's words, in answer to "what are the two workers doing, and why is none of them using fast
halogen?":**

> "okay, let's add one more rule -- we can use it for jobs like W14 too -- essentially when you are
> starting only one worker, use it"

**The rule.** When the main agent is about to start **exactly one** worker, that worker runs on
`halogen` (`halogen/halogen-qwen3.8-flash-next`) — including mid-size analysis jobs, which the previous
policy text (`AGENTS.md` §"Two endpoints, three classes of work") turned away with "not the quick
lookup". When **two or more** workers are running, `halogen` takes exactly one of them — the single most
expensive job — and the rest go on `tensorfold`. The existing caps are untouched: ≤1 on `halogen`, ≤3 on
`tensorfold`, 4 of the LAN server's 5 streams, the fifth never opened; two `halogen` workers is never
correct.

**What it supersedes.** The "expensive job, not the first job" heuristic. It was a throughput argument
masquerading as a safety rule: it was never unsafe to put a small job on halogen, merely wasteful when a
bigger one might need the slot. The user has now priced that trade differently, and correctly — a slot
sitting idle while a worker runs 35 t/s across the LAN is not free.

**What it does not change, and this is the part that stays load-bearing.**

- **The device window still needs zero active workers and halogen down.** A `halogen` worker is not a
  token stream, it is a co-resident client of the same memory pool, so while it lives the box is not
  measurement-clean and halogen cannot be stopped. Under D-15 the window therefore **waits for** the lone
  worker rather than running beside it. That consequence is accepted, not hidden; if measurements start
  being starved for want of a window, the main agent raises it rather than quietly reverting the rule.
- **The stream lock still covers the worker's whole life** (`/tmp/pi_halogen.lock`, tagged per
  `tools/gpu_lock.sh`), so `tools/halogen_gate.sh admit` can refuse and `down` can see the holder.
- **Subagent device discipline is unchanged**: correctness-only gates under ~60 s, PASS/FAIL, never a
  timing, `gpu_lock` held, no `ms` figure in a report — and a `halogen` worker's device gate still may
  not be run inside a timing window.
- **Continuity still beats throughput on a live run.** D-15 governs where a worker is *started*. An
  already-running `tensorfold` worker is not migrated, because a run's endpoint cannot be changed without
  discarding its context — the arithmetic that made W6 and W13 resumses rather than replacements, and
  which has held four times.

**Applied retroactively? No.** W14 and W15 were both started under the old rule while two workers were
live, which is the correct placement for a two-worker board anyway. D-15 binds from the next launch.

**Lone-worker refinement (user, 2026-10-26, later the same day).** *“If there is only one worker, use halogen
for it.”* The “expensive job, not the first job” heuristic was written for a board with several workers, where
halogen is a scarce slot that must not be spent on a quick lookup. With **one** worker that premise is gone: the
cap is not binding, the queue is empty, and the only cost of using halogen is the 22.62 GiB it charges to the
pool — which is already being paid whenever the container is up. So: **one worker → halogen**; **two or more →
halogen takes exactly one, the single most expensive job**, rest on `tensorfold`. Two `halogen` workers remains
never-correct. The measurement rule is unchanged: a halogen worker still blocks the device window, so a lone
halogen worker means no timings that session — the same trade already recorded above, now with the choice of
endpoint made by the user rather than by me.

**Cap revision (user, 2026-10-26, same day, after two lanes died at their 2 h deadline).**
*“it seems like three tensorfold workers is too much, let's limit it by two, so total will be: one hallogen, one
main on tensorfold, and two workers on tensorfold.”* So the steady state is **1 main + 2 workers on
`tensorfold` + ≤1 on `halogen` = 3 of the LAN server's 5 streams, with 2 slots held in reserve**.

The reserve is not slack to be used when a lane finishes early; it is what keeps the server's own headroom
intact while three streams are generating at once. Two consequences are recorded so nobody rediscovers them:
**a new lane queues when both slots are busy** — the D-15 launch shape never included “start it anyway”; and
**throughput is not what the cap is trading against**. Two lanes were killed today by their own run deadline
after going idle mid-task, and the answer is a shorter deadline plus mandatory periodic checkpoints (both now
in the standing brief), not a third slot. The number of streams is a stability budget; a stalled worker is a
deadline-and-observability problem, and buying a third stream to survive stalls would spend the margin to
paper over a harness assumption that had already failed.

## D-16 — Does a whole-buffer cross-arm check inherit the lane's declared field rule? — **DECIDED 2026-10-26 by the user: A**

**User's words (2026-10-26):** *“A seems to be fine + maybe we can somehow detect that this is rounding
error? We had cases like that in the past, where the numbers are +-1 from each other. I'm a bit tired of
approving cases like that — maybe we can just have common machinery that will always accept +-1 from the
target (only in Q8 and above coding! for Q4 that could be fatal!)”*. The first clause decides this entry; the
rest is **D-17**.

**What A commits to.** `prototype/moe_gather/`'s cross-arm check compares declared-exact fields for bit-equality
(`gate`, `up`) and, for the 18 `field_exact=gate_up` cases, counts-and-reports rather than asserts on `s` and
`raw`. Three conditions ride on it, and they are why this is A-with-teeth rather than a blanket pass:

1. the differing-word census stays **printed** on every run — acceptance is visible, never silent;
2. outside the declared fields and cases, exactness is still required and the count must stay **zero**;
3. accepted divergences must be **attributed**, not merely small — the `expf` glibc-vs-libdevice attribution in
   `work/diag_armdiff.py` is the template. "Small and unexplained" is not a category this lane may hold.

**Original question, kept for the record.** `prototype/moe_gather/` declares `field_exact=gate_up` for 18 cases: the gate and up fields are
held to bit-exactness, `s` and `raw` are not, because they carry an `expf` that this lane declares
counted-and-never-banded (glibc on the host, ROCm libdevice on the device). The lane's cross-arm check —
*host arm and device arm agree word for word* — is written over the **whole** output buffer, ignoring that
declaration. It fails with **6,631 differing words of 306,176**, attributed by
`prototype/moe_gather/work/diag_armdiff.py` as follows: all of it in `s` (206) and `raw` (6,425), only in those
18 cases, while `gate`/`up` differ **0** times between arms and **0** from the oracle in both.

**Why this is the user's and not mine.** The two readings are both defensible and they are not a bug report,
they are a policy choice about what the gate is allowed to bless:

| option | what it means | what it costs |
|---|---|---|
| **A — the check inherits the field rule** | cross-arm exactness is asserted per declared field: exact where declared, counted-and-reported elsewhere. The lane goes green today | the gate stops objecting to *any* divergence in `s`/`raw`, including one that is not libdevice. Mitigation: keep printing the differing-word census, and require it to stay zero outside the 18 cases |
| **B — whole-buffer exactness stands** | the cross-arm check keeps its current teeth; the lane stays red until `expf` matches across arms | a red gate that no kernel change can fix without reimplementing `expf` bit-for-bit on both routes — which buys nothing, since the dumps already settle the numerics (D-14's ruling is the nearest precedent: a recomputed transcendental is not a bit-exactness obligation) |

**Evidence, so the choice is informed.** `prototype/moe_gather/results.md`,
`prototype/moe_gather/evidence/moe-gather-device-2026-10-26.log` (`CHECKS ran=908 failed=1`), and the
attribution artifact above. Nearest precedent: **D-14** (a 1-ulp transcendental difference on two `ssm_a`
cases was ruled acceptable, scoped narrowly to those cases — the scoping is the part that argues for option A
*with the census kept printed*, rather than A as a blanket pass).

**What I will not do.** Not loosen the check to make the gate green — the standing rule is that floors, bands
and must-catch sets are never moved to fit what the current build produces. The lane stays committed at
`3f1b977` with `RESULT FAIL … findings=1` until this is answered. Nothing was papered over, and the red is the
evidence that the check still has teeth.

## Agent-delegated decisions (D-18 onward)

Made by the agent under the user's 2026-10-26 delegation, recorded here in the same form as user
decisions so that a later session can tell who ratified what.

| id | question | status |
|---|---|---|
| D-18 | may an agent decide engineering questions alone? | **DECIDED (user grant, 2026-10-26)** — yes, with attribution; reserved list stays |
| D-19 | consult llama.cpp before deciding? | **DECIDED (user instruction, 2026-10-26)** — yes, adopt when comparable; refuse to import across error budgets |
| D-20 | which Higham form bands an `lm_head` dot? | **DECIDED (user, 2026-10-26)** — γ<sub>2K−1</sub> where products are measurably inexact, γ<sub>K−1</sub> where exact |
| D-21 | under what licence are GDN decode rows judged? | **DECIDED (agent, 2026-10-26)** — `recurrent|pinned`, *conditioned* on an inside-the-arm pad witness |
| D-22 | what does "comparable to llama.cpp" mean? | **DECIDED (agent, 2026-10-26)** — ≥ 48 t/s = 90 % of the measured 53.4 t/s median, identical recorded conditions |
| D-24 | what does M1's device read-back actually verify, and against what denominator? | **DECIDED (agent, 2026-10-26)** — per-run D2H sha256 vs the artifact digest, denominator derived from the plan (`463 stageable + 1 draft-refused`), device checks a superset of check-only; witnessed against the stub, **not yet against real HIP** |
| D-23 | what reduction form does `gdn_sumsq` ship with? | **DECIDED (agent, 2026-10-26)** — pairwise f32 tree; wide fp64 accumulator kept behind a define; the band was right and was not touched |

### D-18 — may an agent decide engineering questions alone? — **DECIDED (user grant, 2026-10-26)**

The user's words: *"I give you permission to apply your own judgement, but write it down in DECISIONS
explicitly as something like you made."* So kernel shapes, reduction order, derivable band **form**,
gate design, file layout, fixture strategy and sequencing are agent-decidable. Two conditions ride
on it. (1) **Attribution is load-bearing**: an agent decision must be labelled `DECIDED (agent, date)`
and never presented as user ratification, because a fabricated user ratification cannot be falsified
from inside the repo and would corrupt every later session that trusts it. (2) **A reserved list
survives the delegation**: shipping widths and quantization shape (D-01, D-01a, D-08), the ±1-ULP
policy and any further input quantization (D-17), project scope (D-04), **relaxing** a band rather
than re-deriving it, replacing pinned fixture material, and stopping/starting halogen. An agent that
convinces itself one of those must bend to reach the goal stops and asks.

### D-19 — consult llama.cpp before deciding? — **DECIDED (user instruction, 2026-10-26)**

The user's words: *"every time take a look at llama.cpp, and if there is a similar thing, apply
decision from there."* Adopted as a step in the protocol, with a boundary that turned out to matter
immediately. llama.cpp is consulted and cited as `path::function`. Where its situation is comparable,
its approach is adopted; where it reaches a good property **by paying in a different error budget**,
importing it would be a category error. Worked both halves on the Q8_0 dot: its
`ggml/src/ggml-cpu/quants.c::ggml_vec_dot_q8_0_q8_0_generic` accumulates `int sumi += x->qs[j]*y->qs[j]`
over 32 terms, so per-term products are **exact** (|sumi| ≤ 32·127² = 516,128) — a genuinely better
property, adopted as the model for bandwidth-side kernels. But it obtains that only because it first
quantizes the **activation** to Q8_0 (`vec_dot_type`; the measured GPU path shares the code via
`ggml/src/ggml-cuda/mmvq.cu`), trading our one-ULP question for a ~2⁻⁸ perturbation of the input.
That trade is D-17 territory, so it was **not** imported, and `llama.cpp` remains a speed and
structure ruler rather than a numerical oracle.

**The rule correcting itself, same day.** The M1 lane found that the GGUF load path cited in its
brief — `ggml/src/ggml-llama.cpp` — **does not exist** in `engines/llama.cpp`; the real sites are
`src/llama-model-loader.cpp::load_all_data`, `::init_mappings`, `::load_data_range` and
`src/llama-model-loader.h::llama_tensor_weight`. Those are adopted as the staging precedent (map or
bulk-read the contiguous tensor region once, validate its size against the header — not per-tensor
re-opens), and the wrong cite is recorded here rather than quietly replaced: the D-19 rule is only
worth having if a cited coordinate is checked to resolve, including by the person receiving the cite.

### D-20 — which Higham form bands an `lm_head` dot? — **DECIDED (user, 2026-10-26), option A**

K = 2048 terms, u = 2⁻²⁴. γ<sub>K−1</sub> = γ<sub>2047</sub> = **1.22025596e-04** is valid only if every
product is exactly representable; γ<sub>2K−1</sub> (=γ<sub>4095</sub> = 2.4414061e-04) is required when
each product also rounds. `floor_probe` measured it rather than arguing: the two encoded tables carry
`weight_max_bits=18 × hidden_max_bits=8` = up to **26 bits**. Over **every** banded word (not a sample):
table 2 = real Q8_0 `lm_head`, **7,727 of 409,600** terms inexact; table 3 = real Q8_0 `embd`, **3,474 of
368,640** inexact; both priced at γ<sub>4095</sub>. Every bf16 family measures `products_inexact=0` and
stays narrow at γ<sub>2047</sub>. Two corrections to this entry as first written, both caught by the
implementing lane rather than by me: 1.22085215e-04 was the value at index **K**, the superseded form, not
at K−1; and the earlier denominators (245,760 / 204,800) were sample-sized. The strongest evidence that
the rule is a measurement and not a label is **table 7 = synthetic Q8_0, which measures EXACT and is
therefore priced narrow** — a "Q8_0 ⇒ wide" rule would have mispriced it and nothing would have noticed. The old claim ("Products are exact; sums are not", `lm_head_kernels.h`) proved
**the wrong product**: the dequant `d·q` is exact at 19 bits, the dot term `h·(d·q)` is not. The user
chose A — adopt γ<sub>2K−1</sub> for the inexact families, keep γ<sub>K−1</sub> for the exact ones —
with four safeguards that make this a *re-derivation* and not the relaxation D-18 forbids: the index is
selected from measured `products_inexact` (no literal 2047/4095), `gate.py` re-derives and compares
`work/band.f64` element-wise so any hand-edit fails, an anti-drift check FAILs if an exact family is
priced with γ<sub>2K−1</sub>, and the doubled band must still reject an injected perturbation. Headroom
before the change: `worst_ratio=0.001646`, `headroom_x=607.6` — roughly 304× after doubling, so no
verdict is expected to move; if any row lands under 5× the work stops and returns to the user.

**As implemented and independently re-verified by the main agent** (`floor_probe.py` → `RESULT PASS lm_head
floor probe cases=66 words=1280 arms=5 worst_ratio=0.0009657`, findings **0** where it was FAIL/2; `gate.py
--check-only` → `RESULT PASS lm_head host-only checks=1517 findings=0`; `--selftest` → `checks=44
findings=0`; `v.check` call sites 63 → 68). Every safeguard was seen firing in the main agent's own run,
not taken from the report: tampering `band.f64` narrower gives `BANDTAMPER mode=unpin-narrow words_changed=380`
and **5 named checks red**, including the element-wise `band == gamma_index(products_inexact) * fsum` check;
one word widened is caught (`band-word-wide`); an injected output perturbation is **still rejected under the
doubled band** (`band-perturb case=16` → 10 red checks), which is what makes this a re-derivation rather than a
loosening; and forcing the wide index onto an exact family is refused (`FORCEDFORM table=0
measured_products_inexact=0` → 8 marker lines). `gamma_index` is never called with a literal index (AST-checked),
`floor_probe` imports the same `gamma_index` the judge uses so instrument and judge cannot drift, and
gpu_lock references are 0 across all three drivers.

### D-21 — under what licence are GDN decode rows judged? — **DECIDED (agent, 2026-10-26)**

Slice 3 proved `chunked|pinned` is **unlicensed** at every decode pass — `gdn_ref.probe_arms_for
("decode","pinned")` → `recurrent,reassoc,pinned`, and §04b.1 gives a cached decode step the recurrent
program, so no chunked probe record exists to license it — while the same pair **is** licensed at prefill
across all 7 stages, which is what makes the 84-row census a measurement rather than a constant.
Three ways out: force a chunked probe at a decode pass (licenses a program the model never dispatches —
rejected: a ruler certified against a referent that does not exist), leave it `QUALIFIED` forever
(rejected: it turns the very red we built the census to catch into wallpaper), or judge the rows under
`recurrent|pinned` — the licence that exists for the program the dispatch actually specifies. **Chosen:
the third, conditionally.** The bridging claim ("a C=64 chunked kernel with 63 zero pads implements the
recurrent single-token step") must itself become witnessed, because chunk pads are synthesised *inside*
the arm and a driver-side mutation cannot move them — so a driver-side pad check could never fire, which
this repo forbids. The decision is therefore gated on one experiment: an arm-internal
`-DGDN_BROKEN_PAD` (same pattern as `host_probe_xpose`'s `-DGDN_BROKEN_STATE_TRANSPOSE`, which moved
8,548,098 bytes on live plans and read `INVISIBLE_AT_ZERO` at zero state) must turn the comparison RED.
If it fires, the 84 rows get credited under `recurrent|pinned`; if it cannot fire, the fallback is
`QUALIFIED` with a named `CANNOT_BE_ESTABLISHED`, never option A. llama.cpp consulted per D-19: it has
no chunked/recurrent duality for a linear-attention state (its GDN path is recurrent-only), so no
precedent to import here.

**Outcome, executed and verified by the main agent (2026-10-26).** The condition was met: an arm-internal
`-DGDN_BROKEN_PAD` moved **4,644,568** bytes on the zero-state prefill plan (`pad_nonzero` 0 → 3712) and
12,027,499 / 12,395,650 on the two decode plans, and `pad_witness_visible_at_zero_state` reads **VISIBLE**
where the state-read control reads `INVISIBLE_AT_ZERO` — pads really are read as data, so the bridge claim
is testable rather than assumed. Crediting therefore proceeded: `LICENCEVERDICT banded_rows=116
credited=74 unlicensed=24 licensed_form_unprobed=18`, the carry scope at **60 of 84 credited** under
`recurrent|pinned`, reconciled by a second independent reader (`carry_rows_credited_eq_licence_file
credited=60 derived_from=geometry+gdn_ref.probe_arms_for+band_file_json`), and `credit_refused_without_pad_witness
credited_with_witness=60 credited_without_witness=0` — **no row can be credited unless the pad witness fired
in that same run**. The 24 still-unlicensed rows stay unlicensed: honest residue, not a rounding error.
`GATE FAIL findings=2` remains, being only the two `q_hat` rows, and `GATE PASS` is still unreachable while
42 banded rows are uncredited.

Two defects had to be fixed to get there, and neither was the lane's fault to catch: (1) the witness ran
**after** the judging loop, so classification saw an unfired witness and credited 0; moving it required
hoisting the honest prefill baseline (`h0`/`cs0`) with it, so the zero-state pad plan and the xpose
comparison share one reference run rather than two copies. (2) `carried = bool(carry[i])` treated
`carry=-1` ("state read from file") as carried, contradicting this gate's own established convention
(`sum(1 for v in carry_idx if v >= 0)`). The obvious one-line fix — `>= 0` — was itself a trap: the
non-carried plan passes a constant per case, so a bare `>= 0` would have reclassified every non-carried
case as carried and produced a *different* wrong must-catch set. Both sides were corrected together
(`[-1] * n` for the non-carried plan), and the must-catch set became `must_catch_rows=30
should_move_but_did_not=0 moved_though_upstream=0`, derived from `STAGE_CONSUMERS` rather than adjusted to
pass. Default-path codegen proven unchanged: 179 `STAGE` lines **byte-identical** to
`evidence/gdn-slice3-gate-2026-10-26.log`, determinism 178/178/178 `IDENTICAL`.

### D-22 — what does "comparable to llama.cpp" mean numerically? — **DECIDED (agent, 2026-10-26)**

The user set the goal as performance "comparable with llama.cpp" without a number, and a goal without a
number is unfalsifiable. Fixed at **≥ 48 t/s** = 90 % of the measured llama.cpp median **53.4 t/s**
(53.4/53.4/54.1, 41/41 offloaded, warm run discarded, `devices/evidence/device-window-2026-10-26b.md
§3`), same model artifact, same recorded prompt, `-n 128`, greedy, seed 42, and the same clean-window
conditions (zero workers, halogen down, `gpu_lock` held, blocking). 90 % chosen because the remaining
10 % is where n=3 sampling noise and prompt-length differences live — the morning window's
44.9/49.1/51.4 is *not* strictly comparable (its prompt was never recorded), so a bar tighter than 90 %
would be measuring our harness rather than our engine. Ceiling context: policy A's roofline is ~95 t/s,
so 48 t/s is 51 % of achievable and llama.cpp sits at 56 %. Two riders: the bar is meaningless until
M3's greedy-argmax gate is green, and any figure quoted outside a clean window is not a measurement.
### D-23 — `gdn_sumsq` reduction form — **DECIDED (agent, 2026-10-26)**

**What was red, and which branch it was.** `layer0/q_hat` and `layer1/q_hat` measured `max_abs =
2.980e-08` against a licensed `2.9000121387623423e-08` while `mean_rel` (7.0e-08 vs 1.157e-07) and
`1-cos` sat inside. The band is a fold of its own probe — every licensed key is `BAND_FACTORS[key]` ×
the worst record in the population, and every record is an arm's distance to the f64 anchor — so the
real test is the arm's **own** distance to that anchor against the population maximum
`1.4500060693811712e-08`, not merely its distance to the pinned bytes. Measured: the shipped
ascending-f32 sum of 128 squares sat at `2.6261e-08` / `2.7572e-08` against the anchor — **outside
every record that licensed the band** — while the reference's own f32 arm sits at `1.0003e-08` /
`1.3538e-08` and the pinned dumps reproduce torch's reduction order bit for bit (`ndiff=0`, so the
program measured is the program that wrote the file). **Branch (b): the kernel was wrong and the
licence is correct and unchanged.** At the worst element (`layer0 h20 t21 d49`) the observed error
decomposes exactly: `2.1287e-08 (arm−anchor) + 8.5153e-09 (reference−anchor) = 2.9802e-08`. Cause:
the 128-term ascending chain accumulated in the stored width is 8 ULP off the correctly-rounded sum
(`0.090579554438591003` against `0.090579614043235779`); the products themselves are exact except for
64 squares underflowing into the f32 subnormals at layer1, so the additions were the only rounding
site.

**What was refused.** Licensing the old kernel needed `2 × 2.6261e-08 ≈ 5.25e-08`, i.e. nearly
doubling a bound — a **relaxation dressed as a re-derivation**, which D-18 reserves to the user. It
was not asked for and not done. The band file, its probe block and its digest
(`27a68c7e82c9aa05…`) are byte-unchanged; `gate.py` now re-folds all 28 licensed numbers from the
probe log's own per-record table (`printed=512 parsed=512 equal=1`, `q_hat folded_records=102`) so
the licensed value is re-derived rather than trusted.

**The decision: pairwise f32 tree, not the wide fp64 accumulator.** llama.cpp was consulted per D-19:
`ggml/src/ggml-cpu/ops.cpp::ggml_compute_forward_rms_norm_f32` reduces a row of squares with
`ggml_float sum = 0.0; sum += (ggml_float)(x[i00] * x[i00]);` and `ggml/src/ggml-cpu/vec.h` typedefs
`ggml_float` to `double` — wide accumulator, ascending, one accumulator, products still f32. A
follow-on lane first shipped exactly that shape, and the main agent overrode the choice on hardware
grounds: `ggml_float` is nearly free on a CPU, whereas this target is a Strix Halo iGPU (`gfx1151`)
with heavily derated fp64 and `gdn_sumsq` sits on the per-token path that has to clear D-22's ≥ 48 t/s
bar. **This is D-19's own boundary applied to D-19's own example**: llama.cpp reaches a good property
(combine with more headroom than the storage offers) by paying in a budget that is not ours, so the
property is adopted and the spelling is not. Both candidates are inside the population *and* the
licence at the **same** `max_abs`, so nothing in the numbers forces the choice:

| spelling of the 128-square sum | `q_hat` vs pinned (licence 2.900e-08) | arm vs f64 anchor (population 1.450e-08) | verdict |
|---|---|---|---|
| ascending f32 accumulator (pre-fix) | 2.980e-08, ndiff 71,750 / 74,172 | 2.626e-08 / 2.757e-08 | **outside both — REJECTED** |
| f32 products, `double` accumulator (llama.cpp shape) | 1.490e-08, ndiff 33,922 / 36,356 | 9.987e-09 / 1.001e-08 | inside both — available behind `GDN_SUMSQ_WIDE_F64` |
| **pairwise f32 tree (shipped)** | **1.490e-08, ndiff 36,830 / 36,242** | **1.263e-08 / 8.311e-09** | **inside both — shipped default** |

The tree is within a factor of ~1.26 of the fp64 form on the anchor distance (further at layer0,
closer at layer1) and identical on `max_abs`. It is a *declared* tree: the association is spelled in
the source (halving levels over the f32 products), the `-ffp-contract=off` / no-fast-math rule still
applies, and both compiled arms still agree at zero tolerance (`PAIR … out.bin 35,143,680 B
ndiff_bytes=0` on both plans). No input's numeric width moved anywhere, so D-17 is not in play.

**How it is checked rather than believed.** `gate.py`'s `qhat_*` checks re-measure all three rows of
that table every run from a numpy ruler that reproduces the compiled arm's `q_hat` words at zero
tolerance (262,144 words), and eight witnesses fire in the same run: `build/host_probe_sumsq`
(`-DGDN_BROKEN_SUMSQ_ASC32`, the pre-fix spelling compiled) is RED at the unchanged band and fails the
population test; a band tampered wider *and* narrower is refused by `load_band`; a
probe-consistent inflation — the one edit `load_band` cannot refuse — is caught only by the pinned
digest; and a perturbation at a rung derived from the band's own `mean_rel` (rung 20 = 8.24×) is still
rejected, and accepted only under an inflated band. Full run: `GATE QUALIFIED … credited=74
uncredited=42`, findings **0**, `named_checks 26 → 40` with `named_checks_red=0`, and 124 of 178
STAGE lines moved with the only two verdict flips being the two `q_hat` rows (`evidence/
gdn-slice4-gate-2026-10-26.log`, `evidence/gdn-slice4-stage-line.diff`).

**Deferred, named and not decided here.** The tree holds `n` f32 partials in a local array (128
words), which is a local-memory question for a device-codegen slice in a way fp64 ALU pressure is
not; the fp64-versus-tree comparison has never been measured on `gfx1151`. `GDN_SUMSQ_WIDE_F64` exists
so that measurement can be made later instead of argued, and the licence does not care which of the
two wins.

## D-17 — May a shared comparator accept ±1 ULP automatically, above some width? — **PRINCIPLE APPROVED by the user 2026-10-26; the guardrails are engineering and are recorded here**

**User's request.** Stop escalating isolated last-bit disagreements for approval: have one piece of machinery
that accepts ±1 from the target — **but only at Q8 and above; for Q4 it could be fatal**. The width veto is the
user's and is structural, not a tunable.

**Why it cannot be written before this is settled.** "±1" is not one concept, and this repo already contains
**two incompatible rulers**:

| where | what it calls one ULP | weakness |
|---|---|---|
| `speccheck/moe_ref.py` §ulp_of | `2^(exp−7)` — a float quantum at `\|v\|`, with the bf16 mantissa width **typed** as 8 bits | edge cases at exact powers of two, subnormals and ±0; and a typed width is exactly the class of error `tools/ggml_block_geom.py` was written to kill |
| `prototype/moe_router/gate.py` §ulpdist_bits | **adjacency in the bit image** — integers differ by 1 after sign-magnitude ordering | sound, but only if the sign transform is right: see `pitfalls/30-numerics.md` §A BF16 ULP comparator written on a `uint16` view has a dead branch, where `np.where(x < 0, …)` is never true on an unsigned array and the gate passes **vacuously** |

Five files carry their own ULP logic (`tools/audit_published_quant.py`, `tools/moe_stats.py`,
`speccheck/moe_ref.py`, `speccheck/selftest_moe.py`, `prototype/moe_router/gate.py`). Checked today: none
currently has the dead-branch bug, but three of the five do no sign-magnitude comparison at all. One ruler,
derived from a file, is therefore the first thing to build; the acceptance rule is the second.

**The rule as built** — the guardrails are mine, and each is enforced by a sabotage that was watched to fire:

1. **Isolated, not systematic.** Accept only a small number of elements at *exactly* 1 ULP, with **zero**
   elements at ≥ 2 ULP. A wrong rounding mode or an off-by-one in a reduction produces ±1 on **every** element,
   which is a different function and not rounding noise. The systematic-shift fixture must be **rejected**; that
   arm is what makes "accept ±1" safe, so it is mandatory rather than optional.
2. **The width veto comes from the artifact, not from a string.** ≥ Q8 is evaluated from the tensor's own ggml
   type in the pinned file (`ggml_block_geom.py` derives geometry, `gguf_width_table.py` is the independent
   ruler). Q4_K/Q5_K/Q6_K are refused **even at 1 ULP**, with a test of the refusal itself — because "which
   quantization is this" is the one thing a policy like this must never take on trust.
3. **Fields are declared per lane and reviewed.** Nothing benefits unless the lane names it; the gather lane's
   18 `field_exact=gate_up` cases are the first users.
4. **NaN is never within 1 ULP of a finite value**, with explicit arms for subnormals, ±0 and infinities. A
   comparator that calls NaN-vs-3.7 a rounding difference is worse than no comparator.
5. **Accepted ≠ uncounted.** Every acceptance prints a census — element counts, fields, the ULP histogram — and
   the run still fails if anything outside the declared set moves at all.

**What this replaces, and what it does not.** It replaces the practice of escalating individual ±1 findings for
approval (the D-14 pattern). D-14 is **not** retroactively widened: its ratification stays scoped to the two
`ssm_a` cases it named. The machinery covers future cases of its own kind; it does not enlarge past decisions.

### D-24 — what does M1's device read-back verify, and against what denominator? — **DECIDED (agent, 2026-10-26)**

**What was wrong.** Window `c` found that `--device` printed `RESULT PASS … verified=464 device=on` while the
evidence carried **no device-side digest at all** (`grep -cE "dev_sha256|device_sha256|readback_sha256|dev_digest"`
→ 0 over 290,067 B), `controls=0 fired=0`, and its only device check was answered by construction. M1's exit gate
in `AGENTS.md` is per-tensor read-back digests; it had never been executed.

**The four decisions, all agent-owned under D-18.**
1. **Read-back protocol (a D-19 addendum).** For every staging run: copy in, `hipMemcpyDtoH` the region back,
   sha256 what came back, compare against the digest the artifact region independently produces, and write that
   digest into the evidence record for that run. llama.cpp consulted per D-19: `src/llama-model-loader.cpp::load_all_data`
   and `::init_mappings` validate sizes against the header and then **trust the mapping** — it has no read-back,
   because on its path a short copy is a load error. Our path is an artifact-to-device copy through our own staging
   slab, where a silent short copy is invisible to every other check in the driver, so the property is adopted and
   the trust is not.
2. **The denominator is derived, never typed.** `stageable = [r for r in plan if slab(r.slab).source == "gguf"]`;
   `expected = len(stageable)`. Today that is `463 stageable + 1 draft-refused`: the 464th run is the MTP draft
   block, `draft:draft-requantizer-absent`, 1,689,281,536 B, recorded with `dev_read=0 readback_sha256=-` rather
   than deleted. The literal `readback_count == 464` in the task brief is unsatisfiable while `loader.stage_draft`
   refuses draft-expert addressing, and the honest answer to an unsatisfiable literal is to derive the population
   and print both halves — not to relax the count to whatever the build happens to produce.
3. **A short read is judged by bytes returned vs bytes requested**, not by a return code, and `device-short-d2h`
   is a control that must redden it.
4. **Device mode must be a check-superset of `--check-only`** (`device_mode_checks_superset_of_check_only`). The
   old inversion — device running 31 checks where check-only ran 34 — is the shape a "more rigorous" mode takes
   when it silently drops the checks it cannot satisfy.

**Witnessed, and where the witness stops.** Against the Python stub: `RESULT PASS mode=device checks=40
findings=0 controls=10 fired=10 run_set=464 plan=464 readback=463/463 verified=464 residue=0 uncovered=0
device=on`, and the window-`c` grep now returns 486 matches, 477 `READBACK` records, 33 `fired=RED`. The two
corruption controls corrupt a device buffer and must restore it — a control that cannot restore exits the run
red rather than leaving a slab wrong. **The real HIP path has still never executed**: real per-run digests of
device memory, real 1-byte corruption and the 464-run device plan are untested on hardware until the main agent
runs the window. Every device check above is a stub witness, and no sentence in this repo may imply otherwise
until that run exists.

**Also verified here, because a prohibition check is the easiest thing in the repo to fake.** `copy-path-owns-no-lock`
passes when `LOCK_INVOKED` and `HIP_IMPORT` find nothing in the driver — and `LOCK_INVOKED` is written as a split
literal (`r"gpu_" r"lock\.sh"`) so the file cannot self-match, which also means a broken pattern would print the
same PASS. `engine/probe_lock_check_vacuity.py` feeds each pattern probes it must fire on: 9/9 correct
(`gpu_lock.sh`, `import hip`, `from rocm import util`, `import amdsmi` fire; `import ctypes`,
`HIP_LIBRARY = "/opt/rocm/lib/…"` and the lockdir path do not), and neither fires on the driver itself. Note
`import ctypes` is deliberately **not** prohibited — the HIP library is `dlopen`ed only inside `HipStaging.__init__`,
which the CPU path never constructs — so the prohibition targets importing a HIP/ROCm module, not touching ctypes.

### D-25 — what the M2 device arm ships, and what a log is allowed to prove — **DECIDED (agent, 2026-10-26)**

Six agent decisions from the M2 device-arm slice, consolidated here from `engine/HANDOFF.md` §`DECIDED (agent,
2026-10-26)`, which holds the full reasoning. None is a user decision; none touches a band, factor or licence.

1. **Width veto compares declared types, not materialized dtypes.** `Loader.handle(name)["type_name"]` against
   `PolicyA.WIDTH` — a Q8_0 tensor legitimately materialises as `float32` after dequant, so comparing a torch dtype to
   a declared quant type can only ever fail (this was a live checker bug, not a hypothetical).
2. **Activation dtype comes from `text_config.dtype`, not from a band.** `topk_weights` is banding-exempt because its
   dtype is declared by the model; the rejected alternative was to band `topk_weights`, which would have manufactured
   a tolerance for a fact the config already states.
3. **Device claims require a device; a log can only be a precondition.** `--device` refuses with
   `tag=live-device-path-absent`; evidence-parsing lives in `--staged-evidence` and prints `PRECONDITION … ADMITTED`.
   See `pitfalls/40` §"A device claim assembled by parsing another tool's log".
4. **The residual-glue rows stay emitted and unlicensed** with `referent=absent`, because no manifest pins the layer-0
   residual stream. Creating that fixture is reserved to the user, so M2's terminal verdict is `QUALIFIED`, not `PASS`.
5. **There is no controls-only mode in this gate.** The old `--selftest` starved the controls of the real plan; the
   controls now run inside the real witness, which is why `controls=9 fired=9` means something.
6. **Compiled-pair zero tolerance extends to the device TUs** — `PAIR … record.f32 bytes=22,532 ndiff_bytes=0` and
   `PAIR … router_record.f32 bytes=3,232 ndiff_bytes=0`, host twin vs `hipcc` arm, `-ffp-contract=off` both sides.

**The correction that matters more than the decisions.** An earlier commit (`890fb88`) cited
`TOPKFORM form=weights-f32-carry ndiff=0` as a leg confirming prediction 4 (cast-before-multiply). It is not a leg:
that value is produced by `engine/block_forward.py`, the same torch program that produced the
`layer0.topk_weights_f32` referent, so `ndiff=0` is an identity and not a test. The independent leg is
`layer0.routed_sum` **refusing** the f32 carry — `ndiff 1326, max_ulp 32416` against cast-first's `0` on the same 2048
pinned values (`engine/HANDOFF.md` §`DISCLOSURE-TOPKFORM-F32-CARRY`, cited verbatim there). What the device arm adds is
three genuinely distinct forms inside one record: `RTFORM` cast-first `ndiff=0`, f32-carry `ndiff=8 max_ulp=2`
(which `prototype/moe_router/moe_router_kernels.h` §R5 predicts for an independent fp32 route and does **not** require
to be zero), leak-form `ndiff=4 max_ulp=1`. A row that reads 0 is enforcing R5, not landing on 0 by coincidence — and
the arm's `topk_ids`/`topk_weights` rows are ACCEPTED at `ndiff=0` under the spec §10.6 exactness licence, the first
licensed numerical evidence from a device-side arm.

Also recorded because it is the interesting negative: a bf16 round-trip of the router span changes **0 of 524,288**
values (`RTVOID … verdict=VOID-ON-THIS-ARTIFACT`), so the "lossy copy" control could never have fired and was
reported rather than kept; the F32 premise is carried structurally (loader type + element count + byte length + the
plan's `elt!=4` refusal) and, for row sensitivity, by a one-ULP low-bit flip that moves **246 of 256** fp32 logits in
the arm's own `logits_f32` while the bf16 `router_logits` row moves 0 of 256 — bf16 rounding hides 2⁻²⁴, which is
exactly why the fp32 field carries that claim.

### D-25 (agent, 2026-10-26) — the pinned fixture set is per-block synthetic-stimulus; a chained forward is added additively

**The finding, first, because it is bigger than the decision.** While checking whether M2's
`block_output` referent was missing or merely renamed (the hypothesis being that `layer1.input`
*is* layer 0's block output in a decoder loop), the bytes refuted it: `layer0.input`,
`layer1.input` and `layer39.input` are **byte-identical**, sha256
`f82d402d574a785b0d310a043ff6f3d8d6a03c19bd4ef46f3fb5b48f1f052ac3` (recorded in
`checkpoints/dump-manifest-2026-10-26.json` as `source_tensor: make_input(seed, T)`,
`source_shard: "seeded fixture, not a shard"`). The dumper feeds **every** block the same
synthetic tensor independently — it never chained. The manifest holds 104 records over
`layer0, layer1, layer39, mtp0` and **no** embedding, final-norm, `lm_head` or `logits`
record of any kind.

Consequences, stated plainly: (i) M2's two `stage=block_output` rows were honestly
`referent=absent`, not mislabelled; (ii) nothing pinned so far has ever validated the residual
chain, so no result to date says anything about chaining, only about per-block behaviour under
one synthetic stimulus; (iii) **M3 as written — pinned logits and greedy argmax — was
unreachable with the current fixture set**, and would have been discovered as blocked only when
M3 started.

**Why this is not the reserved act.** D-18 reserves *replacing or adding* a pinned fixture, and
I had filed this under that for three turns. The reservation exists so that the agent cannot
choose what "correct" means. That is not what is happening here: the oracle is installed
`transformers` (`qwen3_5_moe`) running the real checkpoint, the same program that produced the
existing records, and I cannot manufacture its outputs. What is being chosen is **coverage of a
stage that already exists in the model**, not the semantics of correctness. The user's standing
grant (D-18: "apply your own judgement, but write it down") and their words on 2026-10-26 —
*"So... you have a question that I should approve? I thought that I give you authority to judge
and continue"* — settle the remainder.

**Decision.** Extend `tools/dump_reference.py` with an **additive** chained-forward mode.
The existing per-block fixtures and manifest stay exactly as they are and remain the authority
for every per-block stage already licensed against them.

**Guards, because an agent that generates its own answer key needs more scrutiny than one that
reads it.**

1. **Append-only, write-once.** New dump directory and a new manifest filename; the mode refuses
   to run if either exists. Existing `data/dumps/2026-10-26/` and
   `checkpoints/dump-manifest-2026-10-26.json` are never opened for writing.
2. **Chaining must be auditable from the manifest, not assumed.** Every chained record carries
   `input_from: {file, sha256}` naming the predecessor it was read from; per-block residency is
   kept (one block loaded at a time) so peak host RAM stays near one block plus embeddings and
   `lm_head`, not 66 GiB.
3. **`inputs_not_collapsed`** — the 40 block inputs must be pairwise distinct by digest. This is
   the check that catches the defect found here, so it must be witnessed red against the
   **existing** manifest: the current per-block set is the negative control, which makes the
   check non-vacuous and the finding permanent in one stroke.
4. **`chained_input_equals_previous_output`** bit-exact, verified from the files on disk.
5. **Perturbation witness** — re-run the chain with one weight in a middle block perturbed; every
   downstream input must change and no upstream input may.
6. **Two implementations still** — chained stage values are re-derived with the independent
   oracles (`speccheck/moe_ref.py`, `speccheck/gdn_ref.py`) and compared; the dumper is never its
   own oracle.
7. **Bands are derived, never fitted** — M2's `block_output` band comes from a floor probe under
   the D-20 rule, and any widening later is a re-derivation with witnessed rejection.

**What this does not license.** It does not retire the existing 178/104-record pins, does not
re-price any existing band, does not authorize quantizing anything beyond D-01/D-01a, and does
not make a chained fixture authoritative for a stage without its own floor probe. If a chained
record ever **disagrees** with an already-pinned per-block record where the two are supposed to
coincide, that is a finding to be reported, not a discrepancy to be absorbed.

**Cost note.** Host has 94.1 GiB available and 32 cores; the CPU oracle path already runs the
installed `transformers` forward without the weights on the GPU (`speccheck/results.md`). A
full 40-block chain at `T=32` is CPU-only work and is timed only in host wall-clock — no device
rate enters any document from it.

**Outcome (P1, 2026-10-27, verified by the main agent, not by the lane's word).**
`tools/dump_chain.py` produced `checkpoints/dump-manifest-chain-2026-10-26.json` (28 records,
4 blocks `layer0→layer1→layer38→layer39`) and `data/dumps/2026-10-26-chain/`. The main agent
re-audited it with an **independent** implementation (`tools/verify_chain_d25.py`, which imports
neither the dumper nor its gate): all 28 record digests and lengths re-hashed clean, chain
equality verified from files on disk (3/3 pairs), block inputs pairwise distinct (4/4), the
**same predicate rejecting the old fixture** (3 blocks → 1 distinct digest), `missing_layers`
re-derived from the safetensors index as 36 layers `2..37` and matching the manifest exactly,
`contiguous:false` on all 28 records with none claiming `true`, and the old pins byte-identical
(`f82d402d…`) so the run wrote nothing it should not. Witnesses re-hashed rather than read: the
`layer38` `down_proj` perturbation changes `layer38.out`, `layer39.x` and `layer39.out` and
leaves every upstream input untouched; zeroing one substituted `down_proj` row changes 9 records
starting at `layer38.moe_out` and leaves 19 unchanged, which is what makes the substitution
load-bearing rather than decorative; and the unmodified-HF materialised peer agrees with the
meta-built chain on **28/28** digests. Measured cost: full 40 blocks ≈2.5 min warm / ≈4.5 min
cold, ≈97 % of it safetensors materialisation — so P2 is download-bound, not compute-bound.
Two audit findings in the first pass were **probe bugs, not artifact bugs** (my probe looked for
`missing_layers` at top level instead of under `block_set`, and for the seed in record names
instead of `manifest.input`); one printed `OK` beside a hardcoded message claiming the opposite,
which is the same class of defect this repo bans in gates and which is why verdict strings are
never cited when bytes can be re-hashed. The chain remains **non-contiguous** and is not the
model's post-layer-37 activation; M3 still needs shards 3–24.

### D-26 (user, 2026-10-27) — llama.cpp is the sole oracle; HF is retained as a comparison instrument on the four acquired blocks

**The user's words:** *"sole oracle, but for first 4 block out of 40 that we downloaded, we should compare all that
we can"* — following *"why you cannot use quantized gguf and llama.cpp on top of it?"* and *"I propose to use
llama.cpp as oracle and inject the code that will deliver us the intermediate results"*.

**Decision.** `engines/llama.cpp` (vendored `d327bf1f`, tag `b11869`), instrumented to dump named graph tensors
(`tools/llama-instrument.patch` (planned), CPU-only build), is the **sole oracle** for engine acceptance across all 40 blocks.
Installed `transformers` on the original safetensors is retained — on `layer0`, `layer1`, `layer38`, `layer39`, the
four blocks whose shards are on this box — as a **comparison instrument**, and every stage that can be compared
there must be (three-way: HF × llama.cpp × engine).

**This clause outranks everything below it.** Adopting a quantized oracle may only ever **add** checks. No band,
floor, factor or must-catch set changes because llama.cpp would tolerate more; where an HF-derived limit is tighter,
the HF-derived limit stays. "Never loosen a gate to make it pass" is not rescoped by changing the oracle, and any
band change is still a re-derivation with witnessed rejection (D-18/D-20). An agent quoting D-26 to justify a
looser number has misread it.

**A premise I asserted and then falsified, recorded so it cannot be re-imported.** I first wrote that our engine
and llama.cpp share a common mode because *"the GGUF my own converter produced"* sits between them, and made a
conversion audit mandatory on that basis. **False.** The artifact is a third-party published quantization:
`models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, sha256 `ac0e2c11…53e31a61`, read twice identically and equal
to the HF LFS etag (`quantization/evidence/audit-published-quant-2026-10-26.md`). We never converted it. There is
no converter step to audit, and a check written against a step that does not exist cannot fail — which is the same
vacuity class this repo bans. It is replaced by the real requirement in clause 2.

**Named consequences, accepted knowingly.**

1. **Cross-checked is not licensed.** For layers 2–37 there is no unquantized reference on this box, so a policy-A
   row's limit there is a *cross-check* limit — divergence observed against another quantized implementation — and
   must print as `licence=llama-crosscheck(...)`, never as `floor-probe(...)`. A quantized reference prices
   disagreement, not quantization error. Rows already licensed from HF floor probes (`prototype/moe_router`,
   `prototype/gdn`, the act-domain band) keep their stronger provenance.
2. **The oracle must read the bytes we ship.** The published quant is unsloth **UD-Q4_K_M**, which is *not* policy A
   (D-01/D-01a/D-08): llama.cpp reading that file would make it an oracle over a different quantization than we
   ship, and every diff would then mix an implementation difference with a width difference. The fix is the one
   already planned in `quantization/evidence/q4km-published-width-map-2026-10-26.md`: produce the policy-A GGUF with
   **upstream** `llama-quantize --tensor-type-file` (per-tensor type control CONFIRMED in
   `quantization/evidence/llama-cpp-build-and-flags-2026-10-26.md` §Verdict 1) and have llama.cpp read *that*, so
   both sides consume identical bytes and the oracle's only remaining freedom is the forward pass. The audit that
   replaces the fictitious one is therefore a **width audit**: every tensor in the produced GGUF must carry the
   type policy A specifies, derived from the policy table, with a deliberate mis-typed tensor that must redden it.
3. **The prediction-4 precedent is why clause 1 is worded that way.** HF casts renormalised top-k weights to the
   logits dtype *before* the expert multiply; our engine carried them in f32 and was wrong. Had llama.cpp been the
   oracle from the start and carried them in float, it would have **licensed the defect** — the pin that caught it
   is `layer0.routed_sum` refusing the f32 carry (`ndiff 1326` vs `0`). So on the four HF blocks, whenever HF and
   llama.cpp disagree, silently adopting llama.cpp's side is forbidden: the disagreement is a finding to report to
   the user and the tighter limit ships.
4. **Determinism is a precondition.** Two identical llama.cpp runs must be bit-identical or the instrumentation lane
   must say they are not, in writing; a non-deterministic oracle changes what a diff means.
5. **Unestablished implementation.** llama.cpp's correctness for this architecture is not established by anything we
   control — it already *guesses* a reasoning template for this checkpoint (smoke doc) and its GDN state handling,
   rope placement and MoE multiply order are its own. Sole-oracle status is a decision, not a finding, and clause 3
   is the tripwire for it.

**Unchanged.** D-12's separation of roles survives in substance; the ≥ 48 t/s bar (D-22) is still measured against
llama.cpp under identical conditions; D-25's chain fixture and its `contiguous: false` disclosure still stand, and
nothing here makes the 4-block chain an authority for the other 36.

### D-27 (user, 2026-10-27) — ship the published UD-Q4_K_M; self-produced policy-A weights deferred, reopenable only on a measured performance reason

**The user's words:** *"do a) now, documented that we potentially would like to do b somewhere in the future if
there will be a performance reason for that"*.

**Decision.** The shipping artifact is the published, sha-pinned third-party quantization:
`models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, `ac0e2c11…53e31a61` (audited in
`quantization/evidence/audit-published-quant-2026-10-26.md`, width map in
`quantization/evidence/q4km-published-width-map-2026-10-26.md`). No download and no `llama-quantize` step. The
instrumented llama.cpp (D-26) reads **the same file** our engine reads, which is what makes the same-bytes
comparison in D-26 clause 2 achievable today.

**A premise I asserted and then falsified, second instance this session.** Writing the previous entry I stated
that the published file's attention was "Q6_K-class" and that D-01a's Q8_0 therefore no longer described reality.
**False** — checked against the file's own tensor-info header: routed experts gate/up **Q4_K** ✓, shared expert
**Q8_0** ✓, full attention q/k/v/output **Q8_0** ✓, GDN `attn_qkv` / `attn_gate` / `ssm_out` **Q8_0** ✓,
embeddings **Q8_0** ✓. Policy A and the artifact differ on **exactly two families**, both already handled:

| family | published | policy A | effect |
|---|---|---|---|
| expert `down_proj` | **Q5_K** ×37, **Q6_K** ×3 | Q4_K | wider; ~1.3 GiB above our arithmetic. The loader's floor rule admits *wider* without a waiver |
| `lm_head` (`output`) | **Q6_K** | Q8_0 | narrower; 6.5625 bpw, 1.0037 GiB stored vs 1.2960 modelled — the existing explicit waiver |

So D-01/D-01a still describe what ships, everywhere except those two families, and `engine/loader.py` already
carries both cases (`accepted=("Q6_K",)` for `lm_head`; the wider-`down_proj` floor rule). Nothing is re-labelled
and no band moves.

**The audit that D-26 clause 2 needs is therefore a conformance audit, not a production step.** It must parse the
artifact's own tensor-info header, derive the per-family type map, and compare it to the width map above — with
populations derived from the file, never typed, and a deliberately mis-typed record that must redden. It must also
assert that the engine's loader agrees with the file per tensor, so a stale hard-coded width anywhere in the engine
surfaces as a finding rather than as a silent numeric difference.

**On (b) — deferred, with a trigger so deferral cannot become silence.** Producing policy-A weights ourselves
(`llama-quantize --tensor-type-file`, per-tensor type control CONFIRMED in
`quantization/evidence/llama-cpp-build-and-flags-2026-10-26.md` §Verdict 1) is **not** to be done for tidiness,
bandwidth or principle. It is reopened **only** if a measured performance shortfall at M5 (below the ≥ 48 t/s bar
of D-22) is attributed to a width choice — the plausible candidates being `down_proj` at Q5_K/Q6_K rather than
Q4_K (≈1.3 GiB, hence ≈1.3 GiB/token of extra reads on a bandwidth-bound path) or `lm_head` at Q6_K. Attribution
must be arithmetic against the per-token read budget, not vibes. Cost if ever taken: ~67 GiB of weights plus a
quantize run, against 2.4 TB free. Reopening it is a user decision again, because it changes what ships.

**Unchanged by this entry.** D-26's outranking clause (a quantized oracle may only add checks; never loosen a
band), the `licence=llama-crosscheck(...)` spelling for layers 2–37, the four HF blocks as comparison instrument
with disagreement escalated rather than absorbed, oracle determinism as a precondition, and the ≥ 48 t/s bar.

