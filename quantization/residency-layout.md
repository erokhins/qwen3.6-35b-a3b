# Residency layout — does policy A fit, and where does it stop fitting?

Status: **analysis, not measurement.** Everything here is arithmetic over three committed sources
(`checkpoints/Qwen3.6-35B-A3B.config.json`, `checkpoints/Qwen3.6-35B-A3B.safetensors.index.json`,
`checkpoints/Qwen3.6-35B-A3B.gguf-pin.json`) plus the pinned artifact's own header, against device
numbers that are already in this repo (`devices/measured-facts.md`, `pitfalls/10-device-memory.md`).
No device run was made to produce any figure in this file, and no figure here is estimated. The
calculator is `tools/residency_budget.py` (`--selftest` passes 17 witnesses and fires on 9 injected
faults); this file is what it says and what it means for step 3.

Why it is written now: step 1 pinned the artifact, step 3 will lay out residency, and this is the
first model in this lineage where **capacity binds** — bf16 text weights are 66.13 GiB
(§5 finding 1) against a box-wide device-visible pool of 55.5–56.5 GiB. An allocator that discovers
this at 128k discovers it after the layout decision.

Reproduce with:

```
python3 tools/residency_budget.py                      # table + verdicts, exit != 0 if a row cannot fit
python3 tools/residency_budget.py --frontier --give-up # the concurrency grid and the savings ladder
python3 tools/residency_budget.py --selftest           # witnesses + fault injections
```

---

## 1. The two budgets, written as brackets

`devices/measured-facts.md` §"Device-visible memory" (one `hipMalloc` of 32 GiB ok / 40 GiB fails;
8 × 8 GiB → 4th ok, 5th fails; 24 × 2 GiB → 17 ok = 34 GiB; `GPU_MAX_HEAP_SIZE`/`GPU_MAX_ALLOC_SIZE`
do nothing) and §"Residency above the pool" (23.5 + 32 granted, 23.5 + 34 refused, two processes
asking 20 GiB each could not coexist) give one model: a **box-wide pool of ≈55.5–56.5 GiB which a
co-resident server is part of**. `totalGlobalMem` reporting ~110 GiB is not a budget
(`pitfalls/10-device-memory.md` §"The device has no VRAM").

| deployment | arithmetic | budget |
|---|---|---|
| co-resident with halogen (today) | pool 55.5–56.5 − flash_serve 22.62–23.66 | **31.84 – 33.88 GiB** |
| engine owns the box | pool, halogen stopped | **≥ 55.5 GiB, unmeasured** (D-11) |

Two things follow from writing it as a bracket instead of "≈32":

- **A row between 31.84 and 33.88 GiB has no determinate answer today.** The pool bound is razor-thin
  (55.5 granted / 57.5 refused) and halogen's footprint has two live readings — 22.62 GiB
  (`rocm-smi --showpids`) and 23.53–23.66 GiB (`gtt_used`), which `devices/measured-facts.md`
  §"The footprint has two readings" rightly refuses to average. So the calculator prints a distinct
  verdict for that band instead of rounding it away.
- **Design rule for step 3: choose a context/concurrency budget whose worst case is under 31.84
  GiB**, not under 32, and allocate the pool once at that size. The alternative is learning which
  side of the bracket the box is on from a failed allocation at runtime.

Host-pinning is not an escape: pinned host memory is charged to the same pool and reads at ≥ device
speed (215.8 pinned vs 198.6 device GB/s in one window, `devices/measured-facts.md` §"Residency
above the pool"), so a two-tier layout is bandwidth-viable and capacity-neutral. That is D-02, and
it is why this file talks about *residency* rather than *placement*.

## 2. Per-component residency, derived

Policy A = routed experts Q4_K (D-01/D-08), attention + GDN + shared expert Q8_0 (D-01a), `lm_head`
and embeddings Q8_0 (D-10), router/norms/GDN controls exact, draft block uniform with the stack it
drafts for (D-09). Widths are `tools/ggml_block_geom.py`'s, derived from the pinned `ggml-common.h`
structs. Elements come from the 712 text tensor names in the shard index × config-derived shapes;
that census is witnessed three independent ways (§6).

| component | bf16 elements | policy-A bytes | what fixes the number |
|---|---|---|---|
| routed experts (41 × 256) | 33,017,561,088 | **17.2969 GiB** | `[256,1024,2048]` + `[256,2048,512]` per layer × 41 (40 + MTP) |
| GDN matrices (30 layers: `in_proj_qkv/z/a/b`, `out_proj`) | 1,010,565,120 | **1.0000 GiB** | qkv rows = `2·h_k·d_k + h_v·d_v` = 8192 (verified against the shard header) |
| token embeddings | 508,559,360 | 0.5032 GiB | `[248320, 2048]`, Q8_0 per D-10 |
| `lm_head` (**untied**) | 508,559,360 | 0.5032 GiB | `tie_word_embeddings: false` — it is a second full table |
| full attention (10 layers + the draft's) | 299,898,368 | 0.2968 GiB | `q_proj` is `[8192,2048]` because `attn_output_gate` doubles it |
| shared experts (41) | 129,058,816 | 0.1277 GiB | `shared_expert_intermediate_size` 512 |
| routers (41) | 21,495,808 | 0.0400 GiB | `[256,2048]`, never quantized |
| norms + `mtp.fc` | 8,564,736 | 0.0160 GiB | 86 norm tensors + the draft's `[2048,4096]` fusion projection |
| GDN controls (`conv1d`, `A_log`, `dt_bias`, `ssm_norm`) | 988,800 | 0.0018 GiB | exact per `quantization/README.md` §"How a quantized path gets gated" item 3 |
| **weights, main stack** | | **19.3170 GiB** | |
| **MTP draft block (its own row)** | 844,640,768 | **0.4686 GiB** | see below |
| **WEIGHTS TOTAL** | | **19.7856 GiB** | |
| KV per token | 1024 el/token/layer | 20,480 B/token | `2(K,V) × 2 kv-heads × 256 dim × 10 full-attn layers × 2 B` |
| draft KV per token | 1024 el/token | 2,048 B/token | the draft block is a full-attention layer, so it keeps its own rows |
| GDN recurrent state | 524,288 el/layer | 60.00 MiB/sequence | `[h_v,d_k,d_v] = [32,128,128]` × 30 layers × 32 bits (`mamba_ssm_dtype: float32`) |
| GDN conv window | 24,576 el/layer | 1.406 MiB/sequence | `conv_dim 8192 × (k_conv 4 − 1)` × 30 layers × 16 bits |
| logits buffers | | 1.9 MiB | `vocab 248320 × f32`, one for the target pass and one for the draft |
| **per-token read** | | **2.5260 GiB** | every non-expert family + 8/256 of the routed experts |

`specification/04-gdn-linear-attention.md` §04.10 (written by the spec worker, same derivation)
independently agrees on 60.00 MiB recurrent and 1.875 / 1.406 MiB conv at 4 / 3 taps.

**The MTP draft block is its own row because its bytes have nowhere else to come from.** 19 tensors,
844,640,768 elements = **1,611.02 MiB bf16** = 0.4686 GiB at policy-A widths, of which the block's
own 256-expert MoE is 1.5000 GiB bf16 (D-05's figure, confirmed). The published artifact carries
**zero** draft tensors — 733 tensors, layer indices 0..39
(`quantization/evidence/q4km-published-width-map-2026-10-26.md`, re-checked by the witness
`GGUF carries no draft tensors`) — so the draft bytes must come from the HF shards
(`models/w1-mtp-extract/mtp.safetensors`, whose header this tool reads and whose element count
matches the pin's authority exactly).

**Three dtypes in that table are decisions, not facts, and are labelled as such:**

- **KV at bf16 (2 B/elem).** Nothing in `DECISIONS.md` covers KV width — D-01/D-01a/D-10 are weight
  widths. bf16 is the checkpoint's own `dtype` and the oracle's dtype, so it is the conservative
  baseline; `--kv-type q8_0|q4_k|iq4_xs` prices the alternatives (their block overhead comes from
  the ggml type's own bpw, so a Q4_K KV pool is 4.5/4 bits, not 4).
- **Recurrent state at fp32** — this one *is* settled by the config (`mamba_ssm_dtype: float32`),
  and `specification/04-gdn-linear-attention.md` §04.4 records the sharper version of it: fp32 is
  the *scan* dtype, not necessarily the storage dtype, and §04.6 states the state as `S[h_v,d_k,d_v]`
  in F32. Storing it in bf16 would halve 60 MiB per sequence, which at these numbers is not worth an
  accumulation-risk argument; nobody should do it before a quality case exists.
- **Conv window at bf16, `k_conv − 1` taps** — registered as A-09 in the spec's assumption list;
  the whole spread (4 taps, or fp32) is 1.41 → 3.75 MiB per sequence, i.e. noise against a 19.8 GiB
  weight set. Recorded, not argued.

**Deliberately absent, because inventing them would be worse than an empty row:** compute/activation
buffers, sampler scratch, KV block-table overhead, allocator fragmentation, engine code. See §7.

## 3. The budget table

Policy A, KV bf16, **one sequence**. Headroom is against the *pessimistic* co-resident budget
(31.84 GiB).

| context | weights | draft | KV | GDN state | other | total | headroom | verdict |
|---|---|---|---|---|---|---|---|---|
| 4k | 19.317 | 0.469 | 0.086 | 0.060 | 0.002 | **19.933** | +11.907 | **FITS co-resident** |
| 16k | 19.317 | 0.469 | 0.344 | 0.060 | 0.002 | **20.191** | +11.649 | **FITS co-resident** |
| 32k | 19.317 | 0.469 | 0.688 | 0.060 | 0.002 | **20.535** | +11.305 | **FITS co-resident** |
| 64k | 19.317 | 0.469 | 1.375 | 0.060 | 0.002 | **21.222** | +10.618 | **FITS co-resident** |
| 128k | 19.317 | 0.469 | 2.750 | 0.060 | 0.002 | **22.597** | +9.243 | **FITS co-resident** |
| 256k | 19.317 | 0.469 | 5.500 | 0.060 | 0.002 | **25.347** | +6.493 | **FITS co-resident** |

**The answer to the question as posed: policy A fits at every context in the checkpoint's own window,
with 6.5–11.9 GiB to spare, and context is not what kills it.** Most of that is arithmetic this
model's shape makes: only 10 of 40 layers carry KV (30 are GDN), so KV is 22,528 B/token counting
the draft's rows. A dense 40-layer model of the same width would cost **83,968 B/token (3.7×)** and
would run out of the same pessimistic budget at **~153k tokens**, where this model does not run out
inside its native window at all. The GDN state, conversely, is 61.4 MiB per sequence *regardless of
context*, which is why this model trades a small KV slope for a fixed per-lane tax.

Which is why the frontier that matters is **sequences × context**, not context alone (weights are
shared between lanes, KV and state are not):

| | 4k | 16k | 32k | 64k | 128k | 256k |
|---|---|---|---|---|---|---|
| **1 seq** | 19.93 | 20.19 | 20.53 | 21.22 | 22.60 | 25.35 fits |
| **2 seq** | 20.08 | 20.59 | 21.28 | 22.66 | 25.41 | 30.91 fits |
| **4 seq** | 20.37 | 21.40 | 22.78 | 25.53 | 31.03 fits | **42.03 → halogen-down-only** |
| **8 seq** | 20.95 | 23.02 | 25.77 | **31.27 fits** | **42.27 → halogen-down-only** | **64.27 DOES NOT FIT** |

and the crossover the allocator should carry, from `tools/residency_budget.py` at 4 sequences:
**140,755 tokens total across lanes** (20.027 GiB fixed, 90,112 B of KV rows per token once all four
lanes are counted) against a native window of 262,144. Rows within 31.84–33.88 GiB print as "fits
only in the lucky half of the bracket" and should be treated as *not* fitting (§1).

And the variants, which is where the middle verdict genuinely lives:

| variant | weights | resident at 1 lane × 4k | verdict co-resident | verdict halogen-down |
|---|---|---|---|---|
| policy A (DECIDED) | 19.7856 GiB | 19.933 | fits | fits |
| policy B (attention/GDN bf16) | 21.9306 | 22.078 | fits | fits |
| policy G (`lm_head`/embed Q6_K) | 19.5562 | 19.704 | fits | fits |
| policy E (attention + shared Q4_K) | 18.8859 | 19.034 | fits | fits |
| policy F (Q8_0 throughout) | 35.1606 | 35.308 | **no — fits only with halogen down** | fits |
| bf16 everything | 66.1337 | 66.281 | **no** | **no — above the whole pool** |
| policy A at 8 lanes × 256k | — | 64.267 | **no** | **no** — batching, not width, is what breaks it |

Policy F is the interesting row and it says what `DECISIONS.md` D-11 already suspected in smaller
numbers: Q8-throughout is *impossible co-resident and possible dedicated*, so the D-11 measurement
does not change what we build (policy A either way) but does change what stays available if quality
asks for it later. Note the corrected number: the repo quotes 34.97 GiB
(`devices/measured-facts.md` §"Roofline for the target model"), which is **35.16** once GDN `out_proj`
is in the census (§5 finding 1) — still above the co-resident bracket at every context, so the
conclusion stands and the margin is 3.3 GiB, not 3.0.

**256k is not a YaRN question.** `max_position_embeddings` is 262,144 in the config, so 256k is
inside the checkpoint's native window and needs no extension; what is out of scope (`AGENTS.md`
§SCOPE, and D-04's neighbourhood) is going *beyond* it. The 256k row is therefore a real row, with
one caveat that belongs to the spec worker rather than here: the GGUF declares rope *dimension
sections* (MRoPE-style, `quantization/evidence/q4km-published-width-map-2026-10-26.md`), and a text
path that assumes a plain 0..63 rotation will be wrong at long context regardless of bytes.

## 4. What to give up, cheapest first

Savings are derived (`--give-up` prints them for any row), not asserted. **Who decides** matters more
than the size: a width or scope change is the user's, per the D-01/D-01a/D-10 pattern, and this repo
may not spend it.

For a row that is over the co-resident budget (say 8 sequences at 256k, 32.43 GiB over):

| lever | saves | who decides | what it really costs |
|---|---|---|---|
| context 256k → 64k | 33.00 GiB | **user** | capability, not engineering |
| KV container bf16 → iq4_xs / q4_k | 32.31 / 31.63 GiB | **user** | a KV width has never been decided here; attention width was decided *against* Q4 in D-01a for evidence-based reasons, and KV is the same kind of surface |
| halogen down (budget, not bytes) | +23.66 GiB of budget | **ops** | D-11; **restart cost measured 2026-10-26: stop 14 s, ready 17 s** (`devices/evidence/halogen-restart.log`), so this lever is cheap to pull. Measured engine demand at 41/41 offload is **≈26.9 GiB** GTT, not the modelled 19.78 (`devices/evidence/device-window-2026-10-26.md`); the ≈55.5–56.5 GiB ceiling itself is still a D-02 allocation-probe figure |
| one fewer concurrent sequence | 5.56 GiB/lane at 256k | **ops/user** | throughput |
| drop the draft layer's own KV rows | 4.00 GiB at 8×256k | engineering | risks the draft reading rows it did not write — only an option if the draft shares the target's KV lanes |
| drop the MTP draft block | 0.4686 GiB + 2 KiB/token | **user** | reverses D-05 |
| stream one layer's routed experts | 0.4219 GiB/layer | engineering | **a capacity lever, never a speed lever**: 0.0132 GiB/token of expert traffic would arrive through a 54–58 GB/s host→device copy (`pitfalls/10-device-memory.md` §"Allocation recipe") instead of a 237 GB/s resident read — ~4.2× slower for that traffic |
| policy G (`lm_head`/embed Q6_K) | 0.2294 GiB | **user** | not taken (D-10); ~4% of ceiling is the price, per `quantization/README.md` |
| policy E (attention + shared Q4_K) | 0.8997 GiB | **user** | not taken (D-01a); reversible in one direction only, per D-01a's own reasoning |

Reading the table against §3: **at 1 sequence none of these levers is needed.** They are the answer
to "what do we spend if we want lanes", and the cheap ones (context, KV width, lanes) are all
user/ops calls. That is worth saying plainly now rather than discovering at the step-3 gate: the
engineering levers available are either expensive per token (streaming) or already decided against
(policy E/G).

## 5. Arithmetic in this repo that does not add up — reported, not fixed

These are findings. I did not edit any of the files involved (the brief forbade it and three are live
with other workers).

1. **`tools/quant_scenarios.py`'s census is missing GDN `out_proj`, and its `norms_misc` carries a
   phantom term.** Its `gdn_linear_attn` expression has no `out_proj` ([2048, 4096] × 30 =
   251,658,240 elements = 0.4687 GiB bf16) and its `norms_misc` contains `3*h*(3*h)` = 37,748,736
   elements that match no tensor in this checkpoint, while omitting `mtp.fc` and the MTP block's
   norms (net −29,356,032 elements). Text bf16 total: **65.7198 GiB claimed, 66.1337 GiB actual** —
   and the actual figure is not my opinion, it is the number that reconciles (§6).
   Consequences, all quoted elsewhere in the repo:
   - policy A resident **19.59 → 19.7856 GiB** (+0.1951 GiB, reconciled term by term in §6);
   - **per-token read 2.331 → 2.5260 GiB**, so the decode ceiling at the repo's own 237 / 210 / 190
     GB/s is **87 / 77 / 70 tok/s, not 95 / 84 / 76** — the "~95 tok/s" figure in
     `devices/measured-facts.md` §"Roofline", `quantization/README.md` and `DECISIONS.md` D-11/D-08
     is ~8 % high, and it is the number `quantization/README.md` tells readers to multiply by 0.6;
   - Q8-throughout 34.97 → **35.16 GiB** of weights (draft block included), bf16 65.72 → 66.13 GiB.
     The fit verdicts do not change; the margins do;
   - `quant_scenarios.py`'s own docstring says the gap to the shard index is "1.246 GiB (1.86 %) …
     the vision tower plus small tensors". Vision is **0.8318 GiB**; the other 0.4138 GiB is this
     defect, not small tensors. See §6.
2. **KV per token is understated by a factor of 2 in two places.** `devices/measured-facts.md`
   §"Roofline for the target model": "KV is 10 layers × 2 KV heads × 256 dim × 2 B = 10 KiB/token,
   ~0.7 % of a bf16 step" — that product stores K *or* V. Storing both is
   `2 × 10 × 2 × 256 × 2 B` = **20,480 B = 20 KiB/token** (22 KiB with the draft layer's rows), so
   it is **1.5 %** of a corrected bf16 step at 4k, not 0.7 %. `quantization/README.md` §"Open items
   this file hands back" inherits the same slip: "KV ignored … ~+0.04 GiB at 4k" is **+0.0859 GiB**
   at 4k (with draft rows) — still small, but the direction of the error is "the growing part is
   smaller than it is".
3. **`DECISIONS.md` D-09's price table does not reproduce from the checkpoint.** The draft block is
   1.5733 GiB bf16 (844,640,768 elements). Its "uniform / Q8-experts / bf16" rows are 19.59 / 20.17
   / 20.67 GiB, i.e. +0 / +0.58 / +1.08 GiB over uniform. Recomputed: uniform draft = 0.4686 GiB,
   whole-block-Q8_0 draft = 0.8358 GiB (**+0.367, not +0.58**), whole-block-bf16 draft = 1.5733 GiB
   (+1.105, which does match its +1.08 row within the rounding of the 19.59 base). Its "bytes per
   draft token 0.0234 GiB" is not reproducible from any width assignment I can build: the draft
   block's own experts (8/256) + attention + shared + router + norms at policy-A widths is
   **0.0599 GiB per draft token**, and that excludes `lm_head`, which a drafted token must pay for
   unless the engine shares the target pass's logits. The decision itself (uniform) is unaffected —
   it is the cheaper option either way — but the price tag quoted to justify it is wrong in the
   middle row and ~2.6× low on the per-draft-token line.
4. **The stale total in the width-map evidence.** `quantization/evidence/q4km-published-width-map-2026-10-26.md`
   prints "total weights 20.676 GiB (22.20 GB)" and "+1.09 GiB" over our arithmetic; re-parsing the
   pinned file locally now prints **20.604 GiB / 22,123,538,944 B**, which reconciles to the file
   size exactly. The 74.00 MiB difference is the already-documented Q5_K-at-5.5625 error
   (`quantization/evidence/audit-block-geometry-2026-10-26.md`, fixed in commit `3de5e6d`), so the
   `.md` is quoting the pre-correction `.txt`. The corrected delta against our 19.7856 GiB is
   **+0.82 GiB**, not +1.09.
5. **A framing point, not an error.** The brief for this analysis suggested treating 256k as a
   question because "YaRN is out of scope per D-04"; the config's own `max_position_embeddings`
   settles it — 256k is native, no extension involved (§3).

Places where a repo fact was *missing* rather than wrong: no KV geometry is written down anywhere
(this file is the first), no per-sequence state pool size, no compute-buffer allowance, and
`DECISIONS.md` has no decision covering any of them. `devices/measured-facts.md` has no number for
allocation *count* limits, cold-start disk rate, or Managed at anything other than ~32 GiB.

## 6. What witnesses the calculator passes, and what they prove

`python3 tools/residency_budget.py --selftest` → **17/17 witnesses pass, 9/9 injections fire.** The
three that make the census claim real:

- **config-derived shapes vs real safetensors headers**: 91 of the 712 text tensors have their shape
  checked against a real shard header (`models/qwen36-src/*.safetensors`), 0 mismatches.
- **artifact element total == our census minus the draft block**: the pinned GGUF's 733 tensor
  records sum to **34,660,610,688 elements** and our config-derived census excluding the draft block
  sums to the same number, **delta 0 bytes**. The published artifact and the checkpoint config agree
  on the served shape exactly, and they are independent sources.
- **text + vision == the shard index's own `total_size`, exactly** (66.1337 + 0.8318 GiB vs
  71,903,645,440 B, delta 0 B) — so the whole index is accounted for, and the 1.246 GiB that
  `quant_scenarios.py` attributes to "vision + small tensors" is 0.8318 vision + 0.4138 of its own
  omissions.
- **policy A vs `quant_scenarios`' 19.59 GiB, term by term**: gap 0.1951 GiB = out_proj at the
  attention width (+0.2490) + `mtp.fc` at the exact width (+0.0156) − their phantom 3h×3h (−0.0703)
  − their extra norm slots (−0.0002) + their norm slots riding the attention width (+0.0001) +
  their 1-D GDN controls quantized where ours stay exact (+0.0009) → **residual 0 B**. If anyone
  fixes `quant_scenarios.py`, this witness fails on purpose and says "retire the reconciliation".

And the injections (`--selftest` output), each perturbing one number, each caught by a witness that
does not read the perturbed value:

| injected fault | total drift | caught by |
|---|---|---|
| KV row computes to 0 | −0.172 GiB | "no component sums to zero" + "KV assembly == geometry" |
| GDN recurrent state dropped | −0.059 GiB | "GDN state pool: assembly == geometry" |
| GDN conv window dropped | −0.001 GiB | same witness, different component |
| per-token read forgets the untied `lm_head` | −0.503 GiB/token | the two independent per-token routes disagree by exactly 540,344,320 B |
| `out_proj` missing from the census (the real defect) | −0.249 GiB | record count 682 ≠ 712 + route disagreement |
| draft block excluded (contradicts D-05) | −0.469 GiB | "draft block is resident because D-05 has it in scope" + the policy-A reconciliation |
| draft's own KV rows not reserved | −0.016 GiB | "draft rows present for the MTP layer" |
| one full-attention layer lost from the inventory | −0.001 GiB | record count 711 ≠ 712 |
| `lm_head` counted twice | +0.503 GiB | artifact element total, delta −50,855,936 |

Record counts are printed and checked (`parsed 712 tensor records from the index, 712 derived from
config layer counts`), and a zero-valued component is an error rather than a blank cell. The
required three faults (KV row 0, a GDN state component dropped, a per-token read ignoring the untied
`lm_head`) are the first, second and fourth rows above.

## 7. UNMEASURED — for the main agent's device window

Every item is a real gap in §1–§4 that this repo cannot fill from what it has. **None of these was
run here**; they are the queue, with the command and what a PASS would mean. Two of them
(`gpu_lock.sh`) need the lock held and halogen's rules respected; the halogen ones are the main
agent's alone (`AGENTS.md` §"GPU execution").

Build the probes first (they are in this repo):

```
hipcc -O2 -Wl,-rpath,/opt/rocm/lib -o /tmp/mem_mode_probe tools/mem_mode_probe_hip.cpp
hipcc -O2 -Wl,-rpath,/opt/rocm/lib -o /tmp/hostmap_probe  tools/hostmap_probe.cpp
```

1. **The dedicated pool, i.e. D-11.** With the lock held and halogen down:
   ```
   tag="35b pool-dedicated"; tools/gpu_lock.sh acquire "$tag"
   GPU_LOCK_TAG="$tag" tools/halogen_gate.sh down
   /tmp/mem_mode_probe 32 5 ; /tmp/mem_mode_probe 48 5 ; /tmp/mem_mode_probe 56 5
   tools/halogen_gate.sh up ; tools/gpu_lock.sh release "$tag"
   ```
   PASS = a single `hipMalloc` of 48 GiB is granted and its checksummed read is ≥190 GB/s. That
   converts "≥55.5 GiB, assumed" into a number, licenses policy F (35.31 GiB) as a real fallback,
   and — per `devices/measured-facts.md` §"Residency above the pool" — separates a 56-GiB GTT-half
   cap from `enforce_isolation` sharing. If 48 GiB still fails, §1's dedicated row is wrong and
   every "fits only with halogen down" verdict in §3 is worthless.
2. **Allocation *count*, not just size, while halogen is resident.** §1's ladder is four points
   (1×32, 8×8, 24×2 GiB). An engine with 712 tensors is neither shape:
   ```
   /tmp/mem_mode_probe 0.027 5 22528 many 712      # 712 allocations of ~28 MiB, halogen up
   ```
   PASS = all 712 granted. FAIL means step 3 must lay weights into a small number of slabs (the
   `pitfalls/10-device-memory.md` recipe already assumes this) and the per-tensor path is dead — a
   layout decision, and cheaper to make now than after the engine exists.
3. **`hipMallocManaged` at the sizes we would actually use.** The repo's only Managed measurement is
   at ~32 GiB, where it silently reads **51.6 GB/s** (`pitfalls/10-device-memory.md` §"`hipMallocManaged`
   degrades silently near the ceiling"); policy A's footprint is 19.9–25.3 GiB, which has never been
   probed:
   ```
   /tmp/mem_mode_probe 20 5 22528      # its `man` row, and again at 24 and 26
   ```
   PASS = Managed at 20 GiB reads ≥190 GB/s, i.e. the cliff is about the *ceiling* and not about
   Managed per se. FAIL (≈51.6 GB/s at 20 GiB) closes off Managed entirely and makes the
   `hipMalloc` + explicit-copy recipe the only allowed path. Either way it must be measured before
   someone "just tries Managed" and ships an engine that is 4× slow with no error — the exact failure
   mode that trap documents.
4. **Cold-start cost of the real artifact** (file → pinned stage → device slab, 22.12 GB):
   ```
   sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
   time cat models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf > /dev/null
   ```
   PASS = a disk rate, which §2's arithmetic does not have. The repo's measured 54–58 GB/s
   host→device copy and ~35 GB/s first-touch bound the *copy* part (a 19.79 GiB slab is ≈0.4 s of
   copy and ≈0.6 s of page-touch — **derived from those rates, not measured**), but nobody has a
   number for the disk, and startup wall time is disk-dominated.
5. **A reference compute-buffer allowance.** Everything in §2 excludes activation scratch. llama.cpp
   prints its own KV + compute buffer sizes and is a CPU run if `--n-gpu-layers 0`, so it prices the
   gap without touching the iGPU:
   ```
   tools/build_llama.sh && engines/llama.cpp/build-hip/bin/llama-cli \
     -m models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -p hello -n 4 -c 4096 --n-gpu-layers 0
   ```
   PASS = llama.cpp's printed `compute buffer size` and `kv buffer` at 4k/32k/128k give an
   independent floor for the row this file leaves at "not modelled". It also gives the KV
   block-table overhead for free.
6. **Halogen restart cost**, which §4's "halogen down" lever assumes is expensive:
   `tools/halogen_gate.sh up 1800` appends to `devices/evidence/halogen-restart.log`, which
   `devices/measured-facts.md` records as empty. PASS = a logged cold and warm restart time, which
   decides whether "down" is a per-gate option or a once-a-session window.
7. **Two dtype calls that are the user's, not measurements:** KV container (§2), and — for
   completeness — whether the GDN conv window may live in bf16 at 3 taps (A-09 in
   `specification/04-gdn-linear-attention.md` §04.10). A-09 is worth 2.3 MiB per sequence, so it is
   not a capacity question at any lane count in §3.

## 8. What step 3 should take from this

- Design against **31.84 GiB**, the pessimistic co-resident budget, and treat 31.84–33.88 as unknown
  until D-11 runs.
- Policy A fits **one** lane at the native 256k window with ~6.5 GiB spare; **four** lanes at
  **137k aggregate**. That is the allocator's number, and it is set by KV×lanes, not by weights.
- Lay weights as a small number of slabs (`hipMalloc` + bulk copy, `pitfalls/10`), warm before any
  timing, and do not use Managed until item 3 answers for 20 GiB.
- If a variant is ever needed: policy F needs the dedicated pool (35.16 GiB of weights, 35.31 on a
  4k lane), policy E/G are user calls worth 0.90/0.23 GiB, and expert streaming is a capacity lever
  that costs ~4.2× the traffic rate.
- The per-token read this repo should quote is **2.5260 GiB** and the ceiling **87/77/70 tok/s** at
  237/210/190 GB/s — ceilings, not forecasts, and ~8 % below what the current prose says
  (§5 finding 1).
