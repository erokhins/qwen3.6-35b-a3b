# PLAN — the funded queue

**Step 1 is APPROVED (user, 2026-10-26) and NOT STARTED.** Nothing below has been downloaded, and no
gate below has been run. This file records what will happen when it is started; it is not evidence
that any of it happened. Anything written here after a run must cite the artifact that proves it.

## Step 1 — acquire the artifact, and witness it (approved: `published + mtp + audit`)

**The approved shape.** Ship a **published quant** — `unsloth/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` — rather
than converting the checkpoint ourselves; **add the MTP draft block**, which that file provably lacks
(`quantization/evidence/q4km-published-width-map-2026-10-26.md`: layer indices 0..39, zero `mtp`
tensors); and **audit the published file against the checkpoint** on a sample, so that the artifact
becomes evidence instead of an assumption. Total fetch ≈ **31 GiB** against 2.4 TiB free, and none of it
touches the iGPU, so per D-13 it may overlap active workers. `tools/mtp_shard_cost.py` priced every line
of the fetch list from the committed index.

### Why an audit exists at all

With only a published file on disk, the two readers of that file are llama.cpp and our own engine, and
they read the **same bytes**. A converter bug — wrong scale order, transposed expert stack, a router row
that landed in the wrong layer — makes both agree and makes every gate green. That is this repo's
oldest failure shape ("two self-consistent layouts joined by a branch", `pitfalls/30`) moved up a level,
and llama.cpp can never witness it because llama.cpp agrees with itself. The audit compares the file
against the authority (the HF bf16 tensors) where the authority is reachable.

### Fetch list

| what | source | ≈ size | kept afterwards |
|---|---|---|---|
| `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | `huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF` | 20.7 GiB (weights) | yes — it *is* the shipping artifact |
| `model-00025-of-00026`, `model-00026-of-00026` | `huggingface.co/Qwen/Qwen3.6-35B-A3B` | 6.07 GiB actual (3,832,888,256 + 2,231,416,848 B) | extract `mtp.*` — **19 tensors, 844,640,768 elts = 1,611.02 MiB bf16** — then shards may go. The 79.9 MiB this line claimed when the plan was approved was **wrong by 20.14×**, corrected 2026-10-26 from the raw shard headers; `README.md`'s own census (61.50 GiB = 41 layers × 1.50 GiB) already contradicted it, because the draft block carries its **own 256-expert MoE stack**: `mtp.layers.0.mlp.experts.gate_up_proj` `[256,1024,2048]` = 536,870,912 elts and `…experts.down_proj` `[256,2048,512]` = 268,435,456 elts, together 95.34 % of the block |
| `model-00001-of-00026`, `model-00002-of-00026` | same | ~5.2 GiB | audit input for layers 0, 1; then may go |

Layers **0, 1, 39** are the sample, not arbitrary: a `use_more_bits`/dynamic recipe upgrades the first
⅛, the last ⅛ and every third layer between, so 0 and 39 exercise the producer's *upgraded* path and 1
exercises the baseline path. Spreading the value-witness over all 40 layers instead costs **17 shards
≈ 44 GiB** — a full download in disguise, which is the argument for sampling. Sizes are the index's
`total_size ÷ shard count`; the real numbers come from the file listing at fetch time.

### Sub-steps, in order

Each has a pass condition, because a step whose check can be skipped is a step that will be skipped
(`pitfalls/40`).

| # | step | pass condition | device? |
|---|---|---|---|
| **1.0** | Write the audit gate **before the bytes exist** — `tools/audit_published_quant.py` (planned), so the gate is designed from the spec and not reverse-engineered from the data it will judge | `--selftest` on synthetic Q4_K/Q8_0/F32 blocks passes, and every injected bug in its own must-catch set fires | no |
| **1.1** | Fetch the GGUF; record `sha256`, HF revision, `general.file_type`, `general.quantization_version` | sha256 re-reads identically; header parses to 733/733 tensor records (already the expected count from the range-read) | no |
| **1.2** | Fetch the 4 shards | each shard's declared tensor set matches the committed `weight_map` for its contents | no |
| **1.3** | **Structure pass over all 733 tensors** (free — needs no shard): names, rank, dims under GGUF's reversed-dim convention (`ffn_down_exps [512,2048,256]` ↔ HF `[256,2048,512]`), dtype, per-family counts vs the `checkpoints/` census | reconciles tensor-for-tensor; any unreconciled name, dim or dtype aborts before any value is judged | no |
| **1.4** | **Exactness pass** on the unquantized families — router `ffn_gate_inp`, norms, GDN control (`conv1d`, `ssm_alpha/beta`, `ssm_dt`, `ssm_a`, `ssm_norm`), stored **F32** in the file — for the sampled layers, vs bf16 widened to F32 | **bit-exact**, zero tolerance. A wrong router row is the worst silent failure this model has: wrong top-k selects the wrong experts and still prints fluent text | no |
| **1.5** | **Dequant comparison** on the quantized families (experts, attention, GDN, `lm_head`) using **our own** reader written from `ggml-common.h`, not from llama.cpp, against the bf16 originals, with bands sized by a floor probe | inside the floor-probe bands; every injection fires **only** where it belongs — read-Q4_K-as-Q4_NL (drops the `min` term), swap the two halves of gate/up, transpose the expert stack, shift block offset by one | no |
| **1.6** | **Shared-storage check + pinning**: where the index records two names sharing storage, their quantized bytes must agree; then commit `sha256` + revision + the audit table as evidence | identity holds; `checkpoints/` records the pin, so a silent re-upload under the same name becomes a hash mismatch | no |

**Built into 1.0 as hard requirements**, from this repo's own trap index: the gate **refuses to print a
table when a section has zero cases** (an empty parse that reports PASS is the recurring failure here);
must-catch **and** must-not-catch sets are **derived from geometry**, not hand-listed; and the reader
under test is never its own reference — the F32 families judge the loader, the bf16 originals judge the
dequant, and llama.cpp judges nothing in this step.

### Amendment to 1.4 — decided 2026-10-26 by the main agent (`DECISIONS.md` D-14; user-ratifiable)

1.4 as approved demanded bit-exactness of every F32-stored family **including `ssm_a`**. That predicate
was mis-specified for that one tensor: **HF does not store `ssm_a`** — the reference module computes
`A = -exp(A_log.float())` at init, so the file's `ssm_a` is a *recomputation*, not a copy, and there is
no stored authority for it to be bit-exact against. The audit found exactly one element outside zero
tolerance (`blk.1.ssm_a` element 21, 1 f32 ulp from `-exp(A_log)` with two independently computed
correctly-rounded routes agreeing with each other), which is the signature of the producer's libm, not
of a converter bug — and the engine reads `ssm_a` from the file and never recomputes `exp`, so
reproducing that last bit is not an engine fidelity property either. Zero tolerance therefore splits:

- **1.4a — copied families**: routers, all norms, `A_log`, `dt_bias`, `alpha`, `beta`, `conv1d`, the V
  reorder, the gate/up split. **Zero tolerance, unchanged, no new code path.**
- **1.4b — the single documented transcendental route `-exp(A_log)`, and nothing else**: the two
  reference routes must agree bit-for-bit **with each other** (a reference disagreement is a FAIL, never
  "within tolerance"), and the file must lie within **exactly 1 ulp**, the ulp taken from `nextafter` on
  the value's own exponent — no epsilon constant, no general tolerance knob anywhere. Every element
  judged by 1.4b is printed by index beside all three bit patterns and is never called "bit-exact"; the
  verdict reads "26 bit-exact + 1 within-route".

Must-catch **grew**: ±2 ulp, sign flip, element permutation and a wrong-`A_log` read on that same tensor
must all still fire. That is what separates this from the forbidden move ("loosen a gate to make it
pass", `AGENTS.md`): the bound is derived from the *route*, applies to no other family, is witnessed
non-vacuous in the direction adjacent to the slack, and the original zero-tolerance FAIL line stays in
`quantization/evidence/audit-exactness-2026-10-26.json` next to the new verdict.

**Step 1 buys:** the shipping artifact, the MTP weights (D-05 stays alive), the first witness for our
own Q4_K/Q8_0 reader against the authority, and a pinned hash to build on.

**Step 1 does not buy**, and no reading of it should imply otherwise:

- **quantization quality** — unmeasurable here. We inherit the producer's own KLD claim (0.5478,
  `quantization/evidence/unsloth-qwen3.5-35b-a3b-2026-10.md`) unverifiably, and every numeric claim this
  repo can make is "agrees with llama.cpp on identical bytes", never "this is what the model said";
- **width freedom** — `lm_head` is Q6_K, which silently reverses the **D-10** decision to protect it at
  Q8_0 on tail-risk grounds; **policy E** (IQ4_XS, 1.663 GiB/token, ~133 tok/s) — the lever keeping the
  32k plan alive — is unavailable unless someone publishes it;
- **full coverage** — values are sampled, so a defect confined to an unsampled layer's *values*
  survives. Structure is verified for all 40 layers; values for three of them plus MTP.

### Deferred by this decision: the full bf16 checkpoint (67 GiB, 26 shards)

Not cancelled — parked with named re-triggers, so it is bought by a question rather than by taste:

1. the first numerics question that llama.cpp agreement cannot adjudicate (both sides carry it);
2. deciding to pursue policy E / IQ4_XS, or any change to a width decision including D-10's `lm_head`;
3. any public **accuracy** claim, which needs a distance from the model rather than from llama.cpp;
4. the **Flash-Next** step, where no published quant will exist on day one and the quantizer capability
   built by 1.0/1.5 is a prerequisite, not a luxury.

## Step 2 and beyond — named, not yet sequenced

Sequencing these is a separate decision; they are listed so step 1 is not mistaken for the plan.

- `prototype/q4k_reader` — the Q4_K/Q8_0 dequant gate on device, graduating from 1.5, with the same
  derived must-catch sets. **First device work in the repo**, so it inherits the full D-13 protocol
  (gpu_lock held, zero active workers, halogen down).
- Dense/MoE forward path: router top-k, expert gather, shared expert, then the attention and GDN forms —
  each landed as kernel + gate + `results.md` (AGENTS commit-by-topic rule).
- MTP: draft block from the extracted `mtp.*` tensors, acceptance measured against our own target model
  (self-referential, so it needs no bf16).
- Residency and 32k context: the queue that D-11 (pool ownership, now reachable since D-13 made halogen
  stoppable) and policy E both feed.
