# Using `../qwen3.5-0.8b` as the reference implementation

> **This file is a reading list, not a fact store.** Per `pitfalls/60-docs-and-citations.md`, every
> number, trap and protocol this project needs has already been **copied** into `pitfalls/`,
> `devices/measured-facts.md` and `tools/`. What follows points at *code shape and precedent* — go
> and look, then write the version that fits this model. If you find a fact you need that exists
> only over there, copy it in and add a provenance line; do not cite it from here.

The sibling repo is not background reading — it is a working, gated, measured engine for the
same architecture family at a smaller size. Most of what this project needs already exists
there in some form. This file says **what to take, what to adapt, and what must not be
carried over.** All paths below are relative to `/home/erokhins/erokhins_projects/`.

## Read first, in this order

**In this repo — this is the working set:**

| # | file | why |
|---|---|---|
| 1 | `AGENTS.md` | scope, GPU policy, the self-containment rule, decisions that are the user's |
| 2 | `README.md`, `DECISIONS.md` | target facts, sizing, and the open user decisions |
| 3 | `pitfalls/README.md` then the topic file you need | the trap library — copied in, with numbers and correct forms |
| 4 | `devices/measured-facts.md` | this box's measured ceilings and the roofline arithmetic for this model |
| 5 | `reference-implementation.md` (this file) | where the reusable code lives, before writing a new op |

**In the sibling repo — precedent and code shape only, never as a fact store:**

| # | file | why |
|---|---|---|
| a | `qwen3.5-0.8b/AGENTS.md` | how these rules came to be, and the one paragraph that does *not* transfer (see our `AGENTS.md`) |
| b | `qwen3.5-0.8b/engine/performance-status.md` §1 + §6 | §1 is that engine's live ledger; §6 is the long form of `pitfalls/00-measurement.md`. For "what have we already tried", its header points at `qwen3.5-0.8b/progress/2026-10-14/speedup-routes-2026-10-14.md` |
| c | `qwen3.5-0.8b/specification/{00,01,04,05,07,10}` | the spec form to imitate: math + exact shapes only, every non-obvious behaviour tagged `[REF]` or `[ASSUMPTION A-xx]` |
| d | `qwen3.5-0.8b/speccheck/` | the independent torch-CPU oracle and the gate suite pattern |
| e | `qwen3.5-0.8b/prototype/README.md` | the per-op convention (standalone kernel + reference dump + `results.md` + scripted device gate), plus build flags |
| f | `qwen3.5-0.8b/engine/results.md` (4,980 lines) | the journal. **grep it, do not read it linearly** — and copy anything load-bearing before quoting it |

## Reuse table

| area | source | verdict |
|---|---|---|
| measured box facts (bandwidth curve, pool ceiling, warm-up penalty, compute rates, mixed traffic) | **already copied here** → `devices/measured-facts.md`, traps in `pitfalls/00-measurement.md` + `pitfalls/10-device-memory.md`, probes in `tools/` (see `tools/PROVENANCE.md`) | **do not re-measure**; the upstream docs remain available for provenance only |
| GPU lock | `qwen3.5-0.8b/tools/gpu_lock.sh` | **already copied here.** Machine-wide (`/tmp/pi_gpu.lock`) so both repos exclude each other |
| HIP build pattern | `qwen3.5-0.8b/prototype/*/build.sh`, `engine/build.sh` | **reuse**, `-O3` and `-Wl,-rpath,/opt/rocm/lib` are not optional; never time an `-O0` build |
| safetensors loader + weight-shape contract | `qwen3.5-0.8b/engine/{loader.cpp,loader.h,weights_contract.py,weights_reference.py}` | **adapt.** The rank-collapsing contract idea is essential here: the expert tensors are rank-3 (`[E, 2I, H]`), which the 0.8B contract never had to express |
| GDN (Gated DeltaNet) spec + kernels | `specification/04-gdn-linear-attention.md`, `engine/{kernels_gdn.cpp,gdn_rec_*.h,gdn_collapse_kernels.h,gdn_conv_*.h}`, `prototype/{gdn_decode,gdn_chunk,gdn_collapse}/` | **reuse the form.** The recurrence is identical; widths go 16→32 V heads, 18→30 layers. Expect grid/block-shape changes, not math changes |
| gated GQA attention | `specification/05-gated-full-attention.md`, `engine/kernels_attn.cpp`, `prototype/attn_decode/` | **reuse the form.** Same head_dim 256, same rope (0.25 partial, theta 1e7, mrope [11,11,10]); Q heads 8→16, KV 2→2, layers 6→10. The *wide/tiled* variants were tuned to the 0.8B's wave geometry — re-derive before trusting one |
| SwiGLU GEMV = the expert body | `prototype/mlp_gemv/` (+ its `floor_probe.py` and `--negative` gate), `engine/kernels_mlp.cpp` | **closest relative of the new work.** An expert is a 3-matrix SwiGLU with I=512; copy the gate pattern (floor probe, cancellation band, derived must-catch set) with it |
| projection GEMVs / grouped launches | `prototype/proj_gemv/`, `prototype/gemm_algo_sweep/`, `engine/{kernels_proj.cpp,proj_group_kernels.h}` | adapt. Note the sweep lesson: price only the shapes the engine can actually emit |
| logits head | `prototype/lm_head_gemv/`, `engine/kernels_lmhead.cpp`, `lm_head_{batched,xbf16}_kernels.h` | adapt: this model is **untied**, so the table is a separate 1.0 GiB and every token reads it whole |
| prefill / decode drivers, stage clock, wave planner | `engine/{prefill.cpp,decode_step.cpp,decode_run.cpp,stage_clock.*}` | adapt. The `--time-stages` burst is the only trusted decode ruler (§6) |
| MTP draft + accept loop | `specification/09-mtp-draft-layer.md`, `speccheck/mtp.py`, `engine/mtp_{draft,accept,verify,meter}.cpp`, `prototype/mtp_*` | **reusable and relevant** — 3.6-35B-A3B ships a real MTP head ("trained with multi-steps"), which the 0.8B project descoped. The `mtp_*` prototypes also hold the llama.cpp harness traps (`--log-verbose`, `-p` not `-f`, locale commas) |
| chat/tokenizer path | `specification/02-*`, `qwen3.5-0.8b/tools/chat_render.py`, `qwen3.5-0.8b/speccheck/textio.py`, `engine/chat.py` | adapt: same vocab (248320) but **re-derive the stop rule from this checkpoint**, whose `eos_token_id` is 248044 |
| dumper + fixtures | `qwen3.5-0.8b/tools/reference_dump.py` (transformers-CPU, pins tensors into `data/refs/<tag>/` with a `manifest.json`), `speccheck/fixtures/` | adapt: see "CPU oracle at 67 GiB" below |
| gate driver patterns | `speccheck/run_all.py` (`--quick` = 11 gates, full = 26), `speccheck/gates/run_*.py`, `engine/gate.py`, `prototype/*/run.py` | **reuse the discipline**, not the gate list |
| cross-check engine | `qwen3.5-0.8b/engines/llama.cpp` already has `LLM_ARCH_QWEN35MOE` (`src/llama-model.cpp:336`) and native MTP for it | **available immediately** as a second opinion once GGUFs exist; the vendored tree is gitignored/read-only there — clone your own under `engines/` |
| doc/provenance tooling | `qwen3.5-0.8b/tools/{doc_ref_check,citation_rot_check,citation_census,archive_doc,commit_msg_rewrite}.py` | import if this repo adopts the same citation rules (it says it does in `AGENTS.md`) |
| the 0.8B Python env | `qwen3.5-0.8b/.venv` (torch 2.14 CPU + transformers 5.19) | transformers 5.19 **already contains both `qwen3_5_moe` and `qwen4_exp`**, so the reference classes for both this project and the end goal are present. Either point at that venv or build a fresh one; do not assume CUDA — it is CPU-only |

## Do NOT carry over

- **Any numerics band, floor, route allowance, or `MAXSC`-style admission.** They were bought
  against the 0.8B's pinned dumps and its geometry. New model ⇒ new pins ⇒ new floors, sized by
  a floor probe and a geometry-derived must-catch set. Copy the *method* (`prototype/mlp_gemv/floor_probe.py`,
  `prototype/attn_tile_model.py`'s `neg_expect`), never the number.
- The 0.8B's engine kernel variants tuned to its wave/tile geometry (`attn_wide_*`, `attn_fattn_*`,
  `attn_m2`, GDN collapse/hoist/unroll families). They are useful as *style* references and as
  sources of measured instruction prices, not as drop-ins.
- `progress/` journals from that project: history of a different model. Read the routes index,
  skip the rest, unless the question is explicitly "has this failure mode been seen before".
- The quantization decisions in `qwen3.8-flash-next-old/QUANTIZATION.md`. Its *analysis method*
  (component inventory, per-component active bytes, the double-quantization source-chain
  lesson) is worth reading before writing D-01; its numbers are for a 360 GB model.

## CPU oracle at 67 GiB (the practical plan, pending D-01/D-03)

The bf16 checkpoint is 64.6 GiB and this box has 123 GiB of host RAM, so a torch-CPU reference
run is possible **without the weights being on the GPU at all** — which means every MoE kernel
can be gated at full fidelity before any residency or quantization decision is made. Two
practical notes:

1. Prefer **layer-sliced pins**: a GDN layer, a full-attention layer, one MTP layer, and the
   final norm/logits, rather than all 40 layers. The sibling's `qwen3.5-0.8b/tools/reference_dump.py` already has
   `--lean`, `--no-layer-io`, `--no-mlp-io`, `--deep-steps`, `--logit-steps` for exactly this
   kind of trimming; extend rather than fork.
2. Keep the sibling's provenance rules: a carried file is **named and skipped**, never silently
   ignored (the `CARRIED_NOT_PINNED` lesson — a directory that had been priced could never
   reproduce because `--verify` compared the price file too).

## Gate discipline to imitate (the part worth more than the code)

Every op that lands here ships with all four, or it is not done:

1. a **reference dump** from HF transformers CPU, pinned under `data/refs/<tag>/` with a manifest;
2. an **independent** torch-CPU implementation in `speccheck/` (written from the spec, not by
   calling transformers — the two agreeing is what makes the spec trustworthy);
3. a **scripted device gate** `prototype/<op>/run.py` that prints one `RESULT` line, asserts
   the parsed record count equals the printed record count, and runs its negative controls;
4. a **floor probe** and a **geometry-derived** must-catch/must-not-catch list, so the band is
   justified by arithmetic and a control can actually fire.

Measurement rules for this box, condensed (full version: `pitfalls/00-measurement.md`):
hold `tools/gpu_lock.sh`, zero subagents, foreground only, warm pages first, quote medians with
their spread, and never compare a number taken in one window against a number from another.
