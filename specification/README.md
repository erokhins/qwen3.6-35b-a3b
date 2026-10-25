# Specification index

Math-only, library-agnostic definition of the **Qwen/Qwen3.6-35B-A3B text-only forward pass**
(prefill + decode + MTP draft). No library calls, no kernel names, no engine code, no timings.
Every tensor shape is exact; every arithmetic claim names the command that reproduces it.

This directory is the document every later kernel, gate and `speccheck/` file is derived from.
`speccheck/` is the independent implementation and the oracle, `engine/` is the deliverable
(`AGENTS.md` §"Standing rules"); neither may contradict this file without an entry in
`ASSUMPTIONS.md`.

## Ground truth, in precedence order

A conflict is resolved by this list, top wins. Where the top two are silent, the question goes into
`ASSUMPTIONS.md` rather than being answered by taste (`AGENTS.md` §"Before you write anything,
decide nothing alone").

1. **The committed metadata and the real shard headers.**
   `checkpoints/Qwen3.6-35B-A3B.config.json` (the committed `config.json`) and the safetensors
   headers of `models/qwen36-src/model-00001-of-00026.safetensors`,
   `model-00002-of-00026.safetensors`, `model-00025-of-00026.safetensors`,
   `model-00026-of-00026.safetensors` — those four shards hold layers 0, 1, 38, 39 and the whole
   `mtp.*` block, so their headers are a *real witness* for those tensors' exact names, shapes and
   dtypes, and a few stored values are read to settle questions a shape cannot answer (which RMSNorm
   form, what `A_log` actually holds). Read-only; headers and small slices only. Provenance of the
   files: `README.md` §Download, `progress/2026-10-26/step1-kickoff.md` §Acquisition (2026-10-26).
2. **HF-ref — the checkpoint's own reference implementation.** `transformers 5.19.0`,
   `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` (+ `cache_utils.py` for state shapes).
   This is authority for *form* — what is computed, in what order, in what dtype — because it is
   what the checkpoint was trained and published against. Every formula this spec takes from it is
   **quoted here** with the class/function name, per the self-containment rule
   (`pitfalls/60-docs-and-citations.md` §"The self-containment rule"); a reader of this directory
   never needs to open that path. The installed copy used on 2026-10-26 was the one in the sibling
   project's venv (torch 2.14.1+cpu, transformers 5.19.0), which is a *reading* location only and is
   not cited as the source of any fact.
3. **llama.cpp at the pinned commit `d327bf1f0780f87b8c01b367b61853b0c291d7c4`**
   (`engines/llama.cpp`, read-only, never built or run here): `src/models/qwen35moe.cpp` for the
   arch's op order and GGUF tensor names, `src/llama-model.cpp` for `LLM_ARCH_QWEN35MOE` and the
   tensor-creation shapes, `conversion/qwen.py` for the HF→GGUF name map and the value transforms.
   Authority for **GGUF naming and byte layout**, and for the MTP block's composition (which HF-ref
   does not implement at all — see `ASSUMPTIONS.md` A-01).
4. **This repo's own instruments**, which are the only thing that makes a number in this directory
   checkable: `specification/derive.py` (this directory) and the committed
   `tools/quant_scenarios.py`, `tools/gguf_header_types.py`, `tools/mtp_shard_cost.py`.

## Tag conventions

| tag | meaning |
|---|---|
| `[CFG]` | read directly out of `checkpoints/Qwen3.6-35B-A3B.config.json` |
| `[WITNESS]` | read out of real bytes — a safetensors header, real stored values, or the published GGUF's header/data — by a `specification/derive.py` subcommand named inline |
| `[REF]` | not stated in the config; taken from a reference implementation, **quoted here**, with the source's file and symbol named and the disagreement/uncertainty registered in `ASSUMPTIONS.md` |
| `[ASSUMPTION A-xx]` | a decision this spec makes where the sources are silent or disagree; registered in `ASSUMPTIONS.md` with source, risk and the test that would falsify it |
| `[MEASURED]` | a number measured on this box or on this checkpoint and already recorded in this repo, with its date — this spec adds no new measurement (CPU-only, no timings) |

Nothing in this directory is a timing. No `ms`, no GB/s, no tok/s appears except figures already
recorded in this repo with a date and an instrument (`devices/measured-facts.md`, 2026-10-26).

## Files

| file | contents |
|---|---|
| `00-notation-and-hyperparameters.md` | symbols, every constant with the file it comes from, derived sizes, precision classes, the per-token/state summary |
| `01-weights-and-shapes.md` | the complete text-path weight inventory: HF name, GGUF name, shape, dtype, count, bytes, reconciled to the byte against the committed index |
| `02-tokenization-and-embedding.md` | the embedding lookup, the untied `lm_head`, positions, and what the stop rule is (and is not) |
| `03-block-glue-and-normalization.md` | the two RMSNorm forms, the decoder-layer skeleton, residual order, the logits head |
| `04-gdn-linear-attention.md` | Gated DeltaNet + short conv: 30 layers, recurrent and chunked forms, control path |
| `04b-gdn-chunked-form.md` | the prefill GDN scan: chunk width 64 and its padding rule, the UT solve and its two legal branches, the sequential scan over chunks, and the gate that shows it equal to §04's recurrent form |
| `05-gated-full-attention.md` | gated GQA attention: 10 layers, the output gate's exact position, partial interleaved mrope |
| `06-prefill-pass.md` | whole-request pass as an ordered op list, including state initialisation |
| `07-decode-step.md` | single-token step as an ordered op list, with the router named as the serial point |
| `08-state-management.md` | KV + recurrent + conv inventory, per-request budget arithmetic, lifecycle |
| `09-mtp-draft-layer.md` | the draft block: its 19 tensors, its input composition, its own MoE and KV |
| `10-reference-dumps-and-gates.md` | what `speccheck/` must pin for a Q4_K-expert / Q8_0-attention engine, and how the bands and must-catch sets are derived |
| `11-moe-block.md` | the MoE block: router R1–R5 with the cast order, top-k selection semantics, the SwiGLU expert and its `chunk(2)` split, the K=8 sum in ascending expert-id order, the shared expert, the router as the per-token serial point |
| `ASSUMPTIONS.md` | every `[REF]`/`[ASSUMPTION]` with source, risk, falsifying test — including the contradictions found against committed docs |
| `derive.py` | the arithmetic witness for every number above; stdlib-only, CPU-only |

## Scope

**In**: text-only prefill and decode, greedy decoding, the recurrent/KV state model, the MoE block
(router, top-k, grouped experts, shared expert), the MTP drafter's math, and the width bookkeeping a
quantized engine reads (`§10`).

**Out** (no spec, no code — `DECISIONS.md` D-04, `README.md` §Target):

- the vision tower (`model.visual.*`, 333 tensors, 893,142,496 B — measured,
  `python3 specification/derive.py --families`), and with it `image_token_id` 248056,
  `video_token_id` 248057, `vision_start_token_id` 248053, `vision_end_token_id` 248054 and every
  mrope branch that only fires on grid positions;
- multi-user batched serving, paged-attention scheduling, vLLM API parity;
- YaRN context extension beyond `max_position_embeddings` 262144 (`DECISIONS.md` D-04);
- sampling beyond greedy (temperature, beam, speculative policies other than the single MTP draft);
- the three Flash-Next-only subsystems (Gated Residual / hyperconnections, QSA indexer, n-gram PLE),
  which this checkpoint does not contain at all.

## What this model is not

The 0.8B project's specification is a structural sibling and was read for **form and register only**.
None of its numbers transfer: it is 24 layers, 8 Q heads, 16 V heads, an 18.0 MiB state and a tied
embedding, and every one of those is a fact about a different checkpoint. `AGENTS.md` §"Traps that do
NOT transfer" is the rule; `pitfalls/60-docs-and-citations.md` §"The self-containment rule" is why no
file here cites it. Where this spec needs a method (floor probes, geometry-derived must-catch sets)
it cites this repo's own `pitfalls/30-numerics.md` and `pitfalls/40-gates-and-admissions.md`.

## How to check this directory

```sh
python3 specification/derive.py --check        # 30 header-witnessed assertions, non-zero exit on drift
python3 specification/derive.py --families     # inventory vs the committed index, delta must read 0 B
python3 specification/derive.py --per-token    # the three per-token columns §00.5/§10 print
python3 specification/derive.py --state        # the state rows §00.6/§08 print
python3 specification/derive.py --mtp          # the 19 draft-block tensors
python3 specification/derive.py --values <tensor-name>   # real stored values, for the [WITNESS] rows
python3 specification/derive.py --gguf-vs-ckpt models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
python3 tools/gguf_width_table.py models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --check-header
                                       # the byte table in §01.4: every width re-derived from
                                       # engines/llama.cpp/ggml/src/ggml-common.h and the printed bytes
                                       # pinned by sum(row_bytes) + data_start == file size
```

Three committed docs disagree with this spec in four places (text bf16 total, KV per token, the MTP
block's element count, and what "exact" costs in bytes). Each one is written up in
`ASSUMPTIONS.md` (A-07, A-08, A-10, A-11) with both numbers and the command that adjudicates it, and
reported; none of those files was edited from this task.
