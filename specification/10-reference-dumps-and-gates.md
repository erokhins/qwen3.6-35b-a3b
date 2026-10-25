# 10 — Reference dumps, bands and must-catch sets: what `speccheck/` must pin

This file is the contract between the spec and the gates. It says what has to be pinned from the
checkpoint's own reference, how a band is derived, and what a gate must be able to catch before its
green counts as anything. It borrows no number from another project
(`AGENTS.md` §"Traps that do NOT transfer") and it takes no timing (`specification/README.md` §"Ground
truth…" — this directory contains no measurements).

Ground rule, restated because it is the one that gets violated: **`speccheck/` is the independent
implementation and the oracle; the quantized kernel is judged against the bf16 reference, never against
another quantization and never against what the shipped file happens to produce**
(`quantization/README.md` §"How a quantized path gets gated"). llama.cpp judges nothing here — it reads
the same bytes as our engine, and two readers of one artifact agreeing is this repo's oldest failure
shape (`pitfalls/40-gates-and-admissions.md` §"Two self-consistent layouts joined by a branch are
invisible to both sides").

## §10.1 What is already pinned, and by what

`checkpoints/dump-manifest-2026-10-26.json` (written by `tools/dump_reference.py`) records 104 dump
files across four blocks — `layer0`, `layer1`, `layer39`, `mtp0` — over 32 generated tokens
(`seed 20261026`, torch 2.14.1+cpu, 8 threads), and it carries `agreement_at_write_time` for each
block. Per block it pins the MoE block's tensors:

| group | tensors | why it is on the list |
|---|---|---|
| block I/O | `input`, `router_logits`, `topk_ids`, `topk_weights`, `expert_raw`, `routed_sum`, `shared_raw`, `shared_out`, `combined_out` | the chain §11.2 → §11.6, each stage separately so a failure is attributable |
| the cast-order arms | `topk_weights` (**cast-before**, the reference's form), `topk_weights_f32` (pre-cast), and the `_spec` / `_exact` arms of `expert_raw`, `routed_sum`, `combined_out` | this is `AGENTS.md` prediction 4 made measurable; §11.2's verdict is checkable because both arms are on disk |
| weights | `mlp.gate.weight`, the three shared-expert matrices, `mlp.shared_expert_gate.weight` | the small exact tensors a quantized build must not move |
| referenced, not dumped | the two fused expert tensors (1.5 GiB per block) | the manifest carries shape/dtype/shard/`data_offsets` instead, so a gate reads the selected experts straight out of the shard |

| manifest rule | source |
|---|---|
| payload is raw little-endian element bytes, no header; `bytes == elements × itemsize`, and a record whose size cannot hold its declared shape is refused | `pitfalls/30-numerics.md` §"Read a dump's dtype off its file size, not off the comparator's assumption" |
| BF16 is read by `uint16` view then bitcast, never by casting `uint16` to float | same file, §"`Tensor.to(torch.uint16)` casts; `.view(torch.uint16)` bitcasts" |
| `expert_raw`/`routed_sum` are indexed by **top-k slot**, and the `_exact` arm's slots follow its own rank order, so a tie can permute; compare along the slot order sorted by `topk_ids_exact` | the manifest's own `reading_rules` — and the reason §11.3's tie row exists |
| `topk_weights` is authoritative because it is the cast-before tensor | the manifest's `reading_rules.authoritative`, matching §11.2 R5 |

Instruments that already exist and should be extended rather than forked: `tools/dump_reference.py`,
`speccheck/moe_ref.py`, `speccheck/cross_check_hf.py`, `speccheck/selftest_moe.py`,
`tools/audit_published_quant.py`, `tools/gguf_reader.py`, `tools/ggml_block_geom.py`,
`tools/moe_stats.py`. The published-artifact audit has been run and its verdicts are in
`quantization/evidence/audit-published-quant-2026-10-26.md` with the raw JSON beside it.

## §10.2 Two families of tensor, two kinds of check

Width assignment comes from `specification/01-weights-and-shapes.md` §01.5. The check type follows the
width:

| family | stored | check | tolerance |
|---|---|---|---|
| router `ffn_gate_inp` | F32 (wider than "exact") | bit-pattern compare against the bf16 original widened to F32, with the documented converter transforms | **zero** |
| all norms (Form A, `+1` in F32) | F32 | same, and the `+1` must be reproduced in F32 | **zero** |
| `linear_attn.norm` (Form B, **no** `+1`) | F32 | same | **zero** |
| `ssm_conv1d`, `ssm_alpha`, `ssm_beta`, `ssm_dt.bias` | F32 | same | **zero** |
| `A_log` | — | the checkpoint value, not the file's `ssm_a` | **zero** |
| **`ssm_a` = `-exp(A_log)`** | F32 | the **1.4b** predicate: two independently computed correctly-rounded routes must agree bit-for-bit **with each other** (a route disagreement is a FAIL, never "within tolerance"), and the file must lie within **exactly 1 ulp**, the ulp taken from `nextafter` on the value's own exponent | 1 ulp, route-derived |
| V-head reorder, gate/up split | — | the permutation must reproduce exactly, element for element | **zero** |
| experts `gate_up`, `down` | Q4_K (published: down Q5_K/Q6_K) | dequant with **our own** reader written from `ggml-common.h`, compared to the bf16 originals | floor-probe band, §10.3 |
| attention q/k/v/o, GDN `attn_qkv`/`attn_gate`/`ssm_out`, shared expert, embeddings | Q8_0 | same | floor-probe band |
| `lm_head` | Q6_K in the shipped file, Q8_0 per D-10 | same, **plus** the decision-level row (§10.3) | floor-probe band + argmax/top-5 |

The 1.4a / 1.4b split is `PLAN.md`'s §"Amendment to 1.4" (`DECISIONS.md` D-14, user-ratifiable) and it
exists because `ssm_a` is **not a stored quantity**: HF computes `A = -exp(A_log.float())`, so the file
holds a recomputation with a producer-libm history. No element judged by 1.4b is ever called
"bit-exact", and the original zero-tolerance FAIL line stays in the evidence file beside the new
verdict. Must-catch on that tensor grew, not shrank: ±2 ulp, a sign flip, an element permutation and a
wrong-`A_log` read must all still fire.

## §10.3 How a band is derived — never from the build under test

| step | rule | source |
|---|---|---|
| 1 | run a **floor probe**: re-run the reference's own F32 math with its summation re-associated, over the pinned rows, per family | `pitfalls/30-numerics.md` §"A max-ULP band cannot see a whole-tensor 1-ULP error" |
| 2 | stage-local near-exact ops (norms, residual adds, rope) get `ndiff` with **no** cancellation band — their floor is 1 ULP | same |
| 3 | long dots (expert GEMV, `lm_head` GEMV, the GDN scan, attention PV) get **cos + mean-rel aggregates plus a cancellation band**, both sized by the probe. A chained dot's legal `ndiff` is 29–47 %, so `ndiff` there measures nothing | same |
| 4 | the argmax that decides every token is judged **decision-level**: top-1 and top-5 agreement on the pinned rows, over a corpus at 4k context | same, and `quantization/README.md` §"How a quantized path gets gated" item 4 |
| 5 | any byte-identity claim states the **width** it covers; at longer widths, decision-level aggregates + pooled 1−cos | `pitfalls/30-numerics.md` §"Byte-identity at short widths is not byte-identity at long context" |
| 6 | a widened band is admissible only if it still catches **every** injection in the geometry-derived must-catch set. A gate is never loosened to make a build pass | `AGENTS.md` §"Standing rules", `pitfalls/40` §"A negative control with an empty must-catch set is not a control" |
| 7 | where a "byte-identical or die" admission dies on a tie, adjudicate in three parts (flipped token is the flag-off row's runner-up; top1−top2 margin under one BF16 ULP of that row's top logit; who emitted it) and print every margin | `pitfalls/30-numerics.md` §"Two legal bf16 implementations agree only ~0.88 of the way" |

No band, floor, route allowance or wave rule may be imported from the sibling project: **new model ⇒
new pins ⇒ new floors** (`AGENTS.md` §"Traps that do NOT transfer").

## §10.4 Must-catch sets, derived from geometry

Each injection below is generated from a shape or an index in this directory — never typed from memory
(`pitfalls/40` §"An admission over a fixture list typed from memory is not an admission") — and each
must fire **only** on the tensors that carry it, with a witnessed non-zero counter beside it.

| injection | derived from | must fire on |
|---|---|---|
| read a Q4_K block as Q4_NL (drop the `min` term) | `ggml-common.h`'s Q4_K struct, `tools/ggml_block_geom.py` | expert `gate_up`/`down` only |
| swap the two halves of gate/up | `ne[1] = 2·I_e` split at `I_e` (§11.4) | `ffn_gate_exps`/`ffn_up_exps`, fused spelling included |
| transpose the expert stack | GGUF `ne` reversal (§01.3) | all three expert tensors |
| shift a block offset by one | `block_bytes(type)` from the same header | every quantized tensor |
| wrong sub-block scale / wrong `d` super-block scale | the Q4_K super-block layout (`quantization/README.md` §"Which 4-bit format") | experts |
| wrong expert index in the gather | `E = 256`, `K = 8` (§00.2) | `expert_raw`, `routed_sum` |
| a router that picked different experts | perturb one logit across the 8/9 boundary | `topk_ids`, and downstream only |
| cast routing weights after the multiply instead of before | §11.2 R5 | `routed_sum`, `combined_out` |
| accumulate the 8 experts in rank order instead of ascending expert id | §11.5 | `routed_sum` |
| read `attn_q` at `n_q·d_head` rows | §05.1's factor of 2 | `attn_q`, and the gate's `sigmoid` input |
| take the gate as the tensor's upper half instead of per-head | §05.1's row arithmetic | the attention output |
| V-head order: read tiled as grouped (or the reverse) | §04.9's permutation with `h_k = 16`, `h_v / h_k = 2` | `attn_qkv` V third, `attn_gate`, `ssm_alpha/beta`, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d` V channels, `ssm_out` columns |
| apply `+1` to a GGUF norm (double-offset) | §03.1's artifact rule | every Form-A norm, and nothing in Form B |
| add 1 in BF16 instead of F32 | §03.1 | norms, at the bit level |
| read the conv state as outputs instead of inputs | §04.3 consequence 3 | the conv window, and every subsequent step |
| `chunk_decay` off by one position; missing `-inf` mask; unitriangular diagonal dropped; `k_beta` applied to `v`; `q` decayed twice; state stored transposed | §04b.3–§04b.5 | the GDN output and state |
| swapped `e`/`h` halves in `concat(e_norm, h_norm)` | §09.2's column arithmetic | the draft block only |

Every one of these already has a home in the audit's own must-catch set for the artifact
(`quantization/evidence/audit-injections-2026-10-26.json`, `audit-width-negative-controls-2026-10-26.json`);
the new rows are the ones this directory introduced that the artifact audit had no reason to test — the
accumulation order, the `+1`-in-BF16 row, and the draft-block half swap.

## §10.5 Dump-manifest fields a gate is allowed to depend on

| field | why it is mandatory |
|---|---|
| producing command line, interpreter, torch version, thread count | CPU-thread count changes association; a re-run that silently differs in threads is a different arm (`pitfalls/50-harness-and-shell.md` §"Two arms that differ in anything but the thing under test price the difference") |
| seed, token count, dtype, input-generation method | the pinned population must be reproducible, and a fixture list typed from memory is not an admission |
| the **GDN chunk width `C`** that produced any chunked-arm tensor | §04b.7: two widths in one run is how a self-consistent-but-wrong state hides |
| which triangular-solve branch produced a chunked tensor | §04b.4: the two legal branches are different numbers |
| per-record file, bytes, shape, dtype, elements, sha256 | lets the size/dtype rule of §10.1 be enforced rather than trusted |
| `referenced_not_dumped` with shard, tensor, shape, dtype, `data_offsets` | a carried tensor is **named and skipped**, never silently ignored |
| `counts.dumps == counts.expected`, and a section with zero cases **refuses to print a table** | `pitfalls/40` §"A driver whose parser matches nothing prints PASS"; a counter that reads exactly 0 is vacuous until witnessed non-zero where it must fire |

## §10.6 The gate list `speccheck/` owes this directory

| gate | spec file it derives from | headline assertion |
|---|---|---|
| router | §11.2 | F32 softmax over 256, top-8, F32 renorm, **cast to the logits dtype before the multiply** — `topk_weights` bit-exact against the pinned arm |
| selection | §11.3 | `topk_ids` set-equal per row and slot-order equal, compared along the arm's own slot order |
| expert body | §11.4 | the `chunk(2)` split at `I_e` with gate first, in both fused and split spellings |
| routed sum | §11.5 | ascending-expert-id accumulation reproduces `routed_sum` |
| shared expert | §11.6 | scalar sigmoid gate, plain add, no scaling constant |
| norms | §03.1 | Form A vs Form B distinguished; `+1` applied per artifact, in F32 |
| GDN recurrent | §04.4–§04.7 | decay → delta → write → **read post-update state**, all in F32 |
| GDN chunked | §04b | equal to the recurrent form within the probe's band, and the prefill/decode handoff row of §04b.6 |
| attention | §05 | F32 softmax then downcast before PV; gate applied after reshape, before `o_proj` |
| rope | §05.4 | only dims 0..63 rotate, pairing `(i, i+32)`; mrope inert for text |
| state | §08 | `S` and the conv window change when they must and not when they must not |
| MTP | §09 | greedy spec+verify emits exactly the plain-greedy token sequence (§09.5) |
| dequant | §10.2 | our own reader, band from §10.3, every §10.4 injection firing only where it belongs |

Each ships with a `RESULT` line, a parsed-record-count assertion, and its negative controls, or it is
not done (`reference-implementation.md` §"Gate discipline to imitate").
