# ASSUMPTIONS — every `[REF]` and `[ASSUMPTION A-xx]` this spec makes

This is the registry `specification/README.md` §Tag conventions points at: one entry per id, each with
its statement, where it came from, what it risks, the test or gate that would falsify it, and the
file§label that consumes it. Where the sources are silent or disagree, the question is **registered
here rather than answered by taste** (`specification/README.md` §"Ground truth, in precedence order",
`AGENTS.md` §"Before you write anything, decide nothing alone"). Nothing in this file decides a
product, architecture or width question; every entry below transcribes a position the authoring lane
already took in the file that cites it.

**Statuses used below.** *recorded* — the sources settle it and the risk is in the reading;
*open — witness pending* — the rule is settled and the measurement that would prove it bites has not
been seen yet; *unresolved — not this spec's to decide* — the sources are silent, no gate on this box
can close it, and it goes to the user (`DECISIONS.md`) or to a fetch. A-18 and A-19 are unresolved and
are written as unresolved: they are gaps, not decisions.

**Ids are append-only.** Never renumber and never reuse. An id whose question closes keeps its entry
with a dated close-out line, because a gate that cited `A-05` in a commit message has to keep meaning
the same thing.

## Index

| id | subject | status |
|---|---|---|
| A-01 | MTP is not represented in the HF reference | recorded |
| A-02 | `transformers` 4.57.1 → 5.19.0 drift on *form* | recorded |
| A-03 | `topk`'s tie-break is unspecified | open — witness pending |
| A-04 | the two-witness stop rule | recorded, deliberately undecided |
| A-05 | the router's cast order (prediction 4) | open — witness pending |
| A-06 | `mamba_ssm_dtype` is the scan dtype, not the storage dtype | recorded |
| A-07 | census disagreement on the text bf16 total | recorded — adjudicated 2026-10-26, selftest open |
| A-08 | KV bytes per token | recorded, both numbers kept |
| A-09 | conv-window tap count and dtype | recorded, engine's choice |
| A-10 | the MTP block's element count | recorded, header-witnessed |
| A-11 | what "exact" costs in bytes | recorded, three numbers kept |
| A-12 | GDN V-head tiled↔grouped pairing | recorded, audited bit-exact (1.4a) |
| A-13 | `ssm_a` = `-exp(A_log)`, a recomputation | recorded (D-14) |
| A-14 | the rope-width witness | recorded, probed |
| A-15 | mrope is inert for text | recorded — cited inline, never by id |
| A-16 | the two triangular-solve branches | recorded |
| A-17 | the greedy accept rule | recorded |
| A-18 | which tensor is the draft block's `h` | **unresolved** |
| A-19 | how a second draft step is seeded | **unresolved** |
| A-20 | whether drafted stop ids participate | **unresolved** |
| A-21 | two `A-01` namespaces exist | recorded |

## A-01 — the MTP block is not in the checkpoint's own reference implementation

**Statement.** `transformers 5.19.0` lists `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` on
`Qwen3_5MoePreTrainedModel` / `Qwen3_5MoeForConditionalGeneration`, so loading this checkpoint
**discards all 19 `mtp.*` tensors**. The authoritative *form* of the draft block in this spec is
therefore the only shipped implementation read here — llama.cpp at the pinned commit
`d327bf1f0780f87b8c01b367b61853b0c291d7c4`, `src/models/qwen35moe.cpp`, `graph_mtp` — and not HF.

**Where it came from.** `specification/09-mtp-draft-layer.md` §intro and its §09.7 row A-01. The
capability is witnessed, not asserted: `speccheck/evidence/hf-capability-2026-10-26.txt` prints
`CAP mtp_class_in_module: OK NO_MTP_CLASS_IN_MODULE` and
`CAP keys_to_ignore_on_load_unexpected: OK (['^mtp.*', '^model.visual.*'], ['^mtp.*'])`, produced by
`speccheck/hf_capability_probe.py` through `speccheck/cross_check_hf.py`.

**What it risks.** Every form claim in `specification/09-mtp-draft-layer.md` — the §09.2 concat order,
the 18 steps of §09.3, the §09.4 rule that the block owns exactly one KV layer and no recurrent state
— rests on one producer implementation. If that graph diverges from the trained block, verification
still holds (so the text stays right) and the drafter just gets worse: the symptom is an acceptance
rate nobody explains, not a wrong token.

**Falsifying test.** Two, and they differ in kind. (1) A-17's gate: a spec+verify loop must emit exactly
a plain greedy run's token sequence. That protects output, not attribution, so (2) a `transformers`
release that ships an MTP class for this `model_type` supersedes llama.cpp as the authority for form
(`specification/README.md` §"Ground truth, in precedence order") and §09 must be re-derived from it.
Re-run `speccheck/hf_capability_probe.py` and require `mtp_class_in_module` to stay
`NO_MTP_CLASS_IN_MODULE`; if it does not, this entry is stale.

**Consumers.** `specification/09-mtp-draft-layer.md` §intro + §09.7,
`specification/01-weights-and-shapes.md` §01.7,
`specification/10-reference-dumps-and-gates.md` §10.6's MTP row.

## A-02 — every `[REF]` here was read from `transformers` 5.19.0; the weights are 4.57.1's

**Statement.** The committed config says `transformers_version: 4.57.1`; every `[REF]` quote in this
directory was read out of the installed `transformers 5.19.0` copy of `modeling_qwen3_5_moe.py`. The
spec's form statements are therefore 15 minor versions newer than the version that produced the weights.

**Where it came from.** `specification/00-notation-and-hyperparameters.md` §00.5's last row and
`specification/README.md` §"Ground truth, in precedence order" item 2, which also records that the
installed copy's location is a reading location and is not cited as the source of any fact.

**What it risks.** Form drift is invisible here in a way weight drift is not: the CPU oracle and the
pinned dumps both come from the *same* installed copy, so a changed cast position, a moved multiply or a
re-ordered op would agree with itself and disagree with the checkpoint. The load-bearing places are the
ones where one cast position is the whole claim — §11.2 R5, §04.7's intermediate cast, §03.1's F32
`(1 + w)` — and no band or floor from another project or another version transfers
(`AGENTS.md` §"Traps that do NOT transfer").

**Falsifying test.** `checkpoints/dump-manifest-2026-10-26.json` records `interpreter.executable`,
`interpreter.version`, `interpreter.torch` and `threads`, and
`specification/10-reference-dumps-and-gates.md` §10.5 makes those fields mandatory. A dump produced by a
different interpreter than the one a `[REF]` was quoted against is not comparable to it: re-run
`speccheck/cross_check_hf.py` (which archives under `speccheck/evidence/`) and re-quote the symbol, or
re-dump and re-probe. This registry records the exposure; **no 4.57.1-vs-5.19.0 quote comparison has
been run**, so no entry here claims the drift is benign.

**Consumers.** `specification/00-notation-and-hyperparameters.md` §00.5 and every `[REF]` row in the
directory; `specification/10-reference-dumps-and-gates.md` §10.5's interpreter row.

## A-03 — `topk`'s tie-break is not specified by this spec

**Statement.** Behaviour of `topk(probs, 8)` on an exact tie among the 256 F32 softmax probabilities is
not specified here, and this spec defines no tie-break of its own. Two correct implementations can
therefore select different 8-expert sets, or the same set in different slots, with no arithmetic error
anywhere (`specification/11-moe-block.md` §11.3).

**Where it came from.** `specification/11-moe-block.md` §11.3's "ties in `probs`" row; the manifest's
`reading_rules."slot axis"` — the `_exact` arm's slots follow its own fp64 router's rank order and can
permute a tie relative to the bf16 router, so those arms are compared along the slot order sorted by
`topk_ids_exact`.

**What it risks.** An expert swap on a near-tie row changes every downstream bit of that token's MoE
output. Slot-order comparisons across arms fail on rows that are not bugs, and a gate that claims to
cover ties without a tie row in its population is vacuous
(`pitfalls/40-gates-and-admissions.md` §"A negative control with an empty must-catch set is not a
control").

**Falsifying test.** Stated by §11.3 and still owed: **the pinned dump must contain a row whose 8th/9th
margins are under one BF16 ULP of the 8th value before any gate claims to cover ties.** The population
is witnessed non-zero, but by a different instrument: `speccheck/evidence/cross-check-2026-10-26.txt`
prints `top-k id rows checked 128, set-disagreements 0` against the reference, and by arm 6/128 rows
(4.69 %) set-disagree for `exact` and 6/128 for `softmax_bf16` while `ref`, `spec`, `cast_after` and
`no_renorm` hold at 0/128. That is evidence that near-ties exist in the 32 × 4 pinned rows; it is not
the 8th/9th margin measurement, so this entry stays open until that margin is printed.

**Consumers.** `specification/11-moe-block.md` §11.3,
`specification/10-reference-dumps-and-gates.md` §10.1's slot rule and §10.3 step 7,
`specification/09-mtp-draft-layer.md` §09.5's draft/verify compare.

## A-04 — the stop rule: two committed witnesses, neither promoted

**Statement.** This spec does not decide which token id stops generation. It defines the stop set as a
**named input to the decode loop, `STOP_IDS`**, with both committed witnesses recorded and neither
promoted: `text_config.eos_token_id` = **248044** (which is also `bos_token_id`) per the model config,
and `tokenizer.ggml.eos_token_id` = **248046** per the published artifact's own tokenizer metadata. Ids
are named by number only and never printed literally, in this file or in any gate output or log
(`AGENTS.md` §"Never print chat-template special tokens").

**Where it came from.** `specification/02-tokenization-and-embedding.md` §02.4 (the id table and the
paragraph that ends "this spec does not pick a winner"),
`specification/00-notation-and-hyperparameters.md` §00.2's last rows,
`specification/07-decode-step.md` §07.5's `STOP_IDS` row.

**What it risks.** Both directions fail silently and legibly-but-wrong: too narrow and the model
generates past the real end of turn; too wide and it truncates legitimate output the first time 248044
appears mid-turn — and 248044 being *also* bos is exactly what makes the wide set tempting. The sibling
project's rule ("248046 stops") was a fact about *its* checkpoint and may not be carried over
unexamined (`README.md` §Target).

**Falsifying test.** Stated verbatim in §02.4: fetch this checkpoint's `generation_config.json` (or read
a pinned HF generation run) and require `STOP_IDS` to equal its `eos_token_id` list; if the two
witnesses disagree with it, **the config wins** and the GGUF row is annotated as a producer-side
artefact. The adjudicating file is not on disk — `checkpoints/` holds `config.json`, the shard index,
the GGUF pin and this manifest (`checkpoints/README.md`).

**Consumers.** `specification/02-tokenization-and-embedding.md` §02.4,
`specification/07-decode-step.md` §07.5, `specification/09-mtp-draft-layer.md` §09.5 (through A-20),
`specification/00-notation-and-hyperparameters.md` §00.2.

## A-05 — the router's cast order, and whether the pinned rows actually separate it

**Statement.** `w = cast_to_dtype_of(logits)(w)` **before** the expert-output multiply (§11.2 R5, as
`Qwen3_5MoeTopKRouter.forward` writes it). Two things are separated deliberately: the *rule* — the
`speccheck/` oracle reproduces R5 — is settled; the *prediction* that the pinned rows show the two forms
disagreeing is not. The tempting rewrite (renormalise in F32, multiply in F32, cast after) is a
different number: it keeps 8 mantissa bits the reference threw away, changing all 8 scaled outputs, and
on a near-tie row it can change which 8 experts were picked (A-03 is the mechanism).

**Where it came from.** `specification/11-moe-block.md` §11.2 (the `[REF]` quote and the R1–R5 table),
`specification/00-notation-and-hyperparameters.md` §00.4's "routing-weight multiply" row, and
`AGENTS.md` prediction 4.

**What it risks.** An engine that "just" keeps the weights wide is not faster by anything this spec can
promise, and is a different model. The reverse risk is the empty-gate shape: a router gate that passes
because both arms agree on the pinned rows **measures nothing**.

**Falsifying test.** Both arms are already on disk: `checkpoints/dump-manifest-2026-10-26.json` carries
`topk_weights` (cast-before, authoritative per its own `reading_rules."authoritative"`),
`topk_weights_f32` (pre-cast), and the `_spec` / `_exact` arms of `expert_raw`, `routed_sum`,
`combined_out` — 104 dumps with `counts.dumps == counts.expected == 104`. Falsification is a dump in
which the two arms are bit-identical on every row **and** `routed_sum_spec == routed_sum_exact`: that
would mean the pinned rows never exercise R5, `AGENTS.md` prediction 4 stays unfalsified, and any claim
that the router gate "covers" the cast order is vacuous. The gate itself is
`specification/10-reference-dumps-and-gates.md` §10.6's router row (`topk_weights` bit-exact against the
pinned arm), and §10.4's injection "cast routing weights after the multiply instead of before" must fire
on `routed_sum` and `combined_out`.

**Consumers.** `specification/11-moe-block.md` §11.2 and §11.8,
`specification/10-reference-dumps-and-gates.md` §10.1's cast-order-arm row,
`specification/00-notation-and-hyperparameters.md` §00.4.

## A-06 — `mamba_ssm_dtype` is the scan dtype, not the storage dtype

**Statement.** `text_config.mamba_ssm_dtype: float32` names the dtype of the **scan** — the state, the
decay, the q/k l2norms, the `1/sqrt(d_k)` scaling and the scan output — and says nothing about the
tensors feeding it. `A_log`, `dt_bias`, `in_proj_a/b` and `conv1d` are BF16 in the checkpoint, and the
published file stores that control family F32.

**Where it came from.** `specification/00-notation-and-hyperparameters.md` §00.4's closing paragraph and
`specification/04-gdn-linear-attention.md` §04.4's scan-dtype row.

**What it risks.** Two opposite silent errors. Reading the config as a storage claim and keeping `S` in
BF16 halves the largest state in the project (60.00 MiB per request) and is a numerics change owed a
floor probe, not a resident-size knob (`specification/08-state-management.md` §08.1). Reading the
*weights* as F32 because the config says float32 mis-sizes and mis-loads 150 control tensors.

**Falsifying test.** `python3 specification/derive.py --check` asserts the checkpoint dtypes from real
safetensors headers (and `--families` prints them as witnesses: `A_log`, `dt_bias`, `linear_attn.norm`,
`conv1d`, `mlp.gate` all BF16), while `python3 specification/derive.py --state` prints the F32 recurrent
row at 2,097,152 B per layer. A build whose state pool is sized BF16 while its scan is F32 — or the
reverse — contradicts one of those two prints, and
`specification/10-reference-dumps-and-gates.md` §10.6's state row is the gate.

**Consumers.** `specification/00-notation-and-hyperparameters.md` §00.4,
`specification/04-gdn-linear-attention.md` §04.4, `specification/08-state-management.md` §08.1,
`specification/06-prefill-pass.md` §06.1.

## A-07 — the census disagreement on the text bf16 total (adjudicated 2026-10-26)

**Statement.** The text bf16 total this spec quotes is **66.1337 GiB**: 712 names derived from config
geometry, their name set IDENTICAL to the committed index's 712 text names, and
`index − vision = 71,010,502,912 B = derived text × 2 B`, delta **0 B / 0 elements**. The instrument this
directory was written against, `python3 tools/quant_scenarios.py --census`, printed **65.720 GiB** for the
same population with the index delta explained as 1.86 % "vision tower + small tensors". That figure was
**wrong by −0.4138 GiB**, not merely cut differently, and the disagreement is adjudicated in
`quantization/evidence/bf16-census-adjudication-2026-10-26.md`: 66.1337 GiB is authoritative; the
artifact-side **64.5604 GiB** is a *different population* (the GGUF carries no draft block — exactly the 19
`mtp.*` names, 1.5733 GiB bf16); and the old census held two phantoms and two omissions — an absent GDN
`linear_attn.out_proj`/`ssm_out` (−0.46852 GiB), an absent `mtp.fc.weight` (−0.01563 GiB), a
`norms_misc` term `3*h*(3*h)` that maps to nothing (+0.07031 GiB), a `+ 2*h` per linear layer that maps to
nothing (+0.00046 GiB), and a norm count of 84 where 86 exist.

**Where it came from.** `specification/01-weights-and-shapes.md` §01.2 and
`specification/00-notation-and-hyperparameters.md` §00.6 (both written against the pre-repair instrument),
and the committed adjudication `quantization/evidence/bf16-census-adjudication-2026-10-26.md` §1
(verdict), §3 (the three rulers reconciled term by term to 0 bytes) and §5 (the term-by-term verdict on
`tools/quant_scenarios.py`). Its calculator is `tools/bf16_census.py`.

**What it risks.** A residency decision taken on a wrong column — the whole width argument turns on a few
GiB, and the adjudication is explicit that D-01/D-11 survive at every candidate number only because 64.56,
65.72 and 66.13 all sit above the ≈55.5–56.5 GiB box-wide pool
(`devices/measured-facts.md` §"Residency above the pool"), while the corrected policy A is 19.7848 GiB
resident. The live risk is now **staleness, not ambiguity**: `01` §01.2 still says the instrument "reports
65.720 GiB and a 1.86 % delta" and `specification/derive.py`'s `--families` footer still prints that
description, so a reader who follows the pointer this entry is named in gets an explanation of a ruler that
no longer prints those numbers.

**Falsifying test.** Three commands, and only the first is allowed to be quiet. (1)
`python3 specification/derive.py --families` must keep printing `delta 0 B (0 elements)` — the identity,
not the GiB figure, because it is the only one of the two that *can* fail. (2)
`python3 tools/bf16_census.py` must reprint the four closures with no residuals. (3)
`python3 tools/bf16_census.py --selftest` must fire **every** injected fault: as this entry was written it
reported `7 faults injected, 5 fired, 2 missed — FAIL`, with `C4 quant_scenarios delta: -444358656` left
unexplained, and `tools/quant_scenarios.py` carried uncommitted edits at the same moment (a repair in flight
in another lane). That selftest going green is the witness that closes this entry, and this file does not
claim it yet.

**Consumers.** `specification/00-notation-and-hyperparameters.md` §00.6,
`specification/01-weights-and-shapes.md` §01.2, `specification/README.md` §"How to check this
directory". `specification/derive.py`'s `--families` footer names this id itself.

## A-08 — KV bytes per token: 20.0 KiB/token, not 10

**Statement.** KV is **2048 B per layer per token** — `2 (K,V) × n_kv 2 × d_head 256 × 2 B` — hence
**20,480 B = 20.0 KiB/token** over the 10 trunk full-attention layers, and 22.0 KiB/token with the draft
block's own layer. `README.md` §"Memory and roofline arithmetic" states 10 KiB/token over the same 10
layers.

**Where it came from.** `specification/05-gated-full-attention.md` §05.7,
`specification/08-state-management.md` §08.1 + §08.2,
`specification/00-notation-and-hyperparameters.md` §00.6, `specification/06-prefill-pass.md` §06.1.

**What it risks.** The concurrency budget is off by a factor of 2 on the KV term: at 4096 lanes that is
80.00 MiB of KV, not 40.00 MiB, so a request's total is 142.81 MiB rather than 102.81 MiB and a 1 GiB
state pool holds **7.2** requests at 4096 lanes, not the ~10 that the halved row implies
(`specification/08-state-management.md` §08.2). Where capacity is the hard constraint (`AGENTS.md`
§SCOPE, `devices/measured-facts.md` §"Residency above the pool"), halving the state estimate is how a 32k
plan gets funded twice over.

**Falsifying test.** `python3 specification/derive.py --state` prints
`10 layers x 2048 B = 20480 B = 20.0 KiB per token` and the whole per-lane table, derived from
`checkpoints/Qwen3.6-35B-A3B.config.json`; the falsifier is a config or header read that makes
`2 × 2 × 256 × 2 ≠ 2048`. That row also contradicts itself on its own terms: it says "10 KiB/token …
instead of 6, so ~1.7× the 0.8B's 12 KiB/token", and `12 × 1.67 ≈ 20`, which is the spec's number
(`--state` prints the same observation).

**Consumers.** `specification/00-notation-and-hyperparameters.md` §00.6,
`specification/05-gated-full-attention.md` §05.7, `specification/08-state-management.md` §08.1 + §08.2,
`specification/06-prefill-pass.md` §06.1. `README.md` keeps its number; nothing outside this directory
was edited.

## A-09 — the conv window: which taps, which dtype, at what price

**Statement.** The GDN conv state is a window of previous `mixed_qkv` **inputs**, never of conv outputs.
`k_conv = 4` columns is what the reference's cache keeps; `k_conv − 1 = 3` is sufficient for the next
output. Its dtype is the activation dtype in the reference (`zeros_like` of the BF16 conv input); F32 is
the other legal choice. Which window and which dtype the engine stores is an **engine choice, not a
config fact** — so this entry registers it, with both prices.

**Where it came from.** `specification/04-gdn-linear-attention.md` §04.3 consequence 3 and §04.10,
`specification/08-state-management.md` §08.1 (the row and the "dtype of the conv window" rule),
`specification/06-prefill-pass.md` §06.1.

**What it risks.** 2× on the row: per request across 30 layers, 4 taps F32 3.750 MiB / BF16 1.875 MiB,
3 taps F32 2.812 MiB / **BF16 1.406 MiB** — which moves 4096-lane requests per 1 GiB from 7.2 to 7.1
(`specification/08-state-management.md` §08.2). The worse failure is orthogonal and silent: pinning or
reading the conv **output** window reproduces a real-looking tensor that is the wrong quantity
(`pitfalls/30-numerics.md` §"A hook that reads `output` can pin the answer under the name of an
intermediate").

**Falsifying test.** `python3 specification/derive.py --state` prints all four rows (4/3 taps × F32/BF16)
and the totals that depend on them. `specification/08-state-management.md` §08.5 requires the window of
one linear layer pinned **before and after** one decode step, and
`specification/10-reference-dumps-and-gates.md` §10.4's injection "read the conv state as outputs
instead of inputs" must fire on the conv window and on every subsequent step, and on nothing else.

**Consumers.** `specification/04-gdn-linear-attention.md` §04.3 + §04.10,
`specification/06-prefill-pass.md` §06.1, `specification/08-state-management.md` §08.1 + §08.5.

## A-10 — the MTP block's element count: 844,640,768, not 41,943,040

**Statement.** The draft block is **19 tensors / 844,640,768 elements = 1,611.024 MiB BF16**, of which
805,306,368 elements (**95.34 %**) are its own two fused expert tensors
(`mtp.layers.0.mlp.experts.gate_up_proj` [256,1024,2048] and `…down_proj` [256,2048,512]).
`PLAN.md`'s fetch table priced it at 41,943,040 elts / 79.9 MiB — **wrong by 20.14×** — because it
omitted the block's own 256-expert MoE stack.

**Where it came from.** `specification/09-mtp-draft-layer.md` §09.1, `specification/01-weights-and-shapes.md`
§01.7, and `PLAN.md` §Step 1's fetch row for shards 25/26, which carries the correction in place (that
file was not edited from this task). `python3 specification/derive.py --mtp` prints the delta explicitly.

**What it risks.** A fetch/staging plan sized for 79.9 MiB of a 1.58 GiB block; and, in the other
direction, `DECISIONS.md` D-09's "uniform with the trunk" width decision being argued from a block whose
dominant term was missing — which is also why `AGENTS.md`'s census rule is that a family number must
name its instrument.

**Falsifying test.** `python3 specification/derive.py --mtp` reads the real safetensors headers of
`models/qwen36-src/model-00025-of-00026.safetensors` and `model-00026-of-00026.safetensors` and prints
`TOTAL 19 tensors 844640768 elems 1611.024 MiB` with `derived-from-config vs header: IDENTICAL names`
and every derived shape equal to the header's. A header that moves that count falsifies this entry and
the two files that cite it.

**Consumers.** `specification/09-mtp-draft-layer.md` §09.1, `specification/01-weights-and-shapes.md`
§01.7, `specification/10-reference-dumps-and-gates.md` §10.4's draft-block half-swap row.

## A-11 — what "exact" costs in bytes, and which set each figure prices

**Statement.** "Exact" is a precision class (never stored in a block format), **not** a byte count, and
the figures quoted for it price different populations — one of which contains a term that maps to no
tensor at all:

| figure | where it is stated | what it actually counts |
|---|---|---|
| 0.11 GiB | `quantization/README.md` §"How a quantized path gets gated" item 3 — "Router, norms, conv1d, `A_log`, `dt_bias` stay exact … they are 0.11 GiB" | `tools/quant_scenarios.py`'s `norms_misc` 0.071 + `routers` 0.040 at bf16 — and `norms_misc` contains the phantom `3*h*(3*h)` = 9h² term worth +0.07031 GiB, so almost none of that 0.071 is a norm (`quantization/evidence/bf16-census-adjudication-2026-10-26.md` §5 row 3) |
| ~0.0403 GiB | `specification/01-weights-and-shapes.md` §01.2's census rows | routers 0.0400 + norms 0.0003 + gdn_control 0.0000 at bf16, `linear_attn.out_proj` counted in `gdn` and `conv1d` in the GDN family — a different set from item 3's list |
| 0.0974 GiB | `python3 tools/gguf_width_table.py … --check-header`, F32 row | the 361 F32 tensors the shipping artifact actually stores: every router, every norm, `ffn_gate_inp_shexp`, `ssm_conv1d`, `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_dt.bias`, `ssm_norm` — at 4 B per element, i.e. 2× the bf16 price the "exact" label carries |

The adjudication names the same ambiguity as a **width decision rather than a bug**: whether "stays exact"
means bf16 or the F32 the published file carries is worth 0.0403 GiB each on residency and per-token, and
whether the GDN control tensors ride in the attention family (Q8_0) or the exact family is worth a further
0.00075 GiB resident (`quantization/evidence/bf16-census-adjudication-2026-10-26.md` §5 rows 6–7).

**Where it came from.** `specification/01-weights-and-shapes.md` §01.4's type census and §01.5's three
"✓ (wider than 'exact')" rows; `specification/10-reference-dumps-and-gates.md` §10.2's zero-tolerance rows;
`quantization/README.md` item 3; the adjudication rows above. `README.md` §"Memory and roofline
arithmetic" carried a matching "router + norms + misc 0.11" until 2026-10-26 and now prints the
adjudicated component split instead, so the 0.11 figure survives only in `quantization/README.md`.

**What it risks.** An engine that allocates 0.11 GiB for the exact families while loading an F32-stored
artifact under-provisions. The gating risk is sharper: a reader who hears "exact" and compares against the
file instead of the checkpoint has dropped the oracle. Exactness (never quantized) and the comparison
authority (the checkpoint's bf16 value widened to F32, with the converter's documented transforms, per
`specification/10-reference-dumps-and-gates.md` §10.2) are different obligations, and conflating them is
how two readers of one artifact agree with each other and both disagree with the model.

**Falsifying test.** `python3 tools/gguf_width_table.py models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --check-header`
prints the F32 census row — 361 tensors, 104,581,632 B, 0.0974 GiB, 32.0000 bpw — under the witness
`sum(row_bytes) + data_start == file size`; `python3 specification/derive.py --families` prints this
model's own bf16 exact-family rows; `python3 tools/bf16_census.py` re-derives the populations and refuses
to close if a phantom term is present (its `--selftest` injects exactly that fault). Any figure quoted for
"exact" must name which population it means, or it is not checkable.

**Consumers.** `specification/01-weights-and-shapes.md` §01.4 + §01.5,
`specification/10-reference-dumps-and-gates.md` §10.2, `specification/07-decode-step.md` §07.4's router
row (F32 storage is why that line reads 0.0801 GiB/token against 0.0400 bf16).

## A-12 — GDN V heads: the file is tiled, HF is grouped, and pairing must follow the order

**Statement.** HF stores V heads **grouped** by their K head (`repeat_interleave(2, dim=2)`, so V head
`j` pairs with K head `j // 2`). The published file stores them **tiled**: its converter reshapes the V
rows to `[n_k, n_v/n_k, head_v_dim]`, swaps the first two axes and flattens, so GGUF V head `j` holds HF
V head `(j % 16) * 2 + (j // 16)` and the K↔V pairing becomes `j % 16`. Head order and pairing must come
from the same assumption: tiled order read with the grouped pairing reads the wrong K head for half the
V heads and raises nothing.

**Where it came from.** `specification/01-weights-and-shapes.md` §01.6 item 1 (with the tensor-by-tensor
list: the V third of `attn_qkv`, all of `attn_gate`, the rows of `ssm_alpha`/`ssm_beta`, `ssm_a`,
`ssm_dt.bias`, the V channels of `ssm_conv1d`, the columns of `ssm_out`) and
`specification/04-gdn-linear-attention.md` §04.4's head-expansion row + §04.9.

**What it risks.** 30 GDN layers × 8 tensor families of mis-homed heads, producing fluent text. And the
mirror error in a loader that starts from HF order and forgets to permute. Both are invisible to any test
that does not compare against the checkpoint — the file is self-consistent either way.

**Falsifying test.** The artifact was audited and the reorder is **bit-exact after the documented
transform**: `quantization/evidence/audit-published-quant-2026-10-26.md` §1.4a (copied families at zero
tolerance — routers, norms, `dt_bias`, `alpha`, `beta`, `conv1d` and the order/reorder questions — PASS)
records 10 of 10 judged GDN tensors in the tiled order. The runtime-side falsifier is
`specification/10-reference-dumps-and-gates.md` §10.4's injection "V-head order: read tiled as grouped
(or the reverse)", which must fire on exactly the tensors §04.9 names and on nothing else, with the GDN
recurrent and state rows of §10.6 still green.

**Consumers.** `specification/01-weights-and-shapes.md` §01.6 item 1,
`specification/04-gdn-linear-attention.md` §04.4 + §04.9,
`specification/10-reference-dumps-and-gates.md` §10.2 + §10.4.

## A-13 — `ssm_a` is `-exp(A_log)`: a recomputation, so it has a 1-ulp history

**Statement.** The file does not store `A_log`; it holds `-exp(A_log)` under `blk.N.ssm_a`, computed by
the producer at conversion time (`conversion/qwen.py`: `data_torch = -torch.exp(data_torch)`). It is a
**derived** quantity with a producer-libm history, so its exactness obligation is
`DECISIONS.md` D-14's 1.4b route predicate and never "bit-exact with the checkpoint".

**Where it came from.** `specification/01-weights-and-shapes.md` §01.6 item 6,
`specification/04-gdn-linear-attention.md` §04.5's "what the file stores" row, and
`specification/10-reference-dumps-and-gates.md` §10.2's `ssm_a` row plus the paragraph after that table.

**What it risks.** Two symmetric failures. Calling it bit-exact with the checkpoint, which has already
been done wrongly once here: `blk.1.ssm_a` element 21 sits one f32 step from the value both
correctly-rounded routes agree on. And the reverse drift: treating a route allowance as licence to loosen
a *copied* family, which D-14 explicitly does not grant — copied families keep zero tolerance.

**Falsifying test.** The 1.4b predicate, run by `python3 tools/audit_published_quant.py`: two
independently computed correctly-rounded routes (a double `exp` rounded to f32, and a 60-digit `Decimal`
`exp`) must agree **with each other** bit-for-bit — a route disagreement is a FAIL, never "within
tolerance" — and the file must lie within **exactly 1 ulp**, the ulp taken from `nextafter` on the
value's own exponent. The must-catch set beside the allowance must keep firing: `ssma_element_perm`,
`ssma_sign_flip`, `ssma_step_2_down`, `ssma_step_2_up`, `ssma_wrong_authority_read` (2/2 each, with the
±1-step cases correctly staying clean) in
`quantization/evidence/audit-published-quant-2026-10-26.md` §1.4b/§1.5. The original zero-tolerance FAIL
line stays in `quantization/evidence/audit-exactness-2026-10-26.json` beside the new verdict.

**Consumers.** `specification/01-weights-and-shapes.md` §01.6 item 6,
`specification/04-gdn-linear-attention.md` §04.5,
`specification/10-reference-dumps-and-gates.md` §10.2, `DECISIONS.md` D-14.

## A-14 — the rope-width witness: `R = 64`, 32 pairs, only dims 0..63 rotate

**Statement.** `R = int(d_head × partial_rotary_factor) = int(256 × 0.25) = 64`; 32 frequency pairs from
`arange(0, R, 2) / R` with `theta = 1e7`; `cos`/`sin` are those 32 values written twice (`cat(f, f)`) and
are computed in F32 then cast to the activation dtype on return; the rotation replaces the **first 64**
dims of each 256-wide head, pairing `i` with `i + 32`, and dims `64..255` pass through untouched.
`specification/05-gated-full-attention.md` §05.4 states all of this and defers its witness here by name
("command and output in `specification/ASSUMPTIONS.md` A-14's witness").

**Where it came from.** `specification/05-gated-full-attention.md` §05.4 and its facts table,
`specification/00-notation-and-hyperparameters.md` §00.3's `R` and pair-count rows,
`specification/06-prefill-pass.md` §06.2 step 2, `specification/10-reference-dumps-and-gates.md` §10.6's
rope row.

**What it risks.** The whole partial-rotary geometry. A build that rotates all 256 dims (reading the
factor as 1.0, which is what the rotary module defaults to when `rope_parameters` is absent) or that
pairs `i` with `i + 128` because it derived `head_dim / 2` instead of `R / 2` produces fluent text and
raises nothing. §05.4's claim that "an engine reading either source gets 64" — the config product versus
the producer's `qwen35moe` rope dimension count — is only true if both are checked.

**Falsifying test — the probe.** CPU-only, loads no weights, builds the checkpoint's OWN rotary module
from the committed config and prints each claim as True/False. Any False falsifies this entry. Run from
the repo root as `python3 <this block saved as a file>`, with the CPU torch interpreter recorded in the
committed `speccheck/evidence/hf-capability-2026-10-26.txt` header argv (a reading location only,
`specification/README.md` §"Ground truth" item 2).

```python
"""specification/ASSUMPTIONS.md A-14's rope-width witness. CPU-only, loads no weights.

Builds the checkpoint's OWN rotary module from the committed config and prints the claims
specification/05-gated-full-attention.md §05.4 makes about it. Every claim prints True/False.
"""
import json
import resource

resource.setrlimit(resource.RLIMIT_AS, (int(8e9), int(8e9)))   # a rotary needs ~0 B; this is a guard

import torch
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

CFG = "checkpoints/Qwen3.6-35B-A3B.config.json"
cfg = Qwen3_5MoeTextConfig(**json.load(open(CFG))["text_config"])
B, T, HD = 1, 5, cfg.head_dim                       # HD = 256
rot = M.Qwen3_5MoeTextRotaryEmbedding(cfg)
x = torch.zeros(B, T, cfg.hidden_size, dtype=torch.bfloat16)
# exactly the [3, B, T] T/H/W block Qwen3_5MoeTextModel.forward hands the rotary for text:
# rows 1..3 of the hard-coded 4-axis (text, temporal, height, width) arange, one integer each.
pos = torch.arange(T).view(1, 1, -1).expand(4, B, -1)[1:]
cos, sin = rot(x, pos)                              # [B, T, R], activation dtype
R = cos.shape[-1]

print("inv_freq pairs      :", tuple(rot.inv_freq.shape), "(expect 32)")
print("cos width / dtype   :", R, cos.dtype, "(expect 64, bfloat16)")
print("cos == cat(c, c)    :", bool(torch.equal(cos, torch.cat((cos[..., :R // 2],
                                                               cos[..., :R // 2]), -1))))
print("cos[7] == cos[39]   :", bool(torch.equal(cos[..., 7], cos[..., 39])),
      "sin[7] == sin[39]", bool(torch.equal(sin[..., 7], sin[..., 39])))

g = torch.Generator().manual_seed(20261026)
q = torch.randn(B, 2, T, HD, generator=g)            # F32 so the checked arithmetic is exact
qc, _ = M.apply_rotary_pos_emb(q, q.clone(), cos.float(), sin.float(), unsqueeze_dim=1)
i, j, t = 7, 39, 3                                   # the pair (7, 39) of §05.4
want = torch.stack([q[0, 0, t, i] * cos[0, t, i].float() - q[0, 0, t, j] * sin[0, t, i].float(),
                    q[0, 0, t, j] * cos[0, t, j].float() + q[0, 0, t, i] * sin[0, t, j].float()])
print("pair (7,39) matches :", bool(torch.allclose(qc[0, 0, t, [i, j]], want, atol=1e-6)))

touched = [d for d in range(HD) if not torch.equal(qc[0, 0, t, d], q[0, 0, t, d])]
print("rotated indices at  :", (min(touched), max(touched)),
      "| dims 64:256 pass through", bool(torch.equal(qc[0, 0, t, 64:], q[0, 0, t, 64:])))

pert = q.clone()
pert[..., 100] += 1.0
pc, _ = M.apply_rotary_pos_emb(pert, pert.clone(), cos.float(), sin.float(), unsqueeze_dim=1)
print("perturb dim 100     :", [d for d in range(HD)
                                if not torch.equal(pc[0, 0, t, d], qc[0, 0, t, d])], "(expect [100])")

# mrope inertness for text: [11, 11, 10] is a no-op when T, H and W carry one integer (A-15).
f = (torch.arange(T).float()[:, None] * rot.inv_freq.float()).cos()      # [T, 32], ONE position row
print("mrope inert         :", bool(torch.equal(cos[0].float(),
                                                torch.cat((f, f), -1).to(torch.bfloat16).float())),
      "mrope_section", rot.mrope_section)
print("R == int(head_dim * partial_rotary_factor) :", R,
      int(HD * cfg.rope_parameters["partial_rotary_factor"]))
```

Witnessed output, re-run for this registry (project day 2026-10-26, the date the other same-session
evidence files carry); every claim line reads True, and `rotated indices at (0, 63)` is the containment
claim — the *count* of visibly-changed dims is q-dependent and is not a spec claim:

```
inv_freq pairs      : (32,) (expect 32)
cos width / dtype   : 64 torch.bfloat16 (expect 64, bfloat16)
cos == cat(c, c)    : True
cos[7] == cos[39]   : True sin[7] == sin[39] True
pair (7,39) matches : True
rotated indices at  : (0, 63) | dims 64:256 pass through True
perturb dim 100     : [100] (expect [100])
mrope inert         : True mrope_section [11, 11, 10]
R == int(head_dim * partial_rotary_factor) : 64 64
```

**Consumers.** `specification/05-gated-full-attention.md` §05.4,
`specification/00-notation-and-hyperparameters.md` §00.3, `specification/06-prefill-pass.md` §06.2 step 2,
`specification/10-reference-dumps-and-gates.md` §10.6's rope row ("only dims 0..63 rotate, pairing
`(i, i+32)`").

## A-15 — mrope is inert for text, and that is a claim, not an absence

**Statement.** `mrope_interleaved: true` with `mrope_section: [11, 11, 10]` selects, for each of the 32
frequency columns, which position axis supplies `pos`: column `j` takes axis T unless
`j % 3 == 1 and j < 33` (axis H) or `j % 3 == 2 and j < 30` (axis W), and `11 + 11 + 10 = 32`. For
text-only the four position rows carry the **same integer**, so the section assignment changes nothing and
the rotation is exactly the ordinary 1-D one of §05.4. The sections start to matter only if a vision/grid
path is added, which is out of scope (`DECISIONS.md` D-04).

**Where it came from.** `specification/05-gated-full-attention.md` §05.4's "mrope for text-only is inert"
paragraph (with the `recomposition_frequencies` `[REF]` quote),
`specification/02-tokenization-and-embedding.md` §02.3 (the hard-coded four axes, and the producer's
degenerate `qwen35moe.rope.dimension_sections` I32 × 4),
`specification/00-notation-and-hyperparameters.md` §00.3's mrope column-assignment row, and
`specification/10-reference-dumps-and-gates.md` §10.6's rope row, whose assertion includes "mrope inert
for text".

**What it risks.** Two engines that both look right today and both break later: one that builds the
section machinery but is fed genuinely different H/W axes, and one that hard-codes the 1-D rotation and
drops the machinery. The silent live variant is a cache rewind or continuation path that updates row 0 of
`position_ids` and leaves the T/H/W rows stale — invisible while text broadcasts one integer, wrong the
moment anything reads them.

**Falsifying test.** Recorded in §05.4 and witnessed by A-14's last probe line: `cos` equals
`cat(cos(pos × inv_freq), cos(pos × inv_freq))` built from a **single** position row, with
`mrope_section [11, 11, 10]`. That check alone cannot fail for the right reason — code that never reads
the H/W rows prints the same result as code whose inertness is genuine — so the companion negative
control listed here as **owed, not done**: hand the rotary a `position_ids` whose H and W rows differ from
T and require `cos` to change. Until that is run, this entry's witness is the §10.6 rope row plus the
probe line, not a proof that the machinery is exercised.

**Consumers.** `specification/05-gated-full-attention.md` §05.4,
`specification/02-tokenization-and-embedding.md` §02.3,
`specification/00-notation-and-hyperparameters.md` §00.3,
`specification/10-reference-dumps-and-gates.md` §10.6's rope row.
**Cited inline in all four, by id in none** — see §"What this registry found and did not change".

## A-16 — the two triangular-solve branches are both legal and are not the same number

**Statement.** `specification/04b-gdn-chunked-form.md` §04b.4's UT solve has two legal implementations in
the reference: `solve_triangular(ut_system, ·, upper=False, unitriangular=True)`, and an export-target
branch that keeps `-ut_system.tril(-1)`, forward-substitutes row by row, adds the identity and
left-multiplies. One is a factorisation, the other a row-ordered accumulation, so they are **not the same
number**. This spec pins the first as authoritative for `speccheck/`, and a dump must say which branch
made it.

**Where it came from.** `specification/04b-gdn-chunked-form.md` §04b.4,
`specification/10-reference-dumps-and-gates.md` §10.5's "which triangular-solve branch produced a chunked
tensor" row, and §04b.7's sibling rule for the chunk width `C`.

**What it risks.** The two-self-consistent-layouts failure shape one level up
(`pitfalls/40-gates-and-admissions.md` §"Two self-consistent layouts joined by a branch are invisible to
both sides"): an engine on the solver compared against a dump made by the substitution branch yields a
state that is right in aggregate, wrong in the last bits, and invisible to every injection that is not
literally "wrong branch".

**Falsifying test.** Provenance, not tolerance: a chunked-arm record whose manifest does not name the
branch is refused (`specification/10-reference-dumps-and-gates.md` §10.5, whose counters must satisfy
`counts.dumps == counts.expected`), and the §04b.6 equivalence rows — chunked against recurrent, with the
per-head maximum beside the cos/mean-rel aggregates — must stay inside the floor probe's band **for the
named branch**. A cross-branch comparison is a FAIL of provenance, never a band question.

**Consumers.** `specification/04b-gdn-chunked-form.md` §04b.4 + §04b.6,
`specification/10-reference-dumps-and-gates.md` §10.5, `specification/06-prefill-pass.md` §06.2 step 14.

## A-17 — the greedy accept rule, and the gate that makes acceptance checkable

**Statement.** Draft/verify acceptance is **greedy exact-match plus a bonus token**: accept the longest
prefix with `d_i == t_i`, emit the target's own argmax at the first mismatch as the bonus, advance trunk
state by accepted + bonus, and truncate the draft KV of the rejected positions. The reference's accept
loop is engine-level, so this rule is **this spec's definition**, not a `[REF]` — the graph defines the
drafter, not the acceptance.

**Where it came from.** `specification/09-mtp-draft-layer.md` §09.5's `[ASSUMPTION A-17]` row and the
strong rule beneath it, `specification/08-state-management.md` §08.4's rollback rules,
`specification/07-decode-step.md` §07.5's MTP interaction row.

**What it risks.** A speculative loop that emits tokens plain greedy would not is not "slightly off", it
is broken; and the failure mode is a drafter nobody trusts, which is how MTP gets dropped from the project
after it was paid for. Acceptance **rate** is deliberately not a spec quantity — it is a measurement owned
by the main agent, and the only thing that would justify widening the block (`DECISIONS.md` D-09).

**Falsifying test — the gate, stated in §09.5.** A spec+verify loop must emit **exactly** the token
sequence of a plain greedy run (`specification/07-decode-step.md` §07.5) at the same temperature-free
setting. Any divergence is a bug in acceptance, state rollback or the drafter, and is never adjudicated
"within band"; `specification/10-reference-dumps-and-gates.md` §10.6's MTP row is its home in the gate
list. It is checkable without a corpus, which is why it was chosen.

**Consumers.** `specification/09-mtp-draft-layer.md` §09.5 + §09.7,
`specification/08-state-management.md` §08.4, `specification/07-decode-step.md` §07.5,
`specification/10-reference-dumps-and-gates.md` §10.6's MTP row.

## A-18 — which tensor is the draft block's `h`: **unresolved**

**Statement.** What this spec cannot say from any source on this box: **which tensor is `h`**, the "target
hidden state" that `mtp.pre_fc_norm_hidden` (`blk.40.nextn.hnorm`) norms in §09.2's fusion block — step 2
of §09.3's op order. The one shipped
implementation takes it as a graph input named `mtp_h_input` and does not define it, and the checkpoint's
own reference ignores `mtp.*` entirely (A-01), so nothing here pins it. Registered as a gap; **not
decided by this file**.

**Where it came from.** `specification/09-mtp-draft-layer.md` §09.7 row A-18, the `h` line of the §09.2
fusion block, and step 2 of §09.3. What §09.7 records as known is transcription, not a verdict: the block
carries its **own** `mtp.norm` (`shared_head_norm`), which the lane noted as *consistent with* the tap
being the last trunk hidden state **before** the final norm. Consistency is not witness, and §09.7 writes
it that way.

**What it risks.** It decides what a draft step reads. If the tap is the pre-final-norm hidden state, a
draft step re-norms a residual-stream vector; if it is a post-norm vector, `hnorm` is applied to something
the trunk already normalised and the two halves of `mtp.fc` are fed different distributions — while
A-17's gate still passes, because verification repairs every wrong draft. The cost of getting it wrong is
therefore paid entirely in acceptance rate, the quantity no gate in this directory owns.

**What would close it — and what would not.** Not acceptance rate: a low rate has many causes and is a
measurement, not a witness (`specification/09-mtp-draft-layer.md` §09.5). Two things could close it — (a)
the producer's graph construction or its GGUF-side feeder, read at the pinned commit, showing where
`mtp_h_input` is taken from; (b) a `transformers` release that implements `mtp.*`, which also
supersedes A-01 and forces §09 to be re-derived. Until one of them exists the candidate readings stay
open and the question is the user's (`AGENTS.md` §"Before you write anything, decide nothing alone").

**Consumers.** `specification/09-mtp-draft-layer.md` §09.2, §09.3 step 2, §09.7.

## A-19 — how a second and later draft step is seeded: **unresolved**

**Statement.** What this spec cannot say: how draft step 2..k is seeded — what its token input is, and
what its hidden input is — given `text_config.mtp_num_hidden_layers: 1` `[CFG]` and a checkpoint described
as trained with multi-steps. There is one block; recursing over it is a runtime decision that no source
here records.

**Where it came from.** `specification/09-mtp-draft-layer.md` §09.7 row A-19, the `e` line of §09.2 (the
candidate token's trunk-embedding gather, marked "see §09.7 A-19") and the `d_1..d_k` loop of §09.5 ("see
A-19 for the multi-step recursion"). What §09.7 records as known: the loader asserts `n_layer_nextn == 1`,
and the two candidate forms differ in whether the second step reads the **trunk** hidden state again or
the draft block's **own** output.

**What it risks.** k > 1 is where drafting earns its keep or does not, and it is untheorised here: the two
forms differ in the hidden half of every step after the first, so they differ in exactly the quantity that
decides whether a third and fourth draft are worth proposing. It also changes what a batched draft step
costs — `mtp.fc` and the second `lm_head` read in §09.6 are paid per candidate either way, which is why
§09.6 ends by arguing for one batched pass over all `k` candidates.

**What would close it.** The same two witnesses as A-18 (the producer's graph, or an HF implementation),
plus `A-17`'s exact-token gate as the admission that any chosen recursion is at least output-preserving.
Until then this spec defines the loop for `k = 1` and leaves the recursion open, and the `k > 1` columns of
§09.6's cost table are read as per-drafted-token, not as a claimed schedule.

**Consumers.** `specification/09-mtp-draft-layer.md` §09.2, §09.5, §09.6, §09.7.

## A-20 — whether drafted stop ids participate at all: **unresolved**

**Statement.** Whether `STOP_IDS` (A-04) takes part in drafting. §09.5 settles half of it — a stop id
inside a draft sequence does **not** end the request, because a drafted stop id the target does not
confirm was never generated — and leaves the other half open: whether the drafter may *propose* one at
all, and whether proposing one wastes a draft slot that could have been verified.

**Where it came from.** `specification/09-mtp-draft-layer.md` §09.5 (the third rule row) and §09.7 row
A-20; `specification/07-decode-step.md` §07.5's "interaction with MTP" row;
`specification/02-tokenization-and-embedding.md` §02.4's closing mechanics, which this entry does not
change.

**What it risks.** Small but silent either way: allowing proposals wastes draft positions and shows up as
an acceptance rate that never quite explains itself; forbidding them requires the drafter to consult a
host-side set every step, which is precisely the host round-trip §11.7 forbids for routing — so an engine
that "just" filters drafted stop ids on the host has reintroduced a per-token sync into the draft path.

**What would close it.** It is an engine-policy question behind A-17's gate: whichever rule is chosen, a
spec+verify loop must still emit exactly plain greedy's tokens, so the gate cannot adjudicate it — the
decision belongs with the user (or the main agent) alongside A-04, not here. Ids are referred to by number
throughout, per `AGENTS.md` §"Never print chat-template special tokens".

**Consumers.** `specification/09-mtp-draft-layer.md` §09.5 + §09.7, `specification/07-decode-step.md`
§07.5, `specification/02-tokenization-and-embedding.md` §02.4.

## A-21 — two `A-01` namespaces exist on this box; do not conflate them

**Statement.** Two independent assumption-id namespaces live in this repo and they collide on the literal
string `A-01`:

1. **`specification`'s A-01** — this file's entry above: the MTP block is not represented in the HF
   reference, so `graph_mtp` is the authority for its form.
2. **`checkpoints`' own A-01** — a literal `[ASSUMPTION A-01]` inside
   `checkpoints/dump-manifest-2026-10-26.json`'s `input.method`, about **fixture generation**: "fp32
   torch.randn from a private torch.Generator('cpu').manual_seed(seed), each row rescaled to RMS exactly
   1.0, cast to bf16. [ASSUMPTION A-01] RMS-1 models post_attention_layernorm output; it does NOT carry
   the (1+weight) modulation of a real activation". That manifest has its own numbering and its A-01 is
   not this file's A-01; it is an id in a different document, about a different subject.

**Where it came from.** The manifest's `input.method` field, verbatim above. The manifest is a committed
artifact written by `tools/dump_reference.py` and was not edited from this task; the collision was found
while writing this registry, and minting a new id for the *namespace* is the fix that does not touch a
frozen file.

**What it risks.** A reader who conflates them trusts the wrong witness in both directions: reading the
manifest's `[ASSUMPTION A-01]` as "the MTP assumption" makes a fixture-generation choice look like an
authority about the model (RMS-1 rows are a shape stand-in, not a real `post_attention_layernorm` output —
the manifest itself warns it does not carry the `(1 + weight)` modulation of §03.1); and reading this
file's A-01 as "the manifest already assumed this" makes an open question about the drafter look settled
by a seed note. A dump's assumption is about how the input was made; a spec assumption is about what the
model computes.

**Falsifying test.** `grep -n 'ASSUMPTION' checkpoints/dump-manifest-2026-10-26.json` must print exactly
one hit, whose text is about input generation. If a future manifest ever cites `A-01` about MTP — or this
file ever cites the manifest's numbering as if it were its own — the two namespaces have truly collided
and this entry is the finding.

**Consumers.** Inside this directory: none, yet — `specification/10-reference-dumps-and-gates.md` §10.1
and §10.5 read that manifest's `reading_rules` and `input` fields without carrying the id, which is why
the closure census below reports A-21 as unreferenced here rather than fabricating a cite. Its real
consumer is the manifest's `input.method` field itself, outside this directory.

## Closure — the census, both directions

A registry is closed only if **both** lists are empty for the right reason, so this file is checked in
two directions:

1. every `A-\d+` token anywhere under `specification/` has a `## A-NN` heading here — a citation with no
   definition is this file's failure, not the citing file's;
2. every id defined here is cited by some **other** file in the directory — an id nobody cites is either a
   stale entry or a file that forgot to cite it.

The reference scan excludes this file on purpose: an id that is only ever written by its own definition
line is not consumed by anything, so counting self-references would make check 2 unfalsifiable. Run from
the repo root:

```sh
SPEC=specification
REG="$SPEC/ASSUMPTIONS.md"
REFS=$(grep -rhoE '\bA-[0-9]+' --include='*.md' --include='*.py' --exclude=ASSUMPTIONS.md "$SPEC" | sort -u)
TOTAL=$(grep -rhoE '\bA-[0-9]+' --include='*.md' --include='*.py' --exclude=ASSUMPTIONS.md "$SPEC" | wc -l)
DEFS=$(grep -oE '^## A-[0-9]+' "$REG" | sed 's/^## //' | sort -u)
NREFS=$(printf '%s\n' "$REFS" | grep -c .)
NDEFS=$(printf '%s\n' "$DEFS" | grep -c .)

echo "citations outside the registry: $TOTAL  distinct ids: $NREFS  defined: $NDEFS"
echo "referenced but NOT defined: [$(comm -23 <(printf '%s\n' "$REFS") <(printf '%s\n' "$DEFS") | tr '\n' ' ')]"
echo "defined but NOT referenced: [$(comm -13 <(printf '%s\n' "$REFS") <(printf '%s\n' "$DEFS") | tr '\n' ' ')]"
# a census that parsed nothing would also print two empty lists, so the population is asserted:
test "$TOTAL" -gt "$NREFS" || echo "FAIL: citations ($TOTAL) not > distinct ids ($NREFS) — parse is vacuous"
test -z "$(comm -23 <(printf '%s\n' "$REFS") <(printf '%s\n' "$DEFS"))" || echo "FAIL: undefined ids cited"
```

Counts as this file was written (project day 2026-10-26):

| measure | value | reading |
|---|---|---|
| citations outside the registry | 46 | non-zero, and 46 > 19, so the scan is not vacuous |
| distinct ids referenced | 19 | `A-01`…`A-20` less `A-15` |
| ids defined here | 21 | `A-01`…`A-21` |
| referenced but NOT defined | 0 | closed in that direction |
| defined but NOT referenced | 2 | `A-15`, `A-21` — explained in the two items below, and **kept**, not deleted |

Non-vacuity and falsifiability were tested, not asserted. Three negative controls were run against
throwaway copies while writing this file, and each turned the census FAILED: deleting the `## A-12`
heading reported `A-12` as referenced-but-not-defined; appending a citation to a non-existent `A-77`
reported `A-77`; and swapping the `--include` patterns so nothing matched flipped the verdict on
`citations (0) not > distinct ids (0) — parse is vacuous` (the rule `pitfalls/40-gates-and-admissions.md`
§"A driver whose parser matches nothing prints PASS" is about). The script is kept inline here rather than
under `tools/` because the task that wrote this file was documentation-only and `tools/` has other lanes
working in it; promoting it is the main agent's call.

## What this registry found and did not change

Listed, not fixed — the 13 numbered files, `derive.py` and everything outside `specification/` were left
untouched by this task.

1. **A-15 is cited by claim and never by id.** mrope inertness for text is stated in
   `specification/05-gated-full-attention.md` §05.4, `specification/02-tokenization-and-embedding.md`
   §02.3, `specification/00-notation-and-hyperparameters.md` §00.3 and
   `specification/10-reference-dumps-and-gates.md` §10.6's rope row, and none of them writes `A-15`; the
   census reports it unreferenced for that reason. The entry stays because the claim is load-bearing, and
   the fix is a one-word edit whoever next opens §05.4 should make.
2. **A-21 has no consumer inside this directory by construction.** Its subject is a field of
   `checkpoints/dump-manifest-2026-10-26.json`, which is committed and was not editable from here.
3. **A-11 is the one id with no inline statement in the 13 numbered files.** It is cited only by
   `specification/README.md` §"How to check this directory"; the nearest inline statements are
   `specification/01-weights-and-shapes.md` §01.5's three "✓ (wider than 'exact')" rows and
   `specification/10-reference-dumps-and-gates.md` §10.2's zero-tolerance rows. The entry above assembles
   the numbers those rows imply, but no file currently states the claim the id registers.
4. **A-02 is cited without a falsifying test at its cite site.**
   `specification/00-notation-and-hyperparameters.md` §00.5 says only "version drift on *form* is A-02".
   The test recorded above is built from `specification/10-reference-dumps-and-gates.md` §10.5's mandatory
   interpreter fields; no 4.57.1-vs-5.19.0 quote comparison has been run, and this file does not claim one
   exists.
5. **Two inline `[REF]` defaults carry no id and may deserve one.** The chunk width `C = 64`
   (`specification/04b-gdn-chunked-form.md` §04b.2 — "the function's default, not a config field", and
   §04b.7 makes changing it a numerics change owed a floor probe) and the GDN l2norm eps `1e-6`
   (`specification/04-gdn-linear-attention.md` §04.4,
   `specification/00-notation-and-hyperparameters.md` §00.2's `[REF]` row — hard-coded in the reference).
   Both are decisions-in-disguise by `specification/README.md` §Tag conventions' own definition and are
   currently carried as plain `[REF]`s. Minting ids is the authoring lane's call, so none was minted here.
6. **Three cite sites describe a ruler that has since been corrected.** This is the one class of finding
   that was not present when `8f7ff53` was committed: the committed
   `quantization/evidence/bf16-census-adjudication-2026-10-26.md` (written afterwards) adjudicated A-07, and
   three places in the spec still speak of the old instrument.
   - `specification/01-weights-and-shapes.md` §01.2 attributes the 65.720-vs-66.1337 delta to "its
     `gdn_linear_attn` group omits `linear_attn.out_proj` and its `full_attn`/`norms_misc` groups are cut
     differently". The adjudication found **two phantoms and two omissions** (§5 rows 1–5), not a cut; the
     omitted `out_proj` half is right and the rest is not.
   - `specification/derive.py`'s `--families` footer — the very block that tells the reader "A-07 records the
     reconciliation this script produces" — still prints `gdn_linear_attn 1.416 GiB, norms_misc 0.071 GiB,
     text total 65.720 GiB … 1.86 %`, while `python3 tools/quant_scenarios.py --census` now prints closures
     and 66.1337. The footer is the stale text, not the tool.
   - `specification/00-notation-and-hyperparameters.md` §00.6 still contrasts its bf16 column with
     `quantization/README.md`'s "5.194 GiB/token", and `specification/08-state-management.md` §08.2 still
     prices policy A at 19.59 GiB resident; that file now publishes 5.61 GiB/token and 19.7848 GiB. Neither
     row's *argument* changes (the §08.2 point is that a 1 GiB state pool is ~5 % of resident weights), but
     both figures are the pre-adjudication ones.
   A-07 carries the adjudicated position; none of these was edited here. The open half of A-07 is
   `python3 tools/bf16_census.py --selftest`, which at the time of writing reported 2 missed faults against
   the in-repair ruler — that entry cannot be called closed from this file.
7. **The probe `specification/05-gated-full-attention.md` §05.4 defers to was not in this repo.** §05.4
   points at "`specification/ASSUMPTIONS.md` A-14's witness" for the command and output, and before this
   file there was no such record anywhere: no rope probe under `tools/`, nothing under
   `speccheck/evidence/`, no stored output. It has been re-derived from the committed config and is
   recorded in A-14 with its printed witness. The authoring lane's original run output remains unrecorded,
   so the witness on disk is the one reproduced here, not a copy of that one.
8. **`specification/README.md` §"How to check this directory" has been overtaken in two of its four
   places.** It promises "Three committed docs disagree with this spec in four places… Each one is written
   up in `ASSUMPTIONS.md` (A-07, A-08, A-10, A-11) with both numbers and the command that adjudicates it,
   and reported; none of those files was edited from this task." Since that commit, A-07 and A-10 have been
   adjudicated (the 65.72 GiB and 79.9 MiB figures corrected in `README.md` and `PLAN.md`), and
   `README.md` no longer carries the "router + norms + misc 0.11" leg of A-11 — so "both numbers" now
   overstates what remains open for two of the four. A-08's "10 KiB/token" leg is unchanged and still live.
   That paragraph was outside the two edits this task was allowed to make to that file.

