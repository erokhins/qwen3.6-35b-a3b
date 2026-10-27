# Cross-oracle: engine × HF chained dumps × llama.cpp dumps (2026-10-27)

D-26 makes llama.cpp the sole numerical oracle for `engine/*` and keeps the HF reference as a
comparison instrument on four blocks (0, 1, 38, 39 — both GDN kinds and both transitions). This file
records the first run of `engine/cross_oracle.py`, which puts the three sides in one report.

Tool: `engine/cross_oracle.py`. Log: `engine/work/cross-oracle-2026-10-27.log`.
Sides: `checkpoints/dump-manifest-chain-2026-10-26.json` (28 stage records),
`checkpoints/llama-dump-manifest-2026-10-27-run-a.json` (15276 rows, 14267 files written),
`engine/block_forward.py` (the engine side).

**No band, floor or tolerance was authored here.** The run prints `no bands authored=no
licence-issued=none`; had any limit been taken from llama.cpp it would have had to print
`licence=llama-crosscheck(observation-only)`, which means an observation from a cross-check and never
a licensed band (D-17's ±1-ULP policy is the user's and was not touched). CPU-only; the report ends
`no timing taken`.

## Headline: the three sides do not share a stimulus, so nothing numeric is comparable

```
align   NOT-ALIGNED-BY-STIMULUS: no llama graph is fed the chain's 32 ids (prefill is 11)
align   unattainable-through-frontends=yes
RESULT  checks=78 findings=0 aligned=False no bands authored=no licence-issued=none
        verdict=STRUCTURAL-OK no timing taken
```

The full alignment evidence — including the measured detokenize→re-tokenize round trip (30 of 32 ids,
first divergence index 3) and the template-free control — is
`quantization/evidence/llama-stimulus-alignment-2026-10-27.md`. The short version: the chain's ids
were *sampled* (`input.kind = embedding_slice`, `method = torch.randint …`), the vendored front ends
take text and only text, and for a byte-level BPE the byte string is forced by the ids, so the single
candidate preimage fails and no text prefill can reproduce them. D2 therefore stands as a structural
comparison, and `numeric` mode refuses with a named reason instead of printing numbers.

Engine-side coverage is derived from `engine/block_forward.py` itself, not asserted:
`BLOCK=0 TOKEN_ROW=0 GDN_MODE=decode0`, 22 computed sub-stages (`gdn_recurrent_step.*`,
`moe_shared.*`, `moe_block.combined_out`), of which only two exist at §03.2 granularity — the mixer
output `y` and the MoE sum `combined_out`.

## Stage mapping (denominator 28 rows = 4 blocks × 7 §03.2 stages)

llama's names are cited to `engines/llama.cpp/src/models/qwen35moe.cpp` at commit `d327bf1f`.

| block | kind (llama/HF) | hf stage | engine stage | llama tensor | status |
|---|---|---|---|---|---|
| layer0/1/38 | linear / linear_attention | `x` | UNMAPPED-BY-ABSENCE(engine) | `model.input_embed`(layer0) / `l_out`(prev) | PENDING-BY-STIMULUS |
| layer39 | full / full_attention | `x` | UNMAPPED-BY-ABSENCE(engine) | `l_out`(layer38) | PENDING-BY-STIMULUS, chain gap |
| all four | — | `h_in` | UNMAPPED-BY-ABSENCE(engine) | `attn_norm` (`:184`) | MAPPED |
| layer0/1/38 | linear | `mixer_out` | `y` | `linear_attn_out` (`:483`) | MAPPED |
| layer39 | full | `mixer_out` | `y` | `attn_output` (`:352`) | MAPPED |
| all four | — | `x1` | UNMAPPED-BY-ABSENCE(engine) | `attn_residual` (`:204`) | MAPPED, except layer39 (see below) |
| all four | — | `h_mlp` | UNMAPPED-BY-ABSENCE(engine) | `attn_post_norm` (`:211`) | MAPPED(2 hits, first kept) |
| all four | — | `moe_out` | `combined_out` | `ffn_out` (`:536`) | MAPPED, except layer39 |
| all four | — | `out` | UNMAPPED-BY-ABSENCE(engine) | `l_out` (`:222`) | MAPPED, except layer39 |

```
mapping   denominator: 28 rows total (4 blocks x 7 §03.2 stages); 20 resolve to a written llama
          tensor, 4 name a real llama row that the instrument wrote no bytes for, 4 are position-only
          pending the stimulus, 20 are engine absences
```

Three findings came out of building this table, none of them guesses:

1. **`ffn_out`, not `ffn_moe_out`, is HF's `moe_out`.** `qwen35moe.cpp:507` registers `ffn_moe_out`
   *before* the shared expert is added (`:518`–`:536`), so `ffn_moe_out` is the routed-only partial and
   `ffn_out` is routed + shared. Mapping HF's `moe_out` to `ffn_moe_out` — the obvious reading of the
   name — would have compared a routed-only sum against a routed+shared one.
2. **In a linear block, `attn_output` is not the mixer output.** Block 0's `attn_output` (node 53) is
   `[128, 32, 11, 1]`, a GDN interior tensor; the counterpart of HF's `mixer_out` there is
   `linear_attn_out` (node 61, `[2048, 11, 1, 1]`). In the full-attention block the name *is* correct
   (`attn_output`, node 3769, `[2048, 11, 1, 1]`), so the mapping is per-block-kind, not per-name.
3. **Layer 39's post-mixer rows exist but hold zero tokens in the prefill graph.** `attn_residual`,
   `attn_post_norm`, `ffn_out` and `l_out` for block 39 are `[2048, 0, 1, 1]`, `written=false`, because
   the last layer's `crop_before_nextn` gather (`:194`–`:196`) leaves an empty token axis. The same
   names are written in the chunk and decode graphs as `[2048, 4, 1, 1]` and `[2048, 1, 1, 1]`. So
   llama's own layer-39 FFN activations exist only for chunk/decode positions, never for a prefill
   position — independent of the stimulus problem.

## Shape, order and distinctness (structural checks that can fail)

```
shape/order denominator: 20 pairs mapped, 16 ordering pairs checked, 16 increasing, checks so far=28
statistics denominator: 48 records profiled (28 hf + 20 llama), 0-element records = a failure and
                        there were none
aliasing/distinctness — denominator 2 groups
  stage mixer_out  blocks=4 distinct_digests=4 OK
  stage moe_out    blocks=3 distinct_digests=3 OK
```

* Hidden axis agrees on every mapped pair (2048 == 2048). The **token axis never agrees** (32 vs 11),
  printed as `STIMULUS-MISMATCH(not a mapping error)` rather than folded into a mapping failure.
* Graph order: for each block the mapped llama nodes increase in §03.2 order
  (`attn_norm < linear_attn_out|attn_output < attn_residual < attn_post_norm < ffn_out < l_out`) —
  16 of 16 consecutive pairs increasing (layer39 contributes 1 pair because its later rows are
  unwritten).
* Statistics are each side's own values read from its own `.bin` (HF bf16 widened by a left shift of
  16 on the raw bits per `pitfalls/30-numerics.md`, llama f32 read as f32), printed side by side with
  **no delta column**, because a delta between unrelated stimuli is not a measurement. Every record
  profiled was finite with `rms > 0`; a 0-element record is a failure and none occurred.
* Distinctness uses the manifests' own digests (no re-hashing): if a stage digest repeated across
  blocks it would mean the instrument's one-row-per-`(graph, node, seq)` dedupe aliased a tensor —
  the `result_output` bug class in `tools/LLAMA_DUMP_NOTES.md` §A digest that changes when a byte does
  not. 4 distinct of 4 for `mixer_out`, 3 of 3 for `moe_out`.

## Pairwise disagreement counts (three per block, 12 lines)

Every line prints its denominator; a zero-element comparison is a refusal, not a pass.

```
pairwise denominator: 3 pairs x 4 blocks = 12 lines; numeric comparisons performed = 0
numeric denominator: 28 stage slots x 3 pairs = 84 possible comparisons, 0 performed
```

| block | engine-vs-HF | engine-vs-llama | HF-vs-llama |
|---|---|---|---|
| layer0 | 0 of 131072 elements, UNCOMPARABLE-BY-STIMULUS | 0 of 45056, UNCOMPARABLE-BY-STIMULUS | 0 of 393216, UNCOMPARABLE-BY-STIMULUS |
| layer1 | 0 of 131072, same | 0 of 45056, same | 0 of 393216, same |
| layer38 | 0 of 0 — layer 38 is outside the single block `BLOCK=0` implements | 0 of 45056, same | 0 of 393216, same + chain gap |
| layer39 | 0 of 0, same | 0 of 22528, same | 0 of 131072, same + chain gap + unwritten rows |

## Refused HF-vs-llama adjudications (D-26 clause 3)

```
refusals denominator: 4 named, 0 adjudicated, 0 bands authored
```

1. **activation width** — HF records activations as bf16, llama records the same stages as f32. Two
   numeric systems on one stage. Calling f32 the reference licenses a wider input than the bf16
   fixture our gates are pinned to; calling bf16 the reference loosens the oracle. A quantized oracle
   may only *add* checks, never loosen one, so this is reported and not settled.
2. **router renormalisation dtype and form** — llama renormalises the top-k weights with an f32 DIV
   (prefill graph, block 0, node 74, `[8, 11, 1, 1]`); the HF form casts the renormalised weights back
   to the logits dtype before the expert multiply, and our §11.6 fixture already refused the f32 carry
   at `ndiff=1326` with a flipped argmax (`engine/HANDOFF.md` §MoE renormalisation dtype). The pinned
   fixture sides with HF and against the oracle; settling that would either move a must-catch set or
   declare the oracle wrong, and both are the user's calls.
3. **softmax domain** — llama takes `SOFT_MAX` over all 256 router logits (node 66, `[256, 11, 1, 1]`)
   then selects and renormalises; §03.2 softmaxes the top-k subset. Mathematically equal after
   renormalisation, bitwise not, and a near-tie is exactly where a banded router path can flip.
4. **chained vs real activations at layers 38/39** — the chain is `layer0 → layer1 → layer38 →
   layer39` with layers 2..37 absent (`block_set.why_not_contiguous`), while llama's graph is one real
   40-layer forward. Even under an aligned stimulus a layer-38/39 delta would mix a chain gap with an
   implementation difference, so the comparable HF-vs-llama scope is layer0 and layer1; 38 and 39 are
   structural-only regardless of stimulus.

## What this leaves for the next rung

To turn any of this into a numeric cross-check someone must first make the stimuli coincide — an
id-input knob in the vendored front end, or a re-dump of the chain on a text-tokenizable stimulus
(the latter replaces pinned fixture material and is the user's call). If that happens, `align` turns
green by construction (`token_ids` equality plus `gguf.matches_pin`), and the numeric path prints
per-pair magnitudes labelled `licence=llama-crosscheck(observation-only)` with no threshold attached.
Until then the honest statement is this file: the mapping is established, the ordering and shapes
agree, and no number has been compared.

---

CPU-only work; no device number appears in this file. **no timing taken.**
