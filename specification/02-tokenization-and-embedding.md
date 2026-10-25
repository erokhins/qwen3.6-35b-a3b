# 02 — Tokenization, embedding, positions, and what the stop rule is not

The text path's entry and exit. What the model consumes is a sequence of integer token ids and the
positions they sit at; everything else in this directory is downstream of those two things.

## §02.1 What the forward pass is given

| input | shape | meaning |
|---|---|---|
| `input_ids` | `[B, T]` integers in `[0, V)` | the token sequence; exactly one of `input_ids` / `inputs_embeds` must be supplied `[REF: Qwen3_5MoeTextModel.forward raises otherwise]` |
| `position_ids` | `[B, T]` (internally broadcast to 4 axes, §02.3) | absolute position per token, used only by the full-attention mixer |
| `attention_mask` | `[B, T]` boolean | 1 = real token, 0 = pad. Reachable only with batching; §04.2's padding rule and §05.3's mask are the two places it enters |
| `past_key_values` | the state bundle | `specification/08-state-management.md` |
| `use_cache` | bool | affects which state is written, not the math |

`B = 1` throughout this spec: multi-user batched serving is out of scope
(`specification/README.md` §Scope), so "batch" appears only where a formula needs a leading axis to be
well-typed.

## §02.2 The embedding lookup

```
x_(0)[b, t, :] = embed_tokens[input_ids[b, t], :]        # [d] = [2048] row gather
```

| fact | value | source |
|---|---|---|
| table | `model.language_model.embed_tokens.weight`, `[V, d] = [248320, 2048]` | `checkpoints/Qwen3.6-35B-A3B.config.json` geometry, real header in shard 1 (`python3 specification/derive.py --header --filter embed_tokens`) |
| GGUF name / width | `token_embd.weight`, Q8_0 in the published file (540,344,320 B) | `specification/01-weights-and-shapes.md` §01.4 |
| gather cost | one 2048-element row = 4096 B at BF16, 2048 B at Q8_0 per token | `python3 specification/derive.py --per-token` prints 0.004 / 0.002 MiB |
| no scaling | the embedding is **not** multiplied by `sqrt(d)` or any learned scale | `[REF: inputs_embeds = self.embed_tokens(input_ids)]`, and no `embed_scale`-shaped field exists in the config |
| no dropout, no norm on entry | the first arithmetic the token sees is Form A of `specification/03-block-glue-and-normalization.md` §03.1 | |
| padding row | `nn.Embedding(V, d, padding_idx=config.pad_token_id)` with `pad_token_id: null` `[CFG]` ⇒ no reserved zero row | a pad *token id* only exists in the tokenizer metadata (248055), and it is not a special row of the table |
| the draft block reuses this table | `mtp_use_dedicated_embeddings: false` `[CFG]` | `specification/09-mtp-draft-layer.md` §09.2; the published file carries no `nextn.embed_tokens` tensor |

The exit is the untied head in `specification/03-block-glue-and-normalization.md` §03.4 — same width,
separate 0.9473 GiB tensor, read whole every token.

## §02.3 Positions, and why four axes collapse to one for text

When `position_ids` is not supplied, the reference builds it as
`arange(T) + past_seen_tokens`, reshaped to `[1, 1, T]` and expanded to `[4, B, T]` — the hard-coded
four being *text, temporal, height, width* `[REF: Qwen3_5MoeTextModel.forward]`. Row 0 is used for the
causal mask and rows 1..3 are handed to the rotary module as its T/H/W axes.

| consequence | statement |
|---|---|
| text-only | all four rows carry the same integer, so the rotary's `mrope_section` assignment is inert and the rotation is the ordinary 1-D one (`specification/05-gated-full-attention.md` §05.4) |
| the producer encodes the same thing | the published file carries `qwen35moe.rope.dimension_sections` as `I32 x 4` — four sections, again degenerate for text |
| decode offset | position of the next token is `number of tokens already in the cache`, i.e. `past_seen_tokens + t`; it is an absolute index and is **not** reset by a sliding window |
| no `pos_embed` table | positions enter only through rope; there is no learned positional embedding to allocate |
| context ceiling | positions must stay `< 262144` (`max_position_embeddings`); beyond that needs YaRN, which is out of scope (`DECISIONS.md` D-04) |

## §02.4 Token ids, and the stop rule this spec does not decide

Ids are named by number only. Never print a chat-template token literally, in any file, gate output or
log (`AGENTS.md` §"Never print chat-template special tokens") — every agent here is served by a
checkpoint whose stop list is its own eos id, so a literal end-of-turn token silently truncates the
message or kills the tool call it landed in.

| id | meaning | witness |
|---|---|---|
| 248044 | `bos_token_id` **and** `eos_token_id` per the model config | `checkpoints/Qwen3.6-35B-A3B.config.json` `text_config.bos_token_id`, `text_config.eos_token_id`; the GGUF's `tokenizer.ggml.bos_token_id` agrees |
| 248046 | end-of-turn in this family; **the published file's tokenizer calls it the eos id** | GGUF `tokenizer.ggml.eos_token_id = 248046` |
| 248055 | padding id | GGUF `tokenizer.ggml.padding_token_id`; `text_config.pad_token_id` is `null` |
| 248053, 248054, 248056, 248057 | vision start/end, image, video | `checkpoints/Qwen3.6-35B-A3B.config.json` — unreachable here, vision is out (`DECISIONS.md` D-04) |
| 248068, 248069 | think-boundary pair in this family | named by number in `AGENTS.md`; re-derive per checkpoint, do not carry them across models |

**The two committed witnesses disagree about the stop id and this spec does not pick a winner.** The
model config says 248044 is eos; the shipping artifact's own tokenizer metadata says 248046; the
sibling project's rule ("248046 stops") was a fact about *its* checkpoint and `README.md` §Target
forbids carrying it over unexamined. The file that would adjudicate — this checkpoint's
`generation_config.json` — is not on disk (only `config.json` and the shard index were committed,
`checkpoints/README.md`), and the GGUF does not carry `eos_token_id` twice with different values in a
way that resolves the question.

So the spec defines **the stop set as a named input to the decode loop, `STOP_IDS`, with both witnesses
recorded and neither promoted**, and registers the question as `specification/ASSUMPTIONS.md` A-04
with the falsifying test: fetch `generation_config.json` (or read a pinned HF generation run) and
require `STOP_IDS` to equal its `eos_token_id` list; if the two witnesses disagree with it, the config
wins and the GGUF row is annotated as a producer-side artefact. What this file *does* settle is the
mechanics: greedy decoding (`specification/03-block-glue-and-normalization.md` §03.4) emits one id per
step, generation halts as soon as the emitted id is in `STOP_IDS`, the stop id itself is not part of
the returned text, and a stop id arriving during an MTP draft step does not end the *verification*
pass — see `specification/09-mtp-draft-layer.md` §09.5.

## §02.5 Tokenizer facts the artifact carries, and the two things this spec refuses to take from it

From the published file's metadata (`python3 tools/gguf_header_types.py <file> --keys "."`,
witnessed 2026-10-26): `tokenizer.ggml.model = gpt2` (BPE), `tokenizer.ggml.pre = qwen35`,
`tokenizer.ggml.tokens` = 248,320 strings, `tokenizer.ggml.merges` = 247,587 pairs,
`tokenizer.ggml.token_type` = `I32 x 248320`, `tokenizer.ggml.add_bos_token = false`, and a
`tokenizer.chat_template` is present.

| the file also says | this spec's stance |
|---|---|
| `general.sampling.temp = 1.0`, `general.sampling.top_k = 20`, `general.sampling.top_p = 0.95` | producer defaults for a chat endpoint. This spec defines **greedy only** (`specification/README.md` §Scope); these three keys are not inputs to anything here |
| `tokenizer.chat_template` (a multi-hundred-line template that renders vision counts too) | the chat surface is a rendering concern for `engine/`, not part of the forward pass; its content is deliberately not reproduced in this directory. Its special tokens are referenced by id, per §02.4 |
| `general.tags` includes image-text, `general.base_model.0.name = Qwen3.6 35B A3B` | confirms the artifact identity pinned in `checkpoints/Qwen3.6-35B-A3B.gguf-pin.json`; the vision branch stays out |
| `quantize.imatrix.*` (dataset `unsloth_calibration_Qwen3.6-35B-A3B.txt`, 76 chunks, 510 entries) | explains *how* the published widths were chosen; it is not a calibration set this repo holds, so no quantization-quality claim in this repo may rest on it (`PLAN.md` §"Step 1 does not buy") |

## §02.6 The text path, end to end

```
input_ids [B,T] -> embed_tokens gather -> x_(0) [B,T,d]
   -> 40 decoder layers (specification/03-block-glue-and-normalization.md §03.2, mixers 04/04b/05, MoE 11)
   -> Form A final norm (03.4)
   -> lm_head [V,d] -> logits [B,T,V] -> argmax -> next id -> STOP_IDS test (§02.4)
```

with the whole-request and single-token orderings in `specification/06-prefill-pass.md` and
`specification/07-decode-step.md`, and the state that makes `past_seen_tokens` mean anything in
`specification/08-state-management.md`.
