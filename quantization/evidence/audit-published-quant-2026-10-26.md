# Audit of the published Q4_K_M against the bf16 checkpoint (2026-10-26)

Artifact: `models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, from `unsloth/Qwen3.6-35B-A3B-GGUF` revision
`a483e9e6cbd595906af30beda3187c2663a1118c`. Authority: `Qwen/Qwen3.6-35B-A3B` revision `995ad96eacd98c81ed38be0c5b274b04031597b0`; the 4 fetched shards carry layers 0, 1, 38, 39
plus the draft block's 19 `mtp.*` tensors. Instrument: `tools/audit_published_quant.py` with
`tools/gguf_reader.py` (stdlib `python3` 3.12.3 — no numpy, no torch, no llama.cpp binary).
Raw output is the JSON set beside this file; the design-before-bytes ordering is evidenced in
`audit-ordering-2026-10-26.log`. CPU only: **no timing taken**.

| step | pass condition (PLAN.md) | verdict |
|---|---|---|
| 1.0 gate authored from the spec | `--selftest` passes on synthetic blocks and every injected bug fires | PASS at 2026-10-25T01:31:24Z |
| 1.1 sha / revision / header | sha re-reads identical, header parses to 733 records | PASS |
| 1.2 shard sets vs committed weight_map | each shard matches; `mtp.*` extracted | PASS |
| 1.3 structure pass over every tensor record | reconciles tensor-for-tensor before any value is judged | PASS |
| 1.4a copied families, zero tolerance | routers, norms, dt_bias, alpha, beta, conv1d and the order/reorder questions bit-exact against bf16 widened (with the converter's documented transforms) | PASS |
| 1.4b derived-transcendental route predicate | `ssm_a` = -exp(A_log): two correctly-rounded routes agree with each other, file within one f32 step of the agreed value | PASS-with-finding |
| 1.5b dequant comparison | inside floor-probe bands | PASS |
| 1.5c injections | each fires only where it belongs | PASS |
| 1.6 shared storage + pin | identity holds, pin written | PASS |

## 1.1 identity

| field | value |
|---|---|
| sha256 (two independent reads, identical: True) | `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61` |
| HF revision / etag equals the computed sha | `a483e9e6cbd595906af30beda3187c2663a1118c` / True |
| `general.architecture` | qwen35moe |
| `general.file_type` / `general.quantization_version` | 15 / 2 |
| tensor records parsed (= declared = expected 733) | 733 |
| header_end / data_start / alignment | 10990035 / 10990048 / 32 |
| sum(tensor row bytes) + data_start == file size | True |

## 1.2 the shards and the draft block

| fact | value |
|---|---|
| shards fetched / sets matching the committed `weight_map` | 4 / 4 |
| `mtp.*` tensors | 19, all present |
| derived `mtp.*` element total | 844,640,768 elts = 1611.0244 MiB bf16 |
| PLAN.md step 1 claimed | 41,943,040 elts = 79.9 MiB → **REFUTED** |
| extraction written | `models/w1-mtp-extract/mtp.safetensors` |

## 1.3 what the file actually stores, tensor-for-tensor

| family | tensors | types stored |
|---|---|---|
| GDN control | 180 | F32 |
| GDN in_proj_qkv | 30 | Q8_0 |
| GDN out_proj | 30 | Q8_0 |
| GDN z-gate | 30 | Q8_0 |
| embeddings | 1 | Q8_0 |
| full attention | 40 | Q8_0 |
| lm_head | 2 | F32, Q6_K |
| norms | 100 | F32 |
| routed down | 40 | Q5_K, Q6_K |
| routed gate/up | 80 | Q4_K |
| router+shexp-gate | 80 | F32 |
| shared expert | 120 | Q8_0 |
| **total records** | **733** | |

Checkpoint names with no GGUF counterpart: {"absent-mtp": 19, "absent-vision": 333}

### A hand-typed width: the 74 MiB, and where my 232 MiB went

Two tools here priced this table from hand-typed bits-per-weight and one row disagreed with
ggml: Q5_K was written 5.5625 bpw where the pinned struct is 176 B per 256 elements = 5.5
bpw. Widths are now read from the header at run time by `tools/ggml_block_geom.py`, which
requires each struct's member sum and that struct's own `static_assert` to agree, and
`tools/gguf_header_types.py` now refuses to print a width table unless the byte identity
below holds. Derived widths and drift rows: `audit-block-geometry-2026-10-26.md`. The same diff re-audited every row of the old table and found exactly one wrong: Q4_K 4.5, Q6_K 6.5625, Q8_0 8.5, Q4_0 4.5, IQ4_XS 4.25 and MXFP4 4.25 all matched the header already. One wrong row in eight is precisely the shape that survives a spot-check, which is why the table is gone rather than corrected.

| stored type | tensors | elements | bytes (block math) | GiB | bpw derived | block |
|---|---:|---:|---:|---:|---:|---|
| Q4_K | 80 | 21,474,836,480 | 12,079,595,520 | 11.2500 | 4.5 | 144 B / 256 elts |
| Q5_K | 37 | 9,932,111,872 | 6,828,326,912 | 6.3594 | 5.5 | 176 B / 256 elts |
| Q8_0 | 251 | 1,913,651,200 | 2,033,254,400 | 1.8936 | 8.5 | 34 B / 32 elts |
| Q6_K | 4 | 1,313,865,728 | 1,077,780,480 | 1.0038 | 6.5625 | 210 B / 256 elts |
| F32 | 361 | 26,145,408 | 104,581,632 | 0.0974 | 32 | 4 B / 1 elts |

The identity that licenses those bytes: `sum(row_bytes) + data_start == file size` = `22,123,538,944 + 10,990,048 == 22,134,528,992` -> **RECONCILES**. A width table that cannot reproduce the file's own byte count is not printed at all, which is the difference between a derived number and a plausible one.

What the wrong row was worth on this file, with the arithmetic shown rather than asserted: the Q5_K family is 37 tensors of 268,435,456 elements each = 1,048,576 blocks of 256 per tensor, 38,797,312 blocks in total. At the derived 176 B/block that is 6,828,326,912 bytes = **6.3594 GiB**; at the retired 5.5625 bpw (= 178 B/block) it would have been 6,905,921,536 bytes. The overstatement is 2 B per block over 38,797,312 blocks = **74.00 MiB** (0.072266 GiB). Cross-checking from the family bytes instead of the block count: 6.3594 GiB x 0.0625/5.5 = 0.0723 GiB, same answer. An earlier draft of this audit said ~232 MiB; that number was wrong, and the only reason it is not still in this file is that this section recomputes it instead of quoting it.

**This does not move the shipping arithmetic.** `tools/quant_scenarios.py` policy rows use Q4_K / Q8_0 / bf16 / IQ4_XS / MXFP4, and Q6_K only for `lm_head` in E and G; **no policy row uses Q5_K**, so policy A's 19.59 GiB resident and 2.331 GiB/token (D-01/D-11) are unchanged, and the tool now aborts if any policy width stops matching a derived struct width. Only the published-file width-map GiB column above changed. Anyone tempted to "correct" a policy row with 5.5 bpw has corrected a row that was never Q5_K.

A derived table nobody has seen refuse anything is a hand-typed table with extra steps, so the machinery is probed four ways (`python3 tools/audit_published_quant.py --selfcheck-widths`, verdict PASS):

| control | fired? | detail |
|---|---|---|
| stale hand row (Q5_K 5.5625) diffed against the header | yes | drift rows: ['Q5_K']; the three correct rows stayed silent |
| Q5_K block 2 B too big cannot survive to a printed number | yes | clean delta +0 B; cross-check against gguf_reader's row sizes rejected it: blk.0.ffn_down_exps.weight: derived row_bytes 186646528 != gguf_reader |
| width-map tool refuses to print over a failing identity | yes | gguf_header_types.py exit code 1 with a lying file size (no width table printed) |
| policy row literal that is no derived width aborts | yes |   A experts Q4_K, rest Q8_0/experts: 5.5625 bpw is no ggml type's derived width |

## 1.4 exactness: 1.4a copied (zero tolerance) and 1.4b derived-transcendental (route predicate)

| what | value |
|---|---|
| cases judged (F32 families, layers [0, 1, 39]) | 27 = 26 bit-exact + 1 within-route |
| 1.4a copied cases / router tensors among them | 25 / 3 |
| 1.4b derived-transcendental cases (`ssm_a` = -exp(A_log)) | 2 |
| matched bit-exactly as stored | 8 |
| matched bit-exactly only after a documented converter transform | 17 |
| 1.4a zero-tolerance failures | 0 |
| zero-tolerance verdict on the whole set (kept, not replaced) | FAIL — blk.1.ssm_a |
| route-predicate totals | 63 elements bit-exact, 1 within-route, 0 outside, 0 reference disagreements |
| 1.4b population pin (`DERIVED_ROUTE_NAMES`, D-14 narrow reading) | PASS — expected 2 cases / 64 elements, judged 2 / 64: `blk.0.ssm_a`, `blk.1.ssm_a` |

The one-step allowance in the row above is authorised for exactly those tensors and for nothing else (`DECISIONS.md` §D-14 §"Ratification 2026-10-26 — the user's words"): the population is a literal in `tools/audit_published_quant.py` (`DERIVED_ROUTE_NAMES`), the expected counts come from geometry (sampled GDN layers x `linear_num_value_heads`), and `--selftest` injects a widened population, a shrunken one and a copied family walked into the route, each of which must fire. The 25 copied families keep zero tolerance.

The GDN tensors carry the **tiled (converter-reordered)** order of the V heads (10 tensors) — the question the reorder exists to answer, because a runtime that assumes the other order reads the wrong heads (`conversion/qwen.py §_LinearAttentionVReorderBase`).

These are bit-exact ONLY after a transform the pinned converter documents (GDN V reorder, `norm+1`, `-exp(A_log)`):

* `blk.0.attn_norm.weight` — as stored + norm+1(f32)
* `blk.0.ssm_conv1d.weight` — V-reorder + no value transform
* `blk.0.ssm_dt.bias` — V-reorder + no value transform
* `blk.0.ssm_alpha.weight` — V-reorder + no value transform
* `blk.0.ssm_beta.weight` — V-reorder + no value transform
* `blk.0.post_attention_norm.weight` — as stored + norm+1(f32)
* `blk.1.attn_norm.weight` — as stored + norm+1(f32)
* `blk.1.ssm_conv1d.weight` — V-reorder + no value transform
* `blk.1.ssm_dt.bias` — V-reorder + no value transform
* `blk.1.ssm_alpha.weight` — V-reorder + no value transform
* `blk.1.ssm_beta.weight` — V-reorder + no value transform
* `blk.1.post_attention_norm.weight` — as stored + norm+1(f32)
* `blk.39.attn_norm.weight` — as stored + norm+1(f32)
* `blk.39.post_attention_norm.weight` — as stored + norm+1(f32)
* `blk.39.attn_k_norm.weight` — as stored + norm+1(f32)
* `blk.39.attn_q_norm.weight` — as stored + norm+1(f32)
* `output_norm.weight` — as stored + norm+1(f32)

**Within-route, not bit-exact** — `blk.1.ssm_a`: 1 of 32 element(s) sit 1 f32 step from the value both correctly-rounded routes agree on (route `V-reorder + -exp`), indices [21]. The zero-tolerance comparison still calls this a FAIL and that line is kept in `audit-exactness-2026-10-26.json` beside the new verdict; the reason 1.4b does not demand bit-exactness is in the next section.

| element index | file (f32) | file bits | route `f32` value | its bits | route `crl` value | its bits | steps away | one step is |
|---:|---|---|---|---|---|---|---:|---|
| 21 | `-0.9096216559410095` | `0xbf68dcf7` | `-0.9096215963363647` | `0xbf68dcf6` | `-0.9096215963363647` | `0xbf68dcf6` | 1 | `5.960464477539063e-08` |


### Why 1.4 has two halves, and who said so

The supervisor authorised this split on 2026-10-26, as a **route-derived predicate, not a tolerance**: bit-exactness was the wrong predicate for `ssm_a` because HF never stores it (it computes `A = -exp(A_log.float())` at module init), so there is no stored authority to be bit-exact against — demanding it would demand that the producer's libm agreed with ours. It is not an engine-fidelity property either: this engine reads `ssm_a` from the file and never recomputes the exp. The underlying defect was PLAN 1.4 lumping a derived transcendental in with the copied families. This is surfaced to the user for ratification; if they overrule it the finding above is intact and the verdict flips back with one flag.

What the predicate is, and what keeps it honest:

* two independent correctly-rounded routes — `exp` in double then rounded to f32, and `exp` to 60 digits correctly rounded to f32 once — must agree with **each other** bit-for-bit on every judged element; if they ever disagree the case FAILS rather than being absorbed;
* the file may then be the agreed value or one `nextafter` hop either side of it, no looser. There is no ulp knob and no epsilon constant in the tool: the bound is the spacing at that value, and `audit_published_quant.py §selftest_route_predicate` witnesses the spacing against hand-worked neighbours and the predicate at 1, 2, sign-flipped, permuted and binade-edge offsets;
* the must-catch set grew with the relaxation: `audit-injections-2026-10-26.json` carries 14 route cases where +/-2 steps, sign flip, element permutation and a wrong-order `A_log` read must all fire and +/-1 step must not. A relaxation nobody probes at two steps is indistinguishable from one that accepts anything.

## 1.5 dequant comparison, bands sized by the floor probe

| family | type | tensors | floor rms | band rms (x1.5) | worst published rms | worst x floor | verdict |
|---|---|---|---|---|---|---|---|
| GDN in_proj_qkv | Q8_0 | 2 | 9.218e-05 | 1.383e-04 | 9.224e-05 | 1.00 | PASS |
| GDN out_proj | Q8_0 | 2 | 9.262e-05 | 1.389e-04 | 9.268e-05 | 1.00 | PASS |
| GDN z-gate | Q8_0 | 2 | 9.283e-05 | 1.392e-04 | 9.289e-05 | 1.00 | PASS |
| embeddings | Q8_0 | 1 | 6.599e-05 | 9.899e-05 | 6.604e-05 | 1.00 | PASS |
| full attention | Q8_0 | 4 | 1.153e-04 | 1.730e-04 | 1.154e-04 | 1.00 | PASS |
| lm_head | Q6_K | 1 | 4.185e-04 | 6.278e-04 | 3.331e-04 | 0.80 | PASS |
| routed down | Q5_K | 3 | 3.863e-04 | 5.794e-04 | 3.677e-04 | 0.95 | PASS |
| routed gate/up | Q4_K | 6 | 1.109e-03 | 1.663e-03 | 1.018e-03 | 0.92 | PASS |
| shared expert | Q8_0 | 9 | 7.577e-05 | 1.136e-04 | 7.579e-05 | 1.00 | PASS |

Floor = error achievable on this data at this width, measured by re-quantizing the
bf16 authority with this repo's own quantizer at the published width
(`tools/gguf_reader.py` §"Reference quantizers for the FLOOR PROBE").

## 1.5c injection matrix

| injection | must-fire | fired | leaked into families it must not touch | min x floor when it fires |
|---|---|---|---|---|
| `block_shift_1` | 30 | 30/30 | 0/0 | non-finite |
| `expert_stack_transpose` | 9 | 9/9 | 0/21 | 9.4 |
| `gate_up_swap` | 6 | 6/6 | 0/24 | 10.1 |
| `q4k_as_q4nl` | 6 | 6/6 | 0/24 | non-finite |
| `q4k_drop_min` | 6 | 6/6 | 0/24 | 15.1 |


The 1.4b allowance probed at its own boundary — the must-catch set grew with the relaxation, which is the only thing that distinguishes a one-step rule from "whatever we later call one step":

| route injection | must-fire | fired | fired when it must not | what it does |
|---|---:|---|---|---|
| `ssma_element_perm` | 2 | 2/2 | 0/0 | the file's own elements rotated by one position |
| `ssma_sign_flip` | 2 | 2/2 | 0/0 | the negation of the agreed value, every element |
| `ssma_step_1_down` | 0 | 0/0 | 0/2 | one hop below the agreed value, every element |
| `ssma_step_1_up` | 0 | 0/0 | 0/2 | one hop above the agreed value, every element |
| `ssma_step_2_down` | 2 | 2/2 | 0/0 | two hops below the agreed value, every element |
| `ssma_step_2_up` | 2 | 2/2 | 0/0 | two hops above the agreed value, every element |
| `ssma_wrong_authority_read` | 2 | 2/2 | 0/0 | reference built from the UNDOCUMENTED order (identity where the converter reorders) |

Cases: 150 band + 14 route; wrong: 0.
## 1.6 shared storage, and the pin

| check | result |
|---|---|
| alias detector self-witness (a known-aliased fixture must be found) | OK |
| GGUF name pairs claiming identical bytes | 0 |
| GGUF tensors with overlapping byte ranges | 0 |
| safetensors alias groups in the fetched shards | 0 |
| untied `token_embd` vs `output` | 65306/65536 bytes differ in the first 64 KiB |
| verdict | PASS |

Nothing in this file shares storage with anything else, and the untied pair is witnessed
DISTINCT rather than assumed: `tie_word_embeddings=False` is a config claim, the byte
comparison is the evidence. The pin is `checkpoints/Qwen3.6-35B-A3B.gguf-pin.json`
(sha256 `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`, 22,134,528,992 bytes, revision `a483e9e6cbd595906af30beda3187c2663a1118c`, 733 tensors, plus this run's section verdicts), so
a silent re-upload under the same filename is a mismatch instead of a new truth. It also
carries the authority revision and the `mtp.*` element count, and it needs a row in
`checkpoints/README.md` when this is committed.

## What this audit changes for the engine (findings)

* **PLAN's draft-block arithmetic is wrong by 20x.** The 19 `mtp.*` tensors are 844,640,768 elements = 1611.0 MiB bf16, not the 41,943,040 elements / 79.9 MiB step 1's fetch list claims. Nothing downstream that budgets residency for the draft block can keep the smaller number. (Evidence: `audit-shards-2026-10-26.json` §mtp.)
* **The published GGUF carries no draft block at all.** 19 checkpoint names have no counterpart, all of them `mtp.*` (plus 333 vision tensors, out of scope by D-04). The QWEN35MOE template does list `NEXTN_*` tensors and none appear. MTP is in scope (D-05), so this file cannot serve it; the extracted `mtp.*` set at `models/w1-mtp-extract/mtp.safetensors` is what the engine has to work from.
* **The GDN V heads are stored in the converter's tiled order** (10 of 10 judged GDN tensors), not HF's grouped order. The engine must read them tiled or permute on load; this is invisible to any test that does not compare against the checkpoint.
* **Norms are stored as `HF + 1`, computed in float32**, and `ssm_a` as `-exp(A_log)` (also float32): a loader that adds 1 in bf16 loses bits the file keeps. Routes are listed in `audit-exactness-2026-10-26.json` §routes_matched.
* **`attn_q` is 2x the query rows**: `attn_output_gate=True`, so `self_attn.q_proj` is [2*nq*hd, H] = [8192, 2048] and the GGUF carries the gate half too. A reader sized for nq*hd rows reads half a tensor and sees no error.
* **`ffn_gate_inp_shexp` is rank 1 in the file, rank 2 in the checkpoint** ([1, 2048] -> [2048]) on all 40 layers: `ggml_n_dims()` (ggml/src/ggml.c) drops trailing singleton dims, so element counts still match. Recorded as its own class in `audit-structure-2026-10-26.json` rather than folded into "dims match".
* **The producer quantizes per tensor, not per file type.** Stored widths across the 733 records: F32 361, Q4_K 80, Q5_K 37, Q6_K 4, Q8_0 251. So `general.file_type = 15 (Q4_K_M)` describes the headline, not the file: it names the width of 80 tensors out of 733, and every norm, router and GDN-control tensor is F32. Any residency or bandwidth model that multiplies 66.7 GB by one Q4_K_M ratio is wrong by construction (family table above; per-tensor list in `audit-structure-2026-10-26.json`).
* **The gate can see the faults it claims to see**: `block_shift_1` 30/30, `expert_stack_transpose` 9/9, `gate_up_swap` 6/6, `q4k_as_q4nl` 6/6, `q4k_drop_min` 6/6; the 1.4b route allowance is probed the same way (`ssma_element_perm` 2/2, `ssma_sign_flip` 2/2, `ssma_step_2_down` 2/2, `ssma_step_2_up` 2/2, `ssma_wrong_authority_read` 2/2, and the +/-1-step cases correctly stayed clean), and none of them leaks into a family it must not touch (`audit-injections-2026-10-26.json`).
* **Two width tables in this repo were wrong on Q5_K and the audit's own prose was wrong about what that was worth.** The pinned struct is 176 B/256 elts = 5.5 bpw, not the hand-typed 5.5625; on this file that is 74.00 MiB over 37 tensors, not the ~232 MiB an earlier draft claimed. Widths are now derived from the header at run time and both tools abort if the file size or a policy width stops matching. Nothing in the shipping arithmetic moves: no policy row uses Q5_K, so policy A stays 19.59 GiB resident / 2.331 GiB per token. Arithmetic and per-type table: §"A hand-typed width" below and `audit-block-geometry-2026-10-26.md`.
* **No family is even close to its band**: the worst published rms is 1.00x its own floor against a 1.5x band, i.e. the published quantization is as good as an honest single-pass quantizer at those widths.

## What is NOT claimed

* This is a byte/structure fidelity audit of one file. It does not license the widths (policy A of `quantization/README.md` and D-01/D-01a already do) and it is not a quality measurement: nothing here says a Q4_K_M token is any good.
* Only 4 of 26 shards were fetched, so the exactness and dequant passes judge layers 0, 1, 38, 39 and the router/GDN tensors of those layers; the structure pass is still tensor-for-tensor over all 733 records because the census carries every name and shape.
* Nothing here was checked against llama.cpp at runtime: no llama.cpp binary was built or run, and the reader here is this repo's own, written from `ggml-common.h` / `ggml-quants.c`. That is deliberate (PLAN §"Why an audit exists at all") — but it means "the file is fine and llama.cpp disagrees" is a failure mode this pass cannot see.
* Element-count-per-block in `tools/ggml_block_geom.py` is a claim about each ggml type, not a reading of a struct; it is witnessed by the file-size identity over all 733 tensors, not by the header text. Every bytes-per-block figure is read from the pinned header.

