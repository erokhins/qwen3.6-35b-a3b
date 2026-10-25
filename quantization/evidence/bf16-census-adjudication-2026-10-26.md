# bf16 text-weight census — adjudication of 65.7198 / 66.1337 / 64.5604 GiB

Status: **settled, with three conventions left as decisions.** Written 2026-10-26 by W10 (`halogen`
lane) at the supervisor's request, after three rulers gave three answers for "text weights, bf16" and
the number turned out to be load-bearing in `AGENTS.md` §SCOPE, `README.md`, `PLAN.md` and
`DECISIONS.md` D-01/D-11.

CPU-only arithmetic. No device call, no timing, no `ms` figure anywhere in this file. Every GB/s
number is a divisor taken from `devices/measured-facts.md`, not a measurement.

Calculator: `tools/bf16_census.py`. Reproduce with

```
python3 tools/bf16_census.py                # the four closures, the verdicts, the recomputed rows
python3 tools/bf16_census.py --selftest     # 5 injected faults, all 5 required to fire
python3 tmp/bf16_ruler.py                   # the supervisor's ruler, unchanged, unchanged verdict
python3 tmp/gguf_suffix_census.py           # the supervisor's suffix census, unchanged
```

Everything below is derived from four files and nothing else:

| source | what it contributes | count, derived not typed |
|---|---|---|
| `models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | the shipping artifact's own tensor records (name, dims, ggml type) | 733 records, header-declared and asserted equal to the parsed count |
| `checkpoints/Qwen3.6-35B-A3B.safetensors.index.json` | the HF name universe and its `metadata.total_size` | 1,045 names, `total_size` 71,903,645,408 B |
| `models/qwen36-src/model-000{01,02,25,26}-of-00026.safetensors` | **real shapes and dtypes** for 424 of those names (layers 0, 1, 38, 39, the whole draft block, the whole vision tower, `lm_head`, embeddings) | 424 names witnessed |
| `tools/quant_scenarios.py` | the ruler under adjudication, imported and called (`QS.census()`), not retyped | 8 families |

`tools/bf16_census.py` also requires every config-derived shape to equal the shard header's shape
wherever they overlap (424 checks, abort on any mismatch), so the config model is used only for the
621 names no shard on disk covers — and only after it has been checked against 424 that they do.

---

## 1. Verdict

**The authoritative text figure is 66.1337 GiB, and W7 is right.** Two different things were being
counted, and one tool is simply wrong:

| ruler | GiB | what it actually measures | verdict |
|---|---|---|---|
| GGUF's own records (`tmp/bf16_ruler.py`) | **64.5604** | 733 tensors = the 40 transformer blocks only. The artifact carries **no draft block at all** | right, but it is *not* "the text model"; it is "the artifact" |
| W7, `quantization/residency-layout.md` §5 | **66.1337** | 712 text names = 693 body + 19 `mtp.*` draft-block tensors | **right; this is the authoritative text-bf16 figure** |
| `tools/quant_scenarios.py --census` | **65.7198** | a config model with four wrong terms | wrong by −0.4138 GiB; it lands between the other two by coincidence, not by approximation |

The dispute's premise — "the config-driven censuses model tensors that do not exist **and/or** the
GGUF omits text tensors the HF checkpoint carries" — is **both**, and each half is now named:

* the GGUF omits **exactly 19 tensors, all `mtp.*`, 844,640,768 elements = 1,689,281,536 B =
  1.5733 GiB bf16**, and nothing else; every one of the other 693 HF text names has an exact
  element-count twin in the artifact, and every one of the 733 artifact records is claimed by exactly
  one HF name (0 orphans either side);
* `quant_scenarios.py` carries **two phantom terms and two omissions**, which together are worth
  −0.4138 GiB and are accounted to the element in §4.

**D-11's conclusion survives at every candidate number**, because 64.56, 65.72 and 66.13 all sit above
the whole ≈55.5–56.5 GiB device-visible pool, and each is above it by 8.1–10.6 GiB. The correction makes
the argument slightly stronger, never weaker (§9).

---

## 2. The four closures, all closing to 0 bytes

Printed by `tools/bf16_census.py` verbatim:

```
C1  text from index 35505251456 elts = artifact 34660610688 + draft block 844640768   residue 0 B
C2  all 1045 index names, sum(elts) x 2 = 71903645408 B vs index metadata total_size 71903645408 B
    residue 0 B
C3  693 text names paired into 733 artifact records; 0 orphans on either side; 424 of 1045 shapes
    witnessed by real shard headers
C4  per-family delta (authoritative - quant_scenarios): unexplained +0 elts
residuals: NONE — every closure closed to 0 bytes
```

C3 is the name-by-name answer the dispute asked for. The pairing table is in
`tools/bf16_census.py::MAP_HF_GGUF` and each pair is checked three ways: element count equal, exactly
one claimer per artifact record, and — for the one fused→split pair — each half must carry the shape
`[E, I, H]`, not merely the right element count. The naming divergence, which is what the artifact
uses and nothing to do with any error:

| HF (`qwen3_5_moe`) | GGUF (llama.cpp) |
|---|---|
| `linear_attn.in_proj_qkv` / `in_proj_z` / `in_proj_a` / `in_proj_b` / `out_proj` / `conv1d` / `A_log` / `dt_bias` / `norm` | `attn_qkv` / `attn_gate` / `ssm_alpha` / `ssm_beta` / `ssm_out` / `ssm_conv1d` / `ssm_a` / `ssm_dt.bias` / `ssm_norm` |
| `self_attn.q_proj` (which is `[8192, 2048]` because `attn_output_gate` doubles it) / `k_proj` / `v_proj` / `o_proj` / `q_norm` / `k_norm` | `attn_q` / `attn_k` / `attn_v` / `attn_output` / `attn_q_norm` / `attn_k_norm` |
| `mlp.experts.gate_up_proj` `[E,2I,H]` (one tensor) | `ffn_gate_exps` + `ffn_up_exps` (two tensors) |
| `mlp.gate` / `mlp.shared_expert.{gate,up,down}_proj` / `mlp.shared_expert_gate` | `ffn_gate_inp` / `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp` / `ffn_gate_inp_shexp` |
| `mtp.*` (19 tensors) | **absent** |

The "counts differ by 21" the dispute quotes is fully explained and is derived, not asserted:
733 records = 693 paired text names + 40 extra halves from the one fused pair (40 layers); the
index's 712 text names = 693 + 19 draft names; 733 − 712 = 21 = +40 − 19.

Corroboration from outside this session, found after the fact: `checkpoints/Qwen3.6-35B-A3B.gguf-pin.json`
§`audit-shards-2026-10-26.json` already recorded `mtp: {names: 19, elements: 844640768, bytes:
1689281536, present: 19}` and `index_tensor_count: 1045`, and the extracted draft file
`models/w1-mtp-extract/mtp.safetensors` has 19 tensors / 844,640,768 elements. So the draft-block
figure now has four independent derivations: this census, that audit, the shard headers of
`model-00025/26`, and the extracted file's own header.

---

## 3. The three rulers, reconciled term by term to 0 bytes

Everything is done in elements first, GiB last, so nothing is hidden by rounding.

```
  artifact (733 records)              34,660,610,688 elts   64.5604 GiB
  + draft block (19 mtp.* names, all shape-witnessed)    844,640,768 elts    1.5733 GiB
  = HF text (712 names)                               35,505,251,456 elts   66.1337 GiB   <- authoritative
  + vision tower (333 names, shapes all witnessed)       446,571,248 elts    0.8318 GiB
  = index total_size  71,903,645,408 B                                   residue 0 B       66.9655 GiB
```

and the tool's error, decomposed (authoritative − `quant_scenarios`, elements):

| term | elts | GiB bf16 | real or phantom |
|---|---|---|---|
| `linear_attn.out_proj` / `ssm_out` `[h, lv·dv]` × 30 layers, **absent from the tool's gdn term** | +251,658,240 | +0.46852 | real tensor, omitted |
| `mtp.fc.weight` `[2048, 4096]`, the draft's fusion projection, **absent from the tool** | +8,388,608 | +0.01563 | real tensor, omitted |
| `norms_misc`'s `3*h*(3*h)` = 9h² | −37,748,736 | −0.07031 | **phantom**, no tensor family of that size exists |
| gdn term's `+ 2*h` per linear layer (×30) | −122,880 | −0.00046 | **phantom**, W7 did not list this one |
| norm count: real 86 h-sized norms vs the tool's `2L+4` = 84 | +4,096 | +0.00001 | undercount |
| **total** | **+222,179,328** | **+0.41379** | unexplained residue **0 B** |

Per family the tool is exact on six of eight (`routed_experts`, `full_attn`, `shared_experts`,
`routers`, `embed`, `lm_head` all match the artifact+draft census to the element, which also confirms
its `nfull+1` / `L+1` treatment of the draft block is otherwise right); only `gdn_linear_attn`
(+0.46852 GiB low) and `norms_misc` (−0.05468 GiB high) are wrong.

Two derived numbers this also fixes:

* `quant_scenarios --census` prints "delta 1.246 GiB (1.86 %) … the vision tower plus small tensors".
  It is **vision 0.8318 GiB + its own error 0.4138 GiB** = 1.2457, and the whole checkpoint has no
  "small tensors" left over: the element sum equals `total_size` exactly.
* `DECISIONS.md` D-07 says the vision tower is "333 tensors but only ~1.25 GiB". 333 tensors is right
  (1,045 − 712); **0.8318 GiB** is the vision tower, and ~1.25 GiB is that same 1.246 GiB delta
  misattributed. `DECISIONS.md` D-07's "the text path is **65.72 GiB**" should read 66.13 GiB.

The authoritative text census by family, for whoever rewrites the `README.md` §table:

| family | elts | bf16 GiB | share | README currently says |
|---|---|---|---|---|
| routed experts (40 blocks + draft) | 33,017,561,088 | 61.5000 | 92.99 % | 61.50 = 91.8 % |
| GDN linear attention (30 layers, **incl. `out_proj`**) | 1,011,553,920 | 1.8842 | 2.85 % | 1.42 |
| `lm_head` (untied) | 508,559,360 | 0.9473 | 1.43 % | 0.95 |
| token embeddings | 508,559,360 | 0.9473 | 1.43 % | 0.95 |
| full attention (10 layers + the draft's) | 299,898,368 | 0.5586 | 0.84 % | 0.56 |
| shared experts (41) | 129,058,816 | 0.2404 | 0.36 % | 0.24 |
| routers `mlp.gate` (41) | 21,495,808 | 0.0400 | 0.06 % | — |
| norms + `mtp.fc` | 8,564,736 | 0.0160 | 0.02 % | — |
| **text total (712 names)** | **35,505,251,456** | **66.1337** | | **65.72** |
| *(artifact only, 733 records, no draft)* | *34,660,610,688* | *64.5604* | | *not stated* |
| *(vision tower, 333 names)* | *446,571,248* | *0.8318* | | *~1.25 (D-07)* |

---

## 4. W7's claims, adjudicated — it is a worker's claim, not a fact

W7 moved the number *away* from the artifact, which is what made it suspect. It turns out to be the
only ruler that was right, because the artifact is missing a whole block. Its three claims, each
checked against the artifact, the shard headers and the config:

| W7's claim | ruling | witness |
|---|---|---|
| `quant_scenarios.py` omits GDN `ssm_out` = 251,658,240 elts | **ACCEPT, exact to the element** | its gdn term has `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `conv1d`, `A_log`, `dt_bias`, `norm` and no `out_proj`; the artifact carries `blk.N.ssm_out.weight` `[4096, 2048]` × 30 = 251,658,240 elts; HF `linear_attn.out_proj.weight` `[2048, 4096]` witnessed in shard 1 and 25 headers |
| it carries a phantom `3h × 3h` term = 37,748,736 elts | **ACCEPT, exact** | `norms_misc = (2L+4)*h + 3*h*(3*h)`; 9·2048² = 37,748,736. All norms in the whole checkpoint are 86 h-sized + 11·512 head-norms + 30·128 `ssm_norm`, i.e. 0.0160 GiB with `mtp.fc`, not 0.0703 |
| it omits `mtp.fc` | **ACCEPT, exact** | `mtp.fc.weight` `[2048, 4096]` = 8,388,608 elts, witnessed in `model-00026` header; no term of `quant_scenarios.py` produces it |
| its own total, 66.1337 GiB | **CONFIRMED** as the authoritative text-bf16 figure | C1 above, residue 0 B |

**Two defects W7 missed**, both needed for the closure to reach 0 rather than "within rounding":

1. a second phantom in the same gdn term — `+ 2*h` per linear layer, 122,880 elts (0.00046 GiB). The
   term has 8 summands for the 9 tensors a GDN layer carries: seven map onto eight of them (`A_log` and
   `dt_bias` share the `2*lv` summand), `2*h` maps onto nothing, and the tensor left with no summand is
   `out_proj`. Counting summands against tensors is the whole check.
2. the norm count is `2L+4` = 84 h-vectors where the checkpoint has 86 (40 `input_layernorm` + 40
   `post_attention_layernorm` + `model.language_model.norm` + the draft's five), so 4,096 elts of real
   tensors are missing.

Neither is worth a GiB; both are why §3 closes at 0 B instead of ±0.0002 GiB. W7's §5 finding is
therefore adopted in full, plus these two.

---

## 5. Verdict on `tools/quant_scenarios.py`

Its census is right on six of eight families, to the element. What it gets wrong, in the order the
fix should take them:

| # | term as written | maps to | effect on the printed text total |
|---|---|---|---|
| 1 | *absent:* GDN `out_proj` | `linear_attn.out_proj` ↔ `ssm_out`, `[h, lv·dv]` × 30 | **−0.46852 GiB** (reads low) |
| 2 | *absent:* `mtp.fc.weight` | `[h, 2h]` = 8,388,608 elts | −0.01563 GiB |
| 3 | `norms_misc: 3*h*(3*h)` | nothing | **+0.07031 GiB** (reads high) |
| 4 | `gdn_linear_attn: … + 2*h` | nothing | +0.00046 GiB |
| 5 | `norms_misc: (2*L+4)*h` | 86 h-norms exist, not 84 | −0.00001 GiB |
| 6 | `POLICIES[*]["router"] = 16.0` | the published artifact stores `ffn_gate_inp`, `ffn_gate_inp_shexp`, all norms, `ssm_conv1d`, `ssm_alpha`, `ssm_beta`, `ssm_a`, `ssm_dt` as **F32** (4 B/elem), and `quantization/README.md` §"How a quantized path gets gated" item 3 says they "stay exact" | policy-A resident and per-token each **+0.0403 GiB** if "exact" means F32-as-published; 0 if it means bf16. **This is a width decision, not a bug** |
| 7 | family width is per *family*, not per tensor | `ssm_conv1d`/`A_log`/`dt_bias`/`ssm_norm` sit in the `attn` class, so policy A prices them at Q8_0 rather than exact | 0.00075 GiB resident. Same class of question as #6; this is also why this tool's policy-A row differs from W7's by 0.0008 GiB |

Its docstring's validation claim ("validated against `total_size`: … a 1.246 GiB delta that is the
vision tower plus small tensors") is the same misattribution as §3 and should be restated as
"vision tower 0.8318 GiB **plus this census's own error**", or better, replaced by a closure that
must reach 0 like C1–C4 here.

**Not touched.** `tools/quant_scenarios.py` was left exactly as found, per the brief: fixing the ruler
before adjudicating it destroys the evidence, and the fix is a width-scope decision (#6/#7) as much as
a bug fix. The full policy table has been reprinted from the corrected census in
`tools/bf16_census.py` §5 (§6 below has A/F/bf16 and the rest), which lets the D-01a margin be checked
rather than assumed: D-01a is recorded as a *margin* (policy A vs E at 2.331 vs 1.663 GiB/token), and
the corrected figures are **2.525 vs 1.740**, so the gap the decision rests on **widens** from 0.668 to
0.786 GiB/token. That strengthens D-01a rather than reopening it, because the missing `out_proj` costs
0.249 GiB/token at Q8_0 and only 0.132 at Q4_K.

---

## 6. The two numbers that decide D-01/D-11, recomputed independently

Widths come from `tools/ggml_block_geom.py` (derived from the pinned `ggml-common.h` structs,
`Q4_K` = 4.5 bpw, `Q8_0` = 8.5, `BF16` = 16, `F32` = 32) and are applied with `row_bytes()`, whole
blocks. Element counts come from the artifact + draft census. Nothing in this section goes through
`quant_scenarios.py`'s census.

**Control first**, to prove the difference below is the census and not the width rule: pricing policy A
from `quant_scenarios.py`'s census with this tool's width rule reproduces the published row exactly —
**19.59 GiB resident, 2.331 GiB/token**. So every difference in the rows below is attributable to the
four census defects and nothing else.

| row | resident GiB | per-token GiB (with one draft pass/token) | per-token (target only) | ceiling @237 / @210 / @190 GB/s | docs today |
|---|---|---|---|---|---|
| **policy A (DECIDED)** | **19.7848** (main 19.3162 + draft 0.4686) | **2.5252** | 2.4653 | 87 / 77 / 70 | 19.59 / 2.331 / 95-84-76 |
| policy A, exact bucket = F32 as published | 19.8251 (main 19.3555 + draft 0.4696) | 2.5655 | 2.5047 | 86 / 76 / 69 | — |
| policy G (`lm_head`/embed Q6_K, not taken D-10) | 19.5554 | 2.4105 | 2.3506 | 92 / 81 / 73 | 19.36 / 2.216 |
| policy E (attention + shared Q4_K) | 18.8846 | 1.7397 | 1.6940 | 127 / 112 / 102 | 18.81 / 1.663 |
| policy B (attention bf16) | 21.9306 | 4.2269 | 4.1405 | 52 / 46 / 42 | 21.52 / 3.813 |
| policy C (experts MXFP4) | 18.8238 | 2.4952 | 2.4360 | 88 / 78 / 71 | 18.63 / 2.301 |
| policy D (experts IQ4_XS) | 18.8238 | 2.4952 | 2.4360 | 88 / 78 / 71 | 18.63 / 2.301 |
| policy F (Q8_0 throughout) | 35.1598 (main 34.3162 + draft 0.8436) | 3.0057 | 2.9340 | 73 / 65 / 59 | 34.97 / 2.811 |
| bf16 everything | 66.1337 (main 64.5604 + draft 1.5733) | 5.6083 | 5.4882 | 39 / 35 / 32 | 65.72 / 5.194 |

Rows B–G are here because `quantization/README.md` prices them and `DECISIONS.md` quotes their margins;
they cross-check against W7's independent variant table in `quantization/residency-layout.md` §3 to
≤ 0.0014 GiB on every row (B 21.9306, E 18.8846 vs 18.8859, F 35.1598 vs 35.1606, G 19.5554 vs 19.5562).

Answers to the question as posed:

* **19.59 GiB resident does not survive, but its verdict does.** It becomes **19.78 GiB** (+0.19 GiB,
  +0.99 %; 19.83 if "exact" means F32). That is inside any sane margin for a *fit* decision: it is
  12.0 GiB below the pessimistic co-resident budget, and the two competing policy rows move the same
  way (policy F 34.97 → **35.16**, which independently confirms the correction W7 made in
  `quantization/residency-layout.md` §3 against `devices/measured-facts.md`).
* **2.331 GiB/token does not survive and its consequence is visible.** It becomes **2.53 GiB**
  (+0.194 GiB, **+8.3 %**) if the draft block is walked once per token, or **2.47 GiB** (+5.8 %) for
  the target stack alone. The quoted ceilings **95 / 84 / 76 tok/s must be republished as 87 / 77 / 70**
  (with draft) — a 8–9 % haircut on every rate claim in `DECISIONS.md` D-01a and `quantization/README.md`
  §"The policies, priced". **This is the most consequential sentence in this file.** It does not change
  which policy is chosen (all rows move up, and the A-vs-E margin widens, §5), but any rate quoted as
  "≈84 tok/s at 210 GB/s" is now ≈77, and the 0.8B-experienced ×0.6 discount on top still applies.
* Whether the per-token row should include a draft pass is a **convention nobody has decided** here —
  D-09 put the draft block in scope, D-12 measured accept 0.678 ⇒ ~1.91 tokens/round, and neither says
  whether the quoted per-token figure is per generated token including drafting or target-only. Both
  numbers are printed above so the choice can be made explicitly; `quant_scenarios.py` has been quietly
  charging one draft pass per token all along, so "with draft" is the apples-to-apples row.

---

## 7. Consequence for `quantization/residency-layout.md` (W7, uncommitted)

**Essentially none, and the file is the closest thing in the repo to already correct.** Its census was
66.1337 (now confirmed), its weights row was 19.3170 + 0.4686 draft = 19.7856, and its per-token was
2.5260 — all confirmed by an independent path within 0.0008 GiB and 0.0008 GiB/token. The residual
0.0008 GiB is §5 item 7: this tool charges `ssm_conv1d`/`A_log`/`dt_bias`/`ssm_norm` at the attention
width (which is `quant_scenarios.py`'s family rule, kept on purpose so the control in §6 is fair), W7
keeps them exact per `quantization/README.md` item 3. 0.78 MiB; either is defensible, one should be
chosen.

What changes, against the file's own constants (budget 31.84 GiB pessimistic co-resident, fixed 20.027
GiB at 4 lanes, 90,112 B of KV rows per token across 4 lanes):

| row in that file | its value | with this census (exact=bf16) | with exact=F32-as-published | verdict changes? |
|---|---|---|---|---|
| weights + draft | 19.7856 | 19.7848 (−0.0008) | 19.8251 (+0.0395) | no |
| headroom at 4k, 1 lane | +11.907 | +11.908 | +11.867 | no — still `FITS co-resident` at every context |
| headroom at 256k, 1 lane | +6.493 | +6.494 | +6.453 | no |
| tightest frontier row (8 seq × 64k) | 31.27 | 31.27 | 31.31 | no — still 0.53 GiB under 31.84 |
| 4-lane aggregate crossover | 140,755 tokens | 140,764 (+10) | 140,284 (−471) | no — ±0.3 %; 1 GiB of fixed bytes = 11,916 tokens |
| `bf16 everything` row | 66.1337, no fit | identical | identical | no |
| policy F row | 35.1606 | 35.1598 | — | no; still above 33.88 co-resident, below 55.5 dedicated |

`tools/residency_budget.py` was not run and not modified (out of scope, and its own `--selftest` is its
gate), so the rows above are arithmetic on the constants W7 printed, reproduced before being adjusted:
(31.84 − 20.027) GiB ÷ 90,112 B = 140,759 tokens where that file prints 140,755 — a 4-token rounding in
its own print, 0.003 %, noted only so nobody re-derives it later.

Two things that file should still take from this: its citation of 66.1337 is now corroborated by
C1/C2 rather than asserted, and the "66.13 GiB" it quotes should be labelled *text including the draft
block* the first time it appears, because the artifact-side figure 64.5604 is walking around the repo
as "the GGUF ruler's answer" and the two differ by exactly the draft block.

---

## 8. What the docs should say

Not applied — `AGENTS.md`, `README.md`, `PLAN.md`, `DECISIONS.md` are the supervisor's and the user's
to amend, and the wording below is proposed text, not an edit. Two decisions are baked into it and
should be confirmed when it is applied: **(a)** the headline bf16 figure is the *text model including
the draft block*, because D-05 put MTP in scope; **(b)** whether policy A's "exact" bucket is bf16 or
F32-as-published (§5 items 6–7), which moves resident by 0.04 GiB and per-token by 0.04 GiB.

**Does D-11's conclusion survive? Yes, and it is stronger at the corrected number.** Every candidate
bf16 figure exceeds the whole device-visible pool: 64.56 (artifact), 65.72 (the old, wrong figure),
66.13 (authoritative) against ≈55.5–56.5 GiB — over by 8.1, 9.2 and 10.6 GiB respectively. Q8-throughout
35.16 is above the co-resident bracket 31.84–33.88 and below the dedicated pool, which is exactly the
row D-11 draws. So "bf16 exceeds the pool, therefore 4-bit routed experts were forced" holds whichever
number wins; nothing in D-01 or D-01a is reopened. Only the *quoted rates* move (§6).

Suggested replacements, verbatim against the current text:

**`AGENTS.md` §SCOPE** — current:
> Here the model is **65.72 GiB bf16 against a device-visible pool of ≈55.5–56.5 GiB

replace with:
> Here the model is **66.13 GiB bf16 in text tensors — 712 tensors: 40 blocks plus the MTP draft block,
> which the pinned GGUF does not carry (its 733 records are 64.56 GiB) — against a device-visible pool of
> ≈55.5–56.5 GiB

and current:
> forced them: bf16 (65.72 GiB) is above the whole device-visible pool, and Q8-throughout (34.97 GiB)

replace with:
> forced them: bf16 (66.13 GiB text, 66.97 GiB with the vision tower) is above the whole device-visible
> pool, and Q8-throughout (35.16 GiB) is above what one process gets while halogen is resident

**`README.md`** — the row
`| text weights, bf16 | **65.72 GiB** (routed experts 61.50 = 91.8 %, GDN 1.42, …router + norms + misc 0.11) |`
becomes:
> | text weights, bf16 | **66.13 GiB** — 712 tensors, 40 blocks + the MTP draft block; the pinned Q4_K_M artifact carries 64.56 GiB of that element set because it has **no draft block**. (routed experts 61.50 = 93.0 %, GDN 1.88 — including `out_proj`, which `quant_scenarios.py` omits — `lm_head` 0.95, embeddings 0.95, full attention 0.56, shared experts 0.24, routers 0.04, norms + `mtp.fc` 0.02) | **above the whole pool, not just above what is free** — capacity is a hard constraint, not a tuning knob |

the next row's `**5.194 GiB**` becomes `**5.61 GiB** (5.49 target-only)`, the roofline row `~42 / ~34`
becomes `~39 / ~32`, and the policy-A row becomes
`**19.78 GiB resident, 2.53 GiB/token → ~87 tok/s ceiling**` (or `2.47` target-only) with a pointer here.

**`DECISIONS.md` D-01** — `bf16 (65.72 GiB) and Q8-throughout (34.97 GiB)` →
`bf16 (66.13 GiB text, 64.56 GiB in the pinned artifact) and Q8-throughout (35.16 GiB)`.

**`DECISIONS.md` D-11** — `bf16 (65.72 GiB) fits neither.` →
`bf16 (66.13 GiB text; 64.56 GiB even with the draft block excluded) fits neither.` and the table cells
`policy A 19.59, policy G 19.36` / `Q8-throughout 34.97` → `policy A 19.78, policy G 19.56` (±0.04 if the
exact bucket is F32) / `Q8-throughout 35.16`; the "2.331 vs 2.811 GiB/token" becomes
"2.53 vs 3.01 GiB/token", and the "still short, by 3 GiB instead of 26" line for the Flash-Next endgame
is unaffected by a correction this small.

**`DECISIONS.md` D-01a / `quantization/README.md` §"The policies, priced"** — the whole policy table
must be reprinted from a fixed census. All eight rows are recomputed in §6 above so the reprint can be
checked against them; the D-01a decision text should say the margin it rests on is unchanged in
direction and larger in size (0.668 → 0.786 GiB/token between A and E, §5).

**`DECISIONS.md` D-07** — `the text path is **65.72 GiB**; the vision tower is 333 tensors but only
~1.25 GiB` → `the text path is **66.13 GiB** (including the 1.57 GiB draft block); the vision tower is
333 tensors and **0.83 GiB**`. The conclusion "excluding non-text saves 1.9 %" becomes 1.24 %, and the
conclusion "there is no partial-download route that meaningfully reduces it" gets stronger.

**`quantization/README.md`** — line "So bf16 (65.72 GiB) is impossible in either budget, and
**Q8-throughout (34.97 GiB)**" → 66.13 / 35.16; the family table's `**text total** | **65.720**` →
`66.134` (and its `5.19 GiB/token at bf16` → 5.61); the policy table's bf16 and A rows as above.

**`devices/measured-facts.md`** §"Roofline for the target model" — `34.97` → `35.16` (W7 already flagged
this in `quantization/residency-layout.md` §3; this census confirms it).

**`PLAN.md`** carries no 65.72 figure and needs no change: its "full bf16 checkpoint (67 GiB, 26 shards)"
is the whole checkpoint at 66.9655 GiB, and its step-1 row already states the draft block as
"19 tensors, 844,640,768 elts = 1,611.02 MiB bf16", which is exactly what C1 derives.

---

## 9. Defects found in files this worker was told not to touch

Reported, not repaired, as instructed.

**`tools/quant_scenarios.py`** — §5 items 1–7. Its `--census` prints a total that is 0.4138 GiB low and
attributes its own validation delta to the wrong cause.

**`tmp/bf16_ruler.py`** — the total is right and its "agrees with NEITHER" verdict is right; its
*category* classifier is wrong in three places and must not be quoted (its own numbers prove it: the
bucket it prints as `lm_head (untied) 12 tensors` has 12 tensors, not 1):

```python
    elif "output" in nm:                      # matches attn_output and output_norm, not just output.weight
    elif "attn" in nm and ("ssm_out" in nm):  # dead branch: no ssm_out name contains "attn"
    elif "attn" in nm:                        # catches the GDN attn_qkv / attn_gate (1.4063 GiB of GDN)
```

Consequence: its printed `full attention 1.7580 GiB` is 0.5078 GiB of real full attention plus 1.4063
GiB of GDN `in_proj_qkv`/`in_proj_z` minus `attn_output`, and its `lm_head 1.1035 GiB` is 0.9473 GiB of
`output.weight` plus `attn_output` (0.1562) plus `output_norm`. `tmp/gguf_suffix_census.py`, which
strips names instead of substring-matching them, is right and needed no correction; it is also the file
that flagged these buckets as untrusted, and that warning is now earned.

**`quantization/residency-layout.md`** (W7, uncommitted) — no figure in it is wrong. It should add the
"66.13 text incl. draft block / 64.56 artifact" label (§7) and can cite C1–C4 for its §5 finding.

**`checkpoints/Qwen3.6-35B-A3B.gguf-pin.json`** — no defect; it independently corroborates the draft
block (§2), which is worth knowing before anyone re-litigates 1.5733 GiB.

---

## 10. Left open — decisions, not residues

The census has no unexplained residue. These are the questions this file refuses to answer because they
are not the census's to answer:

1. **Which bf16 figure a doc must headline.** Both 66.13 (text, incl. draft) and 64.56 (artifact) are
   correct statements about different objects; §8 proposes leading with 66.13 because D-05 put MTP in
   scope, and requires every mention to name which one it means. The supervisor amends.
2. **Policy A's "exact" bucket: bf16 or F32.** `quantization/README.md` item 3 says "stay exact" and the
   artifact stores F32; `quant_scenarios.py` and W7's table charge bf16. Worth 0.0403 GiB resident and
   0.0403 GiB/token, plus 0.0008 GiB of §5 item 7 ambiguity about the GDN controls. Pick one and the two
   files agree.
3. **Does the quoted per-token figure include a draft pass?** +0.06 GiB/token of difference (§6). This is
   D-09/D-12 territory.
4. **Whose numbers the republished policy table should carry.** All eight rows are recomputed here from
   the corrected census and agree with W7 to ≤ 0.0014 GiB, but they are recomputed in *this* tool; the
   canonical table lives in `quantization/README.md` and is printed by `tools/quant_scenarios.py`, which
   this worker was told not to repair. The reprint should come from the fixed `quant_scenarios.py`, and
   these figures used as the cross-check that the fix was right.
5. **`tools/residency_budget.py` was not run.** §7's numbers are arithmetic on the constants
   `quantization/residency-layout.md` prints, not a rerun of W7's calculator. Its own `--selftest` is the
   gate for that file, and re-deriving 20.027 GiB / 90,112 B from scratch would be a second model of the
   allocator rather than a check of this one.

## 11. Non-vacuity, and how to distrust this file

`tools/bf16_census.py --selftest` injects 5 faults and requires all 5 to fire; it prints the witnesses
first so a zero means something (`693 names paired, 424 shapes witnessed against real headers,
19 draft names, 5 named C4 terms`), and it fails if the clean run leaves any residual or if any
population reads 0.

```
clean run: text 66.1337 GiB, artifact 64.5604 GiB, draft 1.5733 GiB, vision 0.8318 GiB,
           C4 delta +222179328 elts
PASS dropped-family fired:      C1 residual +503316480 B
PASS phantom-term fired:        C4 left +75497472 B unexplained
PASS mispair fired:             blk.0.ffn_down_exps.weight claimed by two HF names
PASS swap-half-shape fired:     name pairing does not preserve element counts or shapes
PASS short-record fired:        parsed 732 GGUF tensor records, the header declares 733
selftest: 5 faults injected, 5 fired, 0 missed — PASS
```
(the PASS lines abridged to their first clause; the tool prints the full reason after each colon)

The two that matter most are the ones that could not be caught cheaply: `phantom-term` is exactly the
`3h × 3h` shape of error this dispute was about, and `swap-half-shape` is a mis-pairing that every
element-count check in the repo would pass. `mispair` fires on the double-claim rule, which is why that
rule exists alongside the half-shape rule rather than instead of it.

To distrust this file productively: `python3 tools/bf16_census.py` reprints every population, family
delta and closure from the four source files, and the `--selftest` fails if any of its checks has been
made unable to fail.

---

## Corrigendum (added 2026-10-26 by the supervisor, after W11's review)

Three GiB values in the §3/§4 conversion column were printed wrong. **The element arithmetic — which is
what this file's own rule makes load-bearing ("elements first, GiB last, so nothing is hidden by
rounding") — is correct, and no conclusion moves.** Recorded here rather than silently patched, because
a reader who already cited 0.46852 should be able to see it was replaced and why:

| quantity | elts | printed | correct | check |
|---|---|---|---|---|
| GDN `out_proj` omitted | 251,658,240 | 0.46852 GiB | **0.46875 GiB** | 503,316,480 B ÷ 2³⁰ |
| phantom `+2*h`×30 | 122,880 | 0.00046 GiB | **0.00023 GiB** | 245,760 B ÷ 2³⁰ — the printed value is ~2×, the shape a factor-of-two slip leaves behind |
| total defect | 222,179,328 | 0.41379 GiB | **0.41384 GiB** | 444,358,656 B ÷ 2³⁰ |

The headline −0.4138 GiB is unaffected (it was always the element total converted correctly), and
`tools/quant_scenarios.py --selftest` now independently reproduces each of these defects by injection
at its own magnitude — 251,658,240 / 8,388,608 / 37,748,736 / 122,880 / 4,096 elts — so the numbers
here are no longer the only witness.
