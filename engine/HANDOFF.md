# engine/HANDOFF.md — M2 layer-0 block forward, state at 2026-10-26 (checker-repair pass)

Written by the M2 worker. Records **agent** decisions, labelled `DECIDED (agent, 2026-10-26)` per
`AGENTS.md` §GOAL; nothing here is a user decision and no user decision is quoted that the user did not
make. **No band, floor, route allowance, width or pinned fixture was created, relaxed or retyped**:
`speccheck/evidence/gdn-band-2026-10-26.json` is byte-identical to HEAD (file sha256
`dd68fd953896da0965fa687511eb7527bb7bed1cf8b763f479f1d22dea5a11dd`) and the gate still prints its
digest `27a68c7e82c9aa0507b9bcd677adc0d5d1b0bc865dd1e86bf495ed90f47392ee`. M1's handoff was moved out
of this file verbatim into `engine/HANDOFF-m1.md`; this file is M2 only.

---

## STATUS NOW — green, qualified, no device claim, no timing taken

| run | log in `engine/work/` | outcome |
|---|---|---|
| `engine/block_gate.py` (CPU, no flags) | `blockgate-cpu-2026-10-26-fixed.log` | exit 0, 161 lines, `verdict=QUALIFIED`, `checks=17 findings=0`, `controls=11 fired=11`, `rows=58 parsed=58`, `licensed=12 accepted=12 rejected=0 unlicensed=46` |
| `engine/block_gate.py --staged-evidence specification/evidence/stage-run-device-readback-2026-10-26.log` | `blockgate-cpu-2026-10-26-staged.log` | exit 0, same gate, plus `PRECONDITION … verdict=ADMITTED … tensors=17/17 bytes_rehashed=537215744 log_readback=463/463` |
| the state this pass inherited | `blockgate-cpu-2026-10-26.log` | exit 1: `verdict=FAIL`, `checks=14 findings=5`, `controls=4 fired=3` (silent: `stage-row-count-mismatch`), `parsed=0 printed=58`, `licensed=12 accepted=11 rejected=1` — the five RED lines are exactly bugs (b), (c), (d) below, the parse reconciliation, and that silent control |

Verbatim RESULT lines of the two green runs:

```
RESULT mode=cpu checks=17 findings=0 controls=11 fired=11 rows=58 arms=ref_bf16,policy_a device=off staged=absent no timing taken
RESULT mode=cpu checks=17 findings=0 controls=11 fired=11 rows=58 arms=ref_bf16,policy_a device=off staged=admitted no timing taken
```

`QUALIFIED`, not `PASS`: 46 of the 58 rows carry no licence, and `pass-unreachable-while-rows-unlicensed`
makes PASS unreachable while any row is unlicensed. Nothing in §5 below is a checker problem.

Licensed and **ACCEPTED (12 rows, 0 rejected)**: `ref_bf16` — `q_hat`, `k_hat`, `scan_out`, `state`
(GDN band, `recurrent|pinned` at `layer0/pinned/decode0`), `expert_raw` (floor probe, limit 5),
`router_logits` (floor probe, limit 5), `routed_sum`, `topk_ids`, `topk_weights` (§10.6 exactness, limit
0); `policy_a` — `router_logits`, `topk_ids`, `topk_weights`, all `ndiff=0`, licensed because those three
weights are stored F32-exact (D-01a) and a run-time check asserts the artifact really stores `F32` for
them before that reasoning is allowed to hold.

---

## 1. The four checker bugs, and the witness that each repair can lose

Every repair below is paired with a control that must go red **because of the repair**; `pitfalls/40`
§gates-must-fail-for-something is the whole argument, and a control whose pattern matches only its own
text is not a control.

**(a) The `STAGE` parser matched nothing.** `ROW_RE` expected a field order the emitter never produced, so
`parse_printed` returned 0 records against 58 printed lines. That much was visible — the inherited log
carries `RED parsed-record-count-equals-printed parsed=0 printed=58` — but nothing proved a *repaired*
parser would be awake, and its sibling control `stage-row-count-mismatch` was silent: it duplicated a row
and compared the count against `len(rows)` held in memory, which grows along with the duplicate. The
parser now reads `key=value` fields generically (`band=` last, because `reason=` carries spaces), and both
counts reconcile against the printed text and the derived population. Witnesses
`stage-row-parse-truncation` — `parsed 58 of 58 printed STAGE lines. A parser matching nothing reads 0,
which is what the inherited gate printed (parsed=0, printed=58) and it went red; the subtler failure is a
parser that matches most lines and drops a few silently, and that is what this control measures — dropping
the last printed line parses 57, which the parsed==printed equality refuses` — and
`stage-row-count-mismatch`, now `59 rows against 58 parsed and a derived population of 58`. A third bug
surfaced only once the parser actually matched: the emitter emits two `referent=absent` rows with no
`ndiff` field at all, and a parser that required one silently dropped them — that is the
`parsed=56 vs printed=58` in the intermediate log `blockgate-cpu-2026-10-26b.log`, which no check would
have caught after the first repair, because 56 is a plausible number and only the equality with 58 made it
loud.

**(b) The policy-A width check compared a torch dtype with a quant-format name.** It asked whether the
tensor that arrived (`torch.float32`, because a Q8_0 tensor dequantizes to f32 by design) was named
`Q8_0`, which can never be true, so it could only ever say the wrong thing. The ruler is now the ggml type
the artifact's own record declares: `engine/block_forward.py::Weights` records it per read
(`ggml_type=PolicyA.last_type`, which is `Loader.handle(name)["type_name"]` — the same field
`engine/loader.py` vetoes with), and `width_verdicts` compares that against `PolicyA.WIDTH`. Witness
`policy-a-width-wrong-declaration`: honest declaration compares 17 reads with 0 mismatches; declaring
`blk.0.attn_qkv.weight` as Q4_K against the artifact's own Q8_0 returns
`[('blk.0.attn_qkv.weight','Q8_0','Q4_K')]`. This only became possible because `PolicyA` now records the
type it read per tensor (`block_forward.py::PolicyA.tensor`, `reads[]`).

**(c) The gather census was polluted by control reads.** The counter walked `PolicyA.reads`, which is one
list per process: a control that recomputes a stage, or reads a whole expert tensor to prove the counter
bites, changed the number the shipping pattern was judged by. Reads now carry a `ctx`
(`Weights(…, ctx=…)`, default `forward`, every control names its own context) and the census counts only
`ctx=="forward"`. Witness `gather-counter-catches-a-whole-tensor-read`: honest forward 24 gathered slices
against the derived 24, 0 whole-tensor reads; appending the record a whole-tensor read of
`blk.0.ffn_gate_exps.weight` (Q4_K, 150994944 B) *would* write flips the same census to `ok=False` with
that name flagged — the record, not the 150 MB dequantization, which a CPU control must not do to itself;
and a control-context read stays out of the count (24, ok=True).

**(d) The constant-predicate scanner matched only itself.** The pattern for `if True` was written as one
literal containing the very text it hunts, so the scanner's own source matched and the check could never
report an empty list — it printed a non-empty list of *itself* and passed. The pattern is now a split
literal (`r"\bif\s+Tr" r"ue\b"`), the real `if True` forms that were in control code were rewritten as
real conditions, and the witness `constant-predicate-scan-is-awake` writes an injected copy at
`engine/work/const-predicate-injected-copy.py` and requires the scanner to see `['True', 'if True']`
there and `[]` in the two shipping files. `device_claims()` (job 3's scanner) is built the same way and
carries the same witness (`device-claim-scan-is-awake`).

## 2. `policy_a / topk_weights` — attributed with pinned bytes, then fixed in the kernel

The inherited state printed `REJECTED arm=policy_a stage=topk_weights ndiff=4 max_ulp=1` and the previous
handoff filed it as D-21 (the `gate_scale` fold making the dumped value unreachable in the shipping
shape) and asked for an adjudication change. **That attribution was wrong and no adjudication was
needed.** The four differing entries, with each form compared against the pinned record *of its own
dtype* (the manifest pins the renormalised weights twice: `layer0.topk_weights`, arm=ref, bf16, the value
after HF's cast; `layer0.topk_weights_f32`, arm=spec, f32 — `speccheck/moe_ref.py` with
`Form(weight_cast="after")` — the value before it):

| form | referent | ref_bf16 | policy_a |
|---|---|---|---|
| cast the weight to bf16, multiply by that (emitted) | `layer0.topk_weights` (bf16) | 0 | 0 |
| keep the f32 weight, multiply by that (cast-after / skipped R5) | `layer0.topk_weights_f32` (f32) | 0 | 0 |
| same, but R1 left in the arm's own basis — **the inherited policy-A state** | `layer0.topk_weights` (bf16) | 0 | **4 (max_ulp 1)** |
| the *product* built with cast-first weights | `layer0.routed_sum` (bf16) | **0** | banded, see below |
| the *product* built with f32-carry weights | `layer0.routed_sum` (bf16) | **1326 (max_ulp 32416)** | banded |

Read in order, those rows say three things. The renormalisation itself (R2-R4) is bit-exact in both arms —
the f32 form reproduces the pinned f32 record exactly, so nothing about softmax/top-k/renormalise is in
question. The cast **position** is not separable on the weights row (bf16 of the f32 weight and the cast
weight are the same image — the router lane's `cast-is-robust-not-lucky` slack,
`prototype/moe_router/results.md` §band) but **is** separable on the product: on the arm whose
`routed_sum` licence is exactness, the pinned `layer0.routed_sum` bytes refuse the f32-carry form by
1326 of 2048 elements. So the pinned material carries cast-first, which is AGENTS.md prediction 4
(“the HF form casts the renormalised top-k weights back to the logits dtype *before* multiplying”), and
the licence for this row never had to move: it was `spec-10.6(exactness)` all along and it still is.

The defect was the third row: on the policy-A arm the dequantized router weight arrives as F32, so
`moe_router` computed R1 in F32, R5's cast-to-logits-dtype became a no-op on the way through, and the
softmax saw logits that had never been rounded. **The fix is in `engine/block_forward.py::moe_router`:**
R1's result is put into the model's activation dtype — `cfg["act_dtype"]`, read from the checkpoint
config's `text_config.dtype` (`block_forward.py::load_geometry`), never typed — which is also, and this
is checked, the dtype the pinned `layer0.router_logits` record stores
(`router-logits-dtype-derivation-matches-the-pinned-referent`). Row after the fix:
`STAGE arm=policy_a stage=topk_weights ndiff=0 max_ulp=0 licence=spec-10.6(exactness) band=ACCEPTED`.
Two guards keep it from rotting: `router-r5-cast-is-a-real-cast-on-every-arm` (the weight reaching the
multiply must be bf16 on both arms — pre-fix this reads `torch.float32` on policy A and goes red) and
control `topk-form-probe-distinguishes-the-wrong-form`, which now has to lose on both sides.

One caveat, printed in the gate itself rather than hidden here: if the policy-A arm ever computes R1 in
the model dtype natively (a kernel that reads Q8_0 router data and emits bf16), the leak form coincides
with the emitted form and control 11 has nothing left to distinguish. It must then be **re-derived against
new pinned material, not deleted**.

## 3. The device claim is honest: a precondition is not a result

- `--device` is gone as a claim. `engine/block_gate.py --device` refuses before it computes anything and
  exits 1: `REFUSED tag=live-device-path-absent` plus the sentence explaining why — this file allocates
  nothing, launches nothing and imports no HIP, and `engine/loader.py::device_slabs` still raises
  `device-allocator-absent` — and a `NEXT` line naming the honest flag. Refusing up front rather than
  printing a note beside a green verdict is deliberate: a reader who asked for a device result must not have
  to notice `device=off` inside a `verdict=QUALIFIED` line.
- `--staged-evidence LOG` is the honest successor and it is *not* a device result: it prints
  `PRECONDITION name=policy-a-staging-integrity verdict=ADMITTED … label=staging-integrity-precondition-not-a-device-result`.
  Before admitting, `engine/block_forward.py::staged_precondition` verifies the log is about **the artifact
  this process is reading**: the opened file name and byte size against the pin, and then the sha256 of
  every run region the log names, re-computed from the file, against the digests the log recorded — 17/17
  policy-A tensors this block reads, 537,215,744 B re-hashed, `log_readback=463/463`. A run region is a
  *span* (the plan coalesces the small exact-family tensors), so the requirement is that
  `engine/loader.py`'s own address for each tensor lies inside a verified span, not that the two are
  equal — that containment is what makes the re-hash meaningful rather than tautological.
- Refusals are named, and each was witnessed (`control staged-precondition-refusals-fire`: one full honest
  admit plus six one-field corruptions, written to `engine/work/staged-evidence-*.log`; the corruptions
  all exit before any hashing, so only the honest arm pays the re-hash):
  RESULT line removed → `staged-evidence-result-line`; artifact renamed → `staged-evidence-artifact-name`;
  one run's three read-back digests flipped *consistently*, so the log stays internally self-consistent and
  only the file can contradict it → `staged-evidence-region-digest`; that run's `file_off` shifted one byte
  → `staged-evidence-layout`; only the read-back digest flipped while the line still says `MATCH` →
  `staged-evidence-readback`; that tensor's `TENSOR` record removed → `staged-evidence-incomplete`; its
  `type=` declared Q4_K against the artifact's Q8_0 → `staged-evidence-width`. Exactly one tag is left
  unwitnessed and says so in the control: `staged-evidence-artifact-size`, because witnessing it would mean
  editing the pinned artifact, which no gate in this repo does.
- Both `--selftest` and `--check-only` are now deleted, and the reason is in the source where they were
  parsed. A controls-only roster starves the controls that bite on the printed population, and a
  “certify that nothing was admitted” mode was the default run wearing a flag: what a run admitted is
  carried by `RESULT … staged=absent|admitted|refused:<tag>`, which is printed in every mode.
- `device-claims-come-from-device-only` scans the run's own output for device-result-shaped lines and is
  witnessed by `device-claim-scan-is-awake`, which assembles its probe line at run time so this source file
  cannot self-match (`pitfalls/50` §the self-matching scanner).

## 4. Fixture material this gate needs and does not have — reported, not fabricated

Two rows print `referent=absent … reason=no-pinned-referent` (`block_output` on each arm), and the row
count is **one** stage, not two: §03.2 has two residual adds (step 3 `x1 = x + m`, step 6 `y = x1 + f`),
and `derived_population()` emits one glue row per add from the spec's own step list — with no referent for
either, the population is still 19 GDN + 9 MoE + 1 glue = 29 per arm. Nothing was invented to fill them:
no residual stream is inverted out of the pinned mixer inputs, because that changes the quantity under
test. It is also true, and worth saying before anyone tries it, that the two pinned passes cannot be
recombined to make one: the GDN chain is fed `layer0.decode0.input` and the MoE chain `layer0.input` row 0,
two *different* tokens, both synthetic RMS-1 mixer inputs (§10.1 A-01), so there is no pinned `x` whose
`m` is the pinned `y`.

What would close it, exactly — one coherent **block-level pass** through the installed reference for layer 0
(one decode token, `arm=ref`), with `layer_types[0] == "linear_attention"` so the mixer is the GDN block:

| record the gate needs | shape | dtype | §03.2 line | what it licenses |
|---|---|---|---|---|
| `layer0.block.x` | [1, 2048] | `torch.bfloat16` | the `x` line 1 norms | left addend of the first add |
| `layer0.block.h_in` | [1, 2048] | `torch.bfloat16` | 1 `RMSNormA(x, w_input_layernorm)` | the input norm on a real residual instead of a synthetic RMS-1 row |
| `layer0.block.mixer_out` | [1, 2048] | `torch.bfloat16` | 2 `mixer_0(h_in)` | right addend of the first add |
| `layer0.block.x1` | [1, 2048] | `torch.bfloat16` | 3 `x + m` | the first add's referent |
| `layer0.block.h_mlp` | [1, 2048] | `torch.bfloat16` | 4 `RMSNormA(x1, w_post_attention_layernorm)` | the post-attention norm |
| `layer0.block.moe_out` | [1, 2048] | `torch.bfloat16` | 5 `MoE(h_mlp)` | right addend of the second add, same token as `x1` |
| `layer0.block.out` | [1, 2048] | `torch.bfloat16` | 6 `x1 + f` | the second add, i.e. this gate's `block_output` row |

**Which dumper flag would emit them: none exists today, and this is a request, not a claim.**
`tools/dump_reference.py`'s CLI is `--blocks --tokens --seed --threads --out --manifest --verify`; its stage
set is the module list `tools/dump_reference.py::STAGE_PLAN`, consumed in `dump_block()`, whose entries are
`(suffix, stage, arm, dtype)` and write `f"{block.name}.{suffix}"`. So the request is (i) new `STAGE_PLAN`
entries under a `block.` prefix, filled from a forward hook on `Qwen3_5MoeDecoderLayer` exactly as
`specification/03 §03.3` (“what is handed to a mixer, and what a hook must record”) already prescribes, and
(ii) a flag selecting them (e.g. `--block-pass`) so the 178 pinned M0 records stay byte-identical — the
manifest is pinned and `Dumps.tensor` refuses on any digest mismatch, so bumping existing stage bytes would
break every green gate in the repo, not this one. Choosing among *pinning the block pass*, *deriving a
block-level band from a floor probe*, and *descoping the glue row from M2's exit gate* is not this worker's
call: the first two change fixture or band material and the third changes what M2's exit gate means.

## 5. Why the other 44 unlicensed rows are unlicensed (none of them is a checker defect)

| count | reason | what would license them |
|---|---|---|
| 18 | `policy-a-width-band-absent` | the stage-output floor probe that `quantization/README.md` §“How a quantized path gets gated” item 1 requires, run policy-A-dequantized against the bf16 pins. Deriving a band from a floor probe is agent work; **relaxing** a band is not |
| 16 | `no-band-entry-and-no-probe` — 12 GDN ref-arm stages that are already `ndiff=0` bit-exact (`a_raw`, `b_raw`, `beta`, `g_decay`, `k_pre`, `q_pre`, `v_pre`, `z`, `conv_pre`, `conv_out`, `conv_window`, `mixed_qkv`) plus `shared_gate_prob` (`ndiff=0`), `shared_raw`/`shared_out` (`ndiff=10`, max 1 ULP) and `combined_out` (`ndiff=3`, max 64 scaled-ULP) | the GDN band file's `band_licenses` population names only `q_hat/k_hat/scan_out/state` at `layer0/pinned/decode0`; the rest need their own floor probes or band entries, derived, never guessed |
| 7 | `policy-a-arm-outside-band-population` | the GDN band's licensed population contains the bf16 arm only; a policy-A entry is a band decision, not a code fix |
| 3 | `arm-pair-not-licensed-at-pass` (ref arm `core_attn`, `normed`, `y`) | same: the entry's arm pair does not cover this pass |
| 2 | `no-pinned-referent` | §4 above |

## 6. Controls (11, all fired=RED in both green runs) and checks (17, 0 findings)

`wrong-offset-weight-read` · `swapped-expert-slice` · `skipped-stage` · `stage-row-count-mismatch` ·
`stage-row-parse-truncation` · `policy-a-width-wrong-declaration` ·
`gather-counter-catches-a-whole-tensor-read` · `constant-predicate-scan-is-awake` ·
`device-claim-scan-is-awake` · `staged-precondition-refusals-fire` ·
`topk-form-probe-distinguishes-the-wrong-form`.

Checks: `stage-rows-equal-derived-population`, `parsed-record-count-equals-printed`,
`no-accepted-row-without-a-licence`, `every-unlicensed-row-names-stage-and-reason`,
`band-digest-pinned`, `pass-unreachable-while-rows-unlicensed`, `probe-limits-came-from-their-probes`,
`pinned-digests-verified-on-every-dump-read`, `policy-a-reads-carry-their-declared-width`,
`routed-experts-are-gathered-slices-not-whole-tensors`,
`router-and-norm-family-is-exact-in-the-artifact`, `no-rate-figures-in-output`,
`no-constant-predicates-in-the-gate-source`,
`router-logits-dtype-derivation-matches-the-pinned-referent`,
`router-r5-cast-is-a-real-cast-on-every-arm`, `device-claims-come-from-device-only`,
`staged-precondition-covers-this-blocks-tensors`.

## REMAINING WORK (ordered; nothing here is blocked on this worker)

1. **Policy-A stage-output floor probe** (the 18-row family). Until it exists the shipping shape cannot be
   ACCEPTED at any quantized stage, and M3's “pinned logits matched” gate will inherit the same wall.
2. **The block-pass fixtures of §4**, or an explicit user decision to derive/descope the glue row.
3. M3 sequencing: all 40 blocks + `lm_head` against the pinned prompt, greedy argmax exact. The GDN band is
   licensed at `layer0/pinned/decode0` only, so blocks 1 and 39 need their own band entries or their own
   floor probes before their rows can be judged.
4. The device half is still untouched by this file: nothing in `engine/block_gate.py` allocates or launches.
   Real staging evidence still comes from `engine/stage_run.py --device --verify` (see
   `engine/HANDOFF-m1.md`), and `engine/loader.py::device_slabs` still raises `device-allocator-absent`.
5. No timing has been taken anywhere in this lane and none should be quoted from it
   (`pitfalls/00`); M5's ≥ 48 t/s bar belongs to a clean window owned by the main agent.

## EXACT NEXT-WINDOW COMMANDS

CPU (no device, no lock, safe beside other workers):

```bash
cd /home/erokhins/erokhins_projects/qwen3.6-35b-a3b
V=/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python
$V engine/block_gate.py > engine/work/blockgate-cpu-$(date +%F).log 2>&1; echo exit=$?
$V engine/block_gate.py --staged-evidence \
    specification/evidence/stage-run-device-readback-2026-10-26.log \
    > engine/work/blockgate-cpu-staged-$(date +%F).log 2>&1; echo exit=$?
$V engine/block_gate.py --device             # exits 1, tag=live-device-path-absent; that is correct
```

There is no `--check-only` and no `--selftest` any more; both deletions are explained in the source where
the flags used to be parsed, and `RESULT … staged=absent|admitted|refused:<tag>` carries what a run
admitted.

The re-run costs roughly two full block forwards plus the control recomputations; the
`--staged-evidence` arm and control 10 each re-hash 537,215,744 B of the artifact. That is deliberate:
an admissions check that never reads the bytes is the failure M1's window c found. No timing is claimed
by any of these runs.

Device window (main agent only, zero workers active, halogen down, machine-wide stream lock held under
your tag, blocking, foreground — `AGENTS.md` §“GPU execution is serialized”), where the *staging* evidence
is made by M1's driver and then admitted here as a precondition:

```bash
cd /home/erokhins/erokhins_projects/qwen3.6-35b-a3b
V=/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python
$V engine/stage_run.py --device --verify --keep-evidence \
    specification/evidence/stage-run-device-$(date +%F).log
$V engine/block_gate.py --staged-evidence specification/evidence/stage-run-device-$(date +%F).log \
    > engine/work/blockgate-admit-$(date +%F).log 2>&1
```

`engine/block_gate.py --device` still refuses with `live-device-path-absent` in that window too, and that
refusal is correct: it is a CPU process.

## DECIDED (agent, 2026-10-26) — drafts to transplant into `DECISIONS.md`

`DECISIONS.md` is outside this worker's write scope, so the entries live here with the numbers that
forced them. Each is **DECIDED (agent, 2026-10-26)**; none of them is a user decision.

- **The policy-A width ruler is the ggml type the artifact declares, not the dtype that arrives after
  dequantization.** DECIDED (agent): compare `Loader.handle(name)["type_name"]` with `PolicyA.WIDTH`.
  Forced by checker bug (b): a Q8_0 tensor arrives as `torch.float32` by design, so the old comparison
  (torch dtype vs `Q8_0`) could not be true of anything. Adopted from llama.cpp, where the declared type
  is what the addressing is built from (`engines/llama.cpp/src/llama-model-loader.cpp::create_tensor`
  takes `gguf_get_tensor_type` and sizes rows with `ggml_row_size`/`ggml_type_size`); our loader already
  reads the same field it vetoes with.
- **R1 (the router logit product) lands in the model activation dtype on every arm, derived from
  `text_config.dtype`.** DECIDED (agent): rejected the alternative of banding `topk_weights`, which would
  have traded a derived exactness row for a tolerance the pins contradict. Forced by
  `policy_a/topk_weights ndiff=4 max_ulp=1` and by the two dtype-matched referents (§2): bf16 bytes 0, f32
  bytes 0, leaked-basis bytes 4, product 1326-refused. Cites AGENTS.md prediction 4,
  `prototype/moe_router/results.md` §Prediction 4 (cast-after moved 128/128 real rows),
  `specification/11 §11.2`, `speccheck/moe_ref.py::router` (`weights = vals.to(logits dtype)` — cast
  BEFORE the multiply). llama.cpp check per `AGENTS.md`: its router path
  (`src/llama-model-loader.cpp` aside — `ggml_mul_mat` on `ffn_gate_inp` with `GGML_TYPE_F32` kept exact)
  keeps the logits in the model dtype for exactly this reason; nothing here rests on a different error
  budget, because the pins, not llama.cpp, set the tolerance.
- **Device claims require a device; a log can only be a precondition.** DECIDED (agent): `--device` refuses
  (`live-device-path-absent`); `--staged-evidence` admits only after re-hashing every region the log names
  against the file this process opened (537,215,744 B for this block's 17 tensors) and prints
  `label=staging-integrity-precondition-not-a-device-result`. Forced by the M1 window-c finding (a
  `RESULT PASS` with no device digest anywhere) and by `pitfalls/40` §admissions. llama.cpp has no
  equivalent to check against — its loader reads the same file it will use, so “is this log about my file?”
  never arises — so this rule is ours and the burden of proof is the five refusal tags it must produce.
- **The residual-glue rows stay emitted and unlicensed.** DECIDED (agent): report `referent=absent` with a
  named reason rather than manufacture a residual stream by inverting RMSNorm, which would change the
  quantity under test. The exact fixture ask is §4.
- **Controls get their own read context (`ctx`) and the census counts only `ctx=="forward"`.** DECIDED
  (agent): forced by checker bug (c) — a control that proves the gather counter bites was changing the
  number the shipping pattern was judged by.
- **There is no controls-only mode in this gate.** DECIDED (agent): the old `--selftest` starved the
  controls that bite on the printed population (parse reconciliation, form probe), and a roster that
  cannot fire is a silent red. The flag is deleted and the reason is in the source where the flag used to
  be.
- **`block_forward.py` records what it read** (`reads[]` with `name/type/kind/bytes/read_class/ctx`,
  `PolicyA.last_type`) so three checks have a real ruler instead of a claim. DECIDED (agent), forced while
  fixing (b) and (c).

## Scratch left in `engine/work/` (regenerable, and named so nobody mistakes it for evidence)

`blockgate-cpu-2026-10-26.log` is the **inherited red state**, kept as the before picture;
`blockgate-cpu-2026-10-26b.log` is the intermediate run that witnessed the two-row parser drop
(`parsed=56 vs printed=58`, `controls=10 fired=8`), kept because it is the only record of that defect;
`blockgate-cpu-2026-10-26-fixed.log` and `blockgate-cpu-2026-10-26-staged.log` are the current green
runs. `staged-evidence-*.log` are control 10's corrupted copies of M1's staging log (M1's real evidence is
`specification/evidence/stage-run-device-readback-2026-10-26.log`); `const-predicate-injected-copy.py` is
control 8's injection; `probe_*.py` are the probes that produced §2 and §3.

---

# PASS 2 (2026-10-26, M2 device path) — `engine/block_device.py` and what it does *not* do

Same rules as above: agent decisions only, labelled as such; **no band, floor, route allowance, width or
pinned fixture was created, relaxed or retyped.** This pass prints the same digest
`band_digest=27a68c7e82c9aa0507b9bcd677adc0d5d1b0bc865dd1e86bf495ed90f47392ee` and judges rows with
`engine/block_gate.py`'s own `stage_row`/`adjudicate` — it adds no adjudication code of its own.

## 0. Scope actually delivered — and the part of the assignment that is NOT done

The assignment asked for three subsystem kernels on the device. **One is built: the shared expert.**
The GDN recurrent decode step and the MoE router + routed-expert gather are **not implemented anywhere in
this pass** — no kernel, no rows, no controls. They are listed in §4 below as the remaining work with the
reason each one needs its own pass. The vertical slice that does exist runs end to end: policy-A resident
bytes for layer 0 → compiled pair → record → stage rows → existing licences → verdict, with five controls.

## 1. Files this pass added

| file | what it is |
|---|---|
| `engine/device/m2_se_kernels.h` | the shared-expert stage arithmetic, composed from `prototype/shared_expert/shared_expert_kernels.h` (the prototype's item functions are used verbatim, not re-derived), plus the poison-NaN census and two control faults |
| `engine/device/m2_se_io.h` | host-only plan file parser + checker; refuses bad geometry or out-of-range spans by name. Also where the prototype's seven named wrong-spellings (`SE_MODE_*`) are reachable from the plan's `FORM mode=` word |
| `engine/device/m2_se_main.inc` | the single `main` both arms share: `HDR/PLAN/RUN/WROTE/RESULT`, counted HIP calls (zero on the CPU body) |
| `engine/device/m2_se_host.cpp` | g++ host twin (`-O2`, x86-64) |
| `engine/device/m2_se_arm.hip` | hipcc device arm, gfx1151: three `__global__` kernels + the counted device calls + the oversize-allocation control leg |
| `engine/device/build.sh` | builds both arms with `-ffp-contract=off -fno-fast-math`, one arch flag list, `-I prototype/shared_expert` |
| `engine/device_arm.py` | the support module: `CountedHip` (subclasses `stage_run.HipStaging`, adds only a tally), `parse_se_layout` (geometry + record layout read out of the prototype header, never typed), `residency` (tensor → covering staging run), `run_arm` (parse-or-refuse), `pair_bytes` |
| `engine/block_device.py` | the driver: stages layer 0 through `stage_run`, builds the arm slab with per-span digest binds, runs the arms, emits rows in `block_gate`'s format, runs 5 controls + 17 checks |
| `engine/work/probe_m2dev_layout.py`, `engine/work/probe_m2dev_se.py` | scratch probes (regenerable, not evidence); the second is what found the Q8_0-vs-bf16 magnitude question recorded as decision 5 in §5 |

Nothing outside `engine/` was edited except one side effect worth naming: `engine/stage_run.py`'s span
reader earns the audit self-test token, which appends one line per grant to
`quantization/evidence/audit-ordering-2026-10-26.log` (four lines in this pass, `section=engine/stage_run.py
staging execution`). That is the audit reader's own gate log, written by the shipped code path — not this
worker editing evidence.

## 2. CPU witness (executed; no device, no lock, no timing)

```
$V engine/block_device.py > engine/work/blockdevice-cpu-2026-10-26.log 2>&1; echo exit=$?
exit=0
```

Verbatim head and tail:

```
PAIR pair=host_twin_vs_hip_hostpath file=record.f32 bytes=22532 ndiff_bytes=0 cases=1 zero_tolerance=1 verdict=IDENTICAL
STAGETALLY arms=1 licensed=0 accepted=0 rejected=0 unlicensed=3
BLOCKDEVICE verdict=QUALIFIED block=layer0 subsystem=shared_expert rows=3 parsed=3 licensed=0 accepted=0 rejected=0 unlicensed=3 checks=17 red=0 controls=5 fired=5 band_digest=27a68c7e82c9aa05 note=a row with no licence is never counted as accepted, and PASS is unreachable while any row is unlicensed
RESULT mode=cpu checks=17 findings=0 controls=5 fired=5 rows=3 arms=policy_a device=off staged=not-attempted hip_calls=0+0 launches=0 bytes_to_device=0 bytes_from_device=0 band_digest=27a68c7e82c9aa05 no timing taken
```

All three rows are `UNLICENSED stage=<name> reason=policy-a-width-band-absent`, which is the honest
answer: no licensed band measures a Q8_0 arm against bf16 pins, and the floor probe
`quantization/README.md` §How a quantized path gets gates item 1 requires has never been run for these
stages. `QUALIFIED`, never `PASS`.

The five controls, all `fired=RED`:
`resident-span-corrupt` (one weight byte flipped → bind breaks, the two stages that read that span change
their distance to the pins, the stage that does not read it is unchanged to the bit, zero rows emitted),
`launch-failure-exits-red` (a plan whose span runs past the slab → `REFUSED tag=arm-exec-failed`, zero
rows; the second leg needs `--device`), `poison-census-can-be-nonzero` (skip stage 3 → 2048/2048 poison
where the honest record has 0/2048), `wrong-modes-further-from-pins` (the lane's seven named wrong
spellings, run on these bytes, are each ≥ as far from the pins in every stage and strictly further in at
least one — an ordering claim, no tolerance authored), `device-claim-scan-is-awake` (the scanner that
forbids a device-shaped line without a counted witness is witnessed non-zero).

## 3. EXACT NEXT-WINDOW DEVICE COMMANDS (main agent only)

Zero subagents active, halogen down via `tools/halogen_gate.sh`, machine-wide stream lock held under your
own tag, blocking, foreground. **`engine/block_device.py` never acquires `gpu_lock` itself** — a driver that
self-acquires and is called by a caller already holding it deadlocks silently (`pitfalls/50`), so take the
lock outside it:

```bash
cd /home/erokhins/erokhins_projects/qwen3.6-35b-a3b
V=/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python
tag="m2-device-window-$(date +%F)"
tools/gpu_lock.sh acquire "$tag"; trap "tools/gpu_lock.sh release '$tag'" EXIT
tools/gpu_lock.sh status && tools/gpu_lock.sh util
./engine/device/build.sh                                   # both arms; prints BUILD ok twice
$V engine/block_device.py --device \
    > engine/work/blockdevice-device-$(date +%F).log 2>&1; echo exit=$?
tools/gpu_lock.sh release "$tag"
```

What has to be true in that log, and what to read if it is not:

- `DEVALLOC slab=weights.shared bytes=133693440` and `DEVALLOC slab=weights.exact bytes=104581632`
  (≈227 MiB total, the two slabs that own layer 0's four shared-expert tensors) plus a
  `hipHostMalloc` staging buffer of 2105344 B — the largest covering run.
- `DEVWITNESS hip_objects=2 slabs=2 staged_runs=4 h2d_calls=4 h2d_bytes=5447680 d2h_calls=8
  d2h_bytes=8798208`. If `staged_runs=0` the device branch never ran and nothing in the file is a device
  claim.
- Four `STAGERUN … verdict=VERIFIED`, four `BIND … dev_sha256=<hex> … bind=ok` lines where `dev_sha256` is
  a *read-back* digest, not the file's digest echoed.
- `ARM arm=hip_device path=gfx1151 hip_calls=>0 launches=3 …` and `RESULT mode=device … launches=3`. **[SUPERSEDED — pass 3 §7: the SE arm now launches 4 and the total is 7.**
  The driver refuses (`tag=arm-device-claim-uncounted`) if a `--device` arm reports zero launches, and the
  check `device-claims-come-from-device-only` reddens if a device-shaped line appears without the counted
  witness (or the witness appears with no line).
- `CONTROL name=launch-failure-exits-red fired=RED legs attempted=2 (bad-plan:arm-exec-failed,
  oversize-alloc:arm-exec-failed)` — the second leg is device-only and is the leg that proves a failed
  device allocation cannot be answered by the host twin.
- `RESTORE tensor=blk.0.ffn_gate_shexp.weight run=211 verdict=STAGED` after the corrupt control wrote one
  byte into the resident set with a 1-byte H2D; if the restore does not read back clean the run stops with
  `tag=corrupt-restore-failed` rather than continue on bytes it dirtied.
- Expected verdict `QUALIFIED` with `checks=17 findings=0 controls=5 fired=5`. Any `RED` line names its
  check. **No number in this file is a rate**, and `no-rate-figures-in-output` is one of the 17 checks.

## 4. Not proven by anything in this pass (do not cite any of these as done)

1. **No device execution at all.** Every device-shaped claim above is *prepared*, not *observed*: no
   `hipMalloc`, no launch, no H2D/D2H has run since this pass began (the worker is forbidden from the iGPU).
   `RESULT … device=off hip_calls=0+0 launches=0` is the truthful state of the log this pass produced.
2. **GDN recurrent decode step: not built.** No engine-side kernel, no rows for `recurrent`/`conv_state`/
   `state`. It needs the state-pool question answered first (which buffers are resident vs per-token), and
   `prototype/gdn` is a decode-step lane whose oracle path is not shaped like a single-block stage record.
3. **MoE router + routed-expert gather: not built.** No `router_logits`/`topk_ids`/`topk_weights`/
   `expert_raw`/`routed_sum` rows on the device path. `prototype/moe_router` and `prototype/moe_gather`
   exist and are gated, but the gather needs the 240-dependent-read shape (AGENTS.md prediction 2) and the
   router needs the dtype-order constraint (prediction 4) honoured at the same time; neither fits inside
   this pass's remaining budget without becoming a second vertical slice.
4. **`lm_head`, block glue, attention/GDN prefill:** untouched, as in pass 1.
5. The `policy_a` shared-expert rows' distance to the pins (ndiff 1630–1645 of 2048, `maxabs ≤ 3.9e-3`)
   is *measured*, not judged. Whether that is what Q8_0 should cost is exactly what the missing floor probe
   would answer; nothing here claims it is acceptable.

## 5. DECIDED (agent, 2026-10-26) — pass-2 drafts to transplant into `DECISIONS.md`

1. **The device lane ships one subsystem first: the shared expert.** DECIDED (agent), forced by the
   deadline and by the fact that `prototype/shared_expert` is the only gated lane whose inputs are already
   policy-A widths (`blk.0.ffn_{gate,up,down}_shexp` Q8_0, `blk.0.ffn_gate_inp_shexp` F32-exact), so a
   vertical slice could be built without inventing a dequantization kernel. Consequence, written down
   plainly: the engine's device path covers 3 of layer 0's 29 stage rows and says `subsystem=shared_expert`
   in its first line so nobody reads 3 rows as a block.
2. **The fused gate|up pair is an address arrangement, not a requantization.** DECIDED (agent): the arm is
   given two spans (gate at `off`, up at `off + gate_len`) and the prototype's `SE_MODE_LAYOUT_PAIR_SWAP`
   machinery is the check that the two are not swapped; the bytes are the file's Q8_0 spans, moved by
   pointer, never re-encoded. Forced by `prototype/shared_expert/shared_expert_kernels.h` §SE_MODE_*
   (its `up_first`/`pair_swap` mistakes exist precisely because this arrangement is easy to get wrong) and
   by the arm slab bind lines. llama.cpp check per `AGENTS.md`: `ggml` keeps gate and up as separate
   tensors and fuses at the kernel level (`ggml/src/ggml-cuda/mmix.cu` shapes), i.e. the same "arrange by
   address, do not requantize" stance — adopted; nothing here rests on a different error budget.
3. **The compiled pair is g++-host vs hipcc-device, judged on the record bytes at zero tolerance.**
   DECIDED (agent): `-ffp-contract=off -fno-fast-math` on both, one flag list in `build.sh`, so a difference
   is codegen and not a flag. Witness `PAIR … bytes=22532 ndiff_bytes=0` plus
   `compiled-pair-same-census` (identical read/item/poison census) and `kernel-census-equals-geometry`
   (2097152+1048576+2048 element reads = 2·I·H + H·I + H from the header's own constants). A non-zero pair
   is a finding to report, never a band to absorb.
4. **No host fallback is reachable from a failed device act, structurally.** DECIDED (agent): `run_arm`
   raises before printing anything when the arm exits non-zero, when a report record is missing, when the
   verdict is not `OK`, or when a `--device` arm counts zero launches; the row emitter only ever sees the
   record file of a run that already passed. Forced by the inherited `block_gate --device` refusal pattern
   and by `pitfalls/40` §admissions. Witness `launch-failure-exits-red`.
5. **The Q8_0 shared expert really is far from the bf16 pins, and that is reported rather than banded.**
   DECIDED (agent), from `engine/work/probe_m2dev_se.py`: `arm_vs_cpu_policy_a meanrel 1.17e-2`,
   `arm_vs_pinned meanrel 4.44e-2`, `cpu_ref_bf16_vs_pinned meanrel 2.96e-5`. The row reason is
   `policy-a-width-band-absent`, the same reason `block_gate` gives for the other 18 policy-A rows; no
   threshold was written anywhere in this pass.
6. **The arm reads no address the plan did not name, and the driver binds every span it hands over.**
   DECIDED (agent): `m2_se_io.h` refuses a plan whose span exceeds the slab or whose geometry is not the
   header's, and the driver prints `BIND tensor=… file_sha256=… dev_sha256=… slab_sha256=… bind=ok|BROKEN`
   for all four spans on every run, including inside the corrupt control. Forced by
   `pitfalls/40` §a-check-whose-reference-side-is-derived-from-the-thing-under-test: the bind is what makes
   "the kernel read the file's bytes" falsifiable instead of assumed.
7. **Controls compare *ordering* or *equality*, never a tolerance this worker invented.** DECIDED (agent):
   wrong-modes control uses "≥ as far in every stage, strictly further in one", the corrupt control uses
   "changed vs unchanged" against a dataflow table (`STAGE_INPUTS`, cited from
   `engine/block_forward.py::moe_shared`), the census control uses "0 vs 2048". If the population changes,
   `stage-input-map-stale` refuses until the table is re-derived.

---

# PASS 3 (2026-10-26, M2 device path, router slice) — three router rows licensed ACCEPTED from a compiled arm, and a correction to a claim pass 1/2 cited

Still the same rules: agent decisions only, labelled; **no band, floor, route allowance, width,
tolerance or pinned fixture was created, relaxed or retyped anywhere in this pass.** The driver prints
`band_digest=27a68c7e82c9aa0507b9bcd677adc0d5d1b0bc865dd1e86bf495ed90f47392ee`, the same digest as every
earlier pass, and judges every row with `engine/block_gate.py`'s own `stage_row`/`adjudicate`. No device,
no iGPU, no `gpu_lock`, no launch, no timing: `grep -c gpu_lock` is 0 on `engine/block_device.py`,
`engine/device_arm.py`, `engine/device/build.sh` and all four `engine/device/*.{cpp,hip}`; the lock appears
only in the command line the main agent types in §3 below.

**State: `checks=23 findings=0 controls=9 fired=9 rows=6`, exit 0.** Six rows = 3 licensed ACCEPTED at
`ndiff=0` + 3 UNLICENSED with a named reason. Hard stop after the router: **no routed-expert gather, no
GDN**, both still unbuilt and listed as such in §7.

## DISCLOSURE-TOPKFORM-F32-CARRY — `TOPKFORM form=weights-f32-carry ndiff=0` is NOT independent evidence of the dtype order

*Section name is stable on purpose: cite it verbatim, `engine/HANDOFF.md` §DISCLOSURE-TOPKFORM-F32-CARRY.*

`engine/block_gate.py::topk_form_probe` prints, for the arm whose `routed_sum` licence *is* exactness
(`arm=ref_bf16`, verbatim from `engine/work/mainagent-blockgate-2026-10-26.log`, `grep '^TOPKFORM'`):

```
TOPKFORM arm=ref_bf16 form=weights-f32-carry ndiff=0 max_ulp=0 limit=0 referent=layer0.topk_weights_f32 dtype=torch.float32 n=8
TOPKFORM arm=ref_bf16 form=weights-emitted-cast-first ndiff=0 max_ulp=0 limit=0 referent=layer0.topk_weights dtype=torch.bfloat16 n=8
TOPKFORM arm=ref_bf16 form=product-times-cast-first ndiff=0 max_ulp=0 limit=0 referent=layer0.routed_sum dtype=torch.bfloat16 n=2048
TOPKFORM arm=ref_bf16 form=product-times-f32-carry ndiff=1326 max_ulp=32416 limit=0 referent=layer0.routed_sum dtype=torch.float32 n=2048
```

The second line and the fourth line are the two halves of AGENTS.md prediction 4 that actually bite. The
**first line is not evidence about the cast order at all**: `block_gate`'s f32-carry value is produced by
`engine/block_forward.py` — the same torch program that produced `layer0.topk_weights_f32` — so
`ndiff=0` there is an identity, not a test. Any commit wording that line as "the pinned material also
confirms the f32 form" is over-claiming; **the independent leg is `layer0.routed_sum` REFUSING the f32
carry: ndiff 1326 (max_ulp 32416) against cast-first's 0, on the same 2048 pinned values.** That refusal
is what carries prediction 4 in the torch lane, and it is what pass 1/2 should have cited.

The consequence for this pass: the engine's compiled arm cannot reuse the f32-carry leg as a witness,
because an *independent* fp32 route is not bit-exact against that dump. Measured on the arm
(`RTFORM`, same run as §3):

```
RTFORM arm=policy_a form=weights-cast-first field=wcast referent=layer0.topk_weights ndiff=0 max_ulp=0 n=8
RTFORM arm=policy_a form=weights-f32-carry field=wpre referent=layer0.topk_weights_f32 ndiff=8 max_ulp=2 n=8
RTFORM arm=policy_a form=weights-arm-basis-leak-pre-fix field=wleak referent=layer0.topk_weights ndiff=4 max_ulp=1 n=8
```

`ndiff=8 max_ulp=2` is the documented consequence, not a defect:
`prototype/moe_router/moe_router_kernels.h` §R5 states that an independent fp32 route reproduces the
pinned pre-cast weights to single-digit ULP and *not* bit-exactly. So `CONTROL
name=router-dtype-order-distinguishable-in-arm` makes a claim the arm can actually hold, and prints each
number it uses: cast-first reproduces the pinned bf16 weights exactly (`ndiff=0`, limit from the licensor
via `BG.prototype_limits()`, never typed here), the arm-basis leak is refused (`ndiff=4`), and the three
forms are three different numbers inside one record — cast-vs-carry differ in **8 of 8** slots,
cast-vs-leak in **8 of 8**. That is what "the arm enforces R5" means on a device arm; "f32-carry == 0"
would have been unfireable, and it is now stated as a non-requirement in the control's own text.

## 1. What pass 3 built, and what it deliberately did not

The router slice: layer 0's `blk.0.ffn_gate_inp.weight` (F32-exact, policy A) times the pinned token row,
softmax, top-8, renormalise, cast-before-multiply — compiled twice from one source, run on the host twin
and the hipcc arm's host path, judged row-by-row against the pinned dump. Rows: `router_logits`,
`topk_ids`, `topk_weights`. **All three licensed and ACCEPTED at `ndiff=0`.**

Still not built, unchanged from pass 2 §4: the routed-expert gather (`expert_raw`, `routed_sum`), the GDN
recurrent decode step, `lm_head`, block glue. Hard stop was taken after the router.

## 2. Files this pass added or changed

| file | what it is |
|---|---|
| `engine/device/m2_rt_kernels.h` | the router arithmetic: R1 accumulation over an **F32** matrix (`m2_rt_logit_acc_f32`, new engine code — the prototype's `mr_logit_row` reads bf16 and cannot be used on policy-A bytes), the leak-form softmax (`m2_rt_softmax_f32_row`, new engine code), and the record layout; R2–R5 are the prototype's own functions (see §5 for exactly which) |
| `engine/device/m2_rt_io.h` | plan parser/checker: refuses `elt != 4`, a span that exceeds the slab, a `top_k`/geometry mismatch, and a mode that reaches no field this slice emits |
| `engine/device/m2_rt_main.inc` | the `main` both arms share: `HDR/PLAN/RUN/WROTE/RESULT`, counted HIP calls, the control faults (`tie_forge`, `skip-renorm`, `skip-topk`) |
| `engine/device/m2_rt_host.cpp` | g++ host twin |
| `engine/device/m2_rt_arm.hip` | hipcc arm, gfx1151: 3 kernels (poison, R1 over 256 experts, R2..R5 as one serial item), 3 `hipMalloc`, 1 H2D, 1 `hipMemset`, 3 launches, 2 D2H, 3 `hipFree`, all counted |
| `engine/device/build.sh` | now builds both pairs; adds `-I prototype/moe_router` |
| `engine/device_arm.py` | `parse_rt_layout()` (evaluates the two headers' own `#define` arithmetic with an AST-limited evaluator — field lengths come from the offsets, nothing is typed), `residency_router()` (the structural width proof), `write_rt_plan()`, `rt_record()`, `rt_plan_check` |
| `engine/block_device.py` | both subsystems in one driver: `RTWIDTH`, `RTRECORD`, `RTFORM`, `RTVOID`, the router `PAIR`, the device-vs-host claimed-field `PAIR`, 4 router controls, 7 router checks |
| `engine/device/m2_se_arm.hip` | device path poisons its record with a kernel (`m2_se_k0_poison`) instead of `hipMemset(d_out, 0, …)`, so an unwritten slot reads the prototype's NaN on both paths. **Device launches for the SE arm are now 4, not 3** — see §8 for the pass-2 lines this corrects |
| `engine/work/probe_m2dev_rt.py`, `engine/work/probe_m2dev_rtwidth.py` | scratch probes (regenerable, not evidence). `probe_m2dev_rtwidth.py` is what measured the bf16-roundtrip-void fact in §4 |

## 3. CPU witness — executed, no device, no lock, no timing

```bash
cd /home/erokhins/erokhins_projects/qwen3.6-35b-a3b
./engine/device/build.sh
V=/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python
$V engine/block_device.py > engine/work/bd-r2.log 2>&1; echo exit=$?     # exit=0
```

Compiled pair, both translation units, zero tolerance, same source, two compilers:

```
PAIR pair=host_twin_vs_hip_hostpath file=record.f32 bytes=22532 ndiff_bytes=0 cases=1 zero_tolerance=1 verdict=IDENTICAL
PAIR pair=rt_host_twin_vs_rt_hip_hostpath file=router_record.f32 bytes=3232 ndiff_bytes=0 cases=1 zero_tolerance=1 verdict=IDENTICAL
```

The six stage rows, verbatim:

```
STAGE arm=policy_a weights=policy-A pass=layer0.row0 stage=shared_gate_prob referent=layer0.shared_gate_prob n=1 ndiff=0 dtype=torch.bfloat16 max_ulp=0 maxabs=0.000e+00 one_minus_cos=0.000e+00 meanrel=0.000e+00 perhead=0.000e+00 licence=none band=UNLICENSED stage=shared_gate_prob reason=policy-a-width-band-absent
STAGE arm=policy_a weights=policy-A pass=layer0.row0 stage=shared_out referent=layer0.shared_out n=2048 ndiff=1630 dtype=torch.bfloat16 max_ulp=32917 maxabs=1.953e-03 one_minus_cos=4.552e-05 meanrel=3.212e-02 perhead=1.953e-03 licence=none band=UNLICENSED stage=shared_out reason=policy-a-width-band-absent
STAGE arm=policy_a weights=policy-A pass=layer0.row0 stage=shared_raw referent=layer0.shared_raw n=2048 ndiff=1645 dtype=torch.bfloat16 max_ulp=32915 maxabs=3.906e-03 one_minus_cos=4.413e-05 meanrel=3.210e-02 perhead=3.906e-03 licence=none band=UNLICENSED stage=shared_raw reason=policy-a-width-band-absent
STAGE arm=policy_a weights=policy-A pass=layer0.row0 stage=router_logits referent=layer0.router_logits n=256 ndiff=0 dtype=torch.bfloat16 max_ulp=0 maxabs=0.000e+00 one_minus_cos=0.000e+00 meanrel=0.000e+00 perhead=0.000e+00 licence=floor-probe(logit_dot_route_disagreement_elements) band=ACCEPTED
STAGE arm=policy_a weights=policy-A pass=layer0.row0 stage=topk_ids referent=layer0.topk_ids n=8 ndiff=0 dtype=torch.int64 max_ulp=0 licence=spec-10.6(exactness) band=ACCEPTED
STAGE arm=policy_a weights=policy-A pass=layer0.row0 stage=topk_weights referent=layer0.topk_weights n=8 ndiff=0 dtype=torch.bfloat16 max_ulp=0 maxabs=0.000e+00 one_minus_cos=0.000e+00 meanrel=0.000e+00 perhead=0.000e+00 licence=spec-10.6(exactness) band=ACCEPTED
```

```
STAGETALLY arms=1 licensed=3 accepted=3 rejected=0 unlicensed=3
RESULT mode=cpu checks=23 findings=0 controls=9 fired=9 rows=6 arms=policy_a device=off staged=not-attempted hip_calls=0+0 launches=0 bytes_to_device=0 bytes_from_device=0 band_digest=27a68c7e82c9aa05 no timing taken
```

Why the router rows can be licensed on a policy-A arm and the shared-expert rows cannot: the licence
`engine/block_gate.py::adjudicate` gives a policy-A router row is conditioned on the router weights being
*exact*, and the artifact really does store them `F32` (`RTWIDTH`), so the arm and the dump are arithmetic
over the same numbers. The shared expert is Q8_0 against bf16 pins, which no licensed band measures —
hence `policy-a-width-band-absent` and `QUALIFIED`, never `PASS`.

## 4. `RTVOID` — the measured fact that killed a control leg, printed instead of hidden

```
RTVOID leg=bf16-roundtrip tensor=blk.0.ffn_gate_inp.weight span_bytes=2097152 values=524288 changed_values=0 values_with_nonzero_low16=0 verdict=VOID-ON-THIS-ARTIFACT note=a bf16 copy of this span is the same numbers, so this leg cannot redden anything and is reported, not required; the F32 premise is carried by the plan refusal, the residency width checks in engine/device_arm.py, and the one-ULP flip
```

Every one of the 524,288 F32 values in `blk.0.ffn_gate_inp.weight` has a **zero low halfword**, so
bf16→f32 changes 0 of them. A "requantized copy" perturbation built by rounding the span to bf16 is the
same 2,097,152 bytes, and a control built on it can only ever print `ndiff 0->0`. That is what the first
draft of this control did — it printed `fired=SILENT`, which is how it was caught. The leg stays in the
output as a *reported measurement with a verdict of VOID*, and the F32 claim is carried by:

1. **structurally** — `engine/device_arm.py::residency_router()` refuses a router tensor whose declared
   type is not `F32`, whose `n_elts != MR_E*MR_H`, or whose byte length is not exactly `4*n_elts`, and
   `engine/device/m2_rt_io.h` refuses a plan whose `elt` is not 4 (`CONTROL … router-span-is-f32` proves
   the refusal by handing the arm a plan that says `elt=2`);
2. **by the smallest difference an F32 store can carry** — xor bit 0 of the mantissa of every one of the
   524,288 weights, a difference no bf16 store can express at all. The arm's own fp32 field moves:

```
one-ULP leg: flipping bit 0 of every one of the 524288 weights moves 246 of 256 fp32 logits in the arm's own logits_f32 field, while the bf16 router_logits row moves 0 of 256 (bf16 rounding hides 2^-24, which is why the fp32 field is the one that carries this claim); bind: span-digest vs span-digest holds on the honest copy and breaks on the flipped one (holds/broke)
```

   Note the asymmetry, because it is the honest shape of the claim: `router_logits` (the licensed bf16
   row) does **not** move — 2⁻²⁴ is below bf16's ~2⁻⁸ — so the row-sensitivity claim is made on
   `logits_f32`, the arm's own fp32 field, and the bf16 count is printed rather than required. Requiring
   the bf16 row to move would have been sizing a band by what the build happens to produce.
3. **the bind, compared like against like**: `build_rt_slab`'s `slab_sha` is `sha256` over exactly the
   span's `length` bytes placed in the slab, and `s["file_sha"]` is `sha256` over the same `length` bytes
   read from the file. The control now shows it **holding on the honest copy and breaking on the flipped
   one** (`holds/broke`), which is the pair that makes the comparison falsifiable. (Pass 2's draft
   compared the *perturbed* digest against the span digest without ever showing the equal case, and the
   "lossy" bytes were in fact identical, so it printed `HELD`; the object-level complaint in the mid-run
   handoff was wrong — the digests were the same kind, the data was not different.)

## 5. What the router arm shares with the prototype lane — the independence boundary, named

The arm does **not** call `engine/block_forward.py::moe_router()`; the referent's program and the arm's
program are different programs, which is what makes the three `ndiff=0` rows a comparison rather than a
tautology. But the arm is not independent of the *prototype lane* either, and the honest list is:

| prototype symbol (`prototype/moe_router/moe_router_kernels.h`) | used for | what stops being independent |
|---|---|---|
| `mr_softmax_row` | R2, the bf16-basis softmax | the softmax max-subtraction and exp ordering behind `router_logits` and every weight field |
| `mr_topk_row` | R3, both tie rules | the selection and tie-break that produce `topk_ids` |
| `mr_renorm_row` | R4 | the renormalisation behind `topk_weights` |
| `mr_bf16_to_f32`, `mr_f32_to_bf16` | R1 cast, R2's input cast, R5 | every dtype edge, including the cast-before-multiply this slice claims |
| `mr_exp` | inside `m2_rt_softmax_f32_row` | the leak form's exp |
| `mr_mode_reaches` | the admissibility guard for `FORM mode=` | which controls are even allowed to be asked |

Engine-new code, i.e. the part a prototype gate cannot vouch for: `m2_rt_logit_acc_f32` (R1 over an F32
matrix — the prototype's `mr_logit_row` reads bf16 and would be wrong on policy-A bytes) and
`m2_rt_softmax_f32_row` (the arm-basis leak field). So `router_logits`'s **R1 half is engine code** and its
R2 half is prototype code; `topk_ids` and `topk_weights` are prototype arithmetic over an engine-computed
R1. This is stated because a green row here licenses *the composition on device bytes*, not a re-proof of
the prototype's arithmetic.

## 6. The nine controls, all `fired=RED` in the run above

Shared-expert (5, unchanged from pass 2 and still firing): `resident-span-corrupt`,
`launch-failure-exits-red`, `poison-census-can-be-nonzero`, `wrong-modes-further-from-pins`,
`device-claim-scan-is-awake`.

Router (4, all new in this pass; verbatim `CONTROL` lines are in `engine/work/bd-r2.log`):

- `router-dtype-order-distinguishable-in-arm` — see §DISCLOSURE-TOPKFORM-F32-CARRY. Every number it uses
  is printed (`RTFORM` × 3) plus the two pairwise-distinctness counts from inside one record.
- `topk-selection-perturbations-redden-ids` — two legs, both derived from the prototype's own modes, and
  the reason the tie leg is shaped the way it is: with an exact tie forged at the K boundary, the rule in
  force (lower index) *still* picks the pinned expert, so the honest row cannot move under it — what moves
  is the **other rule's** ids row, `ndiff=3`. That is the load-bearing witness: an arm with the tie rule
  backwards prints a red `topk_ids` row on this very input. The second leg is `mode=topk_ascending`:
  ids row `0->6`, weights row `0->5`, and the fields the prototype's `mr_mode_reaches` says it cannot
  reach (`logits`, `probs`) are bit-identical, checked rather than assumed. Zero rows emitted by either leg.
- `router-span-is-f32` — §4 above.
- `router-poison-census-can-be-nonzero` — with R4/R5 skipped, `wpre` and `wcast` are 8/8 poison while
  `ids` stays written (R3 ran), against 0/8 on the honest record.

Checks: 23, all PASS, 0 findings. They include `rt-geometry-matches-the-config` (`MR_E=256`/`MR_K=8`/
`MR_H=2048` against the checkpoint's `n_experts`/`top_k`/`d_moe` — a geometry this file cannot read is one
it cannot judge), `rt-census-equals-geometry` (524,288 weight reads = E·H, 524,288 activation re-reads,
256 logit items + 1 serial item), `rt-compiled-pair-byte-identical`, `rt-ids-are-a-selection`, and
`rt-poison-census-zero-on-the-honest-record`.

## 7. Exact device-window command, and what must be true in the log

Main agent only: zero subagents active, halogen down through `tools/halogen_gate.sh`, blocking, foreground.
**The driver never acquires the lock itself** (`grep -c gpu_lock engine/block_device.py` = 0) — a driver that
self-acquires and is called by a caller already holding it deadlocks silently (`pitfalls/50`):

```bash
cd /home/erokhins/erokhins_projects/qwen3.6-35b-a3b
V=/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python
tag="m2-device-window-$(date +%F)"
tools/gpu_lock.sh acquire "$tag"; trap "tools/gpu_lock.sh release '$tag'" EXIT
tools/gpu_lock.sh status && tools/gpu_lock.sh util
./engine/device/build.sh
$V engine/block_device.py --device > engine/work/blockdevice-device-$(date +%F).log 2>&1; echo exit=$?
tools/gpu_lock.sh release "$tag"
```

Must be true in that log, or the run is not a device run:

- `DEVALLOC slab=weights.shared bytes=133693440`, `DEVALLOC slab=weights.exact bytes=104581632`, and a
  staging buffer of 2105344 B (the largest covering run; layer 0's shared spans are runs 210/211/212 and
  the router span is run 334 on `weights.exact`).
- Five `BIND … dev_sha256=<hex>` lines where `dev_sha256` is a **read-back** digest, and five
  `STAGERUN … verdict=VERIFIED`.
- `ARM arm=hip_device … launches=4` (shared expert: poison + 3 stages) and `ARM arm=rt_hip_device …
  launches=3` (router: poison + R1 + R2..R5), both with `hip_calls>0`. `RESULT … launches=7`.
- `PAIR pair=rt_hip_device_vs_rt_hip_hostpath file=router_record.f32 bytes=3232 ndiff_bytes=?
  claimed_fields=ids,ids_alt,wpre,wcast claimed_ndiff=… verdict=CLAIMED-IDENTICAL`. The claimed fields are
  exactly the ones the licences measure at exactness. If `ndiff_bytes` is non-zero **while the claimed
  count is zero**, the difference lives in a long-dot field and that is the prototype's documented
  host-vs-device `exp()` scope (§mr_exp) — a printed fact, not a band. If a claimed field differs,
  `rt-device-host-claimed-fields-identical` reddens and that is a defect, not noise.
- `CONTROL name=launch-failure-exits-red fired=RED legs attempted=2 (bad-plan:arm-exec-failed,
  oversize-alloc:arm-exec-failed)` — the device-only leg is the one that proves a failed allocation cannot
  be answered by the host twin.
- `CONTROL name=poison-census-can-be-nonzero fired=RED` with the same counts as the CPU run (2048/2048 vs
  0/2048). If the device's poison kernel had been replaced by `hipMemset`, this would read 0/2048 on the
  device and non-zero on the host, and that is the `lm_head` sentinel-defect class; it is why the poison is
  a launch on both paths.
- `RESTORE … verdict=STAGED` after the corrupt control's 1-byte H2D, or the run stops with
  `tag=corrupt-restore-failed`.
- Expected verdict `QUALIFIED` with `checks=23 findings=0 controls=9 fired=9`. **No rate appears in the
  log** (`no-rate-figures-in-output` is one of the checks).

## 8. Not proven by anything in this pass — do not cite these as done

1. **Nothing device-executed.** `RESULT … device=off hip_calls=0+0 launches=0` is this pass's truth. Every
   device-shaped statement in §7 is a precondition the main agent has to observe, not a result.
2. **`router_logits` at `ndiff=0` is one token row against one pinned row.** It is not evidence that the
   summation-order independence holds across a prompt; the prototype lane's own floor probe is what bounds
   that, and the row here is ACCEPTED under `logit_dot_route_disagreement_elements`, not re-derived.
3. **The bf16 router row's insensitivity is measured, not modelled.** 246/256 fp32 logits move under the
   one-ULP flip and 0/256 bf16 logits do. This pass does not claim to know the ULP-level distribution of
   that effect over many tokens; it claims the arm consumes the F32 bytes.
4. **Routed-expert gather: not built.** No `expert_raw`/`routed_sum` row exists on the device path, so
   AGENTS.md prediction 2 (240 dependent 6 MiB reads per token, grouped launches) is untouched, and
   prediction 4's *product* half (`product-times-f32-carry`, the 1326-ndiff refusal) cannot be reproduced
   on the device path until `routed_sum` exists.
5. **GDN recurrent decode step: not built** (unchanged from pass 2 §4.2).
6. **`topk_ids`/`topk_weights` are 8 elements.** Their exactness licences are satisfied exactly, but the
   must-catch population for a selection row is 256 choose 8, and this pass only proves the rules are
   load-bearing via the two perturbations in §6 — not that every tie case is handled.

## 9. DECIDED (agent, 2026-10-26) — pass-3 drafts to transplant into `DECISIONS.md`

None of these is a user decision; none touches a width, a band, a floor or a route allowance.

1. **The router slice is the first device slice that can be licensed on policy A, and it is the one that
   was built.** DECIDED (agent, supervisor directed the ordering), forced by the arithmetic: the router
   matrix is stored F32-exact, so the arm and the dump compute over identical numbers and `adjudicate`'s
   existing router licence applies; the shared expert is Q8_0 against bf16 pins and no licensed band covers
   that gap, so its rows stay `UNLICENSED`. Witness: three rows `ACCEPTED` at `ndiff=0` vs three rows
   `UNLICENSED reason=policy-a-width-band-absent` in the same run.
2. **R1 is rewritten for F32 weights instead of importing the prototype's bf16 dot.** DECIDED (agent):
   `m2_rt_logit_acc_f32` reads `const float *`, because `mr_logit_row` reads `uint16_t` bf16 patterns and
   policy A does not store the router in bf16. Forced by `engine/device_arm.py::residency_router()`'s
   width proof (F32, `n_elts`, `4*n_elts`) and by `RTWIDTH`. llama.cpp check per AGENTS.md: its Q8_0 dot
   accumulates int8 products exactly by first quantizing the *activation*
   (`ggml/src/ggml-cpu/quants.c::ggml_vec_dot_q8_0_q8_0_generic`, via `vec_dot_type`); **not imported** —
   activation quantization trades our 1-ULP question for a 2⁻⁸ input perturbation, which is D-17 territory
   and the user's call.
3. **The f32-carry leg is removed as a requirement, with the reason printed.** DECIDED (agent), see
   §DISCLOSURE-TOPKFORM-F32-CARRY: the requirement was unfireable on an independent route (§R5), and the
   claim it was standing in for is carried by `layer0.routed_sum` refusing the f32 carry (1326 vs 0). The
   replacement requirement is pairwise distinctness of the three forms inside one record, so the licensed
   `ndiff=0` cannot be a coincidence of this row's margin.
4. **A control leg measured to be vacuous is printed as void, not deleted and not kept as a fake.**
   DECIDED (agent): the bf16-roundtrip leg prints `RTVOID … verdict=VOID-ON-THIS-ARTIFACT` with its
   numbers and is excluded from the fire condition, while the F32 claim moves to the structural legs plus
   the one-ULP flip on `logits_f32`. Forced by the measurement in §4 and by
   `pitfalls/40` §a-check-that-can-produce-no-output-cannot-fail.
5. **Tie-break is witnessed through the *other* rule, not through the honest row.** DECIDED (agent): on a
   forged tie the lower-index rule reproduces the pinned pick, so the honest row is the wrong place to
   look; the fire condition is the higher-index rule's ids row (`ndiff=3`) plus the two rules disagreeing.
   Derived from `mr_mode_reaches`, not from a typed expert index — the boundary is chosen from the row's own
   probabilities.
6. **Device arms poison their record with a launch, on both paths.** DECIDED (agent): `hipMemset` cannot
   write a NaN bit pattern, and a memset-to-zero would make the unwritten-slot census mean "zero" on the
   device and "2048" on the host. SE launches went 3 → 4 to keep this honest. This is the `lm_head`
   sentinel-destroyed-by-bulk-memset defect class, closed before anyone ran a device.

## 10. Corrections to PASS 2's text (read with §7, not instead of it)

- Pass 2 §3 says `ARM arm=hip_device … launches=3` and `RESULT … launches=3`. **Now 4 and 7** (SE poison
  launch added; router arm adds 3). The bullet in pass 2 is superseded by §7 above.
- Pass 2 §1's `m2_se_arm.hip` row says "three `__global__` kernels"; it is four now, for the reason in §9.6.
- Pass 2 §4 item 3 said "MoE router + routed-expert gather: not built". The **router half is built** by this
  pass; the gather half is still not (§8.4).
- Pass 2 §2's `checks=17 controls=5` is superseded by `checks=23 controls=9` for the current file; the
  17-check/5-control run is a real earlier state of a smaller driver, not a stale claim about it.
