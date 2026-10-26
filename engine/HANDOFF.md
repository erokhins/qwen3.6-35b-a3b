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
