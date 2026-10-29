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
| **PASS 4** `engine/block_device.py` (CPU, no device, no lock, no timing) | `blockdevice-cpu-2026-10-26-p4.log` | exit 0, `checks=28 findings=0 controls=10 fired=10`, `rows=6 parsed=6`, `licensed=3 accepted=3 rejected=0 unlicensed=3`, same `band_digest`. See §PASS 4: the `wpre` claim is split and re-derived, the tie control now runs on the path it claims, and §5 replaces both `--device` command blocks |

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

> **SUPERSEDED by PASS 4 §5 — do not paste this block.** It sets no `pipefail`, it lets
> `tools/gpu_lock.sh status` print "someone else holds it" and exit 0 anyway, and it ends with
> `echo exit=$?`, which turns a gate refusal into a number in scrollback. PASS 4 §5 is the block that
> refuses instead.

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

> **SUPERSEDED by PASS 4 §5 — do not paste this block**, for the reasons listed there (no `pipefail`, a
> status print that is not a gate, an `EXIT` trap that may never fire, and `echo exit=$?` where a refusal
> belongs). The "must be true" list below is also corrected by PASS 4 §5 and §7.

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

---

## PASS 4 (2026-10-26, window `e2`'s two REDs) — the `wpre` claim is split and re-derived; the tie control learns which path it is on; the device block stops being able to swallow a refusal

Inherited from the first device execution of the router arm
(`specification/evidence/blockdevice-device-2026-10-26-e2.log`): the accepted facts (licensed router rows
`router_logits`/`topk_ids`/`topk_weights` at `ndiff=0`, `compiled-pair-byte-identical` PASS at 22,532 B
with `ndiff_bytes=0`, four `STAGERUN VERIFIED`, nine `BIND dev_sha256`, `launches=4+3=7`, shared-expert
rows honestly `UNLICENSED`) stayed as they were, and two REDs were fixed. Nothing here touches the band
file, any licence, or the digest: the run below still prints
`27a68c7e82c9aa0507b9bcd677adc0d5d1b0bc865dd1e86bf495ed90f47392ee`.

| run (this pass) | log | outcome |
|---|---|---|
| `engine/block_device.py` (CPU, no device, no lock, no timing) | `engine/work/blockdevice-cpu-2026-10-26-p4.log` | exit 0, `checks=28 findings=0 controls=10 fired=10`, `rows=6 parsed=6`, `licensed=3 accepted=3 rejected=0 unlicensed=3`, `verdict=QUALIFIED` |
| replay of window `e2`'s preserved device records through the new code (no device act) | `specification/evidence/blockdevice-wpre-scope-replay-2026-10-26.log` | device-vs-host field census, attribution, premise, and the bound's rejection legs, all on the real device bytes |
| derivation + attribution write-up | `specification/evidence/blockdevice-wpre-scope-2026-10-26.md` | the three priors, the arithmetic, and what was proven where |

## 1. RED 1 — `wpre` differs between the two compilations, and that was the claim's bug, not the kernel's

`PAIR … ndiff_bytes=237 … claimed_ndiff=ids=0/8,ids_alt=0/8,wcast=0/8,wpre=5/8 verdict=CLAIMED-DIFFER`
came from window `e2`'s own records, preserved byte-for-byte in `engine/work/e2-device-records/`
(`rt_out_device.bin` sha256 `9430f861…bb97`; the host twin and the hipcc host path are the same
`63dccf2c…c36`). Measured on those bytes, fp32 ULP throughout:

| field | ndiff | worst ULP | what it says |
|---|---|---|---|
| `logits`, `logits_f32` (R1) | 0/256 | 0 | R1 bit-identical — the long dot is not the mechanism |
| `probs` (R2) | 232/256 | 3 | the two compilations' `exp()` |
| `wpre` (R4) | 5/8 | **2** | per slot `1,1,1,0,0,0,2,1`; relative `8.5e-08…1.4e-07` |
| `wcast` (R5) | 0/8 | 0 | the bf16 cast absorbs the fp32 spread, exactly as §R5 predicts |
| `ids`, `ids_alt`, `wleak` | 0/8 | 0 | selection and the leak row unmoved |

The attribution is answerable in four separately-checked conditions rather than asserted: R1 identical,
`wcast` identical, **every moving `wpre` slot has its own `probs` addend moving** (5 of 5; the 3 that did
not move had an addend that did not move), and no field outside the `exp()` path moved. All four hold, so
this is `prototype/moe_router/moe_router_kernels.h` §R5/§`mr_exp`'s documented `exp()` scope and **not a
codegen defect** — and each condition is its own check
(`rt-device-host-difference-is-the-documented-exp-scope`), so the next build cannot inherit this
conclusion as a assumption. `wpre` is `p_k / SUM_K p_j` with `p_i = exp(logit_i) / SUM_E exp(logit_e)`,
which is the only route by which a one-step `exp()` difference reaches a `wpre` slot while R1 sits still;
both routes are also inside the licensor's own `weights_vs_pinned_f32_ulp = 5` of the pinned pre-cast
weights (device 4 ULP, host twin 2 ULP). The full write-up is
`specification/evidence/blockdevice-wpre-scope-2026-10-26.md`; the driver line:

```
RTATTRIB pair=rt_hip_device_vs_rt_hip_hostpath r1_fields_bit_identical=True wcast_bit_identical=True every_moving_slot_has_a_moving_probs_addend=True fields_moving_outside_the_exp_path=none premise_ulp_to_pinned_pre_cast=4/2 (band 5) zero_tolerance_ndiff=ids=0/8,ids_alt=0/8,wcast=0/8 verdict=CONSISTENT-WITH-R5-EXP-SCOPE note=§R5 predicts exactly this shape…
```

Format is what `engine/block_device.py` prints in one line; the numbers above are what the **replay of
window `e2`'s preserved records** produces (`blockdevice-wpre-scope-replay-2026-10-26.log`), not a witnessed
device line — a window has to reproduce them.

## 2. The claim, split: three fields stay at zero tolerance, `wpre` gets a derived scope

The claim was **not** deleted — D-18 reserves relaxing a band to the user, and "stop requiring bit-identity
on a field nobody licensed at exactness" is exactly that. Following the D-20 precedent, the requirement was
**re-derived** with a witnessed rejection, from three independent prior sources, none of them this build's
output (`engine/block_device.py::rt_wpre_scope`, recomputed every run; it **refuses** if the wording it
cites has moved out of the headers):

```
WPRESCOPE field=wpre pair=rt_hip_device_vs_rt_hip_hostpath n=8 ndiff=5/8 ulp_per_slot=1,1,1,0,0,0,2,1
  measured_max_ulp=2
  scope_ulp=9 binding=documented §R5 'to single-digit ULP' -> at most 9 ULP
  prior=10 triangle(2 x licensed weights_vs_pinned_f32_ulp=5, prototype/moe_router/evidence/moe-router-host-2026-10-26.json §bands)
  prior=9  documented §R5 'to single-digit ULP' -> at most 9 ULP
  prior=22 propagated §mr_exp 'differ by an ULP or two' (2 ULP/call) through R2 then R4 -> 1.311e-06 relative
  verdict=WITHIN-SCOPE
```

(Line breaks added to read it; the driver prints one line, and the numbers above are the replay's on the
preserved device record — the CPU witness prints the same line with its own pair and `measured_max_ulp=0`.)

The **strictest** prior binds (9), so a document loosening cannot silently widen the bound; the measured 2
is 4.5× inside it, and the bound is itself stricter than the prototype lane's own cross-arm allowance for
the same field (`prototype/moe_router/gate.py::compare_arms` uses `4 × 5 = 20` ULP). The first prior's
premise is measured, not assumed: `rt-wpre-premise-measured-against-the-pin` requires **both** compilations
inside the licensed 5 ULP of the pinned pre-cast record before the triangle inequality is allowed to
license a cross-compilation number. The artifact's grip on the bound is witnessed, not claimed: the
triangle prior is exactly twice `weights_vs_pinned_f32_ulp` as the artifact reads it (5 → 10), and
re-deriving with that number set to 1 in memory makes the prior 2 (`rt-wpre-scope-derived-from-artifacts`);
a bound typed into this file would not move. And `rt-ulp-ruler-matches-the-licensor` forces this file's
per-slot ULP ruler to agree with `engine/block_gate.py::ulp_diff` on the same data, so there is one ruler
in the repo. The checks that
replace the old single `rt-device-host-claimed-fields-identical`:

| check | tolerance | reddens on |
|---|---|---|
| `rt-device-host-licensed-fields-identical` | zero, on `ids`, `ids_alt`, `wcast` | any single differing bit — it cannot hide inside `wpre`'s spread |
| `rt-device-host-wpre-within-derived-scope` | the derived 9 ULP | a `wpre` slot past the scope, or a record whose slot count is not `MR_K` |
| `rt-wpre-premise-measured-against-the-pin` | each route ≤ the licensed 5 ULP | a route that has drifted past its own licence, which voids the triangle step |
| `rt-device-host-difference-is-the-documented-exp-scope` | the four attribution conditions | any of them failing — i.e. a codegen defect masquerading as the `exp()` story |
| `rt-wpre-scope-derived-from-artifacts`, `rt-ulp-ruler-matches-the-licensor` | structural | a bound that is typed, or a ruler that drifted from the licensor's |

## 3. The bound is seen to reject: control 10

`CONTROL name=wpre-scope-rejects-a-bigger-perturbation`, six legs, every perturbation sized from the
**limit** and never from the spread — "a limit of `>= d` rejects nothing and is not a gate"
(`pitfalls/40-gates-and-admissions.md`). CPU witness, verbatim:

```
CONTROL name=wpre-scope-rejects-a-bigger-perturbation fired=RED legs=host_cpubody:1 claims=host_cpubody field=wpre scope=9 ULP; legs: honest-pair pair=rt_hip_hostpath_vs_rt_hip_gpp_twin ndiff=0/8 max_ulp=0 limit=9 -> INSIDE; real-arm-mistake path=host_cpubody mode=softmax_bf16 ndiff=8/8 max_ulp=53110 limit=9 -> OUTSIDE (rejected, as it must be); at-limit moved=9ULP ndiff=8/8 max_ulp=9 limit=9 -> INSIDE; one-past-limit moved=10ULP ndiff=8/8 max_ulp=10 limit=9 -> OUTSIDE; far-past-limit moved=64ULP ndiff=8/8 max_ulp=64 limit=9 -> OUTSIDE; licensed-field-one-bit xor bit0 of one slot each -> wcast=1/8, ids=1/8, ids_alt=1/8 (each must read 1/8, so the zero-tolerance half still bites one bit). No constant predicate: legs 1 and 2 are records the arms produced, 3-5 are perturbations sized from the derived limit, and 6 is a bit flip on the licensed fields. Arm legs executed by this control=host_cpubody:1; rows emitted=0
```

Leg 2 is not an injected bit: `mode=softmax_bf16` is the lane's own named mistake (the bf16-renormalise
error in the engine header's R4 comment), `mr_mode_reaches` says it reaches `wpre`, and the arm produced
the rejecting record. Legs 1–4 and 6 run in a CPU witness too, so the bound is falsifiable without a
device window; the replay of the device record gives the same verdicts at 9/10/64 ULP.

## 4. RED 2 — the tie control guarded one path and claimed the other

It ran both legs on the host twin while the honest record of a `--device` run is the device arm's, so its
"fields that must not move" test compared a host-twin record against a device record, found `probs` moved
— §`mr_exp`'s documented `exp()` scope, not a `topk_ascending` bug — and read **SILENT in exactly the mode
everything above M2 is built on**. Now: every path the run executed gets **both** legs against **its own**
honest record (host legs on the g++ twin, device legs on the device arm, written under
`work/rtselect_device/`), and the leg counts are read off each arm's own `RUN path=` report
(`arm_leg`), not asserted by the control. CPU witness, verbatim:

```
CONTROL name=topk-selection-perturbations-redden-ids fired=RED legs=host_cpubody:2 claims=host_cpubody per path, each leg against that path's own honest record — host_cpubody legs=2: tie ids 0->0, other-rule row 3 (must exceed 0); ascending ids 0->6, weights 0->5, fields that had to stay put and did: none. …
```

Device mode runs the same two legs a second time under `path=device` (tie-forge reddening `ids_alt`, and
`topk_ascending` moving `ids` and the weights), so the expected device line is
`fired=RED legs=device:2,host_cpubody:2 claims=host_cpubody,device` (the leg map prints sorted, so `device`
comes first) — that line is **not yet witnessed**; it needs a window. Two checks back it up:

- `control-legs-cover-the-path-under-claim` — a control that declares a path with zero arm acts counted on
  it reddens, and a control reporting device legs in a run with no device act reddens; the detail names
  which controls declare a path and which do not, so "declare nothing" is visible rather than a loophole.
- `device-path-claims-executed-on-device` — in a device run the path-claiming controls must have executed
  at least two legs under `path=device`. In a CPU run it says it is vacuous and why.

## 5. The device-window command, corrected (supersedes pass 2 §5 and pass 3 §7)

Pass 3 §7's block could report success on a refused gate. Specifically: it set no `pipefail`, so a
`gate | tee`/`| head` would return the last stage's status and throw the gate's away; it ran
`tools/gpu_lock.sh status && tools/gpu_lock.sh util`, which (a) prints `HELD: someone else` and still
exits 0, so "grepped the status" was never a gate, and (b) short-circuits on a non-zero status and sails on
without a lock; it released the lock through an `EXIT` trap in a snippet that may never exit as its own
process, which is how a lock outlives its window; and it ended with
`echo exit=$?` — a number in scrollback, not a refusal anyone had to clear. Halogen-down was asserted by
reading a message rather than the device. Corrected block, main agent only (zero workers **stopped**, not
quiet; blocking, foreground; the driver still never acquires the lock itself, `grep -c gpu_lock
engine/block_device.py` = 0):

```bash
set -uo pipefail                                    # before any pipeline exists; not after the first one
cd /home/erokhins/erokhins_projects/qwen3.6-35b-a3b || exit 1
V=/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python
day=$(date +%F); tag="m2-device-window-$day"; log="engine/work/blockdevice-device-$day.log"
ACQUIRED=0
die() { printf 'WINDOW-ABORT %s\n' "$*"
        [ "$ACQUIRED" = 1 ] && tools/gpu_lock.sh release "$tag"
        exit 1; }

# --- preconditions. Each one is a command that can refuse, and refusing stops the window. ---------
[ -x "$V" ]                                              || die "no venv python at $V"
[ -x engine/device/build.sh ]                            || die "engine/device/build.sh missing"
[ -e /tmp/pi_halogen.lock ]                              && die "a halogen worker holds the stream lock; it is a device activity, so this window cannot open beside it"
[ "${PI_PROVIDER:-}" != "halogen" ]                      || die "this shell is served by halogen; stopping the engine would decapitate the session doing the stopping"
gl=$(tools/gpu_lock.sh status)                           || die "gpu_lock.sh status unreadable"
printf 'gpu_lock status before acquire: %s\n' "$gl"
[ "$gl" = "free" ]                                       || die "gpu_lock is not free ($gl) — do not steal it; wait, or stop the holder"
tools/gpu_lock.sh acquire "$tag"                         || die "gpu_lock acquire refused"
ACQUIRED=1
gl=$(tools/gpu_lock.sh status)                           || die "gpu_lock status unreadable after acquire"
printf 'gpu_lock status after acquire: %s\n' "$gl"
case "$gl" in "HELD:"*"$tag"*) ;; *) die "acquire returned 0 but the lock is not held as $tag ($gl)" ;; esac
grep -qF "$tag" /tmp/pi_gpu.lock/holder                  || die "the lock file does not name our tag"
tools/gpu_lock.sh util
GPU_LOCK_TAG="$tag" tools/halogen_gate.sh down           || die "halogen_gate down refused"

# Engine-down, read off the device, not off a message. `rocm-smi --showmeminfo vram` is printed because
# it is the line a reader expects and because it proves the device answers; it CANNOT be the gate: on this
# APU `mem_info_vram_used` is the 2.0 GiB dedicated BAR and read 1.68 GiB while 21 GiB of weights were
# resident (`devices/measured-facts.md` §"what each instrument can see"), so a threshold on it could never
# fail. The counters that can see the engine are `rocm-smi --showpids` (which is what
# `tools/halogen_gate.sh::resident_bytes` reads) and `mem_info_gtt_used`.
rocm-smi --showmeminfo vram                              || die "cannot read vram"
rocm-smi --showpids > engine/work/showpids-$day.txt      || die "cannot read --showpids"
grep -q flash_serve engine/work/showpids-$day.txt        && die "flash_serve is still resident (see engine/work/showpids-$day.txt)"
gtt=$(cat /sys/class/drm/card*/device/mem_info_gtt_used | sort -n | tail -1)
[ "${gtt:-0}" -lt 24696061952 ]                          || die "gtt_used=${gtt} B is the recorded halogen-resident footprint (23.53-23.66 GiB, devices/measured-facts.md §Residency above the pool); the engine is up whatever the message said"

b=$(./engine/device/build.sh)                            || die "engine/device/build.sh failed"
printf '%s\n' "$b"
[ "$(printf '%s\n' "$b" | grep -c 'BUILD ok')" = 2 ]     || die "build did not print BUILD ok twice (both arms)"

# --- the gate itself: no pipeline, exit status checked before anything reads the log --------------
"$V" engine/block_device.py --device > "$log" 2>&1
rc=$?
printf 'driver exit=%d log=%s\n' "$rc" "$log"
[ "$rc" = 0 ] || die "block_device.py --device exited $rc — the refusal is in $log; a REFUSED tag is a finding, not a retry signal"

# --- the log has to have PRINTED the device claims; exit 0 alone is not a device run ---------------
missing=0
while read -r pat; do
  grep -q -- "$pat" "$log" || { printf 'MUST-BE-TRUE missing: %s\n' "$pat"; missing=1; }
done <<'PAT'
RESULT mode=device checks=32 findings=0 controls=10 fired=10
WPRESCOPE field=wpre pair=rt_hip_device_vs_rt_hip_hostpath
zero_tolerance_fields=ids,ids_alt,wcast
verdict=CONSISTENT-WITH-R5-EXP-SCOPE
scoped_field=wpre
verdict=CLAIMED-IDENTICAL
CONTROL name=topk-selection-perturbations-redden-ids fired=RED legs=device:2,host_cpubody:2 claims=host_cpubody,device
real-arm-mistake path=device
CONTROL name=wpre-scope-rejects-a-bigger-perturbation fired=RED
CHECK name=device-path-claims-executed-on-device verdict=PASS
PAT
# The required lines are exact strings, so a pattern the code no longer prints reddens the window instead
# of quietly becoming decorative. `checks=32` is 28 both-mode + 4 device-only; **if a check is added or
# removed, update this list in the same commit** — see §5's note below the block.
nstage=$(grep -c '^STAGERUN' "$log")
[ "$nstage" = 4 ]                                        || die "expected 4 STAGERUN rows (three shared spans + the router span), got $nstage"
nbind=$(grep -c '^BIND ' "$log")
[ "$nbind" = 10 ]                                        || die "expected 10 BIND lines carrying a read-back digest (the 5 honest spans plus the 5 the corrupt control re-binds, which is what window e2 printed; pass 3 counted only the honest five), got $nbind"
[ "$missing" = 0 ] || die "the run exited 0 without printing what a device run has to print; treat the window as not taken"
printf 'WINDOW ok log=%s no timing taken\n' "$log"
printf 'NOTE halogen is still DOWN. When this session has taken every timing it wants: tools/halogen_gate.sh up\n'
tools/gpu_lock.sh release "$tag"                         || die "release refused; release by hand with the tag $tag"
```

The check count is part of the required line on purpose: a check that silently stops being reached is the
`e2` failure class, and `checks=32` is 28 both-mode + 4 device-only (`rt-device-launch-counted`,
`rt-device-host-licensed-fields-identical`, `rt-device-host-wpre-within-derived-scope`,
`rt-device-host-difference-is-the-documented-exp-scope`). **If a check is added or removed, this block's
pattern is updated in the same commit** — a required line nobody updates is a required line nobody reads.

## 6. DECIDED (agent, 2026-10-26) — pass-4 drafts to transplant into `DECISIONS.md`

None of these is a user decision; none touches a width, a band, a floor, a route allowance, a pinned
fixture, or halogen. Each names the number or failure that forced it.

1. **`wpre`'s cross-compilation requirement is a derived scope of 9 f32 ULP, not bit-identity and not the
   measured 2 ULP.** DECIDED (agent), forced by window `e2`'s 237 differing bytes and by §R5's sentence
   "reproduces the pinned f32 pre-cast weights to single-digit ULP and NOT bit-exactly". Three independent
   priors were computed before the measurement was compared with them — triangle inequality over the
   licensor's `weights_vs_pinned_f32_ulp = 5` (10), §R5's own wording parsed at run time (9), and an error
   propagation from §`mr_exp`'s "an ULP or two" through R2 then R4 (22) — and the **strictest** binds, so
   no source can silently loosen the bound and the number is not this build's output. The licensor's own
   cross-arm allowance for this field is 20 ULP
   (`prototype/moe_router/gate.py::compare_arms`), so this is a tightening. This is a **scope on one
   unlicensed field**, not a band: `ids`, `ids_alt` and `wcast` stay at zero tolerance, `probs`/`logits`
   remain reported-and-not-absorbed, and the rejection control (`wpre-scope-rejects-a-bigger-perturbation`)
   is what makes it a bound. D-20 is the precedent (a re-derivation with a witnessed rejection), D-18 is
   why the claim could not simply be dropped.
2. **The first prior is only admitted while its premise is measured.** DECIDED (agent): the triangle
   inequality needs both routes inside the licensor's band of the pinned pre-cast weights, so
   `rt-wpre-premise-measured-against-the-pin` measures it every run (4 and 2 ULP against 5) and reddens if
   a route drifts past its own licence. Forced by the risk of a bound derived from two licensed legs
   quietly licensing a third number nobody licensed.
3. **A control's path claim is counted off the arm's report, and an unfulfilled claim reddens.** DECIDED
   (agent), forced by `topk-selection-perturbations-redden-ids` reading SILENT in `--device` while both
   its legs ran on the host twin. `arm_leg` tallies each act under the `RUN path=` the arm printed;
   `control-legs-cover-the-path-under-claim` and `device-path-claims-executed-on-device` are the two
   checks. The alternative — leaving the control host-only and saying so in prose — was rejected because
   the licensed rows in a device run are the device arm's, so a tie control that never touches the device
   licenses nothing above M2.
4. **The bound-rejection control runs in the CPU witness as well as in a window.** DECIDED (agent): a
   limit that can only be falsified on a device can only be falsified when a device is free, which is the
   scarcest resource on this box (one window per session, halogen down, lock held). Legs 1, 3, 4 and 6 need
   no device; leg 2 runs whichever paths the run has. Control count 9 → 10, checks 23 → 28 (CPU) / 32
   (device); `findings=0` and `fired == controls` remain the load-bearing invariants, not the counts.
5. **The device-window block is written so a refusal cannot be swallowed, and "engine down" is read off the
   device.** DECIDED (agent): `set -uo pipefail`, no gate inside a pipeline, every precondition a command
   that can abort, the driver's exit status checked before any reader sees the log, and the log audited
   against a required-line list (`exit 0` without the device claims is treated as a window not taken).
   `rocm-smi --showmeminfo vram` is printed but explicitly not the gate: `devices/measured-facts.md`
   §"what each instrument can see" records `mem_info_vram_used` at 1.68 GiB while 21 GiB was resident, so a
   threshold there could never fail; `rocm-smi --showpids` (no `flash_serve`) and `mem_info_gtt_used`
   (< the recorded 23.53–23.66 GiB resident footprint) are the instruments that can see it. The lock is
   released on every abort path and is never self-acquired by the driver.

## 7. Corrections to PASS 3's text (read with §5 above, not instead of it)

- Pass 3 §7's bullet describing `PAIR pair=rt_hip_device_vs_rt_hip_hostpath … claimed_fields=ids,ids_alt,
  wpre,wcast` and `verdict=CLAIMED-IDENTICAL` is **superseded**: `wpre` is no longer in the zero-tolerance
  set, and the line now prints `zero_tolerance_fields=ids,ids_alt,wcast` plus `scoped_field=wpre
  scoped_ndiff=… scoped_max_ulp=… scoped_limit_ulp=9`. The old `rt-device-host-claimed-fields-identical`
  check name no longer exists; §2's table replaces it.
- Pass 3 §7's "Expected verdict `QUALIFIED` with `checks=23 findings=0 controls=9 fired=9`" is superseded:
  CPU witness is `checks=28 findings=0 controls=10 fired=10`, device mode expects
  `checks=32 findings=0 controls=10 fired=10`, both still `QUALIFIED` with `unlicensed=3`.
- Pass 3 §6's "the nine controls, all `fired=RED`" is now ten.
- The `--device` command block in pass 2 §5 and pass 3 §7 is superseded by §5 above; do not paste the old
  one, it can report success on a refused lock, a refused gate, or an engine that is still resident.

## 8. Not proven by anything in pass 4 — do not cite these as done

1. **Every device-mode line in §2–§4 is unwitnessed until the next window.** The `wpre` numbers come from
   window `e2`'s preserved records replayed through the new code (no device act), and the CPU witness
   proves the derivation and the rejection, not the device run. What a window must newly show: the
   `WPRESCOPE`/`RTATTRIB`/split-`PAIR` lines, `real-arm-mistake path=device … OUTSIDE`, and
   `legs=device:2,host_cpubody:2` on the tie control. Two local smokes stand in for the parts of the device
   branch that a window would otherwise be the first to run: `engine/work/probe_pass4_local_smokes.py`
   counts every literal-%-format `print`/`check`/`control` argument against its specifiers (68 calls, 0
   mismatches, which is what makes the device-only `RTATTRIB`/`PAIR` lines safe to print), and then runs
   both rewritten controls **and both path-claim checks** with `device=True` and the arm acts stubbed to the
   preserved records: the tie control tallies `legs=device:2,host_cpubody:2`, the bound control
   `legs=device:1,host_cpubody:1`, and `device-path-claims-executed-on-device` passes on 3 device legs. In
   that second smoke the two controls report `fired=SILENT` **by construction** — every stub "variant" is
   the honest record again — so it is path coverage, not a result, and its lines are prefixed `DRY>` so they
   cannot be mistaken for evidence. The log is `engine/work/pass4-local-smokes.log`.
2. **The scope is one token row's worth of cross-compilation evidence.** 8 slots, one pinned row, one
   prompt token. It says the two compilations of R4 agree inside a derived bound on this row; it does not
   say they agree over a prompt, and `router_logits`'s licence remains the prototype's floor probe, not
   this scope.
3. **The propagation prior is a bound, not a model.** 22 ULP comes from worst-case spacing at each step
   (numerator, denominator, two division roundings); the measured 2 is well inside it and the binding prior
   is the documentary one. Nobody should read `prior=22` as an expected size.
4. **The corrected command block is unexecuted code.** Its non-device parts were exercised locally: the
   block is extracted from this file by `engine/work/extract_device_block.py` (so the text a reader copies
   and the text that gets checked are the same), it passes `bash -n`, and
   `engine/work/test_device_block_audit.sh` runs the required-line audit twice — against the CPU witness
   log, where it must and does report the device-shaped lines missing, and against a synthetic log carrying
   every required line, where it must and does stay quiet. What has NOT run: the halogen-down read, the two
   `rocm-smi` reads, the build step, and the driver invocation itself. Those belong to the main agent's
   window.
5. **The routed-expert gather and the GDN decode step are still not built** (unchanged from pass 3 §8).

## PASS 5 (2026-10-26, the routed-expert gather) — two width paths through one kernel; the census that prices prediction 2; the swap that had to redden

### 1. What was built

Five new files in `engine/device/`, two extended:

| file | what it is |
|---|---|
| `m2_g_kernels.h` | the shared item body. `m2g_item1..4` are the four dependent stages (gate|up matmul → SwiGLU → down matmul → weighted combine) over one (slot, row) item, folding into a caller-owned `struct M2GItem`. **No atomics, no barriers** — the device arm folds into `M2GCounters` with `atomicAdd` inside its own `__global__` bodies, and the host twin folds with plain `+=`. Same split `m2_se_kernels.h` established, so one body is compiled by g++ and hipcc without the shared code knowing which. |
| `m2_g_io.h` | the plan parser and the structural refuse-list. `m2g_read_plan` plus named refusals: `path-kinds`, `whole-tensor-unit-refused`, `mode-unknown`, `mode-unreachable-here`, `fused-span`, `fused-pair`, `unit-span-width`, `unit-geometry`, `expert-index`, `span-out-of-range`, `slots-not-geometry`, `elt-mismatch-for-path`. Also `m2g_census_report` (one `;`-separated, space-free token — a space would be read as a second `key=value` by the driver). |
| `m2_g_main.inc` | the `main()` both binaries share: reads the plan, runs, writes the record, prints `HOST`/`PLAN`/`RUN`/`WROTE`/`RESULT`/`CENSUS`, and the corrupt-byte fault. |
| `m2_g_host.cpp` | the g++ twin (CPU body only). Carries `struct M2GRunStats` and refuses `--device` by name. |
| `m2_g_arm.hip` | the hipcc arm: five `__global__` kernels (`k0_poison`, `k1_gateup`, `k2_silu`, `k3_down`, `k4_combine`), the plan uploaded as a struct pointer (a `M2GPlan` is ~2.7 KB and kernel params are 4 KB), H2D of slab + corrupt byte, D2H read-back, `UNITREADBACK` (device-only leg), and the `alloc-oversize` fault. |
| `engine/device_arm.py` | `parse_g_layout`, `g_ref_spans`, `g_ref_units`, `g_policy_units`, `g_slab`, `write_g_plan`, `g_record`, `parse_census`, `g_census`, `g_read_file_span`, `g_span_digest`. |
| `engine/block_device.py` | the vertical: `gather_prepare`, `run_gather_arms`, `emit_gather_rows`, five controls, `checks_g`, width-leg accounting (`G_LEGS`, `g_arm`, `g_legs_since`, `g_claims`), and two new witness checks. |

The selection is **not** handed to the gather from the pins: `gather_prepare` takes `ids`, `wcast`, `wpre` out of the router arm's own record, so the two compiled arms form a chain (`RTRECORD → gather`) and the pins stay outside both.

### 2. What the CPU witness printed (first execution of this vertical, `engine/work/bd-gather.log`)

Both width paths ran through both compilers, host path only, no device act:

```
PAIR pair=g_ref_bf16_host_twin_vs_hip_hostpath file=gather_record.f32 bytes=122880 ndiff_bytes=0 verdict=IDENTICAL
PAIR pair=g_policy_a_host_twin_vs_hip_hostpath file=gather_record.f32 bytes=122880 ndiff_bytes=0 verdict=IDENTICAL
STAGE arm=ref_bf16  stage=expert_raw  n=16384 ndiff=1     licence=floor-probe…  (licensed limit 5)
STAGE arm=ref_bf16  stage=routed_sum  n=2048  ndiff=0     licence=spec-10.6…    (exact)
STAGE arm=policy_a  stage=expert_raw  n=16384 ndiff=16139  verdict=UNLICENSED policy-a-width-band-absent
STAGE arm=policy_a  stage=routed_sum  n=2048  ndiff=2014   verdict=UNLICENSED policy-a-width-band-absent
```

The licensed half of the vertical is therefore already the expected shape: one expert output inside the
reassociation floor (`ndiff=1 ≤ 5`, `legitimate_reassociation_layer0` from `prototype/moe_gather`), the
weighted sum **exact** against the pin, and the shipping width refusing to be judged by a band that measures
bf16 against bf16 — UNLICENSED by name, not absorbed into the pass count.

Prediction 2's number, from the plan side of the census (`GWIDTH`/`GATHERCENSUS`; the arm side of these two
lines was wrong on this first run and §3 is the correction):

```
path=ref_bf16  distinct_experts=8 units=24 spans=16 bytes=50331648 per_expert=6291456B whole_tensor_reads=0
path=policy_a  distinct_experts=8 units=24 spans=3  bytes=15204352 per_expert=1900544B whole_tensor_reads=0
```

8 experts for K=8 slots, 6291456 B/expert bf16 — the 6.0 MiB AGENTS.md prediction 2 quotes — and
1900544 B/expert policy A (Q4_K gate 589824 + Q4_K up 589824 + Q5_K down 720896). Per-token gather read
50331648 B bf16 / 15204352 B policy A. `whole_tensor_reads=0` on both paths, and the bf16 stacked tensors
that a whole-tensor read would touch are the real 2 GiB/1 GiB figures read out of the safetensors headers,
never typed (`g_ref_spans`, `write_g_plan(..., whole_bytes=…)`).

Controls already RED on first run: `gather-gate-up-swap-reddens` (swapping the halves moved 16310/16384 raw
elements on the bf16 path, and the input-dim re-chunk leg was refused on the path with no fused span),
`gather-width-substitution-refused` (both paths refused the other width's kind with `path-kinds`),
`gather-poison-census-nonzero-and-attributed`, `gather-device-ask-refuses-instead-of-substituting`, and
`cpu-witness-had-no-gpu-context` PASS (`/sys/class/kfd/kfd/proc` never contained this pid; supervisor
condition 1 satisfied by the printed `GPUWITNESS` lines).

### 3. The four defects that run surfaced, and what closed them (`engine/work/bd-gather4.log`)

All four were in the checks and the census, not in the arithmetic: the two compiled binaries agreed
byte-for-byte before and after every fix, and `expert_raw`/`routed_sum` came out with the same ndiff they had
in §2. The witness now runs to completion: `RESULT mode=cpu checks=40 findings=0 controls=17 fired=17`,
zero `fired=SILENT`, `band_digest=27a68c7e82c9aa05` unchanged.

1. **The arm's census under-counted, and the claim it was used to prove was therefore vacuous.**
   `m2g_run` tallied every unit of the row into one item whose note array held 4, and `m2g_item_note`
   `return`ed silently past the capacity — so `unit_reads=4`/`total=8 MiB` for a plan that reads 50 MiB, and
   `whole_reads=0`, the line "the gather reads slices, not whole tensors", was being read off a counter that
   had dropped 20 of 24 units. A counter that counts nothing reports no violations; that is the whole
   objection, and the fix has three parts: (a) the note belongs to the unit — `m2g_unit_gateup`/
   `m2g_unit_down` run one unit's rows with an item of their own, which is also the device's note shape, so
   neither path can tally two units into one item; (b) a second, different note into one item now increments
   `note_conflict`, printed in the census as `note_conflict=0` and required to stay zero, instead of being
   dropped; (c) the census leg is witnessed non-zero — see 5 below. Both paths now report
   `arm_units=24 arm_spans=16`/`24 arm_total=50331648`/`15204352 disagree=none note_conflict=0`.
2. **`policy_a` reported `spans=3` instead of 24.** `g_policy_units` set `span_off`/`span_bytes` once per
   family; they are now the expert's own range from `Loader.expert_range` (`file_off`, `length`), so the
   distinct-span count — the quantity a scheduler merges spans on — is 16 on the reference layout (fused
   pair = one span) and 24 on policy A (three tensors), and the driver and the arm now agree on it.
3. **`KeyError: 'run'` inside `cpu-witness-reports-no-device-acts`.** `ctx["all_runs"]` merged three dicts
   with two different record shapes. Fixed by normalising (`run_facts`) rather than by defaulting the
   predicate away, because a check that raises has not run — the same class as an empty parse printing PASS.
   A record with no RUN block is now a named refusal, and a counter the arm did not print counts as an
   offence rather than a zero. The predicate is witnessed by `control-witness-guard-can-redden` over three
   fabrications (a launch, a 1 GiB H2D, and a RUN line omitting `launches`), all three of which it reads:
   `GWITNESS guard=no-device-acts fabricated=launches=1 offences=['g/g_policy_a_hip_hostpath.launches=1']`.
4. **`gather-combine-choices-are-load-bearing` read SILENT while its own legs fired.** Its second clause was
   `nd >= honest_nd`, applied to both paths: against an honest `routed_sum` ndiff of 0 that `>=` is satisfied
   by anything (vacuous on the licensed half), and against policy A's width-sized honest distance to a bf16
   pin it demanded something the path cannot promise — `policy_a.rank_accum` moves 1232 elements with pin
   distance exactly `2014>2014`. The clause is now two rulers: every leg must move the field its mode claims
   (`field-moved>0`), and only on the licensed path must it land *strictly* further from the pin.
5. **The census's whole-tensor counter is now seen counting.** `fault=whole-note` notes a whole-tensor read
   per unit rather than a flat +1, so the injection reads `whole_reads=24` against `unit_reads=24`, and the
   same driver-vs-arm comparison that passes on the honest plan reddens on it:
   `GWHOLE path=ref_bf16 arm_whole_reads=24 arm_unit_reads=24 honest_whole_reads=0 comparison_reddened=yes
   first_findings=whole=24/0` (same line for `policy_a`). The arm still refuses a whole-tensor *plan* by name
   in both paths, and the cost such a plan claims is computed from metadata, not typed: 21474836480 B/token
   on the reference path and 3892314112 B/token on policy A, against the 50331648 B and 15204352 B actually
   read. A zero denominator is its own named finding (`census-denominator-zero`) rather than a pass.

Also repaired on the way, because the clean witness exposed it: `gather-control-legs-cover-the-width-under-claim`
read a `widths` key no control ever set, so it could only ever print `none` and FAIL. It now asserts what its
own detail claims — every gather control that declares width paths declares both of them with a counted leg
under each, and controls whose only leg is a named refusal are listed instead of counted as legs.

Still not proven by any of this: any device act. `UNITREADBACK`, the `alloc-oversize` fault leg and every
device-path leg remain VOID here, and `--device` is still window-gated.

### 4. Next session's commands, in order

```sh
# 1. commit by topic: the gather arm sources, then the driver vertical (nothing was committed or staged
#    by the passes that wrote §1-§3; the working tree is the only copy of this work)
 engine/device/build.sh                      # both gather binaries rebuild clean as of this pass
 engine/device/build/m2_g_host --help        # sanity: the twin exists and refuses --device
# 2. re-run the CPU witness after any edit to the gather sources; the line to watch is
#    RESULT mode=cpu … findings=0 controls=17 fired=17 … band_digest=27a68c7e82c9aa05
 /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python engine/block_device.py \
     --work engine/work/m2dev 2>&1 | tee engine/work/bd-gather4.log
# 3. only then the device window, main agent, zero workers, halogen down, gpu_lock held
 … engine/block_device.py --work engine/work/m2dev-win --device
```

The witness is CPU-only work and stays allowed without a window: hipcc **compilation** and host-path
execution create no GPU context (supervisor's ruling, recorded as an agent decision in `DECISIONS.md`).
`--device` remains window-gated.

### 5. Agent decisions made in this pass (all `DECIDED (agent, 2026-10-26)`)

* **No atomics or barriers in shared code.** The gather's four stages are separate launches and each folds
  into a caller-owned struct; the device keeps `atomicAdd`, the host keeps `+=`. A barrier version made the
  shared header un-compilable by g++ and would have made the twin a different program.
* **The plan is uploaded as a pointer, not by value.** `sizeof(M2GPlan)` ≈ 2.7 KB against a 4 KB kernel-param
  ceiling; a by-value plan would have broken the first time the unit list grew.
* **The census token is `;`-separated and space-free**, because `run_arm` parses whitespace-separated
  `key=value` facts and a space inside a value would silently become a second fact.
* **`g_ref_spans` reads safetensors headers only** (`data_offsets`, dtype checked `BF16`). Loading the tensor
  to find its offset would make the census's whole-tensor counter lie.
* **Policy A's activation basis is the bf16 image widened to f32** (`g_widen_f32`), a representation change,
  not an arithmetic one, so the two width paths differ in storage width and nothing else. Importing
  llama.cpp's activation quantization (`vec_dot_type`) would trade our 1-ULP question for a 2⁻⁸ input
  perturbation and is D-17 territory — not imported.
* **Whole-tensor sizes come from metadata** (loader rows / shard headers) in every refusal and every line.

### 6. Still out of scope for this pass

GDN decode step, M2 device arm's remaining blocks, M3 and beyond, any timing. The gather is a gather: it
proves which bytes are read and what they produce, and says nothing about how fast.
*(The GDN decode step is delivered by PASS 6 below; the rest of this line still stands.)*

---

## PASS 6 (2026-10-27, the GDN recurrent decode step) — the fourth compiled pair: one form, one state-contraction point, and a licence that prints `chunked|pinned=refused`

PASS 5 §6 named the GDN decode step as out of scope; this pass is that item. Committed as `4b733de`.
Everything here was executed on the **CPU witness only** — hipcc *compilation* plus host-path execution,
no device act, no `gpu_lock`, no timer read. The orchestrator re-ran the witness independently and
reproduced it exactly.

### 1. What was built

| file | what it is |
|---|---|
| `engine/device/m2_d_kernels.h` (550 L) | the shared item body for one cached recurrent decode step (§04b.1 + §04.7 + §04.8). Seven item kinds, `m2d_item_enter/carry/kvmem/write/out/norm/outproj`, plus `m2d_run` (the serial order the **g++ twin** drives) and `m2d_state_dot` (§2 below). No atomics, no barriers — the device folds tallies with `atomicAdd` in its own `__global__` bodies, the host with plain `+=`, same split `m2_se_kernels.h` and `m2_g_kernels.h` established. |
| `engine/device/m2_d_io.h` (356 L) | plan parser and structural refuse-list: `form-not-implemented`, `field-unknown`, `field-missing`, `field-elt`, `field-words`, `field-bytes`, `span-past-slab`, `span-unaligned`, `spans-overlap`, `mode-unknown`, `fault-unknown`, `readback-range`; plus `m2d_census_report` (one `;`-separated space-free token, because `run_arm` parses whitespace-separated `key=value`). |
| `engine/device/m2_d_main.inc` (206 L) | the `main()` both binaries share: plan read, plan/slab size agreement (`slab-size`), run, record write, `HDR`/`PLAN`/`RUN`/`WROTE`/`RESULT` + `DSTATEFINGERPRINT`. It is where `--device` is answered **or refused**: the g++ twin prints `FAIL tag=device-path-absent-in-this-binary` and exits 1 — it never answers `--device` with a host run — and `fault=alloc-oversize` on a binary with no device path prints `fault-device-only` rather than faking the leg. A device failure exits 1 with `verdict=FAIL`; there is no fallback path to `m2d_host_run`. |
| `engine/device/m2_d_host.cpp` (37 L) | the g++ twin: allocates the scratch, executes `m2d_run` — the engine's own host-side GDN body, the same code the hipcc arm's host path runs. |
| `engine/device/m2_d_arm.hip` (267 L) | the hipcc arm, the only binary that can launch. `WITH --device the acts are: four hipMalloc (slab, record, scratch, counters), one H2D of the slab, one hipMemset (the counters only), eight launches (poison, then enter/carry/kvmem/write/out/norm/outproj), two or three D2H, one synchronize, four hipFree` — every one counted in `M2DRunStats`, because the counter and not the flag is what lets the driver print anything device-shaped. **It never calls `m2d_run`.** One launch per stage, not per head: §04b.1's five steps are sequentially dependent, so the stage boundary is a real barrier and the launch is how a device gets one; the largest grid is 4096 threads. |
| `engine/device/build.sh` | the fourth pair, `BUILD ok gdn_host_twin=` / `gdn_hip_arm=`, `-I $ROOT/prototype/gdn` added, and a refusal if that header is absent (the arm takes `gdn_sumsq`, the bf16 bit-image helpers and `GDN_H/GDN_H_K/GDN_H_V/GDN_D_K/GDN_D_V` from that lane **included, never copied**). Still `-ffp-contract=off -fno-fast-math` on both compilers. |
| `engine/device_arm.py` (+407) | `parse_d_layout` (field names/offsets/widths read out of the C header, not typed), `d_spans` (nine pinned spans, each manifest `dtype` checked against the element width the header claims), `d_slab` (with `zero_state`/`flip_state_word`), `write_d_plan`, `d_record`, `d_stage_tensors`, `parse_d_census`, `d_fingerprint`, `d_census_agreement`. |
| `engine/block_device.py` (+969) | the vertical: `d_prepare`, `run_d_arms`, `emit_d_rows`, `d_variant`, `d_metrics`, `d_row_verdicts`, `d_fields_moved`, `d_bits`, `d_torch_body`, `d_claims`, `derived_d_population`, 11 checks, 8 controls, `--d-arm-binary` / `--d-host-binary`. |

**The seven item kinds and their counts.** The driver derives this table from the header's own constants
and requires the arm's `census=` token to match it kind by kind (`DCENSUS … disagree=none`);
`items_total=18496/18496` is the printed sum, not a typed constant:

| kind | items | geometry | spec | one item does |
|---|---|---|---|---|
| `enter` | 32 | `h_v` | §04.4 | the two l2norms and the recurrent arm's q scale (`q / d_k**0.5`; `speccheck/gdn_ref.py::Form.named("recurrent")` is `q_scale="div"`, not the chunked arm's multiply); V head *h* reads K head `h/expansion` |
| `carry` | 4096 | `h_v·d_k` | §04b.1 step 1 | `S = S_old · exp(decay_head)` — the only stage that reads every word of the incoming state, so this is where the state fingerprint is taken |
| `kvmem` | 4096 | `h_v·d_v` | §04b.1 step 2 | the `k·S` read that forms the delta |
| `write` | 4096 | `h_v·d_k` | §04b.1 step 4 | the rank-one write `S += k⊗delta` |
| `out` | 4096 | `h_v·d_v` | §04b.1 step 5 | the output read `q_hat·S` → `scan_out` |
| `norm` | 32 | `h_v` | §04.7 | the gated norm, both bf16 roundings |
| `outproj` | 2048 | `d` | §04.8 | one `out_proj` row of the decode step |
| **total** | **18496** | | | |

**Record geometry.** one f32 array of `M2D_FLOATS = 546816` words = **2,187,264 B**; a bf16-carrying
field is stored *widened* (`M2D_FIELD_IS_BF16 = 0,0,0,1,0,1,1`), so the comparator compares 4-byte
patterns and never guesses a dtype off a file:

| field | offset (words) | length | referent dtype |
|---|---|---|---|
| `q_hat` | 0 | 4096 | f32 |
| `k_hat` | 4096 | 4096 | f32 |
| `scan_out` | 8192 | 4096 | f32 |
| `core_attn` | 12288 | 4096 | bf16 (widened) |
| `state` | 16384 | 524288 | f32 |
| `normed` | 540672 | 4096 | bf16 (widened) |
| `y` | 544768 | 2048 | bf16 (widened) |

Inputs are nine spans from the pinned decode step (`layer0.decode0.q_pre/k_pre/v_pre/beta/g_decay/z`,
`layer0.weights.norm/out`, and the carried `layer0.prefill.state`), `slab_bytes=18899392`,
`state_words=524288 state_bytes=2097152`, host scratch `3·h_v·d_v + h_v = 12320` f32.

### 2. The two decisions that shape the arithmetic

**One contraction point for the state.** `d_k == d_v == 128`, so a state stored `[d_v,d_k]` instead of
`[d_k,d_v]` is the same 16,384 words per head in a different order: in bounds, shape-legal, and a
different number at every live word. The oracle's own injection name for it is `state_stored_transposed`,
and the prototype lane documents why a zero state made it invisible to its first gate. So the whole
contraction lives in exactly one function, and the mode word is spelled inside it:

```c
GDNDAY_DEV static inline float m2d_state_dot(const float *a, const float *S, long sbase, int col,
                                             int mode) {
    float acc = 0.0f;
    int k;
    if (mode == M2D_MODE_STATE_XPOSE) {          /* injected: S read as [d_v, d_k] */
        for (k = 0; k < GDN_D_K; ++k) acc += a[k] * S[sbase + (long)col * GDN_D_K + k];
        return acc;
    }
    for (k = 0; k < GDN_D_K; ++k) acc += a[k] * S[sbase + (long)k * GDN_D_V + col];
    return acc;
}
```

Single, because a transposition must have exactly one answer and exactly one place for a control to
bend. Its two callers are `m2d_item_kvmem` (the `k·S` that forms the delta) and `m2d_item_out` (the `q_hat·S`
that becomes `scan_out`), and both are `GDNDAY_DEV` inlines reached by the serial host body and by the
device stage-kernels alike, so one control leg per path covers the launched arithmetic and the host body
with the same injection. The decode step's incoming state is **not** zero, which is what makes every state
witness non-vacuous:
`DSTATE role=carried record=layer0.prefill.state words=524288 live=509952 sum=2525565901
sumidx=3508630800 nonzero_fraction=0.972656`.

**The arm never calls `m2d_run`.** `grep -n m2d_run engine/device/m2_d_arm.hip` matches only comments.
The g++ twin reaches the items serially, the hipcc arm reaches the same items with seven stage launches,
and `PAIR pair=d_host_twin_vs_d_hip_hostpath … ndiff_bytes=0` is a codegen statement at zero tolerance,
not a band. That is two implementations; a third is printed for free (§3), so no row rests on one arm
being compared to itself.

### 3. The form discipline — `chunked` is not this arm, and the log says so per row

`speccheck/gdn_ref.py::band_licenses` keys on `width|form`, and at the pinned decode pass
(`layer0/pinned/decode0`) the only pair that grants anything is `recurrent|pinned` — `DECISIONS.md` D-21
records why: `probe_arms_for("decode","pinned")` yields `recurrent,reassoc,pinned`, so no chunked probe
record exists at a decode pass to license it. Every row is therefore meaningless without the form that
produced it, and the driver asks the licensor for **both** pairs per stage and prints the answer —
verbatim, all seven rows:

```
DLICENCE stage=q_hat arm_form=recurrent recurrent|pinned=granted chunked|pinned=refused chunked_reason=no probe record for one of the two arms at this pass
DLICENCE stage=k_hat arm_form=recurrent recurrent|pinned=granted chunked|pinned=refused chunked_reason=no probe record for one of the two arms at this pass
DLICENCE stage=scan_out arm_form=recurrent recurrent|pinned=granted chunked|pinned=refused chunked_reason=no probe record for one of the two arms at this pass
DLICENCE stage=core_attn arm_form=recurrent recurrent|pinned=refused chunked|pinned=refused chunked_reason=no probe record for one of the two arms at this pass
DLICENCE stage=state arm_form=recurrent recurrent|pinned=granted chunked|pinned=refused chunked_reason=no probe record for one of the two arms at this pass
DLICENCE stage=normed arm_form=recurrent recurrent|pinned=refused chunked|pinned=refused chunked_reason=no probe record for one of the two arms at this pass
DLICENCE stage=y arm_form=recurrent recurrent|pinned=refused chunked|pinned=refused chunked_reason=no probe record for one of the two arms at this pass
```

`DLICENCE … chunked|pinned=refused` is the only place the word `chunked` appears in the log; the arm
also refuses to be *labelled* into a licence it did not earn — a plan declaring `form=chunked` is refused
by both binaries with `form-not-implemented` before a byte is read. Per-row licence source, exactly as
printed (never a bare number):

| row | licence as printed | verdict as printed |
|---|---|---|
| `q_hat` | `licence=gdn-band(band_licenses\|recurrent\|pinned@layer0/pinned/decode0)` | `band=ACCEPTED`, `DETAIL … why=all four derived keys inside the probe-derived band` |
| `k_hat` | same | same |
| `scan_out` | same | same |
| `state` | same | same |
| `core_attn` | `licence=none` | `band=UNLICENSED stage=core_attn reason=arm-pair-not-licensed-at-pass`, `DETAIL … why=no probe record for one of the two arms at this pass` |
| `normed` | `licence=none` | same shape |
| `y` | `licence=none` | same shape |

`DROWS form=recurrent rows=7 licensed=4 accepted=4 rejected=0 unlicensed=3 licensed_stages=q_hat,k_hat,
scan_out,state unlicensed_stages=core_attn,normed,y`. The four grants are judged by
`band_check_row` on `BAND_FACTORS` (`one_minus_cos`, `mean_rel`, `max_abs`, `per_head_max`) of
`gdn-band-2026-10-26.json`, `band_digest=27a68c7e82c9aa0507b9bcd677adc0d5d1b0bc865dd1e86bf495ed90f47392ee`
— unchanged by this pass, because no band, floor, limit or tolerance was authored here.

### 4. What the CPU witness printed, and what "three implementations" means

`engine/work/m2d-full.log` (reproduced independently by the orchestrator; two runs differ only in pid and
work-dir path):

```
RESULT mode=cpu checks=51 findings=0 controls=25 fired=25 rows=6 gdn_rows=7 arms=policy_a device=off staged=not-attempted hip_calls=0+0 launches=0 bytes_to_device=0 bytes_from_device=0 band_digest=27a68c7e82c9aa05 no timing taken
PAIR pair=d_host_twin_vs_d_hip_hostpath file=gdn_record.f32 bytes=2187264 ndiff_bytes=0 cases=1 zero_tolerance=1 verdict=IDENTICAL
DCENSUS arm=d_hip_hostpath path=host_cpubody form=recurrent items_total=18496/18496 reads_state=524288/524288 writes_state=524288/524288 state_nonzero=509952 state_sum=2525565901 state_sumidx=3508630800 out_elems=4096/4096 outproj_reads=8388608/8388608 disagree=none
DCENSUS arm=d_host_twin path=host_cpubody form=recurrent items_total=18496/18496 reads_state=524288/524288 writes_state=524288/524288 state_nonzero=509952 state_sum=2525565901 state_sumidx=3508630800 out_elems=4096/4096 outproj_reads=8388608/8388608 disagree=none
DSTATE role=carried record=layer0.prefill.state words=524288 live=509952 sum=2525565901 sumidx=3508630800 nonzero_fraction=0.972656 note=a zero state would make every state-reading control below vacuous, so the count is printed before any row is emitted
```

The `DCENSUS` line is the reconciliation, not a decoration: each of `enter/carry/kvmem/write/out/norm/
outproj`, each of `reads_q/reads_k/reads_v/out_elems/outproj_reads`, and both state counters are compared
against the geometry, and the arm's modular fold of the state's bytes is compared against the driver's
fold of the bytes it wrote into the slab. `disagree=none` means both agreed, on both compilers.

Three implementations, none compared only to itself: the **crediting** comparison is the arm against the
pinned decode records; the **pair** is g++ vs hipcc at zero tolerance; and a **third opinion** is printed
with its own distance to the pins, so the first two cannot be two wrong programs agreeing:

```
XIMPL pair=d_arm_vs_engine_torch_body body=block_forward.gdn_recurrent_step stages=7 ndiff_elements=q_hat=1178/4096,k_hat=820/4096,scan_out=3511/4096,core_attn=0/4096,state=85187/524288,normed=0/4096,y=1/2048 verdict=MEASURED note=this comparison is a THIRD opinion, not a licensor: the rows below are judged against the pinned records by block_gate.adjudicate, and a torch-vs-C difference here is reported and investigated, never absorbed into a band
XIMPL pair=engine_torch_body_vs_pinned body=block_forward.gdn_recurrent_step stages=19 ndiff_vs_pin=a_raw=0/32,b_raw=0/32,beta=0/32,conv_out=0/8192,conv_pre=0/8192,conv_window=0/32768,core_attn=0/4096,g_decay=0/32,k_hat=0/4096,k_pre=0/2048,mixed_qkv=0/8192,normed=0/4096,q_hat=0/4096,q_pre=0/2048,scan_out=0/4096,state=0/524288,v_pre=0/4096,y=0/2048,z=0/4096 verdict=EXACT note=the torch body's own distance to the pins, so the line above cannot be read as two wrong programs agreeing with each other
```

Read that honestly: the engine's torch body is **bit-exact** against all 19 pinned stages, so the C arm's
deviations are its own (declared-order differences: one ascending f32 accumulator where torch reduces its
own way), and the licensed stages are inside the band on the band's four keys. `scan_out` shows
`max_ulp=5912` and `state` `max_ulp=8192` while `meanrel` is 1.840e-07 and 8.433e-09 — that is what a
relative-error band measures, and it is the licensor's rule, unchanged. Nobody should cite those four
ACCEPTED rows as ULP-level exactness.

### 5. The eight controls, what each injects, and what reddens

All eight `fired=RED`; zero `fired=SILENT`; `checks=51 findings=0`. A control that cannot redden its own
predicate is a finding here, which is why each row below names the injection and the thing that moves.

| control | injects | what reddens |
|---|---|---|
| `gdn-state-transposition-reddens` | `mode=state_xpose` through `m2d_state_dot`, on both compiled arms | `q_hat`/`k_hat` must stay **bit-identical** (they are computed before the state is touched, so a control moving them is moving something else), all five state-reading fields must move, none may land closer to the pin, the reach set the control measures must name exactly those five (`reach=scan_out,core_attn,state,normed,y`), and the transposed record must come back **REJECTED from the licensor** on at least one licensed stage: `band_reddened_on=scan_out,state`, with both sides printed — `honest_ndiff=scan_out=3511/4096,core_attn=0/4096,state=85187/524288,normed=0/4096,y=1/2048` against `dirty_ndiff=scan_out=4096/4096,core_attn=4094/4096,state=417406/524288,normed=4094/4096,y=2045/2048`. Reddening the *licence*, not a distance this file chose. |
| `gdn-carry-index-reddens` | `mode=decay_head_off1` — decay taken from head `(h+1) % 32` | the state-byte fingerprint must be **exactly unchanged** from the honest run (`2525565901`/`3508630800`, both equal to the driver's own fold) while all five derived fields move. That equality is the clause that makes this about the carry and not the read: had the mode also changed which words it touched, `state_sum` would move and the control would refuse. This is the class that used to pass silently. |
| `gdn-non-carried-state-reddens` | two legs per compiler: driver zeroes the slab's state span; arm runs `fault=zero-state` | zeroed-bytes leg: fingerprint must become **exactly the all-zero fold the driver computes over those bytes** (`arm_sum=0 driver_sum=0`, `state_ndiff 85187->417408`), with `q_hat`/`k_hat` bit-identical. arm-fault leg: `reads_state=0` and a zero fingerprint, which `d_census_agreement(zero_in=True)` **requires** rather than permits — a fault that read anyway is a red control, not an invisible cheap step. |
| `gdn-state-read-is-the-arm-s-own-fingerprint` | one byte flipped (`XOR 0xFF`) in state word 77,777 of a copy of the slab, on both arms | honest legs: arm fold == driver fold on every arm that ran, with `live=509952` as a non-zero denominator. Flip legs: fingerprint moves (`2525565901→2525566144`), the two sides stay equal, the `state` row moves (`state_ndiff 85187->85188`) and nothing gets closer. This is the leg that shows the fingerprint is a fingerprint of *these bytes* and not a constant the arm prints. |
| `gdn-form-not-merged` | a plan declaring `form=chunked`; both pairs asked of the licensor per stage | every arm run's parsed `form` must be exactly `recurrent`; `chunked\|pinned` must grant 0 of 7 (if a future band file ever grants it here, the control says so aloud instead of keeping the claim decorative); both binaries must refuse the chunked plan by name (`form-not-implemented`); every ACCEPTED row's licence must contain `recurrent\|pinned`; and no stage the licensor granted may be missing from the ACCEPTED set. |
| `gdn-poison-census-can-be-nonzero` | `fault=skip-carry` on the twin, `fault=skip-out` on the hipcc arm | honest record carries no poison in any of the seven fields; forbidding one stage makes **its own** field read entirely poison — `skip-carry field=state poison=524288/524288`, `skip-out field=scan_out poison=4096/4096` — with `q_hat`/`k_hat` clean. Poison is the quiet NaN pattern, so a `hipMemset` to zero would have hidden the same bug as a field of exact zeros. |
| `gdn-plan-refusals-fire` | five plans per binary, each with exactly one thing wrong | all ten refusals must arrive by name: `field-elt`, `field-words`, `field-missing`, `mode-unknown` (a chunked-lane mode word), `fault-unknown` (a gather-lane fault word). A plan this arm must not run is refused before a byte is read, which is what makes the honest run's spans worth something. |
| `gdn-device-ask-refuses-instead-of-substituting` | `--device` and `fault=alloc-oversize` against the g++ twin | both must refuse by name (`device-path-absent-in-this-binary`, `fault-device-only`). No host body is offered for a device claim, so a `--device` report from this arm can only come from the binary that launches. |

Eleven checks back them (`gdn-rows-equal-licensor-stage-list`, `gdn-kernel-geometry-matches-the-config`, `gdn-arm-implements-only-the-form-it-claims`, `gdn-census-equals-geometry-and-driver`, `gdn-honest-record-has-no-unwritten-slot`, `gdn-compiled-pair-byte-identical`, `gdn-state-span-binds-the-arm-slab`, `gdn-cross-implementation-comparison-was-made`, `gdn-licensed-rows-name-the-arm-form-and-no-others`, `gdn-unlicensed-counted-not-absorbed`, `gdn-device-path-claims-executed-on-device`), all PASS. The stage list is derived twice and required to be the same seven (`DPOPULATION licensor=… arm_record=… missing=none extra=0`), the kernel's `GDN_*` constants are required to equal the checkpoint config, and `PASS` stays unreachable while any row is unlicensed (the verdict is `QUALIFIED`).

### 6. The legs that did NOT run here — listed individually for a window to pick up

`DVOID` is printed instead of assuming. Three legs, verbatim, and where each one is picked up:

1. `DVOID control-leg=gdn-state-transposition-on-device status=not-exercised reason=needs-a-device-window note=the same mode is compiled into the hipcc binary's device kernels but no leg executed it in this window; it is reported as unexercised rather than assumed from the host body`
   → in a `--device` window `control_d_state_transposition` adds a third leg (`binary=m2_d_arm, device=True`), the control's claim set becomes `claims=host_cpubody,device`, and it prints `DCONTROL control=transposition leg=gdn-device ran=yes`.
2. `DVOID control-leg=gdn-device-readback status=not-exercised reason=needs-a-device-window note=no host body printed a read-back line, and none was allowed to: the host path binds file bytes to slab bytes by digest instead`
   → in a window the arm prints `DREADBACK path=device off=16802240 bytes=32 hex=<64 hex>` — the first 32 bytes of the state span read back off the **device slab**, i.e. the H2D landing area — and the driver **compares** those bytes to the bytes the dump reader certified: `DREADBACK control-leg=gdn-device-readback ran=yes state_off=16802240 bytes=32 file_sha256=<first 16 hex of the sha256 of the pinned state record this span was read from> per_run=d_hip_device[off=16802240 bytes=32 match=yes]`, and it requires `off`/`bytes` to equal the plan's read range as well as the hex to match. `gdn-state-read-is-the-arm-s-own-fingerprint` requires that comparison in a device run.
3. `DVOID control-leg=gdn-alloc-oversize-actually-fails reason=needs-a-device-window note=the host twin refuses to run this leg at all, which is the leg above; the hipcc arm's 1 TiB hipMalloc refusing is witnessed only in a window`
   → **no driver code path runs it on the device today**, so a window must run it by hand; the command is in §7 and it is part of the window's assertions. (Note for accuracy: this line prints `reason=` but no `status=not-exercised` field, unlike 1 and 2 — see §9.)

### 7. Exact next-window device command (main agent only; zero workers stopped, halogen down, lock held, foreground, blocking)

PASS 4 §5's shape is kept and its stale assertion fixed: `engine/device/build.sh` now prints **eight**
`BUILD ok` lines (four pairs), not the two PASS 4 asserted. No gate command sits inside a pipeline, the
lock is matched against the expected tag, and engine-down is read off the device.

```bash
set -uo pipefail                                    # before any pipeline exists; not after the first one
cd /home/erokhins/erokhins_projects/qwen3.6-35b-a3b || exit 1
V=/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python
day=$(date +%F); tag="m2-gdn-device-window-$day"
wd="engine/work/m2dev-gdn-$day"; log="engine/work/blockdevice-device-gdn-$day.log"
D_ARM=engine/device/build/m2_d_arm; D_HOST=engine/device/build/m2_d_host
ACQUIRED=0
die() { printf 'WINDOW-ABORT %s\n' "$*"
        [ "$ACQUIRED" = 1 ] && tools/gpu_lock.sh release "$tag"
        exit 1; }

# --- preconditions. Each one is a command that can refuse, and refusing stops the window. ---------
[ -x "$V" ]                            || die "no venv python at $V"
[ -x "$D_ARM" ]                        || die "$D_ARM missing; run engine/device/build.sh first"
[ -x "$D_HOST" ]                       || die "$D_HOST missing; the twin is the other half of the pair"
[ -e /tmp/pi_halogen.lock ]            && die "a halogen worker holds the stream lock; it is a device activity, this window cannot open beside it"
[ "${PI_PROVIDER:-}" != "halogen" ]    || die "this shell is served by halogen; stopping the engine decapitates the session doing the stopping"
gl=$(tools/gpu_lock.sh status)         || die "gpu_lock.sh status unreadable"
printf 'gpu_lock before acquire: %s\n' "$gl"
[ "$gl" = "free" ]                     || die "gpu_lock is not free ($gl); do not steal it"
tools/gpu_lock.sh acquire "$tag"       || die "gpu_lock acquire refused"
ACQUIRED=1
gl=$(tools/gpu_lock.sh status)         || die "gpu_lock status unreadable after acquire"
case "$gl" in "HELD:"*"$tag"*) ;; *) die "acquire returned 0 but the lock is not held as $tag ($gl)" ;; esac
grep -qF "$tag" /tmp/pi_gpu.lock/holder || die "the lock file does not name our tag"
GPU_LOCK_TAG="$tag" tools/halogen_gate.sh down || die "halogen_gate down refused"

# engine-down read off the device, not off a message (PASS 4 §5: vram_used is the 2.0 GiB BAR and
# cannot see the engine; --showpids and mem_info_gtt_used can).
rocm-smi --showmeminfo vram                              || die "cannot read vram"
rocm-smi --showpids > "engine/work/showpids-$day.txt"    || die "cannot read --showpids"
grep -q flash_serve "engine/work/showpids-$day.txt"      && die "flash_serve still resident; see that file"
gtt=$(cat /sys/class/drm/card*/device/mem_info_gtt_used | sort -n | tail -1)
[ "${gtt:-0}" -lt 24696061952 ] || die "gtt_used=${gtt} B is the recorded halogen-resident footprint (23.53-23.66 GiB, devices/measured-facts.md §Residency above the pool); the engine is up whatever the message said"

b=$(./engine/device/build.sh)          || die "engine/device/build.sh failed"
[ "$(printf '%s\n' "$b" | grep -c 'BUILD ok')" = 8 ] || die "build did not print BUILD ok 8 times (four pairs, two binaries each)"

mkdir -p "$wd" || die "cannot create $wd"
"$V" engine/block_device.py --device --work "$wd" \
    --d-arm-binary "$D_ARM" --d-host-binary "$D_HOST" > "$log" 2>&1
rc=$?
printf 'driver exit=%d log=%s\n' "$rc" "$log"
[ "$rc" = 0 ] || die "block_device.py --device exited $rc; the refusal is in $log — a REFUSED tag is a finding, not a retry signal"

# --- the log has to have PRINTED the device claims; exit 0 alone is not a device run ---------------
missing=0
while read -r pat; do
  grep -q -- "$pat" "$log" || { printf 'MUST-BE-TRUE missing: %s\n' "$pat"; missing=1; }
done <<'PAT'
# The three lines below were rewritten by PASS 7 (2026-10-27) because the build moved under them: the
# RESULT counts changed (+2 checks, +3 controls), and the GDN device pair DIVERGED from its host path
# (52,976 B of 2,187,264), so a pattern demanding `ndiff_bytes=0 ... verdict=IDENTICAL` would have reddened
# the window for the wrong reason. Counts are matched as PREFIXES now, and findings are named in PASS 7
# rather than counted, because a finding count is exactly the number that changes when a band changes.
RESULT mode=device checks=57
controls=29
band_digest=27a68c7e82c9aa05
DCENSUS arm=d_hip_device path=device form=recurrent items_total=18496/18496 reads_state=524288/524288 writes_state=524288/524288 state_nonzero=509952 state_sum=2525565901 state_sumidx=3508630800 out_elems=4096/4096 outproj_reads=8388608/8388608 disagree=none
PAIR pair=d_hip_device_vs_d_hip_hostpath file=gdn_record.f32 bytes=2187264
DREADBACK path=device off=16802240 bytes=32 hex=
DREADBACK control-leg=gdn-device-readback ran=yes
DCONTROL control=transposition leg=gdn-device ran=yes
claims=host_cpubody,device
CONTROL name=gdn-state-transposition-reddens fired=RED
CONTROL name=gdn-state-read-is-the-arm-s-own-fingerprint fired=RED
DROWS form=recurrent rows=7 licensed=4 accepted=4 rejected=0 unlicensed=3
PAT
[ "$missing" = 0 ] || die "exited 0 without printing what a device run must print; treat the window as NOT taken"

# --- DVOID #3: the 1 TiB refusal. No driver path runs it on the device, so the window does. --------
ov="engine/work/d-oversize-$day.txt"
if "$D_ARM" --plan "$wd/d_plan.txt" --slab "$wd/d_slab.bin" --out "$wd/d_out_oversize.bin" \
     --device --fault alloc-oversize > "$ov" 2>&1; then
  die "the 1 TiB hipMalloc SUCCEEDED or the fault never ran; the leg is broken, see $ov"
fi
grep -qF "alloc-oversize refused" "$ov" || die "the oversize leg refused, but not by name; see $ov"
printf 'DVOID-3 witnessed: alloc-oversize refused by the allocator (see %s)\n' "$ov"

nstage=$(grep -c '^STAGERUN' "$log");  [ "$nstage" = 4 ]  || die "expected 4 STAGERUN rows (three shared spans + the router span), got $nstage"
nbind=$(grep -c '^BIND ' "$log");      [ "$nbind" = 10 ]  || die "expected 10 BIND lines with a read-back digest, got $nbind"
printf 'WINDOW ok log=%s no timing taken\n' "$log"
printf 'NOTE halogen is still DOWN. When this session has every timing it wants: tools/halogen_gate.sh up\n'
tools/gpu_lock.sh release "$tag"       || die "release refused; release by hand with the tag $tag"
```

`checks=55` is the CPU witness's 51 plus the four device-only `rt-device-*` checks, and `controls=26` is
25 plus `gather-device-unit-read-back`, which registers only in a window. **If a check or control is
added or removed, update this pattern in the same commit** (PASS 4 §5's rule). If the printed line
disagrees, that is a finding to read, not a pattern to relax: the GDN contribution to the device bill is
eight launches and four allocations per device run, and the fingerprint numbers above are deterministic
because the arm is handed the same certified bytes.

### 8. What is NOT established here — do not cite any of this as done

* **No device leg has ever run for this arm.** Not one launch, allocation, H2D, D2H or read-back. Every
  device-shaped claim above is `DVOID`/`not exercised`, and §7 is the first opportunity any of it is
  executed. `device-claims-come-from-device-only` is what makes a host run unable to launder one.
* **No timing, no rate, no clock read.** Nothing in `m2_d_*` reads a timer; the driver's lines end `no
  timing taken`. Whether this shape is fast is a different question with a different owner (M5).
* **The four ACCEPTED rows are decode-only under one form.** They are `pass=layer0.decode0`,
  `recurrent|pinned`, one cached token, on the pinned bf16 arm. **Nothing here licenses a prefill claim,
  a `chunked` claim, or a policy-A GDN claim**: `chunked|pinned` grants nothing at this pass (printed
  seven times), the prefill passes were not probed for either pair, and `adjudicate` answers a policy-A
  GDN row with `policy-a-arm-outside-band-population` — the arm reads pinned bf16 dump bytes precisely so
  it never produces a row no licence can judge.
* **`core_attn`, `normed`, `y` are measured residue.** They are printed with the licensor's own reason and
  counted (`unlicensed=3`); nothing leans on them, and the driver keeps `PASS` unreachable while any row
  is unlicensed (verdict `QUALIFIED`).
* **The M1 resident set is untouched.** The GDN arm's inputs are host pinned dump bytes read through
  `block_forward.Dumps` (sha-verified per read) and copied into the arm's own allocation; they are not
  policy-A resident tensors, so nothing here says anything about staging or residency.
* **The `max_ulp` numbers are inside a relative-error band, not inside ±1 ULP.** `scan_out max_ulp=5912`
  and `state max_ulp=8192` pass on `one_minus_cos/mean_rel/max_abs/per_head_max`. If D-17's ±1-ULP policy
  is ever meant to reach f32 GDN stages, that is a licensor question for the user, not this arm's.

### 9. Corrections to record while this is fresh (code-side cleanups, not redesigns)

* `control_d_plan_refusals`'s docstring says "Six refusals … and a plan whose state span overlaps `z`";
  `wants` has five legs per binary and the printed evidence is ten (`10 refusals across two binaries`).
  Add the `spans-overlap` leg or drop the clause — the printed line is the truth and it says ten.
* The third `DVOID` line (`gdn-alloc-oversize-actually-fails`) prints `reason=needs-a-device-window`
  without the `status=not-exercised` field the other two carry. A window grepping for
  `DVOID.*status=not-exercised` finds two of three.
* The per-kind item counts are reconciled but not echoed: they live inside the arm's `RUN census=` token
  and reach the log only as `items_total=18496/18496` plus `disagree=none`. A `DCENSUSKIND` echo would let
  a reader see `enter/carry/kvmem/write/out/norm/outproj` without re-deriving them.
* `engine/work/device_window_pass4.sh:41` still asserts two `BUILD ok` lines; build.sh prints eight. Use
  §7's block, and do not paste the old script into a GDN window.

### 10. `DECIDED (agent, 2026-10-27)` — pass-6 drafts to transplant into `DECISIONS.md`

None of these is a user decision; none touches a shipping width, a band, a floor, a route allowance, a
pinned fixture, or halogen. Each names the number or failure that forced it.

1. **One compiled pair per form; a plan naming any other form is refused rather than honoured.**
   DECIDED (agent, 2026-10-27), forced by `band_licenses` keying on `width|form` and granting
   `recurrent|pinned` for 4 of 7 rows while `chunked|pinned` grants 0 of 7 at `layer0/pinned/decode0`
   (the standing licence D-21 chose, and the reason it chose it). An arm that accepted a `form=` word it
   did not implement could be *labelled* into a licence for arithmetic it never ran; `m2d_io.h` refuses
   with `form-not-implemented` and `gdn-form-not-merged` requires the refusal, the parsed
   `form=recurrent` on every run, and `chunked` granting nothing, all at once.
2. **The state is contracted in exactly one function, and the transposition mode lives inside it.**
   DECIDED (agent, 2026-10-27), forced by `d_k == d_v == 128` making a transposed state shape-legal, and
   by the prototype lane's record of a zero state making the injection invisible. One contraction point
   means one answer, one place to bend, and a control (`band_reddened_on=scan_out,state`) that reddens the
   *licence*. The alternative — inlining the contraction in each of the two stages that contract
   (`kvmem`, `out`) — would have needed two injections per path to prove one property.
3. **The state read is witnessed by a fingerprint the arm computes as it reads, reconciled against the
   driver's own fold, and the reconciliation must also hold at zero.** DECIDED (agent, 2026-10-27),
   forced by the same trap class (`reads_state` as a claim, not a constant). The `zero_in` leg *requires*
   `reads_state=0` and a zero fingerprint, so a skipped carry cannot present as a cheap success.
   Witnessed non-zero (509,952 live words) and witnessed moved (one flipped byte ⇒ `2525565901→2525566144`).
4. **`-ffp-contract=off -fno-fast-math` on the GDN pair as on the other three.** DECIDED (agent,
   2026-10-27), following `m2_se_kernels.h`/`m2_g_kernels.h` and D-25 item 6 (zero-tolerance compiled
   pairs extend to the device TUs): the declared order is one ascending f32 accumulator per reduction with
   no fusion, so the g++/hipcc pair comparison stays a codegen question instead of becoming a question
   about FMA contraction.
5. **The recurrent arm takes `gdn_sumsq`, the bf16 bit-image helpers and the geometry constants from
   `prototype/gdn/gdn_kernels.h` (included, not copied), but implements the cached decode step itself.**
   DECIDED (agent, 2026-10-27), forced by the lane implementing the *chunked* scan: no prototype gate
   vouches for the recurrent step's arithmetic, so importing its body would borrow a licence that does not
   exist. What is imported is the reduction form D-23 ratified (`gdn_sumsq`) and the bf16 bit-image
   helpers. The `q` scale follows the recurrent arm's spelling (`Form.named("recurrent") → q_scale="div"`),
   because the two arms' product trees are not the same program — the same distinction D-21 rests on.
6. **The arm reads pinned bf16 decode-step records, not policy-A resident bytes.** DECIDED (agent,
   2026-10-27), forced by `adjudicate` answering a policy-A GDN row with
   `policy-a-arm-outside-band-population`: the only licence at this pass is the bf16 cross-form one, so
   reading Q8_0 projections would produce rows no licence can judge. D-01/D-01a are untouched — this is a
   statement about which record set a *gate* reads, not about what ships.
7. **The third implementation is printed with its own distance to the pins.** DECIDED (agent, 2026-10-27):
   `XIMPL pair=d_arm_vs_engine_torch_body` alone could be two wrong programs agreeing, so
   `XIMPL pair=engine_torch_body_vs_pinned … verdict=EXACT` is printed beside it. The comparison is
   measured and never used as a licensor; the band remains the only thing that credits a row.
8. **Controls carry their path claim with `d_claims(ctx)`, so the device claim only appears with a device
   leg.** DECIDED (agent, 2026-10-27), applying D-25 item 3 ("device claims require a device") to this
   vertical: transposition and read-witness add a `device=True` leg in a window and stop claiming `device`
   in a CPU run. A control whose legs are all host is host-only, in the printed claim as well as in fact.

## D-27 conformance audit + D-26 cross-oracle: what landed, and the agent decisions

Two tools, three evidence files, no gate moved:

- `tools/gguf_conformance_audit.py` — D-27's conformance audit of the shipping artifact. First run:
  `verdict=CONFORMS … findings=0 doc_column_deltas=2 controls=2 fired=2 legs=4/4 no timing taken`,
  written up in `quantization/evidence/gguf-conformance-audit-2026-10-27.md` with the machine manifest
  at `checkpoints/gguf-conformance-audit-2026-10-27.json`.
- `engine/cross_oracle.py` — the engine × HF-chain × llama.cpp three-way report. First run:
  `checks=78 findings=0 aligned=False no bands authored=no licence-issued=none
  verdict=STRUCTURAL-OK no timing taken`, written up in
  `specification/evidence/cross-oracle-three-way-2026-10-27.md`, with the stimulus-alignment evidence
  in `quantization/evidence/llama-stimulus-alignment-2026-10-27.md`.

Both are CPU-only and neither takes a device lock; both end their reports with `no timing taken`.
`engine/block_gate.py` (`checks=17 findings=0 controls=11 fired=11`) and `engine/block_device.py`'s CPU
witness (`checks=51 findings=0 controls=25 fired=25`) are unchanged and still print
`band_digest=27a68c7e82c9aa05`; logs at `engine/work/blockgate-cpu-2026-10-27-crossoracle.log` and
`engine/work/blockdevice-cpu-2026-10-27-crossoracle.log`. Nothing licensed here credits a row.

### `DECIDED (agent, 2026-10-27)` — pass-7 drafts to transplant into `DECISIONS.md`

(Recorded here because `DECISIONS.md` is outside this worker's write scope; the owner promotes them.)

1. **The conformance audit derives, it does not restate.** DECIDED (agent, 2026-10-27): the family →
   type map comes from the artifact's own tensor-info header (`gguf_header_types.Cur`), counts come
   from `engine/artifact.py::expected_records()` (safetensors index × config shapes), bytes come from
   `tools/ggml_block_geom.py`; the width map is *parsed out of*
   `quantization/evidence/q4km-published-width-map-2026-10-26.md` at run time. Forced by D-27: a
   hand-copied table would let a stale width pass as conformance. The one hand-written table in the
   tool maps the document's row labels to family names and contains no types or counts, and an
   unmapped row or unclaimed family is fatal, so the mapping cannot quietly stop covering something.
2. **The engine leg parses through `loader.py`, not `Artifact()`.** DECIDED (agent, 2026-10-27),
   forced by `artifact.py::_c_draft_block`: `Artifact` refuses any artifact whose draft extract is
   empty, and this GGUF carries zero `mtp.*` tensors (width map §What else the file carries item 3;
   `tools/mtp_shard_cost.py` prices the 19-tensor fix at ~5.2 GiB). So the audit uses
   `loader.parse_records` + `loader.ruler_record`, prints the draft absence explicitly, and still
   runs `loader.admit` per tensor so the `output.weight` Q6_K waiver is *exercised* rather than
   assumed. This is not a verdict about whether MTP ships; it is the reason the audit cannot reuse the
   staging object.
3. **The width map's GiB column is reported, not asserted.** DECIDED (agent, 2026-10-27): types and
   counts are asserted exactly (a mismatch reddens the audit and a control proves it), bytes are
   asserted exactly against `quantization/evidence/audit-hash-2026-10-26.json` (`delta=+0`), and the
   document's per-row GiB column is printed as `doc_column_deltas=2` because those figures were typed
   from the hand-written bpw table that
   `quantization/evidence/audit-block-geometry-2026-10-26.md` retired — Q5_K 5.5625 against the
   struct's 5.5, 74 MiB over 37 `down_proj` tensors, which is exactly the 0.0718 GiB gap between the
   document's 20.676 and the file's 20.6042. Asserting that column would assert a table this repo has
   already overruled. **If a reviewer reads this as gate-loosening, the fix is to correct the
   document's column, not to hide the delta.**
4. **`ffn_out`, not `ffn_moe_out`, is HF's `moe_out`.** DECIDED (agent, 2026-10-27) from
   `src/models/qwen35moe.cpp`: `cb(moe_out, "ffn_moe_out", il)` at :507 fires before the shared expert
   is added at :518-:536, whose result is `cb(cur, "ffn_out", il)`. The name that reads like the
   counterpart is the routed-only partial.
5. **`attn_output` means different things per block kind.** DECIDED (agent, 2026-10-27): in a linear
   block it is a GDN interior tensor (`[128, 32, 11, 1]`, block 0 node 53) and the §03.2 mixer output
   is `linear_attn_out` (:483); in the full-attention block `attn_output` (:352, `[2048, 11, 1, 1]`)
   *is* the mixer output. The mapping is keyed on block kind for that reason, not on name matching.
6. **Llama's layer-39 post-mixer activations do not exist in a prefill graph.** DECIDED (agent,
   2026-10-27), read off run-a: block 39's `attn_residual`, `attn_post_norm`, `ffn_out`, `l_out` are
   `[2048, 0, 1, 1]` with `written=false` because of the last-layer `crop_before_nextn` gather
   (`:194-:196`); the same names are written only in the chunk/decode graphs (`[2048, 4, 1, 1]`,
   `[2048, 1, 1, 1]`). The cross-oracle prints these as `UNWRITTEN-BY-INSTRUMENT`, distinguished from
   `UNMAPPED-BY-ABSENCE`, because a fixture absence and a wrong mapping are different failures.
7. **Stimulus alignment is unattainable through the vendored front ends, so the cross-oracle is
   structural-only.** DECIDED (agent, 2026-10-27), measured: the chain's ids were sampled
   (`input.kind=embedding_slice`, `method=torch.randint …`), the instrument has no token-id input
   (`tools/LLAMA_DUMP_NOTES.md` §CLI surface), and the detokenize→re-tokenize round trip returns 32
   ids of which 30 match (first divergence index 3: chain 78392 vs 750) — and for a byte-level BPE the
   detokenized byte string is the *only* candidate preimage. The template is proved not to be the
   obstacle: tokenizing run-a's prompt text yields exactly the prefill graph's inner ids
   `760 6511 314 9338 369`. `align` mode therefore returns non-zero and `numeric` mode refuses with
   denominators rather than comparing unrelated tensors. Making it possible needs a front-end patch or
   a re-dumped chain, and the second replaces pinned fixture material, which is the user's call.
8. **Four HF-vs-llama divergences are named and refused, not adjudicated.** DECIDED (agent,
   2026-10-27), under D-26 clause 3 (a quantized oracle may add checks, never loosen one): activation
   width (bf16 vs f32), router renormalisation dtype/form (llama's f32 DIV vs HF's cast-then-multiply,
   where our §11.6 must-catch set already sides with HF at `ndiff=1326`), softmax domain
   (256-softmax-then-select vs top-k-then-softmax), and the chain gap at layers 38/39. The second one
   in particular is a user decision, not an agent one: resolving it either moves a must-catch set or
   declares the oracle wrong. No band was authored anywhere in this work; the tool prints
   `licence-issued=none`, and any future llama-derived limit must print
   `licence=llama-crosscheck(observation-only)`.

## PASS 7 (2026-10-27, CPU-only) — window `g`'s crash, the control that printed SILENT, and the two device divergences turned into per-field questions

**Done:** the crash (`control_gather_read_back`), the silent guard (`control-witness-can-redden`'s honest
half), and the CPU half of the attribution. **Not done:** every leg that needs the iGPU — they are the
`DVOID`/`GVOID` list in §5 with the command that clears them. No device act in this pass:
`RESULT mode=cpu checks=53 findings=0 controls=28 fired=28 rows=6 gdn_rows=7 arms=policy_a device=off
staged=not-attempted hip_calls=0+0 launches=0 bytes_to_device=0 bytes_from_device=0
band_digest=27a68c7e82c9aa05 no timing taken`; host wall 8.0 s, RSS 544 MB; `grep -c gpu_lock` is 0 in
`block_device.py`, `device_arm.py`, `device/m2_g_*`, `device/m2_d_*`, `device/build.sh`. Full log:
`specification/evidence/blockdevice-cpu-2026-10-27-p7.log`; the numbers below are in
`specification/evidence/blockdevice-device-attribution-2026-10-27.md`.

### 1. The crash — `KeyError: '_bytes'` — fixed by deleting the assumption, not the control

The control asked a unit for `u["_bytes"]`, which only the CPU host-twin path ever attaches (the driver
keeps the bytes it wrote). A `--device` unit comes from the D2H read-back and has no such key, so the
comparison crashed on the first unit of the reference width — before the policy-A leg, before the GDN
vertical, and before any `RESULT` line. The fix is that the control no longer depends on a copy the driver
happens to keep. `g_unit_expectation(ctx, u)` derives the expected bytes from the **same source the pair
check trusts** — the artifact file, addressed by the unit's own `file_kind`/`file_src`/`file_off` — and
`g_read_back_leg(ctx, u, got_hex, slab_head)` returns a named verdict for every way the leg can fail:

| failure mode | what it means | can it be silently true? |
|---|---|---|
| `NO-EXPECTATION` | the unit names no file coordinate, so nothing is being compared | no — it is a fail |
| `SLAB-DIVERGES` | the driver's own slab bytes ≠ the artifact bytes, so a device read-back of them would certify nothing | no |
| `NO-DEVICE-READBACK` | the arm ran on a path that printed no read-back line | no |
| `MISMATCH` | device ≠ slab, or slab ≠ artifact | no |

Witness that it reddens rather than crashing or passing (printed by the CPU witness every run, so it is not
a one-off claim in this file):

```text
GWITNESS guard=gather-read-back fabricated=unit-without-bytes leg_fired=yes NO-EXPECTATION unit names no file_kind/file_src/file_off (got None/None/None), so there is nothing to compare the device's bytes against
GWITNESS guard=gather-read-back fabricated=device-one-bit-off leg_fired=yes MISMATCH device=783c543b253ab6bb283b843b37bc303c artifact=793c543b253ab6bb283b843b37bc303c
GWITNESS guard=gather-read-back fabricated=slab-from-wrong-offset leg_fired=yes SLAB-DIVERGES driver slab=253ab6bb283b843b37bc303c1fbc5c3c but artifact=793c543b253ab6bb283b843b37bc303c: the bytes handed to the copy are not the bytes the artifact holds, so a device read-back of them would certify nothing
```

The control that owns those legs is `gather-read-back-expectation-is-the-artifact` — in a CPU witness it
prints `CONTROL name=gather-read-back-expectation-is-the-artifact fired=RED legs=cpu_predicate:5`, five
comparisons executed (two honest width comparisons against the real plan, three fabrications). No
`try/except: pass`, no "skip when `_bytes` is absent", no default-true predicate anywhere in it: the absence
of an expectation is itself a failure, which is the only form that would have caught the window-`g` state as
a finding instead of a traceback. The CPU witness still reports the device leg itself as
`GVOID control=gather-device-unit-read-back reason=no-device-act-in-this-run` and does not register it —
a host read of host bytes is not a device read-back.

### 2. The guard that printed `fired=SILENT` on the device path — now per path, and silent costs a reason

`control-witness-guard-can-redden` was honest-half + fabrications, where the honest half demanded "no
record has a device act". In a device window that demand is false by construction — five arms are
launching — so the conjunction could not hold and the control went quiet on the one path its fabrication was
supposed to cover. It now splits the honest set by the path the arm itself reported:

- records with `path != device` must be act-free (`no_device_act_offences`, unchanged);
- records with `path == device` must be seen to **have** acts (`device_act_absences`: `launches=0`,
  `bytes_from_device=0` or a missing counter is an offence — a `RUN`/`ARM` line that omits a number cannot
  certify the number was zero);
- a leg with nothing to read prints `status=not-exercised` **with a reason** instead of being counted:
  `GWITNESS guard=no-device-acts leg=device-records-have-acts status=not-exercised reason=no-device-act-in-this-run note=… it is reported unexercised, not counted as passed, and the fabrication below is what shows it can fire`;
- and a fourth fabrication proves the device half can bite without a device:
  `GWITNESS guard=no-device-acts fabricated=device-run-with-no-act offences=['fabricated/device-run-with-no-act.launches=0', 'fabricated/device-run-with-no-act.bytes_from_device=0']`.

Two rules now make a quiet control expensive, and both are checks, not conventions:

- `silent-control-names-its-reason` — a control that did not fire must print a non-empty `reason=`
  (`silent_control_finding` is the single rule so the check and the witness agree). Witnessed non-zero:
  `silent_control_finding(False, None) → 'silent-without-named-reason'`,
  `silent_control_finding(False, 'needs-a-device-window') → None`, `silent_control_finding(True, …) → None`.
- `control-legs-cover-the-path-under-claim` (pre-existing, PASS 4) — a control that claims a path with zero
  counted legs on it is a finding. Together with the above, `fired=SILENT` with no named reason is now
  unreachable: it reddens as a check instead of reading like a clean guard.

In a device window the device records do report acts (`window g`: `ARM arm=hip_device path=device
hip_calls=12 launches=4 allocs=3 bytes_to_device=3354624 bytes_from_device=22556`), so the control's
honest half is satisfiable there and it fires `fired=RED` in both modes. Its own legs are predicate
evaluations over records the arms already produced, so they are named `cpu_predicate` (6 of them in a device
window, 5 in a CPU run where the device half prints `not-exercised`); it claims no arm path, which is what
`control-legs-cover-the-path-under-claim` holds a claiming control to.

### 3. The two unattributed device divergences — per-field counts, injected causes, and what is ruled out

The rule this pass worked under: **nothing was deleted, no field was moved out of a claimed exactness, no
band/floor/factor was authored, and `band_digest=27a68c7e82c9aa05` is unchanged.** Both pair checks
(`gather-compiled-pair-byte-identical-per-path`, `gdn-compiled-pair-byte-identical`) still claim whole-record
byte-identity at zero tolerance and still redden on a device window. Full write-up with all cells:
`specification/evidence/blockdevice-device-attribution-2026-10-27.md`. Short form:

New output: `GATTRIB`/`DATTRIB` (per-field `ndiff_words`, `ndiff_bytes`, `max_bit_step`, `median_bit_step`,
then an accounting line whose `sum_of_fields` must equal the pair's `ndiff_bytes` — held by the new check
`field-attribution-accounts-to-the-pair`, witnessed non-zero when an early `(start,end)`-vs-`(start,length)`
misread printed `accounting=MISMATCH` on 9 tables); `DHEAD` (the GDN `state` field split per head, because
one `exp()` per head moves a whole 16,384-word head-block); `GINJECT`/`DINJECT` (the same tables with a
cause injected **inside the arm**, as an arm mode — `silu_exp_ulp1`, `cast_after`, `acc_bf16`, `rank_accum`,
`byte_shift_1` for the gather; `decay_ulp1`, `decay_head_off1`, `state_xpose` for the GDN); `GEXPSCOPE`/
`DEXPSCOPE` (how faithful the **host** side is at those transcendental sites, glibc `expf` vs
`Decimal.exp()` at 60 digits rounded once to f32, D-14's second route). That counter got two guards of its
own, because an earlier revision of it built the reference route from `struct.unpack(...)` — a tuple — so the
array came back `(n,1)`, broadcast against the host route's `(n,)`, and printed a "disagreement rate" 3,231×
larger than its own population. `exp-route-shape` requires both routes' shapes to equal the population's and
`exp-route-count-exceeds-population` refuses any count outside its printed denominator; the wrong line is kept
as the witness in `specification/evidence/blockdevice-device-attribution-2026-10-27.md` §2c, and the corrected
numbers are `1 of 4,096` (policy A), `2 of 4,096` (ref), `0 of 32` (GDN decay), every disagreement exactly one
bit-image step.

| pair (device) | observed | best-fitting injected cause (host path of the hipcc arm) | verdict of the magnitude test |
|---|---|---|---|
| `g_policy_a_hip_device_vs_hip_hostpath` | 3,744 B | `silu_exp_ulp1` 15,426 B in `s,raw,routed`, `gate=up=0` | **not reproduced** — 4.1× too large |
| ″ | ″ | `cast_after` 4,982 B / `acc_bf16` 5,402 B / `rank_accum` 1,254 B, all `routed`-only | within 0.69–2.99×, so live **and separable** from the row above by field |
| `g_ref_bf16_hip_device_vs_hip_hostpath` | 0 B | `silu_exp_ulp1` 0 B (every `proj_out` weight is exactly 0 at ref width) | reproduced exactly, and it is why "the device arms differ" is already dead |
| `d_hip_device_vs_d_hip_hostpath` | 52,976 B | `decay_ulp1` 264,704 B, fields `scan_out,state` only, 26/32 heads moved | **not reproduced** — 5.0× too large; ≈5 head-blocks fit, so ~5 of 32 decays, not 26 |
| ″ | ″ | `decay_head_off1` 1,533,482 B, `state_xpose` 1,308,457 B (both move `core_attn`,`normed`,`y` too) | **ruled out**: 25–29× too large and they move fields the device pair does not |

Also measured and load-bearing for the "transcendental" story: on the inputs these rows actually use, glibc
`expf` disagrees with a correctly-rounded route on **1 of 4,096** silu sites (policy A), 2 of 4,096 (ref),
and **0 of 32** GDN decay sites — so `ocml` would have to be ~5× less faithful on these inputs than glibc is,
which must be said rather than assumed. And the cheapest device-only index defect (`byte_shift_1`, a
one-byte `rank_off`) moves **110,212 B = 29× the observed 3,744** across all five fields, which is why an
offset/stride defect is not the gather's story.

Pricing a cause is worth nothing if the arm's own reach claim is wrong, so `GINJECT`/`DINJECT` now compare
what a mode moved with the fields its `HDR mode_fields=` names (`GREACH`, `DREACH`): moving an unclaimed
field is always a finding, and claiming a field the mode never moves is a finding when the cause is inert on
every width (at ref width `s`/`raw` are bf16-rounded and swallow a last-bit `exp` whole, which is the
asymmetry, not an inert mode — printed as `claimed_but_unmoved=raw,routed,s`). Two published claims turned
out to be wrong and are corrected against the measurements: `m2g_mode_field(MG_MODE_BYTE_SHIFT_1)` omitted
`routed` (the witness printed `GREACH path=ref_bf16 cause=byte_shift_1 … beyond_claim=routed` and went
`findings=1` before the fix), and `m2d_mode_fields` claimed all five state fields for the decay mode, which
measures exactly `scan_out,state`. Both are tightenings; neither touches a band.

**Mechanism claim, stated as far as the evidence goes:** *unattributed.* No injected cause reproduces both
the size and the field split of either device divergence, and no document declares a scope for this
pair-and-field: D-16 records the prototype `moe_gather` `s`/`raw` glibc-vs-libdevice disagreement as
"counted and never banded", D-14's 1-ULP acceptance is scoped to one `ssm_a` case, and
`prototype/gdn/gdn_kernels.h:61` is a note about not using `__expf`. Relaxing an exactness claim is D-18
reserved to the user, so the checks stay red and the user gets the choice with numbers. The two families
make **different** predictions (`s,raw,routed` with small bit steps vs `routed` only), which is what makes
one window decisive rather than another round of guessing.

### 4. What a CPU run can never clear — printed as `DVOID`, not assumed

```text
GVOID control-leg=gather-device-divergence-attributed status=not-exercised reason=needs-a-device-window
DVOID control-leg=gather-cause-injected-on-device-path status=not-exercised reason=needs-a-device-window
DVOID control-leg=gather-divergence-constant-across-identical-runs status=not-exercised reason=needs-a-device-window
DVOID control-leg=gdn-device-divergence-attributed status=not-exercised reason=needs-a-device-window
DVOID control-leg=gdn-cause-injected-on-device-path status=not-exercised reason=needs-a-device-window
DVOID control-leg=gdn-divergence-constant-across-identical-runs status=not-exercised reason=needs-a-device-window
GVOID control=gather-device-unit-read-back reason=no-device-act-in-this-run
DVOID control-leg=gdn-state-transposition-on-device status=not-exercised reason=needs-a-device-window
DVOID control-leg=gdn-device-readback status=not-exercised reason=needs-a-device-window
DVOID control-leg=gdn-alloc-oversize-actually-fails reason=needs-a-device-window
GVOID control-leg=gather-alloc-oversize-actually-fails reason=needs-a-device-window
```

The three `…-constant-across-identical-runs` legs are the first rung of `pitfalls/00-measurement.md`
§"a divergence count that changes between identical runs is a race": **constant count + adjacent bit images
+ an attributed third side is rounding; a moving count is uninitialised scratch or a race and may never be
banded.** Constancy is a property of two identical *device* runs, so no CPU witness can establish it and the
leg says so.

### 5. Exact next-window device command (main agent only: zero workers active, halogen down, lock held, foreground, blocking)

PASS 6 §7's preconditions are unchanged and still required (halogen worker lock absent, `PI_PROVIDER` nothalogen, `gpu_lock` free then held as `$tag`, engine read down off `--showpids` + `mem_info_gtt_used`, eight
`BUILD ok`). Then, with the same `$V`, `$wd`, `$log`, `$D_ARM`, `$D_HOST`:

```bash
# one window, both verticals, both attribution halves. Run it TWICE into two logs.
"$V" engine/block_device.py --device --work "$wd" \
    --d-arm-binary "$D_ARM" --d-host-binary "$D_HOST" > "$log" 2>&1; rc=$?
printf 'driver exit=%d log=%s\n' "$rc" "$log"
[ "$rc" = 0 ] || die "exited $rc; the refusal is in $log"

missing=0
while read -r pat; do
  grep -q -- "$pat" "$log" || { printf 'MUST-BE-TRUE missing: %s\n' "$pat"; missing=1; }
done <<'PAT'
RESULT mode=device checks=57
controls=29
band_digest=27a68c7e82c9aa05
CONTROL name=gather-read-back-expectation-is-the-artifact fired=RED
CONTROL name=gather-device-unit-read-back fired=RED
CONTROL name=control-witness-guard-can-redden fired=RED
GWITNESS guard=no-device-acts leg=device-records-have-acts records=
GWITNESS guard=gather-read-back fabricated=device-one-bit-off leg_fired=yes MISMATCH
GATTRIB pair=g_policy_a_hip_device_vs_hip_hostpath file=gather_record.f32 ndiff_bytes=3744 sum_of_fields=3744 accounting=ok
DATTRIB pair=d_hip_device_vs_d_hip_hostpath file=gdn_record.f32 ndiff_bytes=52976 sum_of_fields=52976 accounting=ok
DHEAD pair=d_hip_device_vs_d_hip_hostpath file=gdn_record.f32 heads=32 words_per_head=16384
DHEAD pair=d_cause_decay_ulp1_vs_honest file=gdn_record.f32 heads=32 words_per_head=16384
GREACH path=policy_a cause=silu_exp_ulp1 arm_claimed=s,raw,routed beyond_claim=none
GREACH path=policy_a cause=byte_shift_1 arm_claimed=gate,up,s,raw,routed beyond_claim=none
DREACH cause=decay_ulp1 arm_claimed=scan_out,state beyond_claim=none
GMAGMATCH pair=g_policy_a_hip_device_vs_hip_hostpath cause=
DMAGMATCH pair=d_hip_device_vs_d_hip_hostpath cause=
GCAUSE path=policy_a injected_causes=silu_exp_ulp1
DCAUSE injected_causes=decay_ulp1
CONTROL name=gather-divergence-causes-priced fired=
CONTROL name=gdn-divergence-causes-priced fired=
PAT
[ "$missing" = 0 ] || die "exited 0 without printing what a device run must print; treat the window as NOT taken"

# the determinism rung: same work dir is fine, the arms are deterministic given the same certified bytes
diff <(grep '^GATTRIB\|^DATTRIB\|^DHEAD' "$log") \
     <(grep '^GATTRIB\|^DATTRIB\|^DHEAD' "$log2") \
  && printf 'RUNG-1 constant across runs: ok\n'   || printf 'RUNG-1 DIVERGES between identical runs: race or uninitialised scratch, band NOTHING\n'
```

`checks=57` is the CPU witness's 53 (51 before this pass, +`field-attribution-accounts-to-the-pair`,
+`silent-control-names-its-reason`) plus the four device-only `rt-device-*` checks; `controls=29` is 28
(+`gather-read-back-expectation-is-the-artifact`, +`gather-divergence-causes-priced`,
+`gdn-divergence-causes-priced`) plus `gather-device-unit-read-back`, registered only in a window.
**Findings are deliberately not pinned in that pattern**: while both divergences are unattributed the window
is expected to print `CHECK name=gather-compiled-pair-byte-identical-per-path verdict=FAIL` and
`CHECK name=gdn-compiled-pair-byte-identical verdict=FAIL`, and the two `…-causes-priced` controls may print
`fired=SILENT reason=…` naming the unattributed pair. Those are the findings the user is being handed, not a
pattern to relax (PASS 4 §5's rule: a disagreement is a finding to read, not a pattern to edit).

That window clears every leg in §4 except the two hand-run `alloc-oversize` refusals, which stay manual:
PASS 6 §7's `$D_ARM … --fault alloc-oversize` block, and the same leg for the gather arm (the plan and slab
the driver itself wrote, so the refusal is about the real spans):

```bash
engine/device/build/m2_g_arm --plan "$wd/gather_policy_a/plan.txt" --slab "$wd/gather_policy_a/slab.bin" \
    --out "$wd/g_over.bin" --device --fault alloc-oversize   # must FAIL and print `alloc-oversize refused`
```

## PASS 8 (2026-10-27, CPU-only) — the two device divergences attributed by byte-level reproduction; the width hypothesis refuted, the head subset reproduced

The two deterministic divergences PASS 7 turned into per-field questions (`g_policy_a` device-vs-host
3,744 B in `s/raw/routed`; `d_hip_device` 52,976 B in `scan_out/state`, five heads) were attributed on the
saved device bytes, without a device act. Write-up with every number:
`specification/evidence/m2-device-pair-attribution-2026-10-27.md`; the script that produced it:
`specification/evidence/m2_device_pair_attribution.py`; its log: `engine/work/p8-attrib.log`.

**Done.**

* **Hypothesis 1 (width handling on the policy-A gather): refuted.** The read set's formats are now printed
  from the artifact (`GFORMAT`): layer 0 reads `gate`/`up` at Q4_K — same as policy A declares — and
  `down_proj` at **Q5_K where policy A declares Q4_K**, which is D-27's fact stated in bytes (block_bytes
  176, 720,896 B per expert unit, `relation=WIDER-THAN-DECLARED` on 8 of 24 units). Four width causes are
  now priced as arm modes: `q5k_as_q4k` 73,464 B and `width_misread` 73,475 B (both `raw,routed` only — a
  down read cannot reach `s`), `q4k_as_q5k` 113,285 B and `scale_min_naive` 97,085 B (both also move
  `gate`, which the device left bit-identical). All are 19.6x-30.3x the observed total and none matches the
  field split. The positive half: a per-site signed-ULP table on the silu's `expf` — derived from the
  device's own 58 differing `s` words, all explained by **±1 ULP** (42 down, 16 up) — reproduces the
  entire 122,880-byte record with **zero** residual (`GREPRO … verdict=EXACT-REPRODUCTION`). Zero residual
  is what refutes: there is no room left in the record for a width or geometry mechanism. `q6k_d_byte_0` is
  `not-exercised` (layer 0 has no Q6_K unit), not silently inert.
* **Hypothesis 2 (the GDN head subset): mechanism reproduced; the identity of the five heads is a device
  fact.** The full 32-head table is in the note. {9,12,14,15,23} is not contiguous, not an index range, not
  the span-end head, not separated by carried/non-carried state; six heads (3,11,18,19,22,29) are provably
  immovable because their `exp(g_decay)` underflows to 1e-12…1e-40 (the uniform +1 ruler moves exactly the
  other 26). Each of the five is reproduced by **exactly one ULP down** on its own decay and the whole GDN
  record reproduces byte for byte (`DREPRO steps=-1 mask=0x80d200 … ndiff_bytes=0`). No host-computable
  route predicts the set: glibc's `expf` agrees with `Decimal.exp()` at 60 digits on all 32 decays, and the
  naive `exp2(f32(x)*log2e)` spelling differs on 14 heads but lands **+1** where the device landed **-1**.
  Which inputs ocml rounds differently is leg 3 of §6 of the note, with its prediction written down.
* **Gate work.** New: `GFORMAT` + `gather-read-set-forms-come-from-the-file`; the width causes in
  `G_CAUSES` with a derived `GCAUSE-VOID` for geometry-unexercisable causes; `GREPRO`/`DREPRO` device
  reproduction legs that redden on any residual; `decay_ulp_mask_up/down` causes with a mask-vs-all-heads
  equivalence; two `GEXP` equivalence legs per width (zero table moves nothing, all-+1 table ≡
  `silu_exp_ulp1`); `DHEAD`-style per-head ruler now in both signs. New arm modes: `q5k_as_q4k`,
  `q4k_as_q5k`, `exp_ulp_table` (host-path-only by construction), `decay_ulp_mask` with `mask=`/`steps=`.
* **Defects found and fixed, all in the attribution machinery.** `block_device.py:2906` compared a string
  launch count with `> 0` — the crash that killed window `h`; the fingerprint control went SILENT without a
  named finding; `gather-device-unit-read-back` claimed `device` while attributing no legs. In the kernel:
  `m2g_mode_field` under-claimed reach (a mode reaching `raw` reaches `routed`, same record) — the new
  `GREACH` enforcement caught it. In my own helper: `nextafter(e, 1.0)` for a +1 ULP step moves *down* when
  `e > 1`; it left an 11-site, 3,300-byte residual, which the byte-level reproduction caught on the spot.
  The step search now lives once, in `block_device.py::silu_exp_step_table`, and the evidence script calls
  it instead of carrying a second ruler.
* **Gate lines.** CPU witness `checks=54 findings=0 controls=28 fired=28` (`engine/work/p8-after-cpu.log`;
  baseline 53/0/28/28 — the 54th check is the new format check), `engine/block_gate.py`
  `checks=17 findings=0 controls=11 fired=11` (untouched), `band_digest=27a68c7e82c9aa05` unchanged. No
  band, floor, must-catch set or licence was touched; nothing was relaxed to pass, and the `GMAGMATCH`
  factor-two screen still reads the device pairs UNATTRIBUTED — the reproduction legs are additive.

**Not done / not claimed.**

* Nothing here is a device measurement. The two reproductions were run on the arms' **host** paths against
  device records saved from window `h`; `GREPRO`/`DREPRO` inside the driver only fire in a window.
* The Q6_K read path is untouched: layer 0 reads no Q6_K block.
* No band is proposed for either divergence, and neither should be: both are now *attributed*, which is not
  the same question as what to do about them. Whether a last-bit `expf` difference between ocml and glibc
  gets a band, or the device path gets matched by computing the transcendental differently, is a D-17-shaped
  decision for the user once the window confirms the mechanism.

**Next window (device, main agent only, `gpu_lock` held, halogen down, zero workers).** One window, in this
order — the count expectation moves to `checks=58` (54 CPU + the four `rt-device-*`), `controls=29`:

```bash
tools/gpu_lock.sh acquire "$tag"
python engine/block_device.py --device   # 1: GFORMAT, GATTRIB/DATTRIB/DHEAD, then
                                         #    GREPRO ... vs_device_record_ndiff_bytes=0 verdict=EXACT-REPRODUCTION
                                         #    DREPRO steps=-1 mask=0x80d200 ... verdict=EXACT-REPRODUCTION
                                         #    watch control-legs-cover-the-path-under-claim and
                                         #    silent-control-names-its-reason: both were FAIL or unseen in
                                         #    `g`/`h`; their driver-side causes are fixed here, but only a
                                         #    window can show the legs actually ran
python engine/block_device.py --device   # 2: RUNG-1 — diff the GATTRIB/DATTRIB/DHEAD lines against run 1;
                                         #    3,744 and 52,976 must be identical, or band NOTHING
tools/gpu_lock.sh release "$tag"
```

Then the two legs this pass could not execute anywhere: the device's own `expf` bit images
(`DEXPIMAGE`/`GEXPIMAGE`; prediction: 1 ULP down on heads 9,12,14,15,23, ±1 at exactly 58 silu sites,
identical at the other 27 decays and 4,038 silu sites), and a window on a layer whose `down_proj` is Q6_K
(3 of 40, D-27) to replace `GCAUSE-VOID cause=q6k_d_byte_0` with a priced number. no timing taken.

## PASS 9 (2026-10-27, CPU-only) — the four device-arm defects: the launch count typed at its source, the contradiction retired, the SILENT rule moved onto controls

PASS 8's window `i` (`specification/evidence/blockdevice-device-2026-10-27-i1.log`) died at
`block_device.py`'s GDN attribution with `TypeError: '>' not supported between instances of 'str' and
'int'` — no `RESULT` line, rc=1 — and window `h`'s log carried a control that flatly denied what a
reproduction line nine lines above it had just proved. This pass fixes the four device-path-only defects
that made those two logs what they were. Nothing here is a device act and nothing here is a number from
one; host wall and RSS only, and the CPU gate lines are at the bottom. no timing taken.

**Done.**

* **1 — `launches` is typed where it is parsed, not where it is compared.** The parse site is
  `engine/device_arm.py::parse_arm_report`, and it now reads its types out of the printing body's own
  `printf` format strings: `lane_record_field_types()` walks `engine/device/m2_*_*.{cpp,hip}`, follows each
  body's `#include "m2_<lane>_main.inc"`, and classifies every `key=<conversion>` token it finds in the five
  report heads (`RUN`/`HDR`/`PLAN`/`WROTE`/`RESULT`). `%d`/`%u`/`%ld` → `int`, `%x` → `hexint` (the GDN's
  `mode_mask=0x%llx` arrives an `int`, so a control can *assert* the mask the arm acknowledged instead of
  string-matching it), `%d/%d` → `intlist`, anything else → `text`. An unknown binary or a field one report
  spells two ways is a named `Refused`, never a guess. Before, every field was a `str` and three consumers
  hand-cast it:

  ```python
  # before — device_arm.run_arm: every value a str; block_device:2906 etc. cast at the comparison
  if device and (int(run.get("launches", 0)) == 0 or int(run.get("hip_calls", 0)) == 0):
  # after — device_arm.run_arm: the value is already an int or the parse refused
  if device and (run.get("launches", 0) == 0 or run.get("hip_calls", 0) == 0):
  ```

  The comparison-side casts are gone from every site, and the fold that adds the counters to the witness is now
  one function (`witness_fold`) shared by all four verticals — the rehearsal found a second, worse copy of the
  bug there: `g_arm`/`d_arm`/`register` each had their own `int(run["launches"])`, so a fabricated value still
  killed the run before any check could speak. `device-run-counters-are-typed-at-the-parse-site` now scans the
  whole file and requires **zero** casts of a quoted counter *and* zero inline comparisons of one; the second
  half is what would have caught window `i`'s actual crash site, which was a bare `> 0` on a `str`. Fabrications
  are labelled as fabrications and redden the *check predicate*, never the driver:
  `launches="abc"` → `device-launch-counts-are-ints-and-positive FAIL`, `launches=0` on `path=device` → same,
  a missing counter → same, and `device/launches=1` stays clean. The rehearsal's `--mutate launches-abc` leg
  proves the end-to-end version: `MUTATION arm=launches-abc-red target-red=True result-reached=True
  crash=False named-on-result-line=True`.
* **3 — the contradiction is retired and the guard that lets it come back is in the gate.**
  `CONTROL gdn-divergence-causes-priced … reason=no injected cause reproduces the GDN device divergence`
  coexisted with `DREPRO steps=-1 mask=0x80d200 … verdict=EXACT-REPRODUCTION` because its cause table
  predated PASS 8's byte-level probe. The control is now re-pointed at the cause that reproduces: it takes
  the head subset from the device's own `state` diff (`DHEAD`), prices `steps=-1` and `steps=+1` against the
  device record, and its verdict is the byte test — `fired=RED legs=host_cpubody:N` with
  `reproduction=EXACT steps=-1 heads=9,12,14,15,23 mask=0x80d200 withdrawal=steps+1 residual=75206 B`. It
  reddens by name in three withdrawals: no head named by the ruler, `steps=-1` leaving a residual (the cause
  is gone), or `steps=+1` also reproducing (the sign was never pinned, so the −1 match was a fit). The
  factor-two `DMAGMATCH` screen is still printed — it is what rules `decay_head_off1` and `state_xpose` out
  by 29x and 25x — but it no longer gets to state the verdict once a byte-level reproduction exists. New
  control `attribution-probe-is-dialable` proves the probe is not a tautology: withdrawing the mode moves 0
  B and the arm echoes `mode_mask=0x0 mode_steps=0`, an empty mask is refused by name
  (`m2_d_io.h::mode-mask-empty`), the full mask moves 264,684 B across the 24 heads the ruler can move,
  every single-head mask moves strictly between 0 and that, and +1 ULP moves a different byte count from −1.
  In a device window it also freezes the winning combination (`DREPRO-WIN … verdict=EXACT|WITHDRAWN`) and
  requires each single device head to leave a positive residual. General rule, new check
  `no-control-contradicts-an-exact-reproduction`: no SILENT control may deny a pair that some `DREPRO` or
  `GREPRO` line in the same log reproduced exactly, read off this run's own printed lines, and
  `attribution-contradiction-guard-can-redden` demonstrates red by feeding the predicate the stale sentence
  beside the real reproduction line (printed verbatim as `GWITNESS-STALE`, two findings) while a denial
  about a pair nobody reproduced must stay — and does stay — clean.
* **4 — the SILENT rule binds controls, not checks.** `silent-control-names-its-reason` was reading
  `v.checks`; checks have no legs to lose, which is why window `h` printed two quiet `fired=SILENT` control
  lines with no `reason=` past a passing check. The rule now lives in `silent_control_finding`, which
  `Verifier.control` runs on *every* registration, so a control cannot be registered without passing it, and
  the check reads `v.unreasoned`. `silent-reason-guard-can-redden` demonstrates it: four fabricated control
  registrations through the real `Verifier` — quiet-with-reason clean, quiet-no-reason red,
  whitespace-only-reason red, fired-without-reason clean (a reason is owed by a control that could not run,
  not by one that ran). The two controls window `h` saw go quiet are given real legs or a derived reason:
  `gather-divergence-causes-priced`'s device half now distinguishes residual / unresolved-no-exact /
  unresolved-with-exact and prints `reproduced=…` instead of denying; `control_d_state_read_witness` builds
  its refusal per leg (`honest_fails`/`flip_fails`/`back_fails`) and names which sub-condition failed
  instead of printing the canned `nothing: …`. Zero `fired=SILENT` lines remain in the real registry
  (`controls=31 fired=31`); the SILENT lines still in the log are the scratch-Verifier fabrications and the
  quoted stale text, both of which are the point of their control.

**Not done / not claimed.**

* **2 — the device arm reaching `RESULT` is now demonstrated as far as a CPU-only session can go, and no
  further.** `specification/evidence/blockdevice-device-shape-2026-10-27.py` runs the real driver under
  labelled substitutes — a counted host-byte `StubHip` for the six `engine.device.cuda` primitives, an arm shim
  that re-spells the host report's `path=` field as `device` and stamps `REHEARSAL-FABRICATION` on the line that
  says so, and `REHEARSAL-DEVICE-BYTES status=present|absent sources=…` which refuses to let a host stand-in
  adjudicate a device-vs-host pair. It reaches
  `RESULT mode=device checks=63 findings=6 controls=32 fired=27 … hip_calls=19+336 launches=111
  bytes_to_device=98123776 bytes_from_device=11507624 counters_unparsed=0 band_digest=27a68c7e82c9aa05`, and a
  `sys.settrace` line census reports which device-only statements actually ran:
  `DEVONLY census statements=106 exercised=103 not-exercised=3 traced=2759`. The three unexercised legs are
  named with reasons, not counted as passed: `device_path_census:1607` and `check_control_paths:1668` carry
  `reason=guard-held-but-enclosing-condition-false(…)` and `main:5321` (the `--device` refusal banner for a
  missing `hipcc`) the same. Its own audit is `REHEARSAL verdict=PASS no device act occurred, no timing taken`,
  and the six findings it prints are five device-only controls going quiet **with** named reasons plus the
  rehearsal's own expected red — a real window should print fewer. What a rehearsal cannot do is produce device
  numbers: the `GREPRO`/`DREPRO` reproduction legs are therefore `status=not-claimed` here and belong to the
  next window.
* The ±1-ULP `exp` scoping question is drafted below and is **not** in `DECISIONS.md`; it is agent-decidable
  in form but touches the ±1-ULP acceptance policy, which D-17/D-18 reserve to the user.
* No band, floor, must-catch set or licence was authored or moved; `ulp_policy.py` untouched; the three
  pinned verdicts this pass had to leave alone are untouched (`GREPRO … unexplained=0 … ndiff_bytes=0`,
  `DREPRO steps=-1 … ndiff_bytes=0`, `DREPRO steps=+1 … ndiff_bytes=75,206`), `band_digest=27a68c7e82c9aa05`.

**Gate lines (host wall only, no timing taken).** `python engine/block_device.py` →
`RESULT mode=cpu checks=59 findings=0 controls=31 fired=31 rows=6 gdn_rows=7 arms=policy_a device=off
staged=not-attempted hip_calls=0+0 launches=0 bytes_to_device=0 bytes_from_device=0 counters_unparsed=0
band_digest=27a68c7e82c9aa05 no timing taken` (baseline was 54/0/28/28: +5 checks = the launch-count predicate,
the contradiction rule, the device-only census, the parse-site typing rule, the witness-fold rule; +3 controls =
the dial probe, the contradiction guard, the silent-reason guard; `engine/work/p9-cpu-20.log`, host wall 126 s,
RSS 269 MB). `engine/block_gate.py` → `BLOCKGATE verdict=QUALIFIED … checks=17 red=0 controls=11 fired=11
band_digest=27a68c7e82c9aa05` and `RESULT mode=cpu checks=17 findings=0 controls=11 fired=11 … no timing taken`.
`engine/gate.py` → `RESULT PASS checks=27 findings=0 notes=41 arms=40 evidence=not-written`. Both baselines are
unchanged; running any of these appends `stage_run`'s E2 selftest-marker lines to
`quantization/evidence/audit-ordering-2026-10-26.log`, which is an append-only ledger and grew as a consequence
of running the gates, not of a scope change.

**Disclosure — a scratch artifact was destroyed, and it is not restorable from this repo.** The first rehearsal
child ran with the default `--work`, so it wrote into `engine/work/m2dev/` and overwrote the window-`i` scratch
device records there (`out_device.bin`, `d_out_device.bin`, `rt_out_device.bin`, `gather_ref_bf16/out_device.bin`,
`gather_policy_a/out_device.bin`). They are gitignored `*.bin` scratch, not pinned fixtures — nothing under
`specification/` or the cross-check dumps is touched — and every number derived from them survives in
`specification/evidence/blockdevice-device-2026-10-27-i1.log` and
`specification/evidence/m2-device-pair-attribution-2026-10-27.md`. The rehearsal now runs in its own
`engine/work/m2dev-rehearsal`, reads a records snapshot only from a directory it names, and refuses self-copy and
size mismatch; the raw records are regenerable by the next device window, which regenerates them anyway.

### DRAFT (agent, 2026-10-27) — the 1-ULP `exp` difference at host-twin sites: scoped, with a documented bound

Not a decision the agent may finalize: D-17/D-18 make the ±1-ULP acceptance *policy* the user's. What is
agent-decidable is the **scope and shape of the witness**, and that is what is drafted here for promotion.

The question: at the sites where the device's `expf` and glibc's differ by one ULP — the 32 `exp(g_decay)`
evaluations of §04.6 step 1, of which the device moved {9,12,14,15,23}, and the 58 `expf` sites inside
`silu` that `GREPRO` names — is the difference *scoped* by a documented bound, or does the host twin stop
being a byte-identity witness at those sites?

The bound must come from a document, never from the measurement (that is the trap `pitfalls/30`
§cancellation-bands records: a band read off the current build can only ever ratify the current build).
Three documents, all in this repo, all read at run time:

1. `prototype/moe_router/moe_router_kernels.h::mr_exp` — the prior documented allowance, "allowed to differ
   by an ULP or two", i.e. **2 ULP per transcendental call**; this is the third term of the `min(10, 9, 22)`
   in `specification/evidence/blockdevice-wpre-scope-2026-10-26.md`, which is a scope taken from documents
   read at run time and is the only precedent in-tree for scoping a transcendental at all.
2. D-14 — a recomputed transcendental is not a bit-exactness obligation, and a documented-route bound is
   admissible *provided the must-catch set grows with it*.
3. The arms' own reach claims: `m2d_mode_fields`/`m2g_mode_field` now refuse a mode that claims a field it
   cannot reach (`GREACH`), so the site population is an arm assertion, not a driver opinion.

llama.cpp, per D-19 ("every time look at llama.cpp first"): the gated-delta-net decay there is `ggml_exp`
over the same recurrence (`src/models/delta-net-base.cpp:106,135,174,209,216,336`, reached from
`src/models/qwen3next.cpp`), and on every backend it is the platform's own exponential —
`ggml/src/ggml-cuda/unary.cu:60` (`op_exp` → `expf(x)`), `ggml/src/ggml-sycl/gated_delta_net.cpp:79,114,128`
(`sycl::native::exp`), `ggml/src/ggml-cpu/vec.h:955` (`ggml_vec_exp_f32` → `expf`). It never asks a device
backend for byte identity against the CPU reference: `tests/test-backend-ops.cpp` judges with
`mean_abs_asymm` against `max_maa_err()`, default `1e-4` (line 1250, per-op overrides at 3592 and 6064)
plus `nmse` (line 301). **Adopted:** the structure — keep the device `expf`, and let the witness at those
sites be a bounded-agreement one rather than a byte-identity one. **Refused:** the `1e-4` budget. llama.cpp
earns it because *everything* in its reference path is banded; here the GDN and gather records are
claimed-exact everywhere else, and importing `1e-4` would trade a 1-ULP question for a 2⁻⁸ input
perturbation — D-17 territory, and the same reason PASS 8 refused to import its Q8_0 activation quantization.

**The draft.** The host twin stays a byte-identity witness over the whole record, and the transcendental
sites are the only places a difference is admissible at all: the admissible step is **1 ULP per evaluation
site** (inside the 2-ULP-per-call document in (1), and the observed device step is 1, never 2 — the observed
value is never the reason the bound is 1); the population of sites is the arm's own reach claim plus the
`DHEAD`/`GEXP` rulers; the must-catch set that (2) demands is present and non-inert — `±1` signed step
tables in both signs, the per-head mask dial with `mode-mask-empty` refused, `at-limit`/`one-past-limit`/
`far-past-limit` legs on the router's `wpre` scope, and the whole-record residual required to be **exactly
zero**. Zero residual is what makes this a scoping and not a tolerance: at `steps=-1` the device record is
reproduced with 0 of 2,187,264 bytes left over, and if that residual ever becomes non-zero the divergence is
UNATTRIBUTED again and both controls redden by name (`DREPRO-WIN verdict=WITHDRAWN`,
`gdn-divergence-causes-priced fired=SILENT reason=the cause that used to reproduce this pair is withdrawn`).
No factor, ratio or aggregate error measure is admitted, and nothing leaves claimed-exactness except the
single-ULP step at a named transcendental site.

**Next window (device, main agent only, `gpu_lock` held, halogen down, zero workers).** Expect
`checks=60`-ish (56 CPU + the four `*-device-*` device-mode checks) and `controls=33`; findings must be 0.

```bash
tools/gpu_lock.sh acquire "$tag"
python engine/block_device.py --device   # must print, in this order:
    #  ARM arm=hip_device path=device hip_calls=… launches=…   (×4 honest lanes)
    #  DEVICESTAGED / DEVALLOC / DEVWITNESS counters non-zero
    #  GREPRO pair=g_policy_a… probe=exp_ulp_table … unexplained=0 … vs_device_record_ndiff_bytes=0 verdict=EXACT-REPRODUCTION
    #  DREPRO steps=-1 mask=0x80d200 heads=9,12,14,15,23 … ndiff_bytes=0 verdict=EXACT-REPRODUCTION
    #  DREPRO steps=+1 … ndiff_bytes=75206 verdict=RESIDUAL-LEFT
    #  DDIAL probe=device_heads … / DREPRO-WIN probe=decay_ulp_mask steps=-1 mask=0x80d200 residual_bytes=0 verdict=EXACT
    #  CONTROL name=gdn-divergence-causes-priced fired=RED legs=…  (NOT fired=SILENT)
    #  CONTROL name=gdn-state-read-is-the-arm-s-own-fingerprint — if SILENT, it must carry reason= naming
    #      which of honest/flip/back failed; window `i`'s leg numbers show the state did move (120259→120260),
    #      so the likely residual cause is a sub-condition of the flip leg, now named instead of `nothing: …`
    #  CHECK name=device-launch-counts-are-ints-and-positive verdict=PASS
    #  CHECK name=no-control-contradicts-an-exact-reproduction verdict=PASS
    #  CONTROL name=attribution-probe-is-dialable fired=RED legs=…
    #  RESULT mode=device … findings=0 … band_digest=27a68c7e82c9aa05   ← the line window `i` never printed
python engine/block_device.py --device   # 2: determinism — diff GREPRO/DREPRO/DATTRIB/GATTRIB against run 1
                                          #    (3,744 / 52,976 / 0 / 75,206 must repeat, or band NOTHING)
tools/gpu_lock.sh release "$tag"
```

If a second defect surfaces *after* the parse site, fix it in the same place the type or the population is
derived, not at the consumer — that is the whole content of defect 1 and it is worth re-reading before
patching a symptom. no timing taken.

---

## PASS 10 (2026-10-27, CPU-only) — the two window-`j` check-layer defects: the contradiction check reads the whole log, the terminal tally is reconciled

Window `j` (`devices/evidence/device-window-2026-10-27-g.md` §"Two defects the window exposed",
`specification/evidence/blockdevice-device-2026-10-27-j1.log`) reached a verdict for the first time —
`RESULT mode=device checks=63 findings=4 controls=32 fired=30 counters_unparsed=0`, no traceback — and
the four findings are the two attributed divergences plus the two controls that went quiet with named
reasons. Those four are the honest state and were **not touched**. This pass fixes the two defects in the
check layer that the same log exposed. No device act, no device number, host wall and RSS only; the CPU
gate lines are at the bottom. no timing taken.

**Done.**

* **1 — the contradiction check reads the run's complete printed text, and a zero count is now itself a
  finding.** The defect: `no-control-contradicts-an-exact-reproduction` printed
  `verdict=PASS 0 reproduction legs` in a log containing three, because `engine/block_device.py`
  parsed `ctx["text"]`, a snapshot captured at `parsed = BG.parse_printed(out.text())` **before** the
  reproduction controls printed a single leg — `parsed=0 → PASS` reached through *ordering* instead of
  through the parser. The fix, all in `engine/block_device.py`:
  * `run()` hangs the live buffer on the ctx (`live_text=out.text`), and `run_text(ctx)` is the single
    door: the check and its guard read every line printed at the moment of the check; a proof harness
    that hands over a fixed `text` is complete by construction and falls back to it.
  * `control_attribution_contradiction_guard` now runs **after** `control_gather_divergence_causes` and
    `control_d_divergence_causes`, because its `this-run` leg reads the live text — before the move it
    printed `exact_pairs=none` in a log that had printed three legs, the witness reading the wrong half
    of its own log.
  * `reproduction_leg_finding(device, text)` is the explicit anti-vacuity assertion: a **device run that
    parses zero exact-reproduction legs reddens** (the repro controls print a `DREPRO`/`GREPRO` leg on
    every device pass, so 0 means blind parse — the window-`j` shape, 0 parsed against 3 printed — or a
    withdrawn attribution; both are findings, never a clean log). A CPU run prints no legs and passes
    by a *named* exemption, not by vacuity. The check's own detail now prints the counts it actually
    parsed, and in a device-shaped run those counts are non-zero.
  * Non-vacuity, proven two ways, both printed, through the unmodified functions over window `j`'s own
    device log (`specification/evidence/blockdevice-checkfix-proof-2026-10-27.py`, output in
    `specification/evidence/blockdevice-checkfix-proof-2026-10-27.log`; the harness self-audits —
    `HARNESS verdict=PASS … no device act occurred, no timing taken`):
    (a) a fabricated SILENT control denying a pair the run reproduced → the real check line goes
    `verdict=FAIL … Refutations found: … denies d_hip_device_vs_d_hip_hostpath, which this run
    reproduced exactly`; the guard's `leg=fabricated-denial` prints the same reddening at predicate
    level (`findings=2 expected=2`);
    (b) window `j`'s own text with the `DREPRO`/`GREPRO` lines **removed** → the zero-count assertion
    reddens (`verdict=FAIL 0 reproduction legs … device run parsed 0 exact reproduction legs`) instead
    of the check passing; the guard's `leg=zero-leg-text` prints `finding=True`.
    And the device-shaped green line with its non-zero count, verbatim from the harness (window-`j` text
    through the real check):
    `CHECK name=no-control-contradicts-an-exact-reproduction verdict=PASS 2 reproduction legs printed
    `verdict=EXACT-REPRODUCTION` (pairs: d_hip_device_vs_d_hip_hostpath,g_policy_a_hip_device_vs_hip_hostpath)
    and 1 printed a residual (pairs: d_hip_device_vs_d_hip_hostpath)` — with
    `GWITNESS guard=attribution-contradiction leg=this-run lines=583 exact_pairs=d_hip_device_vs_d_hip_hostpath,g_policy_a_hip_device_vs_hip_hostpath
    findings=none` from the moved guard. The guard control is now five legs (`legs=cpu_predicate:5`) and
    still fires RED in both modes.
* **2 — one finding tally, reconciled against the printed record before it is stated.** The defect:
  `RESULT … findings=4` went out in a log that also printed a `findings=2` line — the attribution
  guard's own stale-text detail, a *different denominator* (findings in its fabricated text, not in the
  run) printed with nothing saying so — and three `verdict=FAIL` lines, the third of which is the
  `BLOCKDEVICE` terminal verdict echoing its own `red=4`, not a third divergence. The two real
  divergences are `gather-compiled-pair-byte-identical-per-path` (policy_a ndiff 3,744) and
  `gdn-compiled-pair-byte-identical` (device-vs-host 52,976); the other two findings are the two SILENT
  controls, which print `fired=SILENT` records and are counted — so 4 = 2 FAIL CHECK lines + 2 unfired
  controls, and nothing was adjusted to make that true. Fixes:
  * `reject_tally_findings(red, lines)` (`_REJECT_TALLY`) is one pure reconciliation over the printed
    lines: every failed CHECK name has exactly one printed `verdict=FAIL` record; every registered
    unfired control has its printed `fired=SILENT` record (finding ⇒ record — a printed line cannot
    manufacture a finding); a printed FAIL with no entry in the list is named; and the totals agree.
  * `check_terminal_tally` is the **last check the run prints** — after every CHECK/CONTROL line exists,
    before `BLOCKDEVICE`/`RESULT` state the number — witnessed non-inert by three fabrications through
    the same predicate, printed as `GWITNESS guard=tally-reconciliation … fabricated=missing-record
    flagged=True fabricated=extra-print flagged=True fabricated=clean flagged=True`. Its detail names
    every counted finding and states both denominators (the `2 findings` line's and the tally's) in the
    log itself. `Verifier.red` is the single source: a failed CHECK under its name, an unfired control
    under `control:<name>`; `BLOCKDEVICE red=` and `RESULT findings=` are the same `len(v.red)`.
  * The `witness-counters-all-parsed-from-arm-reports` check moved *before* the `BLOCKDEVICE` line,
    which also closes the `checks=62`-vs-`checks=63` drift between the two summary lines (the same
    class of defect, one line earlier: a count that moved between two prints of "the same" total).

**Not done / not claimed.**

* The next device window must see the contradiction check **green with a non-zero leg count** (2 exact
  legs, `exact_pairs=` naming both pairs) — on this CPU box the device-shaped rehearsal has no GDN or
  gather device records to compare against (`REHEARSAL-DEVICE-BYTES status=present
  sources=e2-device-records/rt_out_device.bin` cannot adjudicate them), so its honest verdict is
  `no-control-contradicts-an-exact-reproduction verdict=FAIL 0 reproduction legs` and the rehearsal
  RESULT reads `checks=64 findings=7` — the 7th finding **is** that expected red, disclosed by
  `REHEARSAL arm=grepro-drepro status=not-claimed`. The parent still prints
  `REHEARSAL verdict=PASS no device act occurred, no timing taken` (`engine/work/p10-rehearsal.log`,
  filtered extract in `specification/evidence/blockdevice-device-shape-rehearsal-2026-10-27-p10.log`),
  stays confined to `--work m2dev-rehearsal`, keeps the no-HIP-library guard
  (`REHEARSAL-CHECK arm=no-hip-library stage=after-run maps_grep=0`), and its tally check reconciles the
  mixed 7-finding log non-vacuously (2 FAIL lines + 5 unfired controls = 7). Supplying the window-`j`
  gather records through the rehearsal's snapshot channel is NOT possible without editing the
  rehearsal: its arm-shim keys saved records by basename, and both gather lanes' device records share
  the basename `out_device.bin` under different subdirectories — flat snapshot entries would
  cross-pollinate the two width paths. Recorded so nobody "fixes" the rehearsal by copying bytes flat.
* No band, floor, must-catch set or licence authored or moved; `ulp_policy.py`, `gate.py`,
  `block_gate.py`, `device_arm.py` untouched; `GREPRO`/`DREPRO`/`DREPRO-WIN` print sites untouched
  (`unexplained=0`, `ndiff_bytes=0` at −1, `75,206` at +1, `DREPRO-WIN` present — window `j` values,
  unchanged because the code that prints them is unchanged); `band_digest=27a68c7e82c9aa05`.
* The two `exp`-scope DRAFT questions from PASS 9 are untouched and still the user's (D-17/D-18).

**Gate lines (host wall only, no timing taken).** `python engine/block_device.py` →
`RESULT mode=cpu checks=60 findings=0 controls=31 fired=31 rows=6 gdn_rows=7 arms=policy_a device=off
staged=not-attempted hip_calls=0+0 launches=0 bytes_to_device=0 bytes_from_device=0 counters_unparsed=0
band_digest=27a68c7e82c9aa05 no timing taken` (was 59/0/31/31: +1 check = the terminal tally
reconciliation; `BLOCKDEVICE … checks=60 red=0` now equals the RESULT count;
`specification/evidence/blockdevice-cpu-2026-10-27-p10.log`). `engine/block_gate.py` →
`checks=17 red=0 controls=11 fired=11`, `RESULT mode=cpu checks=17 findings=0 controls=11 fired=11`.
`engine/gate.py` → `RESULT PASS checks=27 findings=0 notes=41 arms=40 evidence=not-written`.
`tools/doc_ref_check.py --strict` → `findings: 0`. Proof harness:
`specification/evidence/blockdevice-checkfix-proof-2026-10-27.py` → all seven `PROOF name=…
verdict=PASS`. Gate runs appended to `quantization/evidence/audit-ordering-2026-10-26.log` (append-only
ledger, as before). `engine/work/m2dev` was **not** written to (mtimes still 20:20 = window `j`);
scratch discipline held.

**Next window (device, main agent only, `gpu_lock` held, halogen down, zero workers).**

```bash
tools/gpu_lock.sh acquire "$tag"
python engine/block_device.py --device   # must print, in this order:
    #  GREPRO … unexplained=0 … vs_device_record_ndiff_bytes=0 verdict=EXACT-REPRODUCTION   (unchanged)
    #  DREPRO steps=-1 … ndiff_bytes=0 verdict=EXACT-REPRODUCTION / steps=+1 … 75206 RESIDUAL-LEFT
    #  DREPRO-WIN … residual_bytes=0 verdict=EXACT                                       (unchanged)
    #  GWITNESS guard=attribution-contradiction leg=this-run lines=… findings=none
    #      exact_pairs=d_hip_device_vs_d_hip_hostpath,g_policy_a_hip_device_vs_hip_hostpath   ← non-empty now
    #  GWITNESS guard=attribution-contradiction leg=fabricated-denial … findings=2 expected=2
    #  GWITNESS guard=tally-reconciliation findings=4 fail-check-records=2 unfired-control-records=2 …
    #  CHECK name=no-control-contradicts-an-exact-reproduction verdict=PASS 2 reproduction legs printed …
    #  CHECK name=terminal-verdict-finding-count-matches-printed-records verdict=PASS …
    #      2 CHECK verdict=FAIL lines (gather-compiled-pair-byte-identical-per-path,gdn-compiled-pair-byte-identical)
    #      plus 2 registered unfired controls … the BLOCKDEVICE line's own verdict=FAIL is the echo
    #  BLOCKDEVICE verdict=FAIL … checks=64 red=4 …  and  RESULT mode=device checks=64 findings=4 …
    #      ← findings must be exactly the two divergences + whatever two controls go quiet WITH reasons
python engine/block_device.py --device   # 2: determinism — GREPRO/DREPRO/DATTRIB/GATTRIB must repeat
tools/doc_ref_check.py --strict          # findings: 0
tools/gpu_lock.sh release "$tag"
```

Rule of the pass: a count that reaches a verdict must be parsed from the same lines it claims to count,
and a count of zero must be able to say *why* zero is the honest answer — `parsed=0 → PASS` is banned
whether it arrives through the parser or through the ordering. no timing taken.

---

## PASS 11 (2026-10-27, CPU-only) — D-28 executed: the two compiled-pair checks judge **value-space agreement per field**, with the bound derived before the measurement and from sources that are not the build

D-28 retired the byte-identity premise on the two device-vs-host compiled pairs, because byte identity
was the instrument that *found* the mechanism and is now a worse ruler than the mechanism implies. This
pass implements it in `engine/block_device.py`. No device act, no `--device`, no kernel, no `gpu_lock`,
no timing; host wall and RSS only. Every number below is from `engine/work/p11-f3.log` (the pre-D-28
baseline used for the no-regression diff is `engine/work/p11-base.log`), and the whole derivation is
written up with its numbers in
`specification/evidence/blockdevice-d28-value-band-2026-10-27.md`. no timing taken.

**Done.**

* **1 — the two checks changed verdict-basis, and the keeps-exact list became a bound.**
  `gather-compiled-pair-byte-identical-per-path` → `gather-compiled-pair-value-agreement-per-path`
  (policy A judged per field in value space; the `ref_bf16` pair still at byte identity), and
  `gdn-compiled-pair-byte-identical` → `gdn-compiled-pair-value-agreement` (device-vs-host in value
  space; the g++ twin pair still at zero tolerance). Nothing that D-28 left exact moved: host twins,
  the bf16 gather, router `ids`/`ids_alt`/`wcast`, M1 read-back digests, the pinned-oracle licences,
  `band_digest=27a68c7e82c9aa05`. The keeps-exact *fields* — gather `gate`/`up`, GDN
  `q_hat`/`k_hat`/`core_attn`/`normed`/`y` — are now derived bounds of exactly **0** from the arms' own
  reach claims (`engine/device/m2_g_kernels.h::m2g_mode_field`,
  `engine/device/m2_d_kernels.h::m2d_mode_fields`), printed `teeth=zero-tolerance`, so "stays exact"
  is asserted by the band rather than promised next to it.
* **2 — every bound is computed before the pair is measured, from three device-free sources.**
  `BOUNDFROM pair=… metric=… field=… sources=N[…] derived=… binding=… single_source=… teeth=…
  unavailable=…`, one line per (pair, field, currency) — 36 cells — plus a `BOUNDFROM-SITE` line per
  pair naming the probe, the population (gather 4096 = `MG_K·MG_I`; GDN 32 heads), the step and the
  sign patterns. Sources: (a) `prototype/moe_router/moe_router_kernels.h::mr_exp`'s "an ULP or two",
  read out of the sentence at run time (`vb_documented_allowance` refuses if the wording moves) and
  propagated by an arm-executed finite difference on the arm's *host* path at every documented site,
  all-up / all-down / half-population split; (b) the licensor's published width doubled by the triangle
  inequality (`speccheck/evidence/gdn-band-2026-10-26.json`, same shape as prior 1 of
  `specification/evidence/blockdevice-wpre-scope-2026-10-26.md`) — available for `scan_out`/`state`,
  printed as an unavailable reason for policy A (`policy-a-width-band-absent`); (c) the fp64 floor
  probe of `s` and `routed` from the record's own operands (`engine/device/m2_g_kernels.h::m2g_item1`,
  `::m2g_item4`), doubled, with the D-19 §7 premise inside the source label. `min` binds. A count
  bound that derives to ≥1.0 is refused outright (`d28-count-bound-vacuous`) rather than printed. The
  three admitted single-source shapes (zero-tolerance cells, `gather.raw` per D-14, count cells whose
  other two families each print a reason) are named in `d28-bounds-come-from-documented-sources`, and
  that check also catches a printed source count that disagrees with the source list beside it.
* **3 — the witness is its own line and a CPU box says so.** 38 `BOUNDWITNESS` cells
  (`measured/derived/margin/inside/for_leg`), never folded into a PASS: the pair that exists here (the
  g++ twin against the hipcc host path) is measured for real at `measured=0 inside=yes`, and the
  device-vs-host pair prints `measured=not-claimed … the command that takes it is
  engine/block_device.py --work <work> --device`. `d28-value-band-witness-inside-and-edge-legs-reject`
  fails on any `inside=no`.
* **4 — the band can say no.** `BOUNDEDGE … leg=at-limit|one-past-limit bound=… measured=…
  sized_at=… verdict=INSIDE|OUTSIDE`, twelve legs, one pair of legs per currency, constructed in closed
  form *from the bound* (one element at the denominator's floor moved by `t·n·denom`; one element moved
  by `t·‖b‖`; `floor(t·n)` flipped words) and tightened or widened until unambiguously on its own side;
  every leg is labelled `note=FABRICATION, in-memory only` and nothing is written to disk.
  `at-limit → INSIDE` and `one-past-limit → OUTSIDE` are both required per metric.
* **5 — the must-catch set comes from geometry, not from the build.** `MUSTCATCH-DERIVE` prints the
  geometry it used (`MG_K=8`, `MG_I=512`, `MG_H=2048`, gate|up along the output dim, each field's own
  span boundary; `GDN_H_V=32`, `GDN_D_K=GDN_D_V=128`, state span 524,288 words) and the population it
  derived: 3 gather + 5 GDN = 8, printed `MUSTCATCH name=… reddens=RED` — gate/up halves swapped, wrong
  expert id (slot swap), span off-by-one on each of `raw` and `state`, two heads' state permuted, one
  head zeroed, carried state dropped, decay sign inverted (`state[h] *= exp(2·g_decay[h])`, `g_decay`
  read from the slab span, the record-space image of `exp(−g)`). All eight redden, by five to seven
  orders of magnitude in the value currencies; the check reconciles printed against derived and a zero
  population reddens the run.
* **6 — the refuted causes were re-priced, and the one the band cannot refute is named, not hidden.**
  `silu_exp_ulp1` is still outside the derived band (`teeth=yes`, `routed/rel-err 5.65e-07 vs 3.25e-07`).
  `decay_ulp1` is **inside** it (`teeth=none … note=THE BAND IS TOO WIDE`), and the line says why: that
  cause was refuted by a **byte-count ratio against the observed pair** (264,704 B injected against
  52,976 B observed; 15,426 B against 3,744 B on policy A), a +1 ULP whole-population injection is a
  *smaller member of the same family* as the documented ±2 ULP worst case, and the observed pair is
  device data no device-free bound can encode. The live control that keeps it refuted is
  `GMAGMATCH`/`DMAGMATCH verdict=RULED-OUT`, and the REFUTED-CAUSE line now carries that obligation
  explicitly (`refutation_carried_by=…`), naming the required device-window print. If the reading is
  that D-28 §3 must be satisfied *by the value band itself*, then the bound cannot be derived without
  the device and the re-derivation is the user's call — this pass refuses to fake it either way.
* **7 — the diagnostics keep their voice and lose their vote.** `GREPRO`, `DREPRO`, `DREPRO-WIN` carry
  `role=diagnostic gate=no` at their print sites. Because a CPU witness prints none of them,
  `d28-diagnostic-legs-are-labelled-and-do-not-decide` is structural: it walks this file's own AST, finds
  every `print` whose first literal names one of the three legs, requires the label there, and separately
  proves the two pair checks cite their `BOUNDWITNESS` cells and no `vs_device_record_ndiff_bytes`.
* **8 — D-28's escalation clause is a rule, and it is demonstrated.** `d28_mechanism_offences` reads
  the run's own diagnostics for a value band that passed beside a non-`exp` mechanism (residual-left
  repro where one is required, `unexplained != 0`, a `CANDIDATE` cause that is not an `exp`/`decay`
  cause, a device pass with no repro leg), and `d28_qualification_decision` returns
  `ESCALATE-USER(…)` instead of `QUALIFIED` — it is the function that computes the terminal verdict word
  now. `d28-escalation-clause-refuses-qualification` demonstrates the refusal on a fabricated log
  (`residual-left repro + CANDIDATE cause=state_xpose` → 2 refusals) while the same predicate finds 0 in
  this run, so the clause cannot be decorative.

**Gate lines.**

```
BLOCKDEVICE verdict=QUALIFIED … checks=67 red=0 controls=31 fired=31 band_digest=27a68c7e82c9aa05
RESULT mode=cpu checks=67 findings=0 controls=31 fired=31 rows=6 gdn_rows=7 arms=policy_a device=off
    staged=not-attempted hip_calls=0+0 launches=0 counters_unparsed=0 band_digest=27a68c7e82c9aa05 no timing taken
engine/block_gate.py  → checks=17 red=0 controls=11 fired=11 band_digest=27a68c7e82c9aa05
engine/gate.py        → RESULT PASS checks=27 findings=0
tools/doc_ref_check.py --strict → findings: 0
specification/evidence/blockdevice-device-shape-2026-10-27.py
    → REHEARSAL verdict=PASS no device act occurred, no timing taken   (child: checks=71 findings=7,
      the same 7 named in PASS 10; confined to --work m2dev-rehearsal)
```

`checks` 60 → 67 is the 7 new D-28 checks; controls, licences, bands and the digest are untouched.

**Not done / not claimed.**

* **The device leg of the witness is not taken here.** The command that takes it is
  `/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python engine/block_device.py --work engine/work/m2dev --device`
  under the standing device rules, and its required prints are the 36 `BOUNDFROM` lines, 36
  `BOUNDWITNESS` cells carrying `measured=` numbers instead of `not-claimed`, the two
  `REFUTED-CAUSE … teeth=` lines (with `decay_ulp1`'s `DMAGMATCH verdict=RULED-OUT` obligation
  discharged), and the labelled `DREPRO`/`GREPRO` diagnostics. The rehearsal exercises the *shape* of
  that path against its own labelled fabrications, never against real device bytes, so nothing on this
  box has yet shown a real `measured=` value inside these bounds.
* **The value band does not price a byte-count ratio.** See bullet 6: `decay_ulp1` is inside it by
  construction, and the tension between "refuted causes must keep failing" and "no number read off the
  build" is recorded rather than resolved. Resolving it in favour of the band needs device data and is
  therefore the user's call, not this pass's.
* No band, floor, must-catch set, licence or pin was authored or moved; `ulp_policy.py`, `gate.py`,
  `block_gate.py`, `device_arm.py`, `loader.py`, `engine/device/m2_[gd]_*` and `prototype/` are
  untouched; nothing was written to the real device scratch `engine/work/m2dev` (this pass's runs used
  `--work engine/work/p11-*`); no device act occurred and no timing was taken.

Rule of the pass: a bound is only a bound if it can reject something on purpose — so the band is derived
before the measurement, the rejection legs are sized from the bound rather than from the spread, the
must-catch set is derived from geometry, and a limitation the band cannot cover is printed on the line
that claims to check it. no timing taken.

---

## PASS 12 (2026-10-27, CPU-only) — the D-28 close-out: rejection legs on the format's own lattice, refuted causes naming the axis that refutes them, and a reported device command that is exactly runnable

Window `l` (`specification/evidence/blockdevice-device-2026-10-27-l1.log`, `RESULT mode=device checks=71
findings=3`) showed D-28's band reaching a verdict with real device bytes and **two of its twelve rejection
legs failing** — `g/ndiff_frac at-limit=False` and `d/ndiff_frac at-limit=False` — and PASS 11's own lines
carried two more defects: `sized_at=` was a knob nobody could audit, and a `command=` line printed by a
witness running in a scratch dir was not the command the main agent runs. This pass closes all three in
`engine/block_device.py`; the derivation is written up with its numbers in
`specification/evidence/blockdevice-d28-edge-legs-2026-10-27.md`. No device act, no `--device`, no kernel,
no `gpu_lock`, no timing; the CPU witness is 15.1 s wall / 394 MiB max RSS and
`specification/evidence/blockdevice-cpu-2026-10-27-p12.log`
is the full log behind every number below, with its derivation written up by
`specification/evidence/blockdevice-d28-edge-legs-2026-10-27.py`. no timing taken.

**Done.**

* **1 — the `ndiff_frac` legs were sized in the wrong place, and now are not.** The cause: the old
  `build(count)` flipped bit 0 of the first `count` A-words *without regard to whether they already
  differed*, so with a non-zero base divergence the flips that landed on already-differing words added
  nothing while the base was still counted — gather `s` had 58 of 4096 words differing and `k_at = 1508`,
  and the leg landed at `measured=0.374756 > bound=0.368164`; GDN `scan_out` had 466 of 4096 and landed
  at `0.676758 > 0.669678`. The construction is now `k_at = floor(bound*(1+1e-9)*n)` for at-limit and
  `k_at+1` for one-past, and the divergence is *added* by flipping bit 0 of `k_at - base` words drawn from
  the **agreeing** set, so each flip adds exactly one differing word; `base_words=` is printed on the word
  legs so the base is on the line, not inside the arithmetic.
* **2 — one sizing rule for all three metrics: adjacent lattice points, measured granularity.** Every leg
  is now an index on the grid the stored format has — the number of differing **words** for `ndiff_frac`,
  the number of **f32 steps** one element is moved from its B-side value for `rel-err`/`relL2` (IEEE rank,
  so adjacent integers are adjacent representables) — and the two legs of a metric are the adjacent pair
  that straddles the bound. `granularity = measured_one_past − measured_at_limit` is **measured**, never
  typed: the count metric prints `granularity=0.000244140625`, which is `1/4096` and falls out of the two
  legs (the check re-derives `1/word_count` from the record layout and requires the measurement to match).
  Every `BOUNDEDGE` line now carries `lattice_index=`, `lattice_step=`, `distance_to_bound=`,
  `granularity=`, `sized_correctly=yes|no`, and the six per-metric summaries read
  `… at-limit=… one-past=… sized_correctly=…`. An **under-sized** leg — `distance_to_bound > granularity`
  — reddens: a leg that never approached the bound tested nothing, which is the quiet twin of the leg that
  overshot. Non-inertness is printed as `GWITNESS guard=edge-sizing` legs: `zero-granularity` and
  `under-sized-half-bound` (at-limit built at half the bound: `distance 0.184082 > one lattice step
  0.000244141`) both flag, `naive-first-words-derived-base` rebuilds the pre-fix construction under the
  device's own per-field word count and reproduces window `l`'s `at-limit=False` from its own numbers on
  host bytes with no device act, and `fixed-under-derived-base` — the shipped construction under that same
  non-zero base — must stay clean, which is what says the fix and not the flag is doing the work.
* **3 — a refuted cause now names the axis that refutes it.** `REFUTED-CAUSE` prints `carried_by=` (not
  `refutation_carried_by=`) naming `value-band(currencies)` and/or `byte-diagnostic(injected, observed,
  excess, ratio)`, and `carried_by=none` reddens — demonstrated by running a fabricated `carried_by=none`
  line through the same predicate. `decay_ulp1` stays present, stays `teeth=none` in value space, and is
  refuted **on the byte axis with the arithmetic in numbers**: `injected 264704 B - observed 52976 B =
  211728 B > observed 52976 B (4.997x)`, and `silu_exp_ulp1` `15426 B - 3744 B = 11682 B > 3744 B
  (4.120x)`. The injected totals are parsed from this run's printed `GCAUSE`/`DCAUSE` lines and the
  observed total is **re-summed from the printed per-field `GATTRIB`/`DATTRIB` split** and required to
  equal the pair line and its printed `sum_of_fields`; a CPU witness has no device pair of its own, so it
  reads the pinned window-`l` log and names it on the line (`byte_source=`), and the rule is the screen's
  own `within_factor_two` read the other way, so a byte-exact pair cannot make a huge cause look unrefuted.
  Nothing was relaxed to "both axes must fail in value space", and neither band was widened nor cause
  dropped. Read PASS 11's bullet 6 with this rename: the field it printed as `refutation_carried_by=` and
  the `sized_at=` knob on `BOUNDEDGE` are both gone, superseded by `carried_by=` and the
  `distance_to_bound`/`granularity`/`sized_correctly` triple.
* **4 — a `command=` line is now exactly the command the main agent runs.** `window_command()` rebuilds
  from `WORK_DEFAULT` instead of `sys.argv`, stripping an argv `--work <val>` / `--work=<val>` and any
  argv `--device`, so a witness run in `--work engine/work/p12-*` cannot report a device command that
  reads that scratch dir (window `l` also printed `--device --device`). The new check requires every
  printed device command to carry the canonical `--work` and exactly one `--device` — 9 such lines in this
  run, 0 offences — and proves the predicate bites on a fabricated stray `--work` and a fabricated doubled
  `--device`. A CPU witness owes a command on every leg it cannot take, so a zero population there is
  blindness and reddens; a device window owes none (it *is* the window) and any command it prints is still
  held to the rule. The fabrications and the rehearsal keep their own dirs: nothing in this pass writes to
  `engine/work/m2dev`.

**Gate lines.**

```
BLOCKDEVICE verdict=QUALIFIED … checks=69 red=0 controls=31 fired=31 band_digest=27a68c7e82c9aa05
RESULT mode=cpu checks=69 findings=0 controls=31 fired=31 rows=6 gdn_rows=7 arms=policy_a device=off
    staged=not-attempted hip_calls=0+0 launches=0 counters_unparsed=0 band_digest=27a68c7e82c9aa05 no timing taken
engine/block_gate.py  → BLOCKGATE verdict=QUALIFIED … checks=17 red=0 controls=11 fired=11
engine/gate.py        → RESULT PASS checks=27 findings=0 notes=41 arms=40 evidence=not-written
tools/doc_ref_check.py --strict → findings: 0
specification/evidence/blockdevice-device-shape-2026-10-27.py
    → REHEARSAL verdict=PASS no device act occurred, no timing taken   (child: checks=73 findings=7,
      the same 7 named in PASS 10; confined to --work m2dev-rehearsal)
```

`checks` 67 → 69 is the two new checks (`d28-edge-legs-are-adjacent-lattice-points-and-sized-from-the-bound`,
`d28-reported-device-command-is-the-canonical-one`); controls, licences, bands and the digest are
untouched, all 36 `BOUNDWITNESS` cells are still `inside=yes` with `outside=none`, and all 8 `MUSTCATCH`
legs still print `RED` with `printed MUSTCATCH lines=8, derived=8`.

**Not done / not claimed.**

* **The device leg of the witness is still not taken here.** The command that takes it is now printed
  canonically and is exactly runnable:
  `/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python engine/block_device.py --work /home/erokhins/erokhins_projects/qwen3.6-35b-a3b/engine/work/m2dev --device`
  under the standing device rules. Its required prints now include 12 `BOUNDEDGE` legs with
  `base_words=` non-zero on the count legs and `sized_correctly=yes` on all six metrics, the two
  `REFUTED-CAUSE … carried_by=` lines taking their observed bytes from **that window's own** `DATTRIB`/
  `GATTRIB` lines instead of the pinned log, and no `--work` other than the canonical one.
* **The byte-axis observed numbers in a CPU witness are window `l`'s**, read out of
  `specification/evidence/blockdevice-device-2026-10-27-l1.log` and named on the line. They are unchanged
  since window `g`, but they are pinned device data, not a CPU measurement, and the line says which file
  it came from.
* No band, floor, must-catch set, licence or pin was authored or moved; `ulp_policy.py`, `gate.py`,
  `block_gate.py`, `device_arm.py`, `loader.py`, `engine/device/m2_[gd]_*` and `prototype/` are untouched;
  nothing was written to the real device scratch `engine/work/m2dev` (this pass's runs used
  `--work engine/work/p12-*`, the rehearsal its own `m2dev-rehearsal`); no device act occurred.

Rule of the pass: a rejection leg is only a test if it is built on a grid the data can actually land on and
it lands within one step of the bound it claims to touch — so the lattice comes from the format, the step
comes from the measurement, the refutation names the axis that carries it, and a reported command is the
command. no timing taken.

## PASS 13 (2026-10-27, CPU-only) — a refusal is a counted finding, a check-less run is fatal by name, and window `m`'s refusal was a leg-sizing bug and not a bound failure

Window `m` (`devices/evidence/device-window-2026-10-27-g.md` §window `m`,
`specification/evidence/blockdevice-device-2026-10-27-m1.log`) died inside one leg-construction call and
printed `RESULT mode=device checks=0 findings=1 controls=0 fired=0` with **zero** `CHECK` lines, zero
`MUSTCATCH`, zero `REFUTED-CAUSE` — and `grep -E "verdict=FAIL"` on that log returns nothing. The refusal
was `REFUSED tag=d28-edge-leg-wont-size … the scan_out/rel-err inside leg could not be placed on its own
side of the bound 1.46299e-07`, raised out of `run()`, so everything after it never happened. This pass
closes both halves of that in `engine/block_device.py`: the refusal becomes a finding on the leg that raised
it, an empty suite becomes fatal by name, and the refusal's *subject* is measured instead of assumed. CPU
only: no `--device`, no kernel launch, no `gpu_lock` (`grep -c gpu_lock engine/block_device.py` → 0), no
timing read. The witness is 15.07 s wall / 395 MiB max RSS and
`specification/evidence/blockdevice-cpu-2026-10-27-p13.log` is its full log (byte-identical across two
runs apart from the two pid fields `GPUWITNESS` prints). no timing taken.

**Done.**

* **1 — a refusal is now a counted finding on the leg that raised it, and the run continues.** `leg_attempt
  (ctx, family, what, construct)` wraps one construction call; a `DA.Refused` is printed as
  `REFUSED tag=… for_leg=… owning_check=… counted_as_finding=yes suite_continues=yes` and recorded against
  the check that already owns that family (`VB_LEG_OWNERS`: derivation →
  `d28-bound-and-witness-lines-are-parsed-not-assumed`, real edge legs →
  `d28-value-band-witness-inside-and-edge-legs-reject`, fabrications →
  `d28-edge-legs-are-adjacent-lattice-points-and-sized-from-the-bound`, must-catch →
  `d28-must-catch-set-is-derived-and-reddens`, refuted causes →
  `d28-refuted-causes-still-refuted-and-say-what-carries-it`). Every one of those four checks now refuses to
  PASS while a refusal of its own family is on the record, and names it as `refused=<leg>(tag=…) <note>`.
  The catch sites are per **leg**, not per run: one metric's fabrication set, one injected defect, one
  refuted cause, one pair's bound derivation. Proof
  (`specification/evidence/blockdevice-refusal-proof-2026-10-27.py`, its log beside it, each mode a real
  driver run in its own subprocess with one construction poisoned): poisoning a fabrication leg prints
  `checks=71 findings=1` with all 71 `CHECK` lines, the six metric summaries, 12 `BOUNDEDGE` legs, 8
  `MUSTCATCH` and both `REFUTED-CAUSE` lines present, and exits 1; poisoning window `m`'s exact coordinate
  (GDN `scan_out`/`rel-err` at-limit) prints `checks=71 findings=2` — the witness check and the adjacency
  census, which also sees `1 cells missing` — and completes. In both, `grep verdict=FAIL` finds the
  finding; in window `m` it found nothing.
* **2 — "nothing failed" and "nothing ran" are now different strings.** `terminal_fatality(mode, checks,
  controls, escaped=None)` is the single statement of the rule (checks==0 → `no-check-executed`; controls==0
  → `no-control-executed`; escaped → `aborted-before-its-check-suite-completed`), witnessed on the fabricated
  device shapes `(device,0,0)` and `(device,71,0)`. Two places run it. In-run: `run-executed-a-non-empty-
  check-and-control-suite` (the last check but one) plus `d28-value-band-exists-in-this-run`, which reddens
  if no D-28 pair survives its derivation — the case in which the six `d28-*` checks would each have read an
  empty population and PASSed over it. In `main()`: a refusal that still escapes now prints a `CHECK name=…
  verdict=FAIL` record, a `FATAL mode=… reason=…` line and a `RESULT … fatal=…` line, and exits 1 — the
  old handler's bare `checks=0 findings=1 controls=0` is gone. Proof: `zero-check` mode poisons
  `DA.parse_se_layout` so nothing runs, and the driver prints `CHECK name=run-executed-a-non-empty-check-
  and-control-suite verdict=FAIL`, `FATAL … reason=aborted-before-its-check-suite-completed(…)` and
  `RESULT … checks=1 findings=1 controls=0 fatal=aborted-…`, exit 1. The device shape of that predicate is
  printed from the same function, because no `--device` run happens here.
* **3 — what the device substrate does to `scan_out/rel-err`: nothing. The leg was sized from a base the
  metric never sees.** Measured on CPU from window `m`'s own bytes (`engine/work/m2dev/d_out_{device,host,
  hip}.bin`, read as ordinary host files) and cross-checked against its printed lines
  (`specification/evidence/blockdevice-scanout-substrate-2026-10-27.py`):

  ```
  SCANOUT-DIAGNOSIS substrate=device    base=2.43282e-08 bound=1.46299e-07 verdict=construction-wrong
  SCANOUT-DIAGNOSIS substrate=host_twin base=0           bound=1.46299e-07 verdict=inside
  SCANOUT-DIAGNOSIS substrate=headline  base=2.43282e-08 bound=1.46299e-07 verdict=construction-wrong
  ```

  The device base is *inside* the derived bound by a factor of **6.014** (window `m` printed
  `measured=2.43282e-08 derived=1.46299e-07 margin=6.014 inside=yes`, and re-measuring its bytes reproduces
  that digit for digit), so the refusal could not have been evidence about the bound. The arithmetic is in
  `scan_out/rel-err`'s own units: `vb_fabricate_at_limit`'s closed form moved the largest-|b| element by
  `bound*unit`, i.e. it solved for a base of **zero** — the only base a CPU witness can produce, because its
  host-twin pair is byte-identical. On a substrate that has a base the estimate lands at `1.7063e-07`, which
  is `2.43312e-08` past the bound — exactly the base it ignored — and closing that gap takes **1242** one-ULP
  steps of `1.95878951e-11` while the correction loop's budget was **64**; hence the refusal at the 64th
  step. The construction now sizes from `bound - floor_metric`, where `floor_metric` is the metric with the
  moved element's own base contribution erased (`vb_norm_lattice` + `vb_leg_headroom`, one spelling of the
  lattice and of the escalation rule). On window `m`'s device bytes the same builder now lands `j=6226` at
  `1.46282e-07`, inside, distance `1.65e-11` ≤ one step, in 2 correction steps; on the host substrate it
  lands `j=7468` at `1.46282e-07` — the same point it printed before, so every CPU `BOUNDEDGE` number is
  unchanged (`ndiff_frac at-limit 0.368164 == bound`, `one-past 0.368408`, all six summaries
  `at-limit=True one-past=False sized_correctly=yes`). `BOUNDEDGE` now prints `base_metric=` and
  `base_headroom=` on the ulp legs so the base is on the line, and the driver prints the three
  `SCANOUT-DIAGNOSIS` lines itself, with the device base parsed off the pinned window-`m` log and named on
  the line (a CPU witness has no device record of its own).
* **4 — no escalation is owed, and no bound moved.** `verdict=outside` did not happen: no substrate's base
  exceeds `1.46299e-07`, so `escalation=not-required` and D-18 is not engaged. Had it happened, the refusal
  is now named — `d28-edge-leg-base-above-bound`, produced by the same pure predicate
  (`vb_leg_headroom`) that a fabricated `base=2.0e-07` is run through inside
  `d28-value-band-witness-inside-and-edge-legs-reject`, alongside the honest `base=2.43282e-08` which yields
  headroom `1.21971e-07` and refuses nothing. Which derivation would have been short, had it been needed:
  neither the `mr_exp` 2-ULP propagation (it is what produced this bound —
  `mr_exp x host_fd[all_up](steps=2,decay_ulp_mask)`, `single_source=no`, the licensor triangle also present)
  nor the fp64 floor probe, which prints unavailable for this field for a stated reason ("the record does not
  carry the carry/scan operands of this field"); a documented re-derivation would therefore have had to make
  the record carry those operands and re-probe, not adjust a constant. `band_digest=27a68c7e82c9aa05…`
  unchanged, no band, floor, must-catch set, licence or pin authored or moved.

**Gate lines.**

```
BLOCKDEVICE verdict=QUALIFIED … checks=71 red=0 controls=31 fired=31 band_digest=27a68c7e82c9aa05
RESULT mode=cpu checks=71 findings=0 controls=31 fired=31 rows=6 gdn_rows=7 arms=policy_a device=off
    staged=not-attempted hip_calls=0+0 launches=0 counters_unparsed=0 band_digest=27a68c7e82c9aa05 no timing taken
engine/block_gate.py  → BLOCKGATE verdict=QUALIFIED … checks=17 red=0 controls=11 fired=11
engine/gate.py        → RESULT PASS checks=27 findings=0 notes=41 arms=40 evidence=not-written
tools/doc_ref_check.py --strict → findings: 0
specification/evidence/blockdevice-scanout-substrate-2026-10-27.py → exit 0, all_agree=yes
specification/evidence/blockdevice-refusal-proof-2026-10-27.py     → PROOF verdict=PASS
specification/evidence/blockdevice-device-shape-2026-10-27.py → REHEARSAL verdict=PASS no device act
    occurred (child: RESULT mode=device checks=75 findings=7 — the same 7 findings PASS 10 named, +2 checks)
```

`checks` 69 → 71 is exactly the two new guards (`d28-value-band-exists-in-this-run`,
`run-executed-a-non-empty-check-and-control-suite`); `RESULT findings=` is reconciled against the printed
records by the last check as ever, all 36 `BOUNDWITNESS` cells are still `inside=yes` with `outside=none`,
all 8 `MUSTCATCH` legs still print `RED` with `printed MUSTCATCH lines=8, derived=8`, both `REFUTED-CAUSE`
lines still print `carried_by=` (`silu_exp_ulp1` on value-band + byte-diagnostic, `decay_ulp1` on
byte-diagnostic with the factor-two arithmetic), the six `BOUNDEDGE` metric summaries are byte-identical to
PASS 12's, `granularity` is still measured (`0.000244140625 == 1/4096`), and
`d28-escalation-clause-refuses-qualification` still demonstrates its refusal (2 fabrications → 2 refusals,
real diagnostics → 0).

**Not done / not claimed.**

* **The device window has not been re-run — it cannot be from here.** What it should show, and what it will
  show nothing about if it does not: 12 `BOUNDEDGE` legs with `sized_correctly=yes` including
  `d/rel-err … base_metric=2.43282e-08 base_headroom=1.21971e-07`, no
  `REFUSED tag=d28-edge-leg-wont-size` line, three `SCANOUT-DIAGNOSIS` lines with
  `substrate=device … source=this-run BOUNDWITNESS … bound_matches_this_run=yes`, and `checks≈71` (device
  legs add two) with the refusal counted if anything else refuses instead of the run ending. The command,
  canonical as printed by the driver:
  `engine/block_device.py --work /home/erokhins/erokhins_projects/qwen3.6-35b-a3b/engine/work/m2dev --device`
  under the standing rules (zero workers, halogen down, `gpu_lock` held, main agent, foreground).
* The `SCANOUT-DIAGNOSIS` device base in a CPU witness is **parsed from the pinned window-`m` log**, and the
  closed-form landing is then derived as `bound + base` rather than measured (no device record exists in a
  CPU run); the script that measures it from window `m`'s bytes agrees with the derivation to 2e-05 of the
  bound. If window `n` prints a different base, the line says which file it read and the comparison is redone
  against the new log — `VB_SCANOUT_EVIDENCE` is the one place that file is named.
* The rehearsal's device-vs-host pairs still compare host paths (`e2-device-records`, not `m2dev`), so the
  rehearsal exercises the *shape* of the refusal path and not a non-zero device base; the non-zero base is
  exercised by the scanout script on window `m`'s real bytes.
* Nothing was written to the canonical device scratch `engine/work/m2dev` (this pass used
  `engine/work/p13-*`, the rehearsal its own `m2dev-rehearsal`); `device_arm.py`, `block_gate.py`,
  `gate.py`, `ulp_policy.py`, `loader.py`, `engine/device/m2_[gd]_*` and `prototype/` are untouched; no
  device act occurred. no timing taken.

Rule of the pass: a refusal is a claim about one leg, so it is counted on that leg and the run finishes;
"nothing failed" and "nothing ran" are different sentences and only one of them is a pass; and a bound that
cannot be approached from a substrate is evidence about the *construction* until the substrate's own base has
been measured in the bound's own currency — which is the only measurement that may ever point at a
re-derivation, and never at a widening. no timing taken.

## PASS 14 (2026-10-27, CPU-only) — a silent control is a state, not a bit: the two window-`n` controls, and the two checks that make de-claiming cost a sentence

Window `n` (`specification/evidence/blockdevice-device-2026-10-27-n1.log`) was otherwise clean —
`RESULT mode=device checks=75 findings=2 controls=32 fired=30`, 75 CHECK lines, no `verdict=FAIL` except the
two findings themselves, 12 BOUNDEDGE, 8 MUSTCATCH RED, `REFUSED=0` — and both findings were controls printing
`fired=SILENT` while claiming legs. Neither was a dead arm; both were the control's own bookkeeping, and the
mode that exposed them is the mode the controls were written to cover.

**The two defects, and why they were invisible in a CPU witness.**
`control-witness-guard-can-redden` (SILENT on the device path since window `g`) had a single flat must-catch
list shared by three predicates, so it asked `launch_count_offences` — which reads `launches` and nothing else
— to catch a `bytes_to_device` fabrication, and cloned every fabrication from `sorted(honest)[0]`, which in a
window is the *device* record. On CPU that accident fired the control (the victim was a host record with
`launches=0`, so the bytes_to_device fabrication looked caught for a reason that had nothing to do with the
field it bent); on a device window the victim had `launches>0` and the control printed
`uncaught=['device/bytes_to_device=1GiB']`. Each predicate now gets the fabrications its own fields own:
`no_device_act_offences` has acts added to a *host* base, `device_act_absences` has acts removed from a real
*device* base (`bytes_from_device=0`, `launches=0`, the counter unprinted) and `launch_count_offences` gets
the three `launches` shapes plus the `launches=1` shape that must stay clean. Where no device record exists the
base is printed as `synthetic-device-record reason=no-device-act-in-this-run`, because bending a base that was
already an offence catches nothing — that was the accident. Legs are counted under the path of the record each
one read, so the control now claims `device` and carries `device:10` in a device-shaped run.
`gdn-state-read-is-the-arm-s-own-fingerprint` printed `reason=7 leg(s) refused` and all seven reasons were
fabricated by the control: three were `fingerprint-disagreed` printed beside `agrees=True` (the refusal table
listed `("fingerprint-disagreed", not still_agree)`, so it appended exactly when the folds agreed), and four
were a cross-path baseline — every flip leg was judged against `ctx["d_honest_metrics"]`, the headline record,
which in a window is the device arm's `state` ndiff 120259, so the g++ twin's flip printed
`state-row-did-not-move (state_ndiff 120259->85188)` for a flip that moved one word *away* from its own honest
85187. Each leg is now judged against the honest record of the arm that ran it and prints which baseline it
used (`honest=d_out_host.bin`, `d_out_hip.bin`, `d_out_device.bin`), the same correction
`topk-selection-perturbations-redden-ids` needed in window `e2`.

**The state rule.** A control is now in exactly one of three machine-distinguishable states, assigned by
`control_state` at the moment `Verifier.control` prints it: **fired** (`fired=RED` with `legs=` naming the
path it claimed), **not-applicable** (`status=not-exercised reason=<named>` and it claims none of those paths),
**claimed-and-silent** (a finding, named with its sub-shape: `claimed-and-silent`,
`claims-a-path-it-declared-unexercised`, `silent-without-declaration`). Two checks, both in
`check_control_states_and_declaims` and both running after every control has registered:
`control-states-enumerate-and-claimed-and-silent-reddens` enumerates the whole registered population and
reddens on the third state, and `control-claims-do-not-quietly-shrink` compares this run's claim sets against
the sets parsed at run time out of the last recorded log of the same mode — a path may only leave a control's
`claims=` if the line that dropped it prints `declaimed=<path> reason=<named>`, and a blank reason reddens.
Nothing was converted to `not-exercised` to move a count: an unfired control is still a counted finding either
way, which is the clause that makes the marker honest rather than an escape hatch. Both checks carry
fabricated legs (six registrations through the real `Verifier`; five claim-shrink cases) and the baseline parse
is required to see every `CONTROL` line in its source log (35/35 in the CPU baseline) so a guard whose parser
matched nothing cannot pass.

**Evidence.** `specification/evidence/blockdevice-cpu-2026-10-27-p14.log`:
`RESULT mode=cpu checks=73 findings=0 controls=31 fired=31 … band_digest=27a68c7e82c9aa05` (checks grew by the
two new ones; BOUNDWITNESS 38 lines with 36 `inside=yes` and no `inside_no`, six
`at-limit=True one-past=False sized_correctly=yes`, eight `MUSTCATCH name=` RED legs with the derived set equal
to the printed set, `REFUSED=0`, escalation and canonical-command checks PASS), beside
`BLOCKGATE … checks=17 red=0 controls=11 fired=11`, `engine/gate.py` → `RESULT PASS checks=27 findings=0` and
`tools/doc_ref_check.py --strict` → `findings: 0`. Device-shaped, with no device act
(`specification/evidence/blockdevice-device-shape-rehearsal-2026-10-27-p14.log`,
`specification/evidence/blockdevice-control-states-2026-10-27.md`):
`RESULT mode=device checks=77 findings=7 controls=32 fired=28`, rehearsal `verdict=PASS`, and
`control-witness-guard-can-redden fired=RED legs=cpu_predicate:6,device:10 claims=device`. The state check
reddens by name in that rehearsal, which is the required "fabricated claimed-and-silent control went red" seen
on a real population rather than only on a scratch one. `gdn-state-read-…` in the rehearsal keeps exactly one
refusal — `d_hip_device: the arm printed no DREADBACK path=device line` — because a host body is not allowed to
print a device read-back (`engine/device/m2_d_main.inc`), and its three flip legs are the window-`n` numbers
that the two defects had been printing as refusals (host 85187→85188, hip host path 85187→85188, device
120259→120260); the window's own `DREADBACK control-leg=gdn-device-readback ran=yes … match=yes` is the leg
that closes it there. That control's device-run `CONTROL` line is the one line this pass could not print
without touching the iGPU, so it is named as the open item rather than quoted from imagination.

Rule of the pass: a control owes its state in the line, not in the reader's memory of the format — claim a
path and go silent and a check reddens with your name on it; drop a claim and the line has to say why; and a
fabrication must be bent against a base that was clean, or it proves only that the base was dirty. no timing
taken.

## PASS 15 (2026-10-27, CPU-only) — M3 unblocked: one stimulus both oracles can be compared on, a candidate logits fixture, and the logits band put to the user instead of authored

Rule of the pass, unchanged from every earlier one: **no band, floor, tolerance, factor, must-catch set
or pinned fixture was created, relaxed or retyped anywhere here.** The candidate fixture this pass
produces is `status=candidate`, is not registered, and licenses nothing (D-18 reserves registering
pins to the user). `engine/cross_oracle.py` still prints `licence-issued=none` and still refuses to
adjudicate HF-vs-llama disagreement (D-26 clause 3). CPU-only: `grep -c gpu_lock` is 0 on every file
this pass added, the oracle binary is the CPU build gated by `tools/llama_instrument_apply.sh` plus
`tools/llama_tokens_build.sh`, `-ngl 0`, no device act, and no rate or duration appears anywhere in
this section or in the files it cites — host RSS only, read out of the manifests. **no timing taken.**

### 1. What was actually blocking M3, and the one door that was left

`specification/evidence/cross-oracle-three-way-2026-10-27.md` established the blocker: three sides,
three stimuli. HF's chain ran on the 32 sampled ids of `checkpoints/dump-manifest-chain-2026-10-26.json`;
the delivered llama runs (`checkpoints/llama-dump-manifest-2026-10-27-run-a.json`) ran on a
chat-template-wrapped 11-id prefill plus a 4-id assistant header carrying the think-boundary id.
`quantization/evidence/llama-stimulus-alignment-2026-10-27.md` then proved the text door is shut:
detokenizing the 32 ids yields 205 bytes whose re-tokenization reproduces 30 of 32 (first divergence
index 3), and for a byte-level BPE the byte string is forced by the ids, so that candidate is the
*only* candidate. Two exits were named there: re-dump the chain on a tokenizable stimulus (that
replaces pinned fixture material — the user's act, not an agent's), or give the oracle an id input
channel. This pass takes the second.

The vendored front ends have no id channel, and this pass does not patch one into them: **`tools/llama_tokens_dump.cpp` is a client of the vendored library, not an edit of the vendored tree.** Its input path, named because "template-free" has to be falsifiable and not a mood, is:

```
tokens file (decimal ids) -> llama_batch.token[i] (llama_batch_init) -> llama_decode(ctx, batch)
```

It never calls `llama_chat_apply_template` / `common_chat_apply_template`, `common_tokenize`,
`llama_tokenize`, `llama_detokenize` or a BOS insertion, and `tools/llama_tokens_build.sh` proves the
stronger fact from the artifact rather than the source: **`ldd` on the binary shows `libllama-common`
absent** — the library that owns chat templating and tokenization in this tree is not in the process
image, so it cannot run. That gate is witnessed, not asserted: gates 5–7 of `tools/llama_tokens_build.sh`
require `libllama-common` absent from `llama-tokens-dump` **and present in `llama-cli`**, so the check
has to be able to see the thing it is looking for. The dump itself is still `tools/llama-instrument.patch`
arming inside `llama-context.cpp::process_ubatch()` — `engines/llama.cpp` was never made dirty
(`tools/llama_instrument_apply.sh status` at the end of this pass: HEAD `d327bf1f…`, porcelain 0 lines,
patch applied: no).

The front end is a template-free *single ubatch prefill*: `n_ctx 128`, `n_batch 64`, `n_ubatch 64`,
32 tokens → exactly one graph, positions 0..31 (the same positions the HF chain feeds
`position_ids = torch.arange(tokens)` with), `logits[]` marked on the last token only, `load_mtp=false`
so the graph is the 40-block main path, `n_gpu_layers=0`, `no_perf=true` so the library prints no
timings either. `llama_batch_init` `malloc`s `logits[]` and does not fill it, so marking it is not
optional — an uninitialized `logits[]` would have made `n_outputs` garbage and the logits row
meaningless (`llama-batch.cpp::llama_batch_init`).

### 2. The aligned stimulus, and the evidence that it is aligned

Driver: `tools/llama_aligned_run.py run <tag>`. Verbatim lines from the run that produced the fixture
(the ids are printed as numbers, per AGENTS.md §the special-token rule):

```
chain   n_ids=32 token_ids_sha256=f87335bb69b4e413f…9631 encoding=int64-le-per-id recomputes=True
oracle  graphs_captured=1 (cap was 1: one ubatch of 32 tokens is one graph)
oracle  graph 0 n_tokens=32 n_seqs=1 token_ids_sha256=f87335bb69b4e413f…9631
align   token_ids_sha256=f87335bb69b4e413f…9631  chain_manifest_token_ids_sha256=f87335bb69b4e413f…9631
        equal=True (elementwise 32/32, length_equal=True)
template template_applied=no  (API path: file -> batch.token[] -> llama_decode; the binary does not
        link libllama-common …)
template control-class ids (>= 248044) among the 32 ids consumed: none
template witness: the same detector run over run-a graphs[1].token_ids finds 3 control-class id(s)
        [248045, 248046, 248045] — the detector is not vacuous …
template length witness: the template-wrapped prefill carried 11 ids for a 5-token text; this run
        carries 32 ids and every one of them is a chain id, so nothing was added around it
gguf    gguf_sha256=ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61 pinned_sha256=ac0e2c11…
        matches_pin=True
```

Three things about that block are load-bearing and each is a check that reddens, not a sentence:

* **The equality is checked twice over.** The ids the *oracle* recorded consuming come from the
  instrument's own `graphs[0].token_ids` (read from the `llama_ubatch` inside `process_ubatch()`), not
  from the driver's echo, and their sha256 is recomputed under the chain manifest's own encoding —
  sha256 over each id as a little-endian int64, which the driver first verifies against the chain
  manifest's own `token_ids_sha256` field before trusting it. A negative control was run for real:
  feeding the first 4 ids only makes the same check print `equal=False (elementwise 4/32)` and exit
  `verdict=REFUSED tag=token-ids`.
* **`template_applied=no` is proven by construction *and* by evidence**, as required. Construction: the
  API path above plus `libllama-common` absent from the binary. Evidence: no control-class id among the
  ids consumed, with the detector witnessed firing on run-a's template-wrapped prefill, and a length
  argument (a template adds ids; 32 in, 32 out, all of them chain ids).
* **The GGUF digest is read from the artifact and compared to the pin**, `ac0e2c11…53e31a61` =
  `checkpoints/Qwen3.6-35B-A3B.gguf-pin.json`, `matches_pin=true`. This is the D-27 condition: the
  oracle reads the same published artifact our engine reads, so the only freedom left to it is the
  forward pass.

Oracle determinism (D-26 clause 4) is re-established **on the aligned stimulus**, not inherited from
the run-a/run-b pair. Two independent runs (`d3`, `d4`), keyed on
`(graph, block, name, op, dtype, shape, seq)` like `tools/llama_dump_verify.py`, plus one aggregate
digest per run over every written row's sha256, plus a control that runs the identical comparator
against a *different* stimulus so `differing=0` is a property of the oracle and not of the comparison:

```
pair    run d3 rows_written=3639 aggregate_digest=b9f83aa152c1984108d5408aac292f8d3ca8c00bf47298e2d3eca370be72a86c
pair    run d4 rows_written=3639 aggregate_digest=b9f83aa152c1984108d5408aac292f8d3ca8c00bf47298e2d3eca370be72a86c
pair    keys compared=3639 differing=0 only_in_first=0 only_in_second=0 aggregate_equal=True
pair    control: the same comparison against the template-wrapped run-a gives keys_compared=649 differing=229
pair    logits digest equal across the pair=True (n_vocab=248320)
DETERMINISM verdict=DETERMINISTIC pair=d3,d4 tensors_compared=3639 differing=0 aggregate_digest=b9f83aa1…a86c
```

### 3. The candidate fixture (proposed, not registered)

`checkpoints/m3-candidate-llama-fixture-2026-10-27.json`, written by `tools/llama_aligned_run.py
candidate d3 d4`. `status=candidate`, `registration=not-registered`, and it does not touch
`checkpoints/dump-manifest-chain-2026-10-26.json` or any other pinned manifest. Fields: the stimulus
(ids, `token_ids_sha256`, the three-part template-free proof), the oracle (gguf sha + pin + commit +
producing command + instrument-manifest sha), the determinism pair (both aggregate digests, tensors
compared, differing, and the non-vacuity control's numbers), the logits (file, sha, byte count, vocab,
`argmax_id`, top-5, and the definition "argmax over all 248320 logits of the last token, no penalties,
no bad-token mask, no temperature"), the coverage block, and a `not_licensed` block that says no band
came from this file.

Two numbers the gate will eventually be judged on, from the aligned run: `result_output` is
`[248320, 1, 1, 1]`, sha256 `68315e1a0aae1fea3774104d1f3cdab51719fddd98c51e760c1a08fea44b507f`, and the
digest the front end read back through `llama_get_logits` equals the digest the instrument captured for
that row — two different reads of the same graph output agreeing byte for byte. Greedy argmax = id `82`,
top-5 `82:9.12432 7:8.34051 198:8.29955 13:7.93431 62:7.86702`; the margin between first and second
(`≈0.78` in logit units) is printed precisely so a future gate can be told that a near-tie here is *not*
a near-tie, and so nobody has to guess whether the argmax was a coin flip. The digest witness behind
those numbers: `460 of 3639` recorded row digests were re-hashed from the bytes (every written row of
the four HF blocks, every model-level row, every 37th written row), 0 mismatches.

The honest limitation, recorded in the manifest and repeated here: **full-40-block logits/activation
truth exists only as the quantized oracle.** `models/qwen36-src/` holds 4 of 26 safetensors shards
(layers 0, 1, 38, 39), so HF calibrates layer0/layer1 numerically and layer38/layer39 structurally only
(the pinned chain jumps `layer1 → layer38`; see that manifest's `block_set.why_not_contiguous` and
`specification/evidence/cross-oracle-three-way-2026-10-27.md` §Refused adjudications item 4). There is
no bf16 reference for layers 2..37 on this box at all, and there is no HF logits record for any block:
the chain fixture is per-block stages, and `lm_head`/`logits` are in it nowhere (D-25). **So the logits
side of M3 has exactly one candidate referent, and it is a quantized one.**

Coverage, restated as the counts the fixture prints: 40 blocks wrote cross-check tensors
(`counts.files_written=3639`, `bytes_written=2313656464`, one graph, 32 tokens, peak RSS
`33587880 kB` host; `430` cross-check rows carried in the manifest with
name/op/dtype/shape/sha256 per block); layers 2–37 (36 blocks) are cross-check-only; layers 0/1 are the
numeric HF overlap; layers 38/39 are structural-only; `lm_head` has no HF referent.

And the aligned distance between the two oracles, which is what the band question in §6 is really
about. `engine/cross_oracle.py --mode aligned` (22 counted preconditions, then measurement) prints,
for HF bf16 against llama f32 on identical positions:

```
RESULT    mode=aligned checks=23 findings=0 aligned=True escalations=14 bands-authored=none
          licence-issued=llama-crosscheck(observation-only) verdict=ALIGNED-OBSERVED no timing taken
layout  cell=layer0.out relL2[token-major re-view]=2.757e-02 relL2[raw re-view]=1.416e+00
        chosen=token-major margin=51.4x   # the transpose claim is a check, not a sentence
  layer0.x (embedding rows)   max_abs_delta=2.608e-04 relL2=5.367e-03   (bf16 shard rows vs Q8_0 GET_ROWS)
  layer0.h_in                 max_abs_delta=2.630e-02 relL2=5.609e-03
  layer0.moe_out              max_abs_delta=5.528e-03 relL2=7.350e-02
  layer1.h_in                 max_abs_delta=2.919e-01 relL2=2.800e-02
  layer1.moe_out              max_abs_delta=1.039e-02 relL2=8.152e-02
  layer1.out                  max_abs_delta=1.596e-02 relL2=4.043e-02
```

The `layout` line is one of the 23 checks and it exists because the two sides store the same matrix two
ways (HF wrote torch `[tokens, hidden]` in C order, ggml shapes are `[hidden, tokens]` with `ne[0]`
fastest). Both re-views are computed for one cell and the chosen one must be the closer by >5x — a wrong
re-view scrambles element order and lands near √2 relative error, which is 51× worse here. Without that
leg "we transposed" is a sentence, and a sentence cannot fail.

The `layout` line is one of the 23 checks and it exists because the two sides store the same matrix two
ways (HF wrote torch `[tokens, hidden]`, ggml shapes are `[hidden, tokens]` with `ne[0]` fastest). Both
re-views are computed for one cell and the chosen one must be the closer by >5x — a wrong re-view
scrambles element order and lands near √2 relative error, which is 51× worse here. Without that leg
"we transposed" is a sentence, and a sentence cannot fail.

(all 14 cells; the table is the tail of them, full run in
`specification/evidence/m3-aligned-stimulus-2026-10-27.md`). Three readings, none of them a band: the
mapping is corroborated by data rather than names, because `layer1.x` and `layer0.out` agree to the
last printed digit — they are one tensor reached two ways; the disagreement is already present at the
embedding row before any block runs, which is refused-adjudication #1 becoming measurable rather than
settled; and relL2 grows along the stages (~7–8e-2 at `moe_out` on a small-rms tensor) rather than
staying put, which is the quantitative reason a per-block number cannot be extrapolated to a logits
band. `engine/cross_oracle.py` run with no flags is unchanged: `checks=78 findings=0
verdict=STRUCTURAL-OK`, and the only wording change in that path is that the alignment line now says
`unattainable-through-text=yes` **and** `attainable-through-raw-ids=yes`.

### 4. The shape of the comparison, including the two places the topology lies

llama.cpp crops before the last block when there is one output row
(`qwen35moe.cpp` `build_layer`'s `crop_before_nextn(inp_out_ids)` gather at `il == n_layer-1`). In the
aligned run that means block 39's `attn_residual`, `attn_post_norm`, `ffn_moe_*`, `ffn_out` and `l_out`
carry **1 token** while blocks 0..38 carry **32**, and `result_norm`/`result_output` are 1 row. That is
not a defect and not a mapping error — it is llama.cpp choosing to evaluate the last block only at the
output position — but a gate that compares per-(stage, token) ranges must know it, or it will compare a
32-token engine row against a 1-token oracle row and call the mismatch a numerical finding. Two rules
follow, and they belong to M3's gate:

1. **Compare per stage at the token positions the oracle actually wrote.** Blocks 0..38: 32 positions.
   Block 39: position 31 only. `result_output`: position 31 only. A stage whose oracle row has a
   smaller token axis than the engine's is a *coverage* statement, printed with both counts, never a
   silent truncation of the engine side.
2. **Name the activation width on every row.** The oracle's activations are f32 over Q4_K_M/Q8_0
   weights; the HF fixture's are bf16 over the exact shards; our policy-A rows are Q4_K/Q8_0/Q6_K
   (`quantization/evidence/q4km-published-width-map-2026-10-26.md`, D-27's two exceptions). This is the
   refused adjudication in `specification/evidence/cross-oracle-three-way-2026-10-27.md` item 1 and it
   does not get settled by being inconvenient.

### 5. M3's exit gate, as proposed

M3's ladder row is "all 40 blocks + `lm_head`, pinned prompt, pinned logits matched, greedy argmax
exact". The gate is four rungs, run in order, and the run refuses at the first red one.

**G0 — stimulus and identity (preconditions; no numbers compared yet).** Redden unless: the engine was
fed exactly the 32 pinned ids at positions 0..31 and prints them back as numbers with
`token_ids_sha256 == f87335bb…`; the oracle side reports the same digest (already true, §2);
`gguf_sha256 == checkpoints/Qwen3.6-35B-A3B.gguf-pin.json`; the engine's resident set is the same
artifact the oracle read (`engine/loader.py`'s per-tensor digests, not a re-quantize); the oracle's
determinism pair holds (§2/§3); and the engine run is single-tenant and complete — a refused or absent
block is a refusal, not a missing row. `engine/cross_oracle.py align` already implements the stimulus
half of this and returns non-zero when it fails; M3 makes it the first rung instead of a diagnosis.

**G1 — per-block against llama.cpp, layers 2–37 and the parts of 0/1/38/39 HF cannot reach.** For each
of the 40 blocks, the stages the engine emits (`h_in`, `mixer_out`, `x1`, `h_mlp`, `moe_out`, `out` per
§03.2, mapped in `engine/cross_oracle.py::STAGE_MAP` to `attn_norm`, `linear_attn_out`|`attn_output`,
`attn_residual`, `attn_post_norm`, `ffn_out`, `l_out` — `ffn_out` and not `ffn_moe_out`, which is
routed-only). Each row prints
`licence=llama-crosscheck(<what-is-compared>, oracle_sha=<per-tensor digest>, width=f32-vs-policy-A)`
and **never** `floor-probe(...)` (D-26 clause 1). These rows price *disagreement with another quantized
implementation*, so a pass here is `ACCEPTED-BY-CROSSCHECK`, and the aggregate verdict stays `QUALIFIED`
wherever nothing stronger than a cross-check was used. Rows already licensed from HF floor probes
(router, GDN, act-domain) keep their stronger provenance; a cross-check may add a check to them, never
loosen one (the outranking clause).

**G2 — the four HF blocks against HF, under the existing licences.** layer0 and layer1 compare
numerically against the pinned bf16 chain stages with whatever band each stage already carries; layer38
and layer39 print `STRUCTURAL-ONLY(chain gap)` with the reason, and are not counted as numeric passes.
Where HF and llama.cpp disagree on these four blocks, the run prints both sides and stops — it does not
pick. That refusal is `engine/cross_oracle.py`'s existing behaviour and M3 inherits it verbatim; the
`layer0.routed_sum` precedent (prediction 4, `ndiff 1326` vs `0`, argmax moved) is the reason the rule
exists.

**G3 — logits and greedy argmax.** The gate is: greedy argmax **exact** (`argmax_id == 82` for this
stimulus, derived from the fixture rather than typed at the gate), *plus* a magnitude statement about
the logits row. The exact-argmax half is already fully specified and needs no band: it is an
equality check on an integer, and the must-catch set is the one that already exists in spirit — the
`layer0.routed_sum`-class defects (f32 carry instead of cast-first, gate|up pair swap, route-order
error, a skipped renormalisation) must be shown to move the argmax. The magnitude half is the open
question, and it is §6.

**G4 — what the run must print, so the gate cannot pass by being quiet.** Every population derived:
40 blocks × the stages the engine emits = the denominator of G1, printed as `parsed == expected` with a
`missing = expected − |seen|` identity per block; the pairwise numeric denominators as in the three-way
report (`N of M elements compared`, and `0 elements` is a refusal, never a pass); `checks` counted
against an expected count, so a check that stopped running fails; and the terminal line carrying
`no timing taken`. A `0` counter in any of these is reported as vacuous unless a control in the same
run showed it non-zero somewhere it must fire — including the new digest machinery: the aggregate digest
is witnessed by re-hashing 460 of 3639 row files (§2), and the determinism comparator is witnessed
against a deliberately different stimulus.

### 6. THE BAND QUESTION — presented, not answered. **This choice is the user's.**

Logits are the most cancellation-hostile tensor in the model: a `[248320, 1]` row produced by 2048
products per vocabulary row, read from `Q6_K` weights (D-27's narrower-than-policy-A exception), on top
of a 40-block residual stream whose last block the oracle evaluates at one position. D-18 and D-20
forbid a band read off the build, and D-26's outranking clause forbids importing one from llama.cpp's
error budget. So no number is written here. Two derivations are on the table, plus a hybrid; each is
listed with the witness that would have to exist before it could ship.

**Option A — a documented width rule plus an fp64 floor probe (the D-20/D-28 shape).**
Derive, before measuring: the engine's arithmetic contract for the logits path (final RMSNorm form,
`lm_head` GEMV accumulation order and dtype, the width every product is measurably carried in) and the
Higham form that prices it — `γ_{2K−1}` where products are measurably inexact, `γ_{K−1}` where they are
exact, with `K = 2048` (D-20's rule, applied to `lm_head` exactly as it was applied to the `lm_head`
dot). Add the D-28 requirement: propagation from the *input* error the block stream actually carries
(whose size G1 can price per layer) plus a **floor probe** — the same logits row computed in fp64 and
in at least two f32 summation orders, to measure how much of any difference is association rather than
defect. Band = derived bound, and the measurement must be shown to land *inside* it as a separate
witness, never as the definition of it.
*Rejection witness A must produce:* (i) an at-limit / one-past pair on its own lattice, in the shape
`tools` already prints for D-28 (`at-limit=True one-past=False sized_correctly=yes`); (ii) a
geometry-derived must-catch set — a single weight word flipped in `lm_head`'s row for the argmax token,
a gate|up pair swap upstream, an f32-carry renormalisation defect, a swapped expert route — each of which
must land outside the band; and (iii) the refuted-causes leg: if the measured distance is attributable to
a mechanism the entry does not list, the run must not print `QUALIFIED` (D-28's escalation clause).
*Cost, stated honestly:* it needs an fp64 `lm_head` over the aligned stimulus (dequantize `Q6_K`
`output` in host RAM and do a `248320 × 2048` fp64 GEMV, CPU-only), and it prices a bound whose shape is
already accepted. *Why someone might reject it:* a bound that wide may admit every candidate, and a band
that every build passes is a decoration — so leg (i) and leg (ii) are not optional.

**Option B — exact argmax, no magnitude band at all.**
Ship no tolerance for logits. M3 then closes on: argmax exact, plus the ordinal facts around it (the
margin between rank 1 and rank 2, the top-5 ordering), plus byte-identity claims that need no band (the
oracle's own `result_output` digest is reproducible run to run). "Logits matched" is then satisfied as
*decision matched*, and the magnitude rows print `UNLICENSED reason=no-band-authored` — which is
exactly how `engine/block_gate.py` already treats policy-A rows, so it is a familiar verdict:
`QUALIFIED`, never `PASS`.
*Rejection witness B must produce:* a demonstration that the argmax-only rule is not blind. Concretely,
a perturbation class that leaves the argmax fixed while the logits are wrong in a way this project would
call a defect — e.g. a uniform additive offset, or a monotone squash of the whole row, or an injected
error on logits far from the top — must be shown to **pass** argmax-only. If it does (it will), the
option must be paired with a second, cheap, band-free invariant that catches it (digest equality against
the fixture is the only band-free candidate, and it demands the engine reproduce the oracle's arithmetic
bit for bit, which across two quantized implementations is a much stronger demand than the argmax).
Option B fails if that pairing cannot be built; it succeeds if the project is willing to say M3 is a
decision gate and leave magnitude to M5.
*Why someone might reject it:* the ladder row says "pinned logits matched", and B does not match them, it
chooses among them.

**Option C (hybrid, named so it is on the record, not endorsed).** Argmax-exact as the gate (B), and a
magnitude *report* labelled `licence=llama-crosscheck(observation-only)` with no threshold attached,
while the *licensed* magnitude band is derived only on layer0/layer1 from HF's bf16 truth (the one place
an unquantized referent exists) and then carried forward by D-28-style propagation. This is the cheapest
route to "band derived from something that is not the build", but it rests on a propagation claim across
36 blocks that would itself need evidence, and it risks being the option chosen for convenience.

What is **not** on the table: taking llama.cpp's own tolerances (`nmse`, `mean_abs_asymm`, the `1e-4`
form) and calling them ours — D-19's refusal, restated in D-28's drafting note; and "relaxing" any
existing band because a quantized oracle tolerates more, which D-26 explicitly forbids. The choice
between A, B and C changes what M3 can claim, so it is the user's. Until it is made, M3 has G0/G1/G2
specified and G3 half-specified, and that is stated as an open item rather than patched over.

### 7. Files this pass added or changed

| file | what it is |
|---|---|
| `tools/llama_tokens_dump.cpp` | the template-free id-input front end (§1). Reads ids, builds one ubatch, decodes once, writes the logits row, prints argmax and top-5. Refuses an out-of-vocab id, an empty file, a stimulus larger than one ubatch, and a null logits pointer |
| `tools/llama_tokens_build.sh` | compiles it against the instrumented CPU `libllama`, and gates 8 checks including "no ROCm/HIP in ldd" and "libllama-common absent here, present in llama-cli" |
| `tools/llama_aligned_run.py` | the driver: `run` (aligned dump + all §2 evidence, each with its denominator), `pair` (D-26 determinism precondition + non-vacuity control), `candidate` (writes the candidate fixture manifest). `CHECK_ONLY=1` re-checks an existing dump without producing another one |
| `engine/cross_oracle.py` | aligned-stimulus support: a new `aligned` mode that re-verifies every precondition from files (ids, gguf pin, no control-class id, the determinism pair *and* its non-vacuity control, the layout re-view) and then measures the HF-vs-llama distance labelled `licence=llama-crosscheck(observation-only)` with `band=NONE adjudicated=no`; `align` now accepts an aligned manifest via `--llama-manifest` and says `attainable-through-raw-ids=yes`. The default no-flag run is byte-identical to the recorded three-way result (`checks=78 findings=0 verdict=STRUCTURAL-OK`), and the refusal to adjudicate HF-vs-llama is untouched (§8) |
| `checkpoints/m3-candidate-llama-fixture-2026-10-27.json` | the candidate fixture (§3). New file; no existing manifest was modified |
| `specification/evidence/m3-aligned-stimulus-2026-10-27.md` | the narrative evidence file for this pass, with every §2/§3 line quoted verbatim and the 14-cell aligned table |
| `specification/evidence/m3-aligned-run-d3-2026-10-27.log`, `…-d4-…log` | the two aligned dump runs, driver stdout as printed |
| `specification/evidence/m3-aligned-pair-d3d4-2026-10-27.log` | the determinism pair and its non-vacuity control |
| `specification/evidence/m3-cross-oracle-aligned-2026-10-27.log`, `…-align-…log`, `…-baseline-…log` | the aligned-mode run, the `align` re-check against the aligned manifest, and the unchanged 78-check structural baseline |
| `quantization/evidence/llama-stimulus-alignment-2026-10-27.md` | unchanged — it remains the proof that the text route is impossible; this pass is the second door it named, not a correction of it |

### 8. Not proven here, stated so it cannot be cited as done

1. **No engine number exists for M3.** `engine/block_forward.py` is a single-block lane (22 sub-stages,
   one token row, `BLOCK=0 GDN_MODE=decode0`). The 40-block × 32-token prefill G1 assumes does not
   exist yet; G0–G4 are a specification, not a run.
2. **The candidate fixture's bytes are in scratch, not in `data/`.** The byte blobs live under
   `/tmp/m3-align/dumps/<tag>/` with per-tensor digests in the instrument manifest, because `data/` is
   closed to this lane by instruction. Promoting them to `data/dumps/…` is part of registering the
   fixture, and registration is the user's act (D-18). A digest-only candidate is a weak fixture, and
   it is labelled as one; the fixture's manifest names where the bytes are so they can be moved intact.
3. **The HF side was not re-run.** It did not need re-running: the pinned chain *already* used the 32
   ids, which is why the alignment work was all on the oracle side. Any HF re-dump would be new pinned
   material and is therefore out of scope.
4. **`result_output` for a multi-token prefill is one row.** Whether M3 wants logits at more than the
   last position (it does not, for greedy continuation) is settled by the topology in §4, not by us.
5. Nothing here licences anything: no `ACCEPTED`, no band, no must-catch set, no fixture registration,
   and `engines/llama.cpp` is pristine at `d327bf1f…` with the instrumentation reverse-applied.

Rule of the pass: an oracle you cannot feed the same sentence is not a second opinion, it is a second
speaker — so the fix was to build a door for the ids, prove from the link map that the template could
not have run, and prove from a deliberate mismatch that the equality check reddens. And the number that
M3 actually hangs on, the logits band, is the one number this pass was right not to invent. no timing
taken.

## PASS 17 (2026-10-27, CPU-only) — the checkpoint is whole and tied to the Hub's own etags, HF answers all 40 blocks again, and the two agree on the greedy argmax; the registration is prepared and deliberately not applied

D-29's `D→A`, executed. Rule of the pass, unchanged: **no band, floor, tolerance, factor, must-catch set
or pinned fixture was created, relaxed or retyped anywhere here.** `band_digest` is still
`27a68c7e82c9aa05`; no `band_licenses` row moved, and the word `ACCEPTED` appears nowhere in anything
this pass wrote. CPU-only throughout — both new tools assert `torch.cuda.is_available()` is False at
import, `grep -c gpu_lock` is 0 on every file this pass added, `HF_HUB_OFFLINE=1` and
`CUDA_VISIBLE_DEVICES=""` were exported for every run, and the only resource numbers anywhere in this
section are host wall, host RSS and `MemAvailable`. **no timing taken.** The full evidence, with the
raw log of every command, is `specification/evidence/m3-hf-reference-d29-2026-10-27.md` (labels §p1-checkpoint,
§p2-fixture, §p3-crosscheck, §p4-registration, §parent-cited-registration-is-not-in-the-tree,
§reproduce, §what-is-not-shown). Numbering note: this file's previous section is §PASS 15; the gap at 16
is the task's own label and is left as found rather than relabelled.

### 1. The sentence this pass earns, and the one it does not

Earned: **the unquantized reference is no longer 4 blocks deep.** All 26 shards of
`models/qwen36-src/` are on disk, every one of them hashes to the Hub's own LFS OID, and `transformers`
ran the same 32 token ids the pinned chain uses through all 40 blocks and `lm_head`. HF and llama.cpp
agree on the greedy choice: `HF argmax_id=82` (logit 8.875) against `llama argmax_id=82` (logit 9.124),
`n_vocab=248320`. That is the M3 referent in the only shape D-29 licenses.

Not earned: any statement about logit *magnitudes*, and therefore any band. The worst HF-vs-llama
divergence is relL2 3.494e-01 at `moe_out`, which is §10.4's open router-renormalisation question
rather than a new finding, and the logits row differs by max_abs 5.655e-01 (relL2 7.833e-02) while still
argmaxing identically. Measuring a divergence is escalation material; adopting a side is not this pass's
act (D-26 clause 3).

### 2. `tools/checkpoint_integrity_d29.py` — an auditor that imports neither producer nor consumer

Modes `hub-tree` / `build` / `verify` / `witness`; it reads nothing outside scratch and its own output
path. The etag question is the one worth writing down, because an etag is not one thing: of the Hub's 40
tree entries, 27 are LFS-tracked — the 26 shards plus `tokenizer.json` — and for those the etag **is**
`lfs.oid`, i.e. sha256 of the file bytes, so `etag_equals_sha256` is a real equality; the other 13 are
plain git blobs whose etag is a sha1 git-blob hash, recorded under `git_blob_sha1` and never called a
sha256 match. Build: 26/26 `etag_equals_sha256=True`, `bytes == hub size` for all 26, and every count
identity printed with its operands because a bare 0 is a constant predicate and therefore not evidence —
shards `26 - 26 = 0`, tensors `1045 - 1045 = 0`, `sum(tensor_bytes) 71903645408 == index.total_size`,
layers `40 - 40 = 0`, families `{'text': 693, 'visual': 333, 'mtp': 19}` summing to the index's 1045, every
tensor BF16. The file-bytes identity is stated as a subtraction because the two sides differ by the
index's own JSON: `71903776776 - 131368 = 71903645408`. The 4 shards that were already on disk were
re-hashed, not grandfathered.

Four witnesses (`VERDICT witness findings=0`), each asserting both that the verifier's exit code has the
required sign *and* that the victim's name appears in its output — a red verdict naming the wrong shard
witnesses nothing: pristine symlink tree green; one byte flipped inside shard 17's tensor data red by
name; shard 17 deleted from the listing fails the **count** identity (`26 - 25 = 1`) instead of passing
quietly; and a corrupted safetensors header reddens that shard instead of crashing the verifier — that
last one exists because the first version of the witness died inside a hash of a truncated tensor, which
is the difference between an auditor and an error message.

### 3. `tools/hf_full_forward_d29.py` — the referent, and an honest account of what it costs

`Qwen3_5MoeForCausalLM` built meta-first from `checkpoints/Qwen3.6-35B-A3B.config.json` (text tower only:
vision 0.83 GiB and the MTP block 1.57 GiB are not loaded, and D-04 is why), loaded **per tensor** through
`safe_open`, strict bijection 693 checkpoint tensors ↔ 693 module tensors with only the rotary
`inv_freq` buffers having no checkpoint source. Two runs `r1`/`r2` and one deliberate perturbation `p1`:
all three argmax to 82; `r1` and `r2` are **byte-identical across all 282 records**
(`aggregate_digest=eeba4b91c82aef12…`); `p1` — one row of layer 20's `down_proj` expert 7 re-encoded,
512 elements — moves **137 of 282 records and exactly 0 records above layer 20**, which is the witness
that the determinism verdict is not vacuous. The argmax caught none of it, so the argmax is not a
sensitive instrument and no gate may treat it as one (D-26 clause 4).

Memory, both numbers and not the friendly one: 64.56 GiB of text weights were read
(`load shards_read=26 tensors_read=693 bytes_read=69321221376`) while peak RSS was 22,838,968 KB ≈ 21.8
GiB, because mmap-backing never makes the whole tower resident at once; `MemAvailable` was 98,986,917,888 B
≈ 92.19 GiB before the load and ≈91.8 GiB after it, and the tool's own headroom line reads
`headroom_after_text_weights=29,665,696,512 B = 27.63 GiB`. `MemAvailable` is sampled at two points, so
the true trough is between samples and the quoted figure is an upper bound. Fixture:
`checkpoints/m3-candidate-hf-fixture-2026-10-27.json`, **`status=candidate`,
`registration=not-registered`** — 282 records (40 blocks × 7 stages + `normed` + logits row), per-record
sha256, token identity asserted elementwise against the chain
(`token_ids_sha256=f87335bb69b4e413…`, 32 of 32) before the first forward rather than assumed from two
files agreeing on a length.

### 4. `tools/hf_llama_crosscheck_d29.py` — 280 cells derived, 276 measured, 4 refused by name

Standalone: it imports neither fixture's producer and re-declares `STAGE_MAP`, `values_of` and
`token_major` citing `engine/cross_oracle.py` as their source, because a cross-check that shares its
comparator with the thing it checks is not independent. ggml's `[hidden, tokens]` storage is **witnessed
before it is used**: token-major re-view of `layer0.out` gives relL2 2.757e-02 against the raw re-view's
1.416e+00, a margin of 51.4×, and the comparator's own non-vacuity witness scores a same-shape array
with one row of 32 perturbed at 5.631e-01 and an identical pair at 0.0 — so a small relL2 here cannot be
the comparator failing to notice. Worst per stage over the 40 blocks: `x` 3.445e-02, `h_in` 2.081e-01,
`mixer_out` 2.295e-01, `x1` 1.380e-01, `h_mlp` 9.868e-02, `moe_out` **3.494e-01**, `out` 2.570e-01,
`normed` 6.025e-02, logits 7.833e-02 (max_abs 5.655e-01). The 4 refused cells are all layer 39
(`x1`, `h_mlp`, `moe_out`, `out`) and each names its reason: llama.cpp keeps those intermediates only for
its final token, `[2048, 1]` against HF's `[1, 32, 2048]`, so the shapes differ and a delta there would
be element order, not arithmetic. `CROSSCHECK findings=0 cells_measured=276 cells_refused=4
escalations=276 hf_argmax=82 llama_argmax=82 bands_authored=0`.

The unexpected half: HF-full against the **pinned chain**, both bf16, is **byte-identical through
`layer0.h_mlp` (max_abs 0.0)** and then separates at `moe_out` (relL2 3.4e-03), cascading
(`layer0.out` ≈5.4e-03 → `layer1.x` ≈3.2e-03 → `layer1.h_in` ≈1.2e-02), with layers 38/39 at O(1) exactly
as the chain's own `block_set.contiguous=false` predicts. The surviving candidate for the residual is
association order in the MoE block — fixed-slot weighted sum over all 256 experts
(`speccheck/block_ref.py`, the §10.6 form) versus top-k gather-then-scatter (`transformers`) — and it is
reported, not adjudicated.

### 5. The registration: prepared, and stopped at the line

`specification/evidence/m3-registration-d29-2026-10-27.diff` is the patch that would apply the user's
licences —
`d29-hf-bf16-authoritative(prompt=pinned-32-ids, width=bfloat16,
referent=per-block-stage|40-block-argmax|40-block-logits-row)` for the HF fixture,
`d29-hf-bf16-identity(checkpoint=full-checkpoint-manifest-2026-10-27.json,
shared-bytes-with-engine-inputs=none)` for its identity, `llama-crosscheck(observation-only)` unchanged
on the llama rows. **It is not applied.** D-18 makes registering a user act; the working tree still says
`status=candidate` in both fixtures and `git status` still lists the HF fixture as untracked.

What it would buy, measured by applying it in a scratch copy and running both modes there:

| run | structural | aligned |
|---|---|---|
| unpatched | `checks=78 findings=0 verdict=STRUCTURAL-OK` | `checks=23 findings=0 escalations=14 verdict=ALIGNED-OBSERVED` |
| patched | `checks=78 findings=0 verdict=STRUCTURAL-OK` (byte-identical output) | `checks=287 findings=0 escalations=276 verdict=ALIGNED-OBSERVED` |

262 comparison cells that could not be measured before become measurements; the structural mode does not
move by one check. `HF_COMPARABLE` stops being the typed pair `("layer0", "layer1")` and is derived from
the fixture's own records, because a typed pair would keep asserting in code that layers 2..37 have no
unquantized referent, which the checkpoint now contradicts. The fixture preconditions were **reworded,
not loosened**: they tested the word `candidate`, they now test the licence, and a fixture carrying some
other licence or none still refuses — a registration that accepted anything would be the loosening it
looks like.

### 6. The registration the task said already existed does not

The task stated the HF reference was already registered as `m3-candidate-hf-fixture-2026-09-27.json`,
`status=registered-by-user`. **It is not in this repo** — no filename in `checkpoints/` contains `09-27`,
no commit mentions it, the string `registered-by-user` appeared in no file that this pass did not write,
and HEAD is `2baa2a7`, the D-29 decision commit. The commands and their outputs are in
`specification/evidence/m3-hf-reference-d29-2026-10-27.md §parent-cited-registration-is-not-in-the-tree`.
This pass did not assume the date was a typo for `2026-10-27`; it wrote its own candidate and everything
here is measured against that. If a registered HF fixture exists outside this working tree, the two must
be reconciled by digest (`aggregate_digest=eeba4b91c82aef12…`) before either is judged against.

### 7. What this pass refuses to claim

No band was derived, licensed or widened; the relL2 table is a measurement list and a band needs a floor
probe plus a geometry-derived must-catch set (D-20), and the logits band is still the user's question
(P-09). `speccheck/` did not produce this fixture — `speccheck/block_ref.py` remains the only place the
§10.6 MoE form is pinned, and §4 above says the gap to `transformers` is not yet attributed. The
`MemAvailable` minimum is a sampled upper bound. Nothing under `models/` was written by this pass, and it
does not name the actor that placed the 22 shards there (their mtimes predate this pass's first artifact);
what it can say is that they hash to the published checkpoint's own LFS OIDs. And the four gates that
must not move did not: `RESULT mode=cpu checks=75 findings=0 controls=31 fired=31 counters_unparsed=0
band_digest=27a68c7e82c9aa05` for `engine/block_device.py` (no `--device`, so no device act),
`BLOCKGATE verdict=QUALIFIED checks=17 red=0 controls=11 fired=11 band_digest=27a68c7e82c9aa05` for
`engine/block_gate.py`, `RESULT PASS checks=27 findings=0` for `engine/gate.py`,
`checks=78 findings=0 verdict=STRUCTURAL-OK` for `engine/cross_oracle.py`, and `doc_ref_check --strict`
findings: 0 with everything this pass wrote in place.

Rule of the pass: an oracle that can only answer about 5 % of the model is not a second opinion, it is a
third speaker with the same blind spot as the first — so the work was to make the whole checkpoint
attributable before trusting a single one of its numbers, then to refuse twice: once to adopt a side in
the router disagreement, and once to apply a registration the user had not applied. no timing taken.

---

## PASS 18 (2026-10-27, CPU-only, **written mid-pass as a checkpoint**) — M3 leg 1: the engine speaks 40 blocks in its own voice, and a second dequantizer is put under authority before it is trusted

This section is a **checkpoint**, written before the 40-block sweep it exists to serve, because the
deadline is the failure mode (AGENTS.md §GPU execution: 21 of 73 recorded runs died at the run deadline,
most still printing). Everything in §1–§4 is built and checked; §6 is what is still open at the moment of
writing, and §7 says what is *not* claimed.

### 1. The lane: `engine/m3_forward.py` produces, `engine/m3_gate.py` adjudicates, and neither can borrow the other's inputs

`engine/m3_forward.py` runs embedding → 40 decoder blocks → final norm → `lm_head` for the pinned 32
ids and emits the HF fixture's own stage set — `x, h_in, mixer_out, x1, h_mlp, moe_out, out` per block
plus `normed` and the last-token logits row = **282 records**, f32 files with a per-record sha256 and an
`aggregate_digest` over the sorted `name:sha` lines (the same construction the fixture uses, over
basenames so two runs are comparable). It reads the **shipping artifact**
`models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` through `engine/loader.py` addresses under the audit
selftest token, and nothing else: no `transformers`, no HF shard, no fixture byte.

Independence is a checked property, not a comment. The forward installs a `Ledger` around `builtins.open`
for the compute phase, so every path it opened is on the record and the gate can audit that list against
both candidate fixtures' record lists; `--peek PATH` exists so the gate can make that counter **fire**
rather than read a 0 and hope (`pitfalls/40` §a-counter-that-reads-0). The static half — no `transformers`,
no producer tool, no gate import — is asserted at import and printed. The comparator is a separate file
that reads the fixtures and never the other way round.

Every step is cited to its spec file where it is used: §03.1 Form A with no `+1` because the published
file already carries `1+w` (§03.1's artifact rule), §04/§04b chunked delta scan (`C=64`, `-inf` masked
**before** the `exp`, unitriangular UT solve, pads zero-filled so a pad column writes nothing), §05.1's
*interleaved* q|gate head split and per-head QK norm before rope with `R = int(256×0.25) = 64`, §11's R1–R5
including the two casts that change values rather than precision (router logits to the activation dtype,
renormalised top-k weights cast back before the multiply — AGENTS.md prediction 4), and the routed experts
accumulated in **ascending expert id**, not rank. `torch.cuda.is_available()` must be False or the file
refuses; no device call, no lock, no `ms`/`tok/s`/`GB/s` anywhere, host RSS and `MemAvailable` only.

### 2. The second dequantizer, and the trap it walked into

`tools/gguf_fast_dequant.py` exists because the audited decoder walks blocks in Python (~16.85 M
elements/s) and this pass dequantizes ~10⁹ elements per block. That is precisely the
`pitfalls/40` §two-self-consistent-layouts shape, so the answer is not "trust it, it is faster":

* **Authority is stated and enforced.** `tools/gguf_reader.py::dequant` is authoritative — it is the
  decoder `engine/block_forward.py::WeightSource.get` uses on the M2-gated policy-A arm. `engine/loader.py`
  has **no** decoder (addresses only), so "the loader's path" is `loader.py` addresses + `gguf_reader.dequant`.
  The fast decoder is admitted per run only by `Artifact.verify()`, which re-decodes every tensor this
  pass reads and compares bit-for-bit: **unquantized families in full** (norms, router, GDN controls — the
  never-quantized bucket, where a 1-ULP difference is not a rounding difference, so block-sampling them
  would be the wrong strength) and quantized families on first/middle/last block, indices by arithmetic
  from the record. A nonzero `ndiff_bytes` raises `Refused("dequant-disagrees")` and the block stops.
  One line per tensor: `M3DEQUANT-CROSS tensor=<name> fast_vs_loader ndiff_bytes=<n> width=… sample=… blocks=… elements=…`,
  routed-expert slabs grouped per `(layer, role)` with `named_first`/`named_last` so no line is nameless,
  and a trailer carrying `reads`, `tensors_verified` and `reads_verified_equal`.
* **The equivalence leg was too weak and it hid a real bug.** The first `--xcheck` compared single blocks
  only. At `nb=1` a 5-D advanced-index result in the Q6_K path broadcast against a 4-D scale and produced
  a correct-looking 256-element answer; at `nb>1` the same code asks for a `(nb, nb, 2, 4, 32)` temporary —
  32 TiB on the first real multi-block call. Fixed (index vectors carry no leading singleton; shapes are
  pinned), the silent `vals.reshape(-1)[:n]` truncation became a hard `vals.size != n` error, and
  `--xcheck` now compares multi-block runs (up to 64 blocks) per sampled tensor plus a `BADSHAPE` leg.
  Re-run after the fix:
  `XCHECK types_compared=5 of_stored_types=5 (F32,Q4_K,Q5_K,Q6_K,Q8_0) spans=143 elements_compared=503088 bit_mismatches=0 verdict=BIT-IDENTICAL-TO-AUDITED-READER no timing taken`,
  with `control_seen_to_differ=5/5` per type. The lesson is the repo's own: an equivalence check whose
  sample is smaller than the code's failure mode is a control that passes for the wrong reason.
* Neither decoder reads fixture bytes: both take bytes only from `gguf_reader.Gguf.read` on `loader.py`
  addresses, and the ledger proves it for the pass.

### 3. Cost, measured instead of guessed

One block cost ~4–6 min of host CPU, which does not fit a 40-block sweep twice over (determinism pair plus
a perturbation leg). `M3COST block=<n> dequant_s= gemv_s= verify_s= other_s=` lines were added (host wall
only, explicitly not a device timing, no rate derived) and a `cProfile` pass over one block is the
diagnosis. Whatever it names, the two legitimate levers are a dequantized-span cache keyed on
`(name, expert, type, file_off, length)` of the pinned artifact, and running the sweep in chunks of a few
blocks with the residual stream carried between chunks on disk, so a deadline kills a chunk and not the
run. What is **not** legitimate is trimming the pass: no block may be skipped, no weight substituted, and
`--limit-blocks` prints `M3-BLOCKED role=… reason=… blocks_completed=… missing=…` for the honest prefix
rather than pretending (§PASS 15's rule that a refusal is a counted state).

### 4. Non-regression, re-run with this pass's files in the tree

`engine/block_device.py` (CPU): `RESULT mode=cpu checks=75 findings=0 controls=31 fired=31
counters_unparsed=0 band_digest=27a68c7e82c9aa05`. `engine/block_gate.py`: `BLOCKGATE verdict=QUALIFIED
checks=17 red=0 controls=11 fired=11 band_digest=27a68c7e82c9aa05`. `engine/gate.py`: `RESULT PASS
checks=27 findings=0`. `engine/cross_oracle.py --mode structural`: `checks=78 findings=0` with
`bands_authored=0 licence-issued=none`. `tools/hf_llama_crosscheck_d29.py`: `findings=0 cells_measured=276
cells_refused=4 escalations=276 hf_argmax=82 llama_argmax=82 bands_authored=0`. The band digest has not
moved, which is the point: this pass adds a producer and a comparator, not a band.

### 5. What the band PROPOSAL will be, and what it may not be

No band, floor, tolerance or licence is authored in this lane (D-18, D-20). The forward prints **operands**
only, per stage family, at probe layers `{0, 20, 39}` and for `lm_head`: `K`, stored width, `sum_abs_x`,
`quant_term_mean` = `Σ_j |x_j|·half_j` from the stored block scales (the grid half-width `dequant_np`
derives from its own scales — a width statement read out of the file, not a tolerance),
`round_term_mean` = `γ_{2K−1}·Σ_j |x_j w_ij|` with Higham's `γ_n = nu/(1−nu)`, `u = 2⁻²⁴`, and the
association floor measured by re-running the same dot in float64 (`assoc_max_abs_delta`, `assoc_relL2`).
Stages whose defining op reads no quantized byte (both residual adds, all four Form-A norms, the embedding
gather) report `quant_term_mean=0` with the reason named, because their width floor is exactly zero and
saying so is the honest operand list, not a free pass. The gate turns those into a bound marked
`proposal=awaiting-main-agent`; layers 2–37 get no licence from me.

### 6. Open at the moment of writing

The 40-block sweep has not completed, so `M3DIST`, `M3ARGMAX`, the determinism pair and the perturbation
witness are **not yet emitted**, and no distance number is quoted anywhere in this file. `engine/m3_gate.py`
and the candidate checkpoint for this lane are due with those lines. The first smoke run reaches the norms,
GDN scan, MoE and `lm_head` structurally but died inside the operand formatting; that is fixed and the
profiling run is what is in flight.

### 7. What this pass refuses to claim

It has not shown the engine agrees with anything yet — it has shown the engine can *say* 282 things from
the shipping bytes, with the byte decoder pinned to an authority and a ledger that can catch a peek. It
does not author a band, does not license a row, does not register a fixture, does not touch the device, and
does not quote a rate. `no timing taken`.
