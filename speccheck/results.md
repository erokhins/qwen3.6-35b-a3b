# `speccheck/` — the CPU MoE oracle: what it asserts, what it caught, what it does not cover

**Provenance.** Written 2026-10-26 (box clock 2026-10-25T12:00Z) to close the artifact owed since
commit `cb78950` ("Owed, and recorded rather than waived: `speccheck/results.md`, the narrative that
ties this to `specification/10-reference-dumps-and-gates.md`"), which a worker lost to a run deadline
left unwritten. **Two kinds of number appear below and they are labelled as such.** Quoted figures are
read verbatim from a capture under `speccheck/evidence/`, named beside them. Derived figures are sums or
products of quoted ones, with the factors given — the per-block float-stage population **803,104** is
`8192+8192+256+524288+65536+65536+32+65536+65536`, the nine `ndiff=` denominators printed in
`speccheck/evidence/cross-check-2026-10-26.txt` for one block; **3,212,416** is that × 4 blocks; and
**128/128** top-k agreement is the capture's `ordered 32/32` × 4 blocks. Neither kind is recalled from a
prior report, and a derived figure should be re-computable from its stated factors without opening the
script that produced it. **No device work and no clock are involved anywhere in this document**:
every instrument cited here is torch-CPU and prints counts, ratios and norms only, and each capture
ends with its own `no timing taken`.

**Who this is for.** A reader who has never seen this lane and has to decide one question: *can I use
this oracle as the bf16 authority for a MoE kernel or a quantization gate?* §Findings answers it,
§Limits bounds it, and §Contradictions names the five things that would make a careful reader stop.

**Citation form.** `path §label`, per `pitfalls/60-docs-and-citations.md`; the evidence captures are
under `speccheck/evidence/` and are named by filename.

---

## §In one paragraph

Two implementations of the MoE block are run over identical pinned inputs on CPU: ours
(`speccheck/moe_ref.py`, written from the committed config and shard index, no HF import) and the
installed `transformers` **5.19.0** eager `Qwen3_5MoeSparseMoeBlock` (driven by
`speccheck/cross_check_hf.py`). Over 4 blocks × 9 float stages they agree on **3,212,416 elements
with `ndiff = 0`, `maxabs = 0.000e+00`, `>1ulp = 0`**, and top-k ids are ordered-equal on **128/128**
rows. The gate (`speccheck/selftest_moe.py`) is a three-arm matrix — honest / stub / crossed — whose
green state was produced by derivation, not by moving an expectation: the 192-vs-184 element-count
dispute is spelled out in §The 192-vs-184 dispute and is the case study for *derive the population,
never type it*. The oracle reproduces all 40 pinned ref-arm dumps bit-exactly, and 104/104 dump files
still hash to the manifest. What it does **not** certify is long: no quantized path, no attention/GDN,
no norms, no token-level decisions, 4 of 41 blocks, and a tie-break
(`specification/ASSUMPTIONS.md` A-03) that both sides inherit from the same `torch.topk` call.

---

## §Pins, and how they were produced

Everything a re-run needs is pinned by name, count and command line. Nothing here is "the current
build" — the manifest records the interpreter, including its thread count, because CPU matmul
reduction order follows the thread partitioning (`speccheck/moe_ref.py` §set_threads).

| pin | value | recorded in |
|---|---|---|
| geometry source | `checkpoints/Qwen3.6-35B-A3B.config.json` — `E=256`, `top_k=8`, `I=512`, `shared_I=512`, `H=2048`, 40 layers + 1 MTP block ⇒ **41 MoE blocks**, expert = **6,291,456 B = 6.0 MiB** bf16 | `speccheck/evidence/cross-check-2026-10-26.txt` header; `speccheck/moe_ref.py` §Geometry |
| weights on disk | 4 of 26 shards acquired (`model-00001/00002/00025/00026-of-00026`) under `models/qwen36-src/`; the index names 1,045 tensors across 26 shards, of which a 4-shard view keeps **424** | `models/qwen36-src/`; `checkpoints/Qwen3.6-35B-A3B.safetensors.index.json`; `speccheck/evidence/hf-capability-2026-10-26.txt` `POP truncated_index_population` |
| input fixture | 32 tokens, **seed 20261026**, bf16; fp32 `torch.randn` from a private `torch.Generator('cpu')`, each row rescaled to RMS exactly 1.0, then cast | `checkpoints/dump-manifest-2026-10-26.json` `input.method`; `speccheck/moe_ref.py` §Input generation |
| interpreter | `/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python`, python 3.12.3, **torch 2.14.1+cpu**, **8 threads** | `manifest.interpreter`; re-checked this pass |
| reference library | **transformers 5.19.0** (re-checked in that venv; also recorded in commit `cb78950` and in `speccheck/cross_check_hf.py` §docstring, and consistent with `specification/ASSUMPTIONS.md` A-02). *The brief that dispatched this document said 5.13.0; the recorded pin is 5.19.0 and 5.13.0 appears nowhere in the lane.* | `speccheck/evidence/hf-capability-2026-10-26.txt`; commit `cb78950` |
| dump pins | `data/dumps/2026-10-26/`, **104 files** — arms `ref` (40 files / 3,442,816 elements), `spec` (16 / 2,622,464), `exact` (28 / 2,918,400), `stored` (20 / 14,688,256); `counts.dumps == counts.expected == 104` | `checkpoints/dump-manifest-2026-10-26.json`; `speccheck/evidence/pins-reverified-2026-10-26.txt` |
| selftest pins | same seed, `--threads 8`, blocks `layer0,layer1,layer39,mtp0`, `--tokens 16`, `--softmax-tokens 64` | `speccheck/evidence/selftest-2026-10-26-w8.txt` header |

Two consequences of the pin set that a reader must not miss:

1. **The interpreter binary lives outside this repo.** `AGENTS.md` §Self-containment lets you *look*
   at `../qwen3.5-0.8b`; the facts (version strings, thread count, seed) are copied above, but a
   session that cannot reach that path cannot re-run the lane. `.gitignore` already reserves a
   repo-local `.venv/` ("the CPU oracle venv; W4 may create `./.venv`") and none was created.
2. **The pins were cut by the oracle itself.** `tools/dump_reference.py` imports `moe_ref.py` and
   writes its `arm=ref` output as the reference dump — refusing to write a byte unless the HF
   reference agrees exactly on `topk_ids` and `routed_sum`, and within the structural bug detector
   elsewhere (recorded in `manifest.agreement_at_write_time`). So reproducing the dumps proves
   determinism and integrity, not independence; §Limits states what independence actually rests on.

---

## §Findings

### §F1 — per-stage agreement against the installed reference

Driver: `speccheck/cross_check_hf.py --blocks layer0,layer1,layer39,mtp0 --tokens 32` (arms
`ref,spec,exact,cast_after,softmax_bf16,no_renorm`), capture
`speccheck/evidence/cross-check-2026-10-26.txt`, ending `CROSS_CHECK PASS`. Coverage line as printed:
`printed comparison records 240, expected 240 (4 blocks x 6 arms x 10 stages)` — 9 float stages plus
one discrete `topk_ids` line per (block, arm). A parse of those 240 lines accounts for all of them
(216 float records + 24 discrete), so no row of the table below is a subset passed off as the whole.

**The claim-carrying arm is `ref`** — our `Form()` (the form we assert the pinned reference takes)
against the HF eager module:

| stage | population per block | elements over 4 blocks | maxabs | 1−cos | ndiff | >1ulp |
|---|---|---|---|---|---|---|
| `router_logits` | `[32, 256]` bf16 → 8,192 | 32,768 | 0.000e+00 | 0.000e+00 | 0 | 0 |
| `router_probs` | `[32, 256]` f32 → 8,192 | 32,768 | 0.000e+00 | ≤ 6.328e-15 | 0 | 0 |
| `topk_weights` | `[32, 8]` bf16 → 256 | 1,024 | 0.000e+00 | ≤ 2.220e-16 | 0 | 0 |
| `expert_raw` | `[32, 8, 2048]` bf16 → 524,288 | 2,097,152 | 0.000e+00 | 0.000e+00 | 0 | 0 |
| `routed_sum` | `[32, 2048]` bf16 → 65,536 | 262,144 | 0.000e+00 | 0.000e+00 | 0 | 0 |
| `shared_raw` | `[32, 2048]` bf16 → 65,536 | 262,144 | 0.000e+00 | ≤ 2.220e-16 | 0 | 0 |
| `shared_gate_prob` | `[32, 1]` bf16 → 32 | 128 | 0.000e+00 | ≤ 2.220e-16 | 0 | 0 |
| `shared_out` | `[32, 2048]` bf16 → 65,536 | 262,144 | 0.000e+00 | 0.000e+00 | 0 | 0 |
| `combined_out` | `[32, 2048]` bf16 → 65,536 | 262,144 | 0.000e+00 | ≤ 2.220e-16 | 0 | 0 |
| **all float stages** | 803,104 per block | **3,212,416** | **0.000e+00** | — | **0** | **0** |

Ten of those 36 `ref`-arm rows carry a non-zero `1−cos` (five stages show a non-zero maximum, the
largest being 6.328e-15 on `layer39/router_probs`) and the cause is the comparator itself: it widens
both sides to f64 with an explicit `.double()`, where the dot's reduction order differs from the f32
one, and two bit-identical tensors then disagree by ~1e-15 in the cosine. The clamps are printed rather
than hidden (`speccheck/moe_ref.py` §metrics clamps `1−cos` at 0 rather than printing a negative). Bit-
equality, not closeness, is the verdict: `ndiff = 0` over 3,212,416 elements. The per-arm totals in this and the
last table of this section are sums over the named capture lines and are reproducible with `awk` over
that file; no hidden instrument produced them.

`expert_raw` is the stage a naive check gets wrong. The HF expert loop multiplies the routing weight
*inside* the loop and never returns the pre-weight tensor, so the HF side of that comparison is
`speccheck/cross_check_hf.py` §recompute_expert_raw — HF's own ops (`F.linear(...).chunk(2,-1)`,
`ACT2FN["silu"]`, `down_proj`) with the weight multiply omitted. That is a definition, not a witness,
so the pins carry their own guard: `manifest.agreement_at_write_time` records
`recompose_routed_sum_bitexact: true` and `raw_vs_post_weight_elements_differ: 524288` for all four
blocks — `expert_raw` really is pre-weight (multiplying it by the pinned weights rebuilds `routed_sum`
bit-for-bit) and really is not the post-weight tensor (`pitfalls/30-numerics.md` §"A hook that reads
`output` can pin the answer under the name of an intermediate").

### §F2 — top-k selection is a discrete check, and its denominator is named

`topk_ids` is compared by exact equality, ordered and as-a-set, never with a band:

| line from the capture | value |
|---|---|
| `top-k id rows checked 128, set-disagreements 0` | 4 blocks × 32 rows, the reference check |
| `ref` / `spec` / `cast_after` / `no_renorm` ordered | 32/32 on all four blocks |
| `exact` ordered (f64 router) | layer0 26/32, layer1 25/32, layer39 31/32, mtp0 24/32 |
| `softmax_bf16` ordered | layer0 31/32, layer1 30/32, layer39 25/32, mtp0 18/32 |
| selection disagreement by arm (rows whose top-8 **set** differs) | `ref` 0/128, `spec` 0/128, `cast_after` 0/128, `no_renorm` 0/128, `exact` **6/128 (4.69 %)**, `softmax_bf16` **6/128 (4.69 %)** |

Two arms therefore change *which experts run* on 4.69 % of rows: the f64-router arm (whose logits are
computed at a different width) and the bf16-softmax arm. That is the near-tie population showing
itself, and it is why every `expert_raw` figure carries its denominator — `expert_raw` is compared only
on rows whose top-8 set agrees, and that population differs by arm:

| arm | `expert_raw` rows compared, per block | `expert_raw` elements compared |
|---|---|---|
| `ref`, `spec`, `cast_after`, `no_renorm` | 32/32, 32/32, 32/32, 32/32 | 2,097,152 |
| `exact` | 32/32, 29/32, 31/32, 30/32 | 1,998,848 |
| `softmax_bf16` | 32/32, 32/32, 30/32, 28/32 | 1,998,848 |

so the per-arm element totals are **3,212,416** for the four full arms and **3,114,112** for `exact`
and `softmax_bf16`. Reporting "0 of 3,212,416" for `exact` would be reporting a denominator it never
compared (`pitfalls/40-gates-and-admissions.md` §"A requirement written after the object it constrains
is a note, not a forcing" — the unnamed-denominator family). The comparison also canonicalises the slot axis
(`speccheck/cross_check_hf.py` §canon_raw, §rows_same_set): `expert_raw` is indexed by *slot*, slot
order is the top-k rank order, and a tie permutation is not an arithmetic error.

The other arms are in the same capture and are the sanity check that the comparator can see a real
difference at all — summed over each arm's 4 blocks × 9 float stages:

| arm | purpose | elements | ndiff | >1ulp | maxabs |
|---|---|---|---|---|---|
| `ref` | **the claim** | 3,212,416 | **0** | **0** | **0.000e+00** |
| `spec` | our own math form (f32 accumulate, rank-order addends, per-slot GEMM, weights kept wide) | 3,212,416 | 3,146,845 | 32 | 8.766e-03 |
| `exact` | the error reference: same math, no bf16 rounding anywhere (f64 router, f64 accumulate) | 3,114,112 | 3,114,112 | 19,077 | 7.036e-02 |
| `cast_after` | one-switch arm: routing weight cast **after** the multiply | 3,212,416 | 176,548 | 4 | 9.766e-04 |
| `softmax_bf16` | one-switch arm: softmax taken in bf16 instead of f32 | 3,114,112 | 250,382 | 21,148 | 1.670e-01 |
| `no_renorm` | deliberate error: renormalisation over the K omitted | 3,212,416 | 524,347 | 450,044 | 2.952e-01 |

`spec` differing on ~98 % of elements while `>1ulp = 32` is the expected shape of two legal bf16
implementations: nearly every element moves, almost none by more than one ULP of the reference max. Two
of the arms also carry their intermediates wider than the reference does — `spec` keeps its routed
stages in f32 and `exact` computes them in f64 (the manifest stores the `_exact` arms downcast to f32),
where the reference rounds each stage to bf16 — so `ndiff` ≈
100 % is arithmetic talking, and `>1ulp` (counted against one bf16 ULP at the reference's own max, the
aggregate `pitfalls/30-numerics.md` §"A max-ULP band cannot see a whole-tensor 1-ULP error, and `ndiff`
cannot see a chain" asks for) is the column that means something.
`no_renorm` is the opposite shape (450,044 elements over 1 ULP; `1−cos` up to 4.841e-01 on
`combined_out` at mtp0), which is what an actual defect looks like. The structural bug detector that
separates the two shapes is `BUG_DETECTOR_ONE_MINUS_COS = 1e-2`, derived in
`speccheck/cross_check_hf.py` §module header (bf16 relative width 2⁻⁹), applied only to the legal-form
arms and never to the deliberate-error arms.

### §F3 — the pinned dumps: integrity and reproduction (green, re-captured this pass)

The only against-dumps capture that shipped was **red**, and it was red for a reason belonging to the
checker, not the dumps: its 40 failure lines read `dump shape/dtype (32, 2048)/torch.bfloat16 !=
recomputed (32, 2048)/torch.bfloat16` — identical on both sides, an artifact of an intermediate
revision of `check_against_dumps` that was never committed
(`speccheck/evidence/selftest-against-dumps-2026-10-26.txt`, kept as history). The current code
separates shape-vs-manifest, recomputed-shape-vs-manifest, dtype, and bit-equality, and prints metrics
on a value mismatch.

Re-run with committed instruments, condensed here and verbatim in
`speccheck/evidence/pins-reverified-2026-10-26.txt`:

```
$ tools/dump_reference.py --verify --out data/dumps/2026-10-26 --manifest checkpoints/dump-manifest-2026-10-26.json
# verify: 104 manifest records, 104 verified, 0 bad, expected count 104
VERIFY PASS                                                     (exit 0)

$ speccheck/selftest_moe.py --against-dumps
## arm=dumps: 40/40 pinned ref tensors reproduced bit-exactly from the oracle
SELFTEST PASS (against-dumps only). no timing taken             (exit 0)
```

40/40 means all 3,442,816 elements of the ref-arm dumps — `input`, `router_logits`, `topk_ids`,
`topk_weights`, `expert_raw`, `routed_sum`, `shared_raw`, `shared_gate_prob`, `shared_out`,
`combined_out` × 4 blocks — are reproduced by `speccheck/moe_ref.py` today, at `torch.equal`.
`pitfalls/60-docs-and-citations.md` supersedes by naming rather than deleting: this capture supersedes
the red one and both filenames are recorded in the evidence file.

### §F4 — what the installed reference can and cannot do with this checkpoint

`speccheck/hf_capability_probe.py`, run in a subprocess with `RLIMIT_AS = 24 GB` so an attempted 66
GiB materialisation dies inside the probe rather than swapping the box, and driver-counted: the
driver re-derives the probe's own `show("…")` names from the file and refuses to continue unless all
10 printed (`pitfalls/40-gates-and-admissions.md` §"A driver whose parser matches nothing prints PASS").
Capture `speccheck/evidence/hf-capability-2026-10-26.txt`, return code 0:

| probe | verdict |
|---|---|
| `config_from_json` | OK `Qwen3_5MoeConfig` |
| `build_conditional_generation_meta` | OK, **1,026 params** (meta device) |
| `build_text_causallm_meta` | OK, **40 layers** |
| `mtp_class_in_module` | **OK, `NO_MTP_CLASS_IN_MODULE`** |
| `keys_to_ignore_on_load_unexpected` | `(['^mtp.*', '^model.visual.*'], ['^mtp.*'])` |
| `default_experts_implementation_bare_text_config` / `_nested` | `None` / `None` |
| `experts_implementation_after_model_build` | `bare=None after_build=grouped_mm` |
| `from_pretrained_real_index_26_shards` | **FAIL** `FileNotFoundError: model-00003-of-00026.safetensors` |
| `from_pretrained_truncated_index_4_shards` | **FAIL** `DefaultCPUAllocator: can't allocate 1073741824 bytes` (population: kept 424 of 1,045 tensors in 4 shards) |

Three consequences, all load-bearing for how the cross-check is built:

1. **The whole model cannot be loaded here.** Both honest `from_pretrained` attempts fail — one on the
   22 shards never downloaded, one on memory. The lane therefore instantiates the reference's
   *`SparseMoeBlock`* and replaces every parameter with the tensor read from the shard (on the meta
   device first, to avoid allocating 1.07 GiB of `[256, 1024, 2048]` scratch per block before
   overwriting it).
2. **The dispatch key must be pinned.** A freshly built config reports `_experts_implementation =
   None`, and after a model build it is `grouped_mm` — an unpinned `from_pretrained` model would *not*
   run the eager expert loop this spec mirrors. `speccheck/cross_check_hf.py` §build_hf_block sets
   `cfg_text._experts_implementation = "eager"` explicitly and asserts the module's parameter-name set
   equals the seven names it loads, so a `transformers` upgrade that renames or adds a parameter breaks
   the gate instead of silently judging a different binary.
3. **There is no HF MTP class at all** (`NO_MTP_CLASS_IN_MODULE`, and the loader is told to ignore
   `^mtp.*`). The `mtp0` MoE sub-block has an independent reference because it is the same
   `SparseMoeBlock` fed with `mtp.layers.0.*` weights — the draft *layer* around it (fc, pre-fc norms,
   the accept test) has none. Same fact `specification/ASSUMPTIONS.md` A-01 records from the other side.

Separately — and this is the router's load-bearing dtype fact — the driver spies on `F.softmax` while it
calls the reference block once per block, and refuses to continue if the block makes anything other than
exactly one softmax call (`speccheck/cross_check_hf.py` §build_hf_block: "the reference block made N
softmax calls in one forward; the capture below is stale"). What it printed, once per block and again for
the decode-shaped call:

```
reference softmax call: args=1 kwargs={'dtype': 'torch.float32', 'dim': '-1'} -> dtype torch.float32, 8192 elements
reference softmax call: args=1 kwargs={'dtype': 'torch.float32', 'dim': '-1'} -> dtype torch.float32, 256 elements
```

8,192 = 32×256 (the pinned T) and 256 = 1×256 (the decode-shape call of §F5). That is
`specification/11-moe-block.md` §11.2 R2 (softmax over all 256 logits in F32, not softmax-of-bf16)
observed at the source rather than inferred from the dump. `router_probs` is **not** in the manifest's
dump list, so this capture plus the bit-exact `router_probs` comparison in §F1 is the whole of R2's
evidence; see §Contradictions item 3 and §Limits item 10.

### §F5 — host-synchronising calls in the reference's MoE forward (prediction 1, structural half)

`speccheck/moe_ref.py` §SyncCounter wraps `torch.where`, `torch.nonzero`, `Tensor.nonzero`,
`Tensor.item`, `Tensor.tolist` and `Tensor.__bool__`, counts, and never reads a clock — so the
structural half of `AGENTS.md` prediction 1 can be adjudicated without touching the device:

| block | T | `torch.where` | `Tensor.__bool__` | `Tensor.nonzero` | total per block forward |
|---|---|---|---|---|---|
| layer0 | 32 | 142 | 142 | 1 | 285 |
| layer1 | 32 | 147 | 147 | 1 | 295 |
| layer39 | 32 | 121 | 121 | 1 | 243 |
| mtp0 | 32 | 136 | 136 | 1 | 273 |
| **layer0 (decode shape)** | **1** | 8 | 8 | 1 | **17** |

The decode row is the one prediction 1 is about: one token, `8 distinct experts hit = top_k=8`, and
**17 host-synchronising calls per MoE block forward × 41 MoE blocks = 697 per token** if the
reference form were shipped as-is. The capture prints `COUNT ONLY — no timing taken, and the tok/s
consequence is the main agent's measurement`, and the gate fails if that count reads 0 (a counter that
can only be 0 cannot fail). The count licenses the statement that the reference's shape is
host-touching per expert slot; it licenses no throughput figure.

---

## §The three-arm selftest, and why an undifferentiated green is worthless

`speccheck/selftest_moe.py --selftest`; capture `speccheck/evidence/selftest-2026-10-26-w8.txt`
(exit 0, torch 2.14.1+cpu, threads 8, seed 20261026). The case list and every population are derived
from the config geometry — 15 cases per block (`wrong_expert_index` × (top_k slots + 1) = 9, plus
`swapped_gate_up_half`, `swapped_shared_gate_up`, `transposed_expert_stack`, `renorm_omitted`,
`cast_after_multiply`, `router_softmax_bf16`) × 4 blocks = **60 cases**. The driver fails if the
derived injections do not equal the declared set, if any section derives 0 cases, or if the case list
and the population table disagree.

### §What each arm is for

| arm | differ policy | what it is evidence for | recorded values |
|---|---|---|---|
| **honest** | the differ as it ships | must-catch fires **and** must-not-catch holds; carries the localisation claim | `cases 60`, `caught 60`, `localised 60`, `stage_diffs 192`, `ids_flips 4`, `shared_moved_cases 0`, `router_moved_cases 12`, `must_not_catch_violations 0`, `must_catch_missed 0` |
| **stub** | the "did this stage move" predicate replaced by `lambda: False` — "assume no difference exists" | the gate reads the differ and not a constant; every 0 here is asserted against a derived non-zero expectation | `caught 0`, `stage_diffs 0`, `must_catch_missed 192` (= derived), `must_not_catch_violations 192` |
| **crossed** | one must-not-catch stage deliberately perturbed per case (the case's contract says it must stay identical) | the must-not-catch side *can* fail, so the honest arm's 0 is earned | `crossed_breaches_injected 60`, `crossed_alarms_fired 60`, `stage_diffs 252`, `must_not_catch_violations 60`, `localised 0` |

Why the arms exist at all, in this repo's words: **a passing selftest nobody has watched fail is not
evidence — the sabotage is the evidence** (`pitfalls/40-gates-and-admissions.md` §"A requirement written
after the object it constrains is a note, not a forcing" — the witness-by-stubbing rule; same file
§"Gates must fail for *something*").
The failure this design was built out of is on disk: `speccheck/evidence/selftest-witness-2026-10-26.txt`
is the pre-fix witness run in which the differ was blinded and the file still printed
`SELFTEST PASS (witness mode) — 60 derived cases, 0 stage differences, 0 must-not-catch violations`
**five lines below** its own `must_not_catch_violations 192`. Two true numbers about one run, printed
unlabelled, and a PASS line that could not have been false. That is why every count in the current
output is printed with its arm name beside it, why the exit status is computed from the same tallies
that are printed, in one place at the end of the file, and why `crossed − honest = 60` (one breach per
case) is checked rather than admired.

A reader mis-reading the stub arm is the classic trap, so state it plainly: in the stub arm
`must_not_catch_violations` and `must_catch_missed` are **both 192**, because a blinded differ makes
every allowed stage look silent and each silent-allowed stage is listed in both tallies. In the honest
arm the same two counters read `0` and `0`. Same words, opposite meanings, different arms.

### §The sabotage: nine controls, all witnessed

`speccheck/evidence/selftest-controls-2026-10-26-w8.txt`. Each control mutates **one** thing in the
imported module and re-runs the whole three-arm matrix; the shipped file is never edited, so the file
that shipped is the file that was tested. A control counts only if the run exits 1 **and prints the
named reason** — exiting 1 for any reason is not a pass.

| control | fault injected | named reason it must print | verdict |
|---|---|---|---|
| A | the stub does not blind the gate | `the stub did not blind the gate (stage_diffs=192)` + `still caught 60 cases` + `missed 0 allowed stages, derived 192` | FIRED |
| B | `CONDITIONAL_AFFECTED` emptied | `honest/layer*/router_softmax_bf16: topk_ids:moved; expert_raw:moved` ×4, `8 must-not-catch violations`, `only 56/60 cases localised` | FIRED |
| C | conditional table inflated with `shared_raw` | `shared_raw:silent` ×4, `4 must-not-catch violations`, `only 56/60 localised` | FIRED |
| D | the flip fact is lost | the same 8 violations, plus the conditional-never-licensed tail | FIRED |
| E1 | a case carries no flip fact | `recorded 1 flip facts, derived 2 — the stub expectation would be counting a different case set` | FIRED |
| E2 | one arm reads a different discrete fact | `flip fact for ('layer0','router_softmax_bf16','{}') reads False but arm=honest read True` | FIRED |
| E3 | `wrong_expert_index` flips the selection it must report clean | `topk_ids changed but wrong_expert_index declares no conditional stages` | FIRED |
| E4 | no router case flips | `the conditional stages expert_raw/topk_ids were never licensed by a real fact — that half of the contract is untested, not satisfied` | FIRED |
| E5 | **no fault at all** | (must print nothing) | SILENT, as it should be |

Control C is the one that says where the power sits, and the file records it explicitly: inflating the
conditional table is **not** caught by the stub arm — the derived expectation follows the table, so
both move to 196 and still agree. It is caught by `arm=honest`, the arm that compares the contract
against what the differ really saw. Under-declaring (B) is caught by both. A control that only ever
moves the ruler is not a control.

### §What the localisation actually is, and where the positive control is thin

Each case has a contract — the stages its injection is allowed to move — and the verdict is
`allowed == observed`, stage for stage, in both directions. The honest arm's `maxabs` column is the
largest single-element disagreement seen, over T=16 tokens (T=64 for the router case, which must be
shown able to change a *decision*, not only a value):

| injection | moved stages (same for all four blocks) | maxabs layer0 / layer1 / layer39 / mtp0 |
|---|---|---|
| `wrong_expert_index` (9 cases/block) | `expert_raw, routed_sum, combined_out` | 9.961e-02…1.436e-01 / 1.296e-01…1.738e-01 / 4.663e-01…1.654e+00 / 5.332e-01…1.287e+00 |
| `swapped_gate_up_half` | `expert_raw, routed_sum, combined_out` | 1.050e-01 / 7.263e-02 / 6.631e-01 / 8.193e-01 |
| `transposed_expert_stack` | `expert_raw, routed_sum, combined_out` | 1.168e-01 / 1.626e-01 / 1.188e+00 / 1.075e+00 |
| `swapped_shared_gate_up` | `shared_raw, shared_out, combined_out` | 4.531e-01 / 2.781e-01 / 2.988e-01 / 1.497e-01 |
| `renorm_omitted` | `topk_weights, routed_sum, combined_out` | 2.600e-01 / 2.585e-01 / 1.916e-01 / 2.477e-01 |
| `cast_after_multiply` | `topk_weights, routed_sum, combined_out` | 9.766e-04 / 4.883e-04 / 9.766e-04 / 9.766e-04 |
| `router_softmax_bf16` | `router_probs, topk_ids, topk_weights, expert_raw, routed_sum, combined_out` | 2.050e+02 / 1.670e+02 / 1.440e+02 / 2.200e+02 |

The `swapped_shared_gate_up` rows are the whole point of putting the shared branch inside the contract:
a shared-expert defect moves `shared_*` and `combined_out` and must **not** touch the router side. The
capture prints exactly that for those cases — `shared_raw x4, shared_out x4, combined_out x4` moved, no
router stage moved — and `must_not_catch_violations = 0` says the router side stayed in its lane. The
counter named for shared moves, `shared_moved_cases`, reads **0 in all three arms** of the shipped run
because it counts *allowed* shared-stage moves and the honest contract makes the shared stages
*required*; it becomes 4 only under control C, in
`speccheck/evidence/selftest-controls-2026-10-26-w8.txt`. Worth knowing before someone reads a 0 as an
absence. The element population the arm walks is
7,230,240 per block (14 cases × 401,680 elements at T=16 + 1 case × 1,606,720 at T=64), i.e.
**28,920,960 per arm and 86,762,880 element comparisons across the three arms**.

One thinness worth naming rather than glossing: the crossed arm picks its breach as the *first* stage in
`FLOAT_STAGES` that the case's contract immunes and that is currently identical — which in this build
is `router_logits` in all 60 cases (the `breach` column of the firing table, and
`router_moved_cases` 60 there against 12 in the honest arm). The positive control therefore proves the
localisation check can fire, but only ever on one stage. Rotating the breached stage across cases is
the cheap fix; until then, "the must-not-catch side can fail" is witnessed on one stage of ten.

---

## §The 192-vs-184 dispute — the case study in deriving the population

Full working: `speccheck/evidence/selftest-derivation-2026-10-26-w8.txt`, which quotes the red state the
killed worker left behind as its own first line:

```
- arm=stub: stub missed 192 allowed stages, derived 184
```

Two numbers about one quantity, and the temptation that has killed sibling projects twice is to "fix"
it by adopting whichever number the build currently prints. The answer was **192**, and what matters is
the three independent derivations that had to agree before anyone was allowed to say so:

1. **Why 184 was the bug.** `AFFECTED` is the *unconditional* half of each case's contract — the stages
   an injection moves whatever the data does. `|AFFECTED|` summed over the 15 cases is **46 stages per
   block**, and 46 × 4 blocks = **184**. Meanwhile the runtime contract (`allowed_stages()`) had been
   adding two more stages for the bf16-softmax case, from a bare literal the header derivation could
   not see. Two spellings of one contract, one invisible to the check that was supposed to police it —
   the shape `pitfalls/40-gates-and-admissions.md` §"Two self-consistent layouts joined by a branch are
   invisible to both sides" warns about.
2. **Why those two stages, and why only conditionally.** `route()` takes the top-k *out of*
   `softmax(softmax_dtype)`, so a bf16 softmax can reorder the probabilities and change **which**
   experts are in the top-k. When it does, the reported `topk_ids` move — and `expert_gather` indexes
   the `[E, 2I, H]` stack by those ids, so a flipped id reads a **different 6.0 MiB expert**, and
   `expert_raw` moves with it, then `routed_sum`, then `combined_out`. When the selection does *not*
   flip, both stages must stay bit-identical. So they are licensed by a **discrete fact** (did
   `torch.equal(dirty["topk_ids"], clean["topk_ids"])` fail), case by case — never always, never by
   guess. The fact here is genuine and counted: `ids_flips = 4` in the honest arm (4/4 blocks), and
   `topk_ids` appears in the moved-stage column of exactly the 4 router rows and no others:
   `{'router_softmax_bf16': 4, 'wrong_expert_index': 0, …}`. A flip anywhere else is the injection
   leaking, which is why `verdict_flips` fails the run if `wrong_expert_index` — contracted to *report*
   clean ids while reading mutated ones — ever flips. Residual: **2 stages × 1 case × 4 blocks = 8**,
   localised to one named section.
3. **The third witness, from an arm that cannot lie about it.** `arm=crossed` injects exactly one
   must-not-catch breach per case, so its `stage_diffs` must be honest + 60. Printed: **192 + 60 =
   252**. And `arm=honest` reads `must_not_catch_violations 0` with `localised 60/60`, i.e. the moved
   set equals the allowed set case for case — which pins |allowed| at **48 per block**, not 46.

The before/after table that file prints (observed vs contract vs the gate's own derivation):

| reading | pre-fix | shipped |
|---|---|---|
| A differ-observed moved stages, arm=honest | 192 (48/block × 4) | 192 |
| B contract (AFFECTED + licensed CONDITIONAL) | **184** (46/block × 4) | **192** |
| C the gate's own printed derivation | unconditional 184, derived total `None` | unconditional 184, **derived total 192** |
| R residual (moved, not licensed) | **8**, in `router_softmax_bf16` | **0**, in no section |

The structural repair, not the numeric one, is the artifact: the conditional pair now lives in
`CONDITIONAL_AFFECTED`, one table read by **both** `allowed_stages()` and
`derive_expected_allowed_stages()`, and the flip facts are cross-checked across all three arms (same
case set, one fact per case, only sections that declare conditional stages may flip, and the
conditional half must be licensed *somewhere* or the run fails as "untested, not satisfied"). The
derived population is printed with its denominator beside it:

```
  unconditional   184   |AFFECTED[kind]| x 15 cases x 4 blocks
  conditional       8   2 stage(s) expert_raw,topk_ids x 4/4 router_softmax_bf16 case(s) flipped topk_ids
  derived total   192   the blinded differ must miss exactly this many allowed stages, no more and no fewer
```

The general rule this lane now demonstrates rather than asserts: **derive the population, never type
it** — and when a derived count and an observed count disagree, the disagreement is a fact about the
contract tables, not a nuisance to be reconciled by editing the expectation.

---

## §Prediction 4 — the routing-weight cast position, as caught here

`AGENTS.md` prediction 4: the reference casts the renormalised top-k weights back to the logits dtype
**before** multiplying the expert output, and doing the multiply in F32 and casting after is a
different number. Formally `specification/11-moe-block.md` §11.2 R5, quoted from
`Qwen3_5MoeTopKRouter.forward` in that file, and reproduced by `speccheck/moe_ref.py` §route as a
`[REF]` tag with a switch (`Form.weight_cast = "before" | "after"`) so it is measured, not argued.

**Confirmed, with numbers, in this lane:**

| witness | value | source |
|---|---|---|
| cast-before is the reference's form | our `ref` arm: `topk_weights` **maxabs 0.000e+00, ndiff 0/1,024** and `routed_sum` **maxabs 0.000e+00, ndiff 0/262,144** against the HF eager module | `speccheck/evidence/cross-check-2026-10-26.txt` |
| cast-after is a different number | `cast_after` vs HF, layer0: `topk_weights` maxabs 9.332e-04 (`ndiff 256/256`), `routed_sum` maxabs 2.441e-04 with **ndiff 26,550/65,536 (40.51 %)**, `combined_out` ndiff 11,429/65,536 (17.44 %) — same shape on all four blocks (`routed_sum` ndiff 26,284 / 26,659 / 25,350 at layer1 / layer39 / mtp0) | same file |
| the pins carry both arms | `topk_weights` (cast-before, `reading_rules.authoritative`) **and** `topk_weights_f32` (pre-cast) are both dumped, so the claim is checkable from the artifact and not from prose | `checkpoints/dump-manifest-2026-10-26.json` |
| it is a must-catch, not a preference | `cast_after_multiply` is one of the 15 derived cases and must move exactly `topk_weights, routed_sum, combined_out`; observed maxabs 9.766e-04 (layer0/layer39/mtp0), 4.883e-04 (layer1) | `speccheck/evidence/selftest-2026-10-26-w8.txt` |
| independently, on the device lane | the router gate reports `PREDICTION4 cast_after_rows=128` — a different number on 128 of 128 real rows, worst row 1,111 of 2,048 elements — judged against a separately written cast-after combine | `prototype/moe_router/results.md` §"Prediction 4 (the cast position)", box clock 2026-10-25 |

**Not confirmed, and the record should say so.** The same prediction's second clause — that the cast
position "will move the argmax of a near-tie row" — does **not** hold for the cast, and this lane
measured why: R5 happens *after* `topk`, so the cast position cannot change the selection. Measured
directly, `cast_after`'s selection disagreement against the reference is **0/128 rows as a set**, and
the device lane found ids identical between the two cast orders. The mechanism that *does* flip a
selection is the **probs-dtype** change: `softmax_bf16` disagrees on **6/128 rows as a set** and loses
ordered agreement on 32 of them (18/32 at mtp0), and the bf16-softmax case is the only case in the
matrix whose contract licenses `topk_ids` to move. `specification/11-moe-block.md` §11.2 currently
carries the conflated sentence, and the other lane already flagged it (§Contradictions item 2).

And the *other* reading of "moves the argmax" — the model's next-token argmax on a near-tie row — is
**outside this oracle entirely**: no `lm_head`, no sampling, no token decision anywhere in this lane
(§Limits item 5).

---

## §Near-tie and utilisation — synthetic-input statistics, and the follow-up that fixes that

`speccheck/evidence/moe-stats-2026-10-26.json`, produced by
`tools/moe_stats.py --all --tokens 256 --floor-tokens 32 --proxy-tokens 12` at seed 20261026. These
numbers are **properties of the fixture, not of real activations**: `speccheck/moe_ref.py` §Input
generation declares the fixture's rows RMS-1 (its `[ASSUMPTION A-01]`, in the *manifest's* namespace —
see §Tag vocabulary), because the block's input is `post_attention_layernorm(hidden)` and this
family's RMSNorm is `x · rsqrt(mean(x²)+eps) · (1+w)`. The fixture carries the RMS-1 part and not the
`(1+w)` modulation. **The follow-up that makes these statistics real** is named in that same docstring:
replace `make_input` with `post_attention_layernorm` outputs captured from a real prefill pass, re-dump
the pins, and re-cut these statistics — which needs the dense path landed, so it is sequenced, not free.

The population does apply to the pinned rows: `make_input(32, seed) == make_input(256, seed)[:32]` is
`True` (verified and recorded in `speccheck/evidence/pins-reverified-2026-10-26.txt` §capture 3), so the
32 pinned rows are the first 32 rows of the measured 256.

| block | margin(l8−l9) p0 / p50 / p99 | rows with margin ≤ 1 bf16 ULP of l8 | flip radius ‖δx‖/‖x‖ p50 / p99 | MC flips at ε = 0 / 1e-3 / 1e-1 |
|---|---|---|---|---|
| layer0 | 0.000e+00 / 3.906e-02 / 2.832e-01 | **56/256** | 7.618e-04 / 5.966e-03 | 0 / 23 / 247 |
| layer1 | 0.000e+00 / 3.125e-02 / 1.746e-01 | **70/256** | 5.646e-04 / 3.487e-03 | 0 / 28 / 251 |
| layer39 | 0.000e+00 / 1.562e-02 / 1.129e-01 | **54/256** | 6.989e-04 / 4.594e-03 | 0 / 31 / 251 |
| mtp0 | 0.000e+00 / 1.053e-02 / 6.426e-02 | **56/256** | 6.423e-04 / 3.874e-03 | 0 / 27 / 249 |

`p0 = 0.000e+00` in every block says the fixture contains **exact 8/9 ties** — which is what
`specification/ASSUMPTIONS.md` A-03 (the tie-break is not specified by this spec) is about, and what
`prototype/moe_router/results.md` §"A-03, the tie-break: exercised, not adjudicated" measured
independently on the pinned rows (13 rows with a bit-identical 8th and 9th probability; smallest
non-zero margin 12,797 ULP). The flip radius is a closed form, not an estimate: for a bias-free linear
router `Δ = (l8 − l9) / ‖W8 − W9‖₂` is the exact minimum input displacement that can change the top-8
set, and the Monte-Carlo arm re-routes real perturbations against it (predicted-vs-observed printed per
ε, with ε=0 required to flip nothing and ε=0.1 required to flip something, or the section fails as
unable to adjudicate).

Expert **utilisation** over the same 256 tokens (expected 8.0 selections per expert):

| block | experts hit | never hit | min / max | std | χ² vs uniform | entropy |
|---|---|---|---|---|---|---|
| layer0 | 250/256 | 6 | 0 / 25 | 4.776 | 727.0 | 7.7362 / 8 bits |
| layer1 | 251/256 | 5 | 0 / 23 | 4.778 | 727.75 | 7.7251 / 8 bits |
| layer39 | 221/256 | 35 | 0 / 42 | 7.970 | 2024.5 | 7.3077 / 8 bits |
| mtp0 | 249/256 | 7 | 0 / 33 | 6.514 | 1352.5 | 7.5532 / 8 bits |

The skew is real and layer-dependent (layer39 is the outlier), which matters for any gather schedule
that assumes balance — but on 256 synthetic tokens, so it is a direction, not a load factor.

---

## §Floor probe — the distance between *legal* implementations, and two caveats on its ruler

Same evidence file, `sections.floor`, T=32, each arm compared against the f64 `exact` oracle. This is
what any bf16 band must sit **above**, and it is a proposal input, never a band: a band is a
user-visible claim (`specification/10-reference-dumps-and-gates.md` §10.3 — a band is derived, never
from the build under test — and `pitfalls/30-numerics.md` §"Format changes: the gates are the decision
procedure, with two exceptions", which is where int4 quantisation is named as something that must be
asked; `DECISIONS.md` D-01, D-01a and D-08 own the widths).

Max over the 4 blocks, per stage, as `maxabs / 1−cos / Σ>1ulp`:

| arm vs f64 oracle | `topk_weights` | `expert_raw` | `routed_sum` | `combined_out` |
|---|---|---|---|---|
| `ref` (the shipping form) | 2.67e-03 / 9.4e-06 / 3 | 6.99e-01 / 7.4e-02 / 80,381 | 7.02e-02 / 6.3e-03 / 11,170 | 7.04e-02 / 5.2e-03 / 7,896 |
| `spec` | 2.41e-03 / 8.0e-06 / 2 | 7.00e-01 / 7.4e-02 / 80,385 | 6.98e-02 / 6.3e-03 / 11,094 | 6.98e-02 / 5.2e-03 / 7,895 |
| `cast_after` | 2.41e-03 / 8.0e-06 / 2 | 6.99e-01 / 7.4e-02 / 80,381 | 7.02e-02 / 6.3e-03 / 11,170 | 7.04e-02 / 5.2e-03 / 7,910 |
| `accum_f32`, `accum_f64`, `rank_order`, `per_slot` | within one ULP of `ref` on every stage; routed_sum Σ>1ulp 11,088–11,177 | ditto | ditto | ditto |

**Two caveats, and they matter before anyone sizes a band from this file.** Both are two-rulers
problems:

1. **The `routed_sum`/`combined_out` columns are dominated by selection flips, not by rounding.** The
   floor's reference arm runs the router in f64 and selects a different 8-set on 6/128 rows (§F2), and
   one flipped expert changes a whole row. That is why `ref/routed_sum` maxabs reads 7.02e-02 (mtp0:
   7.017e-02, `1−cos` 6.30e-03 — byte-for-byte the numbers `speccheck/evidence/cross-check-2026-10-26.txt`
   prints for `mtp0/exact vs hf_eager/routed_sum`, which is the same f64-vs-bf16 comparison under the other
   driver) while the one stage in the section the router cannot touch, `ref/shared_out`, maxes at 2.07e-03
   (layer0) with a worst `1−cos` of 1.05e-05 (layer39). A band taken from the routed column would be sized
   by a routing-selection artifact.
2. **`section_floor` compares `expert_raw` elementwise without canonicalising the slot axis**, while
   `speccheck/cross_check_hf.py` §canon_raw does (and `manifest.reading_rules."slot axis"` says to).
   The same quantity under the two rulers: the floor's `ref/expert_raw` 1−cos reaches **7.4e-02**,
   while the canonicalised, agreeing-set comparison in the cross-check puts the distance between the
   wide-form expert outputs and the bf16 reference at **6.620e-06 … 7.586e-06** (printed for `spec` and
   `exact` on all four blocks). The gap is slot permutation on tie rows, not arithmetic.
   Before any band is sized from this file, `section_floor` should canonicalise `expert_raw` and print
   the flip-row count separately — recorded as owed in §Open work.

The same file's `quant_proxy` section (`--proxy-tokens 12`) re-quantizes the hit experts to Q4_K with
vectorised packers that are **required bit-exact** against `tools/gguf_reader.py`'s transcriptions
before any proxy number prints (layer0 example: `expert_raw` vs bf16 ref `1−cos` 7.37e-03;
weight-space relative Frobenius `gate_up_Q4_K` 0.0800, `down_Q4_K` 0.0928 over 70 experts). It is a
12-token proxy for width selection and is not this oracle's verdict on the quantized path — §Limits
item 1.

---

## §Limits — what this oracle does **not** establish

Read this as the price of using §Findings. Every line is something that stays false even though every
gate in this lane prints PASS.

1. **No quantized path.** Everything here is bf16/f32/f64 on CPU. Q4_K/Q8_0 appear only as the
   12-token proxy in §Floor probe; widths and bands are the user's (`DECISIONS.md` D-01, D-01a, D-08;
   `quantization/README.md` §"How a quantized path gets gated (no gate gets loosened)").
2. **Reproducing the pins is not independence.** `tools/dump_reference.py` writes `arm=ref` out of
   `moe_ref.py`, so `--against-dumps` green = determinism + file integrity + no drift since pinning.
   The independence is `speccheck/cross_check_hf.py` plus `manifest.agreement_at_write_time` — and even
   there the two sides **share one weight-reading layer** (`speccheck/moe_ref.py` §load_block_weights
   plus the committed shard index), so a wrong tensor *name*, *shape* or *stride* reading would be
   invisible to both. What mitigates it: shapes asserted against config-derived geometry
   (`speccheck/moe_ref.py` §shapes_ok, including bf16 dtype-by-filesize), the manifest's
   shard/byte-offset map as the authority, and HF's parameter-name set asserted equal to the seven
   names loaded. What that does **not** settle is prediction 3 — whether `[E, 2I, H]` with `chunk(2)` is
   the checkpoint's true layout — which the device lane left explicitly **UNRESOLVED** after sweeping
   all 256 experts (`prototype/moe_router/results.md` §"Prediction 3 (the fused `[E, 2I, H]` layout):
   UNRESOLVED, and reported as unresolved": worst 1−cos 1.140 over the "correct" reading, best cosine
   0.065 over the full sweep). This is the largest open item sitting
   under this oracle, and this lane cannot close it: both of its candidate witnesses (pinned
   `expert_raw`, HF's `expert_raw`) are produced through the reading in question.
3. **4 of 41 MoE blocks.** `layer0`, `layer1`, `layer39`, `mtp0` — 9.8 % of the blocks; the gate itself
   prints `blocks available in the acquired shards: 4 of 41 … coverage stated, not implied`. Layers
   2…38 are unpinned because 22 of 26 shards were never acquired.
4. **No attention, GDN, norms, rope, state, embedding, `lm_head`.** Against the gate list in
   `specification/10-reference-dumps-and-gates.md` §10.6, this lane covers the router (§11.2), selection
   (§11.3), the expert body's *form* (§11.4), the routed sum (§11.5) and the shared expert (§11.6), and
   covers **none** of the rows for norms (§03.1), GDN recurrent (§04.4–§04.7) or chunked (§04b), attention
   (§05), rope (§05.4), state (§08), MTP (§09) or dequant (§10.2). §10.6's closing paragraph asks every
   gate to ship a `RESULT` line, a parsed-record-count assertion and negative controls: the five MoE rows
   are the ones this lane has anything to say about, and its verdicts print as `SELFTEST PASS` /
   `CROSS_CHECK PASS` with the record count asserted ("printed comparison records 240, expected 240")
   rather than as a line literally spelled `RESULT`. The other eight rows are untouched here.
5. **No decision-level check at all.** `specification/10-reference-dumps-and-gates.md` §10.3 step 4's
   top-1/top-5 agreement on the pinned rows over a corpus at 4k context is unreachable here: no `lm_head`,
   no sampling, no tokens. Neither reading of "the argmax of a near-tie row" is adjudicable in this lane.
6. **The tie-break is inherited, not verified.** Both sides call `torch.topk`, so agreement on tie rows
   is not independent evidence about ties. `specification/ASSUMPTIONS.md` A-03 stays open, and this
   spec deliberately defines no tie-break of its own.
7. **Two must-catch rows from `specification/10-reference-dumps-and-gates.md` §10.4 are not exercised
   as cases here**: "accumulate the 8 experts in rank order instead of ascending expert id" exists only
   as a *legal form arm* (measured by the floor probe, never injected as a case), and "a router that
   picked different experts — perturb one logit across the 8/9 boundary" is exercised by a softmax-dtype
   change rather than by perturbing a logit. Real gaps in the must-catch set, not phantoms.
8. **The positive control touches one stage of ten** (§What the localisation actually is).
9. **CPU-only, and thread-count-sensitive.** CPU matmul reduction order follows the thread
   partitioning, so pins reproduce only at the recorded 8 threads — the same wall the device lane hit
   (`prototype/moe_router/results.md`: torch's own `F.linear` reproduces the pinned logits at `ndiff 0`
   only at `manifest.interpreter.threads`). Nothing here says anything about any accelerator path, and no
   number in this document is a timing.
10. **`router_probs` has no pinned arm** — it is not in `manifest.dumps`. Its witnesses are the live HF
    comparison and the captured `dtype=torch.float32` softmax call, not a byte on disk.
11. **The block's own residual add is out of scope.** By `specification/11-moe-block.md` §11.1 the block
    returns routed + shared and the decoder layer adds; nothing here pins that add.
12. **Synthetic inputs throughout.** Every near-tie and utilisation figure is a fixture property
    (§Near-tie and utilisation), including the 4.69 % selection-disagreement rate in §F2, which is a
    statement about near-ties *this fixture produces*, not about real activations.

---

## §Contradictions found while writing this

Reported, not repaired: each is another lane's file or a user decision.

1. **`speccheck/moe_ref.py` §route's inline comment says `torch.topk` breaks ties "by lower index".**
   The pinned rows refute it: a lower-index rule reproduces the pinned ids on 105 of 128 rows and
   disagrees on 23 (`prototype/moe_router/results.md` §"Contradictions found, reported and not
   repaired" item 1). The comment states a rule the dump does not follow; the behaviour itself is A-03's
   and the code makes no choice — it is the comment that overstates. Action: correct the comment when
   A-03 is decided, not before.
2. **The cast order and the probs dtype are conflated** in `specification/11-moe-block.md` §11.2 (and
   echoed by `specification/ASSUMPTIONS.md` A-05): the cast cannot change a selection because it happens
   after `topk`; the softmax dtype can. Measured both ways here (§Prediction 4: 0/128 vs 6/128). The
   device lane flagged the same thing; whoever owns that file decides the wording.
3. **`router_probs` is compared but never pinned** (§Limits item 10) — a hole in the manifest's dump
   list, not in the comparator.
4. **The shipped against-dumps capture was red because of a checker bug** and would have been quoted as
   an oracle failure. Now re-captured green (§F3 and
   `speccheck/evidence/pins-reverified-2026-10-26.txt`), with the superseded file named and kept.
5. **`speccheck/README.md` does not exist, and two files cite it.** `speccheck/moe_ref.py`'s opening line
   calls this module "the oracle (§What this is of speccheck/README.md)" and `speccheck/cross_check_hf.py`
   sends the reader to the same missing section. The lane owed a narrative and, until this file, had only
   evidence captures. What those cites promise now lives here (§Pins, §Findings, §Limits), so the repair is
   either a short `speccheck/README.md` that points at this document or re-pointing both docstrings — and
   per `pitfalls/60-docs-and-citations.md` §"Moving or pruning a document is two jobs, not one" a re-point
   is its own commit into `.py` files, which is not this document's remit.

---

## §Tag vocabulary

`speccheck/moe_ref.py` §docstring points here: every non-obvious behaviour in that file carries one of
three tags, so a reader can tell a derived fact from a reference's habit from our own unresolved guess.

| tag | meaning | the load-bearing instances |
|---|---|---|
| `[SPEC]` | derived from the committed config or geometry; if the config changes, the code refuses | `Geometry.n_blocks` = 40 + 1 = 41 (guarded: `load_geometry` raises if `intermediate_size` or `mlp_only_layers` ever appear); `Geometry.expert_bytes` = `[2I,H] + [H,I]` bf16 = 6,291,456 B; header-first shard read (8-byte length + JSON header, no tensor bytes); the `E != 2I` guard that keeps the transposed-stack injection non-degenerate |
| `[REF]` | a **form choice the pinned reference takes**, recorded so the choice is visible rather than inherited — each is switchable in `Form` and therefore measurable | F32 softmax over the 256 logits (`softmax_dtype="f32"`, confirmed at the source by §F4's captured `dtype=torch.float32`); renormalise over the K (`renorm=True`); **cast the weights to the logits dtype before the multiply** (`weight_cast="before"`, R5); bf16 accumulation of the routed sum (`accumulate="bf16"`); addends associated in ascending expert-id order (`order="expert_asc"`); grouped batching, one GEMM per hit expert (`batching="grouped"`); the shared branch added in the model dtype |
| `[ASSUMPTION A-nn]` | our reading, not yet adjudicated | **A-01 (the manifest's namespace)**: fixture rows are RMS-1 because the block's input is `post_attention_layernorm(hidden)`; it does **not** carry the `(1+w)` modulation of a real activation. Consequence, stated where it bites: every near-tie/utilisation figure in this file is labelled a synthetic-input statistic and the follow-up is named (§Near-tie and utilisation) |

**Namespace warning, because it has already confused a reader once.** `specification/ASSUMPTIONS.md` A-21
records that two `A-01` namespaces exist on this box: *specification's* A-01 (the MTP block is not
represented in the checkpoint's own reference implementation) and the *dump manifest / speccheck* A-01
(the RMS-1 fixture). They are different documents about different subjects; this file always says whose
A-01 it means, per that entry's own instruction.

---

## §How to re-run this lane

All CPU-only, all PASS/FAIL, none reads a clock. From the repo root, with the pinned interpreter:

| command | adjudicates |
|---|---|
| `speccheck/selftest_moe.py --selftest` | the three-arm matrix (60 derived cases, 192 licensed stage pairs) |
| `speccheck/selftest_moe.py --against-dumps` | the oracle reproduces the 40 pinned ref-arm tensors bit-exactly |
| `speccheck/cross_check_hf.py --blocks layer0,layer1,layer39,mtp0` | the two implementations, per stage, plus the capability probe and the host-sync count |
| `speccheck/hf_capability_probe.py <config> <sharddir> <index>` | what the installed `transformers` can build/load at all (run standalone it prints the 10 `CAP` lines; the driver counts them) |
| `tools/dump_reference.py --verify --out data/dumps/2026-10-26 --manifest checkpoints/dump-manifest-2026-10-26.json` | every dump hashes and sizes to the manifest |
| `tools/moe_stats.py --all --tokens 256 --floor-tokens 32 --proxy-tokens 12` | utilisation, near-tie, floor probe, quant proxy (writes `speccheck/evidence/moe-stats-2026-10-26.json`) |

Re-dumping (`tools/dump_reference.py` without `--verify`) rewrites the pins and is a decision, not a
maintenance step: it moves what every later gate is judged against.

## §Open work owed by this lane

Ordered by what unblocks the most.

1. **Prediction 3 — the expert layout read** (§Limits item 2). Needs a witness that does not pass through
   `moe_ref.load_block_weights`: read a known expert's bytes straight out of the shard at the manifest's
   `data_offsets` and compare against a GGUF-spelling reader once the engine's reader exists. Until then
   `expert_raw`'s definition is ours, and the device lane's UNRESOLVED stands.
2. **Fix the floor probe's rulers** (§Floor probe, caveats 1–2): slot-canonicalise `expert_raw` in
   `tools/moe_stats.py` §section_floor, and print the selection-flip row count beside `routed_sum` and
   `combined_out` so a band is never sized by a routing-selection artifact.
3. **Add the two missing must-catch cases** (§Limits item 7): rank-order accumulation as an injection,
   and a logit-perturbation selection flip in addition to the dtype one.
4. **Rotate the crossed arm's breached stage** across `FLOAT_STAGES` so the positive control is not
   `router_logits` × 60.
5. **Pin `router_probs`** into the manifest (or record, in `manifest.reading_rules`, that R2 is held by
   the softmax-call capture instead).
6. **A repo-local `.venv/`** for the CPU oracle so the lane's executability does not depend on a path in
   another project (`AGENTS.md` §Self-containment).
7. **A-03 needs the user's word** (`specification/ASSUMPTIONS.md` A-03; also A-05's second clause, which
   §Prediction 4 shows is about the probs dtype, not the cast). This lane supplies the witnesses; the
   adjudication is not ours to make.
8. **Real activations** (§Near-tie follow-up), which needs the dense path.
