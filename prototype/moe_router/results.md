# `prototype/moe_router` — the router as kernel + gate, first green run

**Verdict: PASS.** `RESULT PASS device checks=40 findings=0`. The iGPU (`gfx1151`) computes R1→R5 —
logit dot, softmax, top-k, renormalisation, bf16 cast — and the weighted combine, and every field
agrees with the authoritative arm of `speccheck/moe_ref.py` over the pinned dumps of
`checkpoints/dump-manifest-2026-10-26.json`. Twelve injected misreadings fire only where they belong;
`MISMATCH=0` across 19,344 field comparisons.

Artifact names carry the project date (2026-10-26), matching `quantization/evidence/`; the box clock
at run time read `2026-10-25T09:13:27Z`.

## Device-discipline record (AGENTS.md §"GPU execution is serialized…")

| field | value |
|---|---|
| command line | `./gate` → `python3 prototype/moe_router/gate.py` → `moe_router --plan work/plan.tsv --slab work/plan.bin --out work/out.f32 --report work/report.tsv --expect-cases 3224` |
| sample, declared **before** the device work | `cases=3224 arms=268 rows=140(real=128,synth=12) field_comparisons=19344 elements_compared=8330816 launches=13 slab_elems=4688896 slab_MiB=8.9 modes=13` |
| cut applied | none. `--max-rows` was used during bring-up (it prints `CUT rows_per_block=N of 32` and cuts **sampled rows only**; every mode, arm, field and assertion still runs on the rows kept) and the green run is the full 32 per block |
| wall time (lock record input, not a measurement) | started `2026-10-25T09:13:27Z`, finished `2026-10-25T09:13:46Z` |
| `tools/gpu_lock.sh util` | before `gpu use: 2%`, after `gpu use: 1%` |
| lock | acquired by the `gate` wrapper only, tag `q36 W13 moe_router device gate`, released on EXIT; `LOCKREF gate.py references to the lock tool must be 0: 0`; `grep -rl gpu_lock prototype/moe_router/` names only `gate` |
| timing | **no timing taken**. No timer exists in `gate.py`; the wall pair above is the lock record AGENTS.md asks a device run to carry |
| driver stdout | `work/driver.log`, parsed from the file; the return code is checked directly; nothing is piped |

## What the gate compares, and with what ruler

Six fields per case: `logits, probs, wpre, wcast, ids, routed`. Two arms per real row — `KERNEL`
(computes R1 from `x @ Wg`) and `LOGITS` (feeds the pinned `router_logits` in) — so an R1
disagreement cannot mask a downstream one.

| field | ruler | why |
|---|---|---|
| `wcast` (R5) | **bit-exact** | the floor probe measured the distance from every pinned f32 weight to the nearest bf16 rounding boundary as **164 f32 ULP**, against a worst legal route spread of **5 ULP** — the cast is robust by ~32×, so zero tolerance is arithmetic, not luck (`cast-is-robust-not-lucky`) |
| `ids` | **bit-exact** on tie-free rows; `ALT` where the row's 8th/9th probabilities are bit-identical | the tie-break is A-03, the user's |
| `routed` | **bit-exact** against the pinned `routed_sum`, built from the pinned `expert_raw` | 0 differences on all 128 real rows |
| `logits` (kernel arm) | ≤ 5 elements differing, ≤ 1 bf16 ULP each | `logit_dot_route_disagreement_elements=5` from the floor probe; torch's own `F.linear` reproduces the pinned logits at `ndiff 0` **only at the manifest's recorded 8 threads**, which is why the thread count is read from `manifest.interpreter.threads` and not typed |
| `probs` | the **certificate**: finite, non-negative, summing to `1 ± (2(E-1)+1)·2⁻²⁴/(1-(E-1)2⁻²⁴)` | there is no pinned `router_probs` (see contradictions). The bound is derived from the header's `MR_E` and IEEE binary32; the first draft folded in the 19-ULP exp route spread and the witness refused it (3.05e-4 against a 3.05e-5 bound) — the certificate is normalisation-invariant, so the exp error cancels. `witness-prob-sum-bound` shows a 3×-bound row rejected and a legitimate drift accepted |
| `wpre` (f32 weights) | ≤ 5 f32 ULP | `weights_vs_pinned_f32_ulp=5` |

**Prediction 4 (the cast position).** The authoritative arm is `topk_weights`, which the manifest
records as the value **after** the cast back to the logits dtype. The device arm reproduces it on
**128 of 128** real rows. The `cast_after` injection — multiply by the fp32 weight, cast the product —
produces a **different number on 128 of 128 real rows**, worst row **1111 of 2048** elements
(`prediction4-wrong-order-differs`, `PREDICTION4 cast_after_rows=128`). It is not merely different, it
is different in the direction the reference discards: keeping the fp32 weight retains the 8 mantissa
bits the reference threw away. The judge also lands it on an independently written cast-after combine
(`ALT-EXACT cast_after/routed = 19` rows in the host arm), so "different" here is not "different
because I said so".

**Prediction 1 (the serial spine), from the code, no clock read.** `SERIAL
stages_per_block=5 blocks_per_token=41 dependent_router_stages_per_token=205
dependent_expert_reads_per_token=328 barriers_per_router_row=2 host_values_emitted=0`. R1→R5 are
strictly dependent within a block, one workgroup per row with lane 0 walking the spine, and the
combine's eight 6 MiB gathers per token are reachable only through R5. `host_values_emitted=0` is the
claim that matters for the prediction's host-bound failure mode: nothing in the row path reads a value
back to the host, so the host-bound decode ceiling prediction 1 warns about is not baked into this
kernel's shape. **This is a structural claim, not a measurement** — no throughput figure exists in
this lane, and the number in `AGENTS.md`'s prediction stands or falls on the main agent's measurement.

**Prediction 3 (the fused `[E, 2I, H]` layout): UNRESOLVED, and reported as unresolved.** The probe
reads one selected expert three ways (`chunk2`, `halves_swapped`, `stack_transposed`) and the three
readings are provably three different functions (`prediction3-readings-distinguishable`), but **no
reading reproduces the pinned `expert_raw`**: worst `1-cos` over the correct reading is 1.140, i.e.
uncorrelated. That is not a bug in my SwiGLU — `speccheck/moe_ref.py`'s own `_expert_apply` (which
asserts `out.shape[-1] == 2I`, prediction 3's own check) agrees with a separately written numpy SwiGLU
to the last bit, and sweeping **all 256 experts** for `layer0`/token0/slot0 peaked at cosine **0.065**
(`work/probe15.py`). So `expert_raw` is not `down(silu(gate·x)·(up·x))` for any expert under the
manifest's geometry, and this dump cannot settle the layout claim. What *is* verified from
`expert_raw` is the combine, because `routed_sum` reproduces from it bit-exactly on every real row.
This lane contains no expert GEMM, so no router verdict rests on P3; the question belongs to whoever
owns `tools/dump_reference.py` and what it means by `expert_raw`.

## A-03, the tie-break: exercised, not adjudicated

Derived from the pinned probabilities, over the 128 real rows:
**`class0=83 class1=32 class2=12 class3=1`** (0 = no tie in the top K+1, 1 = tie strictly inside the K,
2 = straddle with one member inside, 3 = straddle with ≥2 inside), and **13 rows have a bit-identical
8th and 9th probability** (margin exactly **0 ULP**); the smallest non-zero margin anywhere is **12797
ULP**, so route noise can never flip a set decision — only a genuine tie can. The synthetic fixtures
(`derived_from=K+1+1+2`) make each immunity class reachable, which the dump alone cannot: no pinned row
has ascending ids.

**Which side my kernel picks: the highest index wins** (`mr_topk_row`, tie `>=`). It is stated here
rather than decided: on the 13 straddle rows the engine and the pinned arm chose different tied experts
on 23 rows overall (`A03 rows_where_the_engine_and_the_pinned_arm_chose_different_tied_experts`), and
those rows' `routed_sum` comparisons are **skipped, with the count derived** (`witness-unjudged-is-
derived`: skips = divergent rows × 2 arms × 1 mode, because only the reference mode compares against
the dump). A-03 remains the user's call; this gate's job was to make the witness exist.

## The injection matrix

`mr_mode_reaches(mode, field, arm)` is the kernel's own claim about which fields a mode can move —
`NO` (a move is a second bug), `YES` (must move somewhere) or `MAY` (reachable but data-dependent).
The gate asks the **compiled** predicate (`moe_router --reach-table`, 156 records, count checked
against `REACHCOUNT`) rather than mirroring it in Python.

| mode | FIRED | ALT | IMMUNE | MAY_MOVED | note |
|---|---|---|---|---|---|
| `softmax_bf16` | 536 | 0 | 268 | 528 | bf16 probabilities move the fp32 weights; whether a near-tie changes the winner is population-dependent |
| `no_renorm` | 0 | 268 | 804 | 530 | lands exactly on the unrenormalised form |
| `cast_after` | 0 | 268 | 1340 | 0 | prediction 4; lands exactly on the independent cast-after combine |
| `rank_accum` | 0 | 265 | 1343 | 0 | immune exactly where ids ascend |
| `w_by_expert_id` | 0 | 266 | 1342 | 0 | immune exactly where `ids[k] % K == k` |
| `row_off_1` | 128 | 0 | 840 | 311 | immune on the arm that never reads `x` |
| `wg_transposed` | 128 | 0 | 840 | 640 | same |
| `tie_high` | 0 | 99 | 705 | 92 | obliged only on the straddle classes |
| `f32_combine` | 0 | 268 | 1340 | 0 | |
| `topk_asc` | 105 | 690 | 545 | 265 | |
| `no_max_sub` | 0 | 0 | 268 | 534 | exact in real arithmetic; moves only where fp32 lets it |
| `poison_nan` | — | — | — | — | leak guard: every slot must be the canonical NaN, and a correct mode must leave none |

`modes_that_never_fired = []`. Classes: `MATCH=1564 ALT=2170 IMMUNE=9635 FIRED=897 MAY_MOVED=2900
MAY_UNCHANGED=2132 MISMATCH=0` — every class populated, and each counter that can read 0 has a witness
that reads it non-zero (`witness-comparator-flip` proves one flipped bit moves exactly one element, so
a comparator here cannot be blind).

Three per-case "did not move" results are classified `ALT` rather than `MISMATCH`, and the reason is
printed rather than hidden (`ALT-EXACT ...:coincides`): on a tied fixture whose weights are exactly
`1/8`, casting before or after the multiply **is** the same arithmetic, and agreeing with an
independently computed alternative form is agreement. `witness-injections-fired` still demands that
every mode move something somewhere.

## Contradictions found, reported and not repaired

1. **`speccheck/moe_ref.py`'s `route()` comment says `torch.topk ... ties by lower index`.** The
   pinned rows refute it: a lower-index rule reproduces the pinned ids on **105 of 128** rows and
   disagrees on **23**. The comment is describing a rule the dump does not follow. Not repaired —
   `speccheck/` is read-only to this lane, and the tie-break is A-03, which is the user's.
2. **`specification/11-moe-block.md` §11.2 / A-05 says the cast-order rewrite "can change which 8
   experts topk picked".** The cast happens **after** top-k, so it cannot; measured, `ids` are
   identical between the two cast orders. The clause is true only for the *probs-dtype* change. Left
   as written, flagged here.
3. **`router_probs` is compared by `speccheck/cross_check_hf.py` but is not in the manifest's
   `dumps` list.** So R2 has no pinned arm and can only be held to its certificate — which is what the
   derived bound above is for, rather than a typed tolerance. (This is also the reason the first draft
   of that tolerance, a hand-typed `1e-5`, is gone.)
4. **`manifest.agreement_at_write_time` claims `topk_ids_bitexact_vs_reference: true` and
   `routed_sum_bitexact_vs_reference: true`.** Re-derived here rather than trusted
   (`manifest-claims-ids-hold`, `manifest-claims-routed-hold`: 4/4 blocks), and the sha256 of all 104
   dumps was verified against the manifest before any comparison (`manifest-sha256`).

## Two kernel bugs the host arm caught before the device was touched

Both are why `--check-only` exists as a mode: it runs the oracle, the plan, the host arm and **the same
judges** the device run uses, with zero device work.

1. **Output widening dropped the shift.** `out[...] = mr_bits_to_f32(st->logits[l])` fed a `uint16_t`
   to a function that reinterprets **32 bits**, producing a denormal instead of the value. Every other
   selftest passed, because the arithmetic used `mr_bf16_to_f32` correctly and only the output path was
   wrong. Now guarded by the driver's `output_widening` selftest (the bf16 fields must land as
   `bits << 16`).
2. **The injection self-comparison double-counted the base.** `mr_row` was handed `out + dst_off` while
   `cmp_off` is an absolute buffer index, so it compared against `buf[dst_off + cmp_off + ...]`: with
   the full 3224-case plan that walks off the end (AddressSanitizer SEGV at that comparison), and with
   a small plan it silently read *another case's* words and printed `bad[f] = 0`. `mr_row` now takes the
   buffer base. The gate's verdicts never depended on it — they come from the words read out of the
   output file — but the kernel's own leak counters were meaningless until this was fixed.

A third, in the oracle arm rather than the kernel: `combine_bits` widened its bf16 weight array with
`.astype(np.float32)` (a numeric conversion: `0x3e59` → `15961.0`) instead of `<< 16` then `.view`, and
the synthetic fixtures combined the **unrounded** fp32 draw rather than the bf16 values the slab holds.
Both are the `.view`-vs-`.to` trap of `pitfalls/30-numerics.md`, and both made a correct kernel look
wrong on ~30% of the synthetic elements.

## Open items

- **No host-vs-device `expf` floor probe exists.** The floor probe measured CPU re-association
  (19 ULP), not `glibc expf` against `libdevice expf`. Under the two injections that keep the fp32
  weight, the arms differ by up to **2 bf16 steps** on 2 elements of 8.3M. Rather than invent a bound
  that fits what the build produced, the gate prints `ARM-SEPARATION mode=cast_after field=routed
  worst_bf16_steps=2` and asserts only that the comparison was made
  (`host-device-f32-carrying-separation-reported`). Everything else is bit-exact across the arms, and
  every verdict against the pinned dumps is `MISMATCH=0`.
- **P3 needs `expert_raw`'s definition** from the dump tooling, as above.
- **A-03 needs the user's word.** This gate exercises ties and states which side it picks; it does not
  claim the choice is correct.
