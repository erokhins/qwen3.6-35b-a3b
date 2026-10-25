# Shared ULP machinery and the ±1-ULP policy — what was built, and the staged migration

**Date.** 2026-10-26. Worker lane `ulp-policy` (single `halogen` worker under `DECISIONS.md` D-15;
stream lock held for the whole life of the turn and released at handback).
**Authority.** `DECISIONS.md` **D-16** (cross-arm check inherits the lane's declared field rule — the
user's option A) and **D-17** (automatic ±1-ULP acceptance above Q8, vetoed below Q8 — principle approved
by the user; the guardrails are engineering and are recorded there).
**No timing appears in this file or in any log it cites.** The `WALL` line in
`prototype/moe_gather/evidence/moe-gather-device-2026-10-26.log` is a lock record and prints
`no timing taken`.

Files this turn produced:

| file | what it is |
|---|---|
| `tools/ulp_policy.py` | the one ruler + the one policy + `--selftest` (99 checks, 0 findings) |
| `specification/evidence/ulp_policy_equiv.py` | equivalence harness: new ruler vs every existing ruler |
| `specification/evidence/ulp_policy_stage_patches.py` | generates the staged migration patches and proves each applies |
| `specification/evidence/ulp_policy_smoke_staged.py` | applies all patches to a mirror tree and runs the patched code |
| `specification/evidence/ulp-policy-{selftest,equiv,staged-patches}-2026-10-26.log` | the three runs |
| `specification/evidence/ulp-policy-migrate-*.patch` | five staged patches (not applied to the repo) |
| `prototype/moe_gather/{gate.py,gate,work/probe_ulp_hist.py,work/probe_cross_arm.py,work/negctl_crossarm.py}` | the first consumer |

---

## 1. The ruler, and where its widths come from

**Distance in the sign-magnitude-ordered bit image**, i.e. adjacency after
`key = (~b & mask) if (b & sign) else (b | sign)` — an integer count of representable steps. This is
the `§ulpdist_bits`/`§bf16_ulpdist` family, not the `§ulp_of` float-quantum family, and the reason is
that only a count can answer "is this element within 1 ULP of that element" at a power of two, at a
subnormal, at ±0 or across zero. The quantum is a *length*: asked how many steps lie between 1.0 and 2.0
it answers "1", the image answers 2^mant (128 for bf16, 8,388,608 for f32).

Format facts are **derived at run time from two independent sources and refused if they disagree** — never
typed:

```
FORMAT name=bf16 bits=16 mant=7 exp=8 bias=127 subnormal_min=9.184e-41 normal_min=1.175e-38 max=3.3895e+38 derived_from=bitimage(torch) crosscheck=torch.finfo
FORMAT name=f16  bits=16 mant=10 exp=5 bias=15  subnormal_min=5.960e-08 normal_min=6.104e-05 max=6.5504e+04 derived_from=bitimage(struct) crosscheck=numpy.finfo
FORMAT name=f32  bits=32 mant=23 exp=8 bias=127 subnormal_min=1.401e-45 normal_min=1.175e-38 max=3.4028e+38 derived_from=bitimage(struct) crosscheck=numpy.finfo
```

* **source A — the format's own arithmetic.** `image(1.0) = bias << mant` and `image(2.0) = (bias+1) <<
  mant`, so `mant = log2(image(2.0) − image(1.0))` and `bias = image(1.0) >> mant`, from `struct` round
  trips for f16/f32/f64 and from `torch.tensor(...).view(torch.uint16)` for bf16.
* **source B — library metadata.** `numpy.finfo` (`nmant`, `minexp`, `bits`) for f16/f32/f64,
  `torch.finfo` (`eps`, `tiny`, `bits`) for bf16. Any disagreement raises `FormatSourceError`.
* **bf16's explicit mantissa width is 7, not the 8 that `speccheck/moe_ref.py` §ulp_of's prose types.**
  The formula `2^(exp−7)` in that function was always right; the comment was the guess. A sabotage arm
  injects a typed 8 and requires it to be caught, in the style of `tools/gguf_width_table.py`'s injected
  stale Q5_K row.

Width classes are a separate derivation: the tensor's **own ggml type in the pinned file**
(`tools/gguf_reader.py`), bits-per-weight from the pinned `ggml-common.h` structs
(`tools/ggml_block_geom.py`), cross-checked against `tools/gguf_width_table.py` as the independent ruler;
`≥ Q8` means `bpw ≥ 8.0` (one constant, which is a *threshold*, not a width). Fixtures come from real
tensors in `models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`:

```
WIDTH tensor=blk.0.ffn_gate_exps.weight ggml_type=Q4_K(id=12) bpw=4.5000  q8_or_above=0
WIDTH tensor=blk.0.ffn_down_exps.weight ggml_type=Q5_K(id=13) bpw=5.5000  q8_or_above=0
WIDTH tensor=output.weight              ggml_type=Q6_K(id=14) bpw=6.5625  q8_or_above=0
WIDTH token_embd.weight                 ggml_type=Q8_0(id=8)  bpw=8.5000  q8_or_above=1
WIDTH output_norm.weight                ggml_type=F32(id=0)   bpw=32.0000 q8_or_above=1
```
`BF16` and `F16` are not tensor types in that file; the selftest prints `WIDTHABSENT` for them rather
than letting a veto fixture disappear from the denominator.

### Finding: a latent disagreement between the two width rulers, on Q4_0

```
WIDTHDRIFT type=Q4_0 ggml_block_geom=4.500000 bpw gguf_width_table=8.500000 bpw in_pinned_file=0
note=latent: no tensor of this type in the pinned file, so the file-size witness in gguf_width_table.py cannot see it
```
`tools/gguf_width_table.py`'s `BLK["Q4_0"]` is `(32, 34, "sizeof(ggml_half) + QK")`, but `block_q4_0`'s
`qs` is `QK/2` bytes — 2 + 16 = **18** bytes per 32 elements = 4.5 bpw, which is what
`tools/ggml_block_geom.py` derives from the pinned struct and what `tools/gguf_reader.py`'s own header
comment says. The witness that makes `gguf_width_table.py` trustworthy ("sum of tensor row bytes +
data_start == file size") cannot see it because no tensor in this file is Q4_0. **Not fixed here**:
`tools/gguf_width_table.py` is not this lane's file, and `WidthClass` already *refuses* any type whose two
rulers disagree, so the policy is fail-closed against it today. For a file that did contain Q4_0 tensors,
the policy would refuse to judge them until this is resolved.

---

## 2. The policy, and its selftest

`judge(field, declared_fields, width, census, attribution)` accepts only if **all** hold; each refusal is
a named branch:

| guardrail | refusal branch | D-17 |
|---|---|---|
| field in the lane's declared set | `refuse-field-not-declared` | (a) |
| the tensor's ggml type is ≥ Q8, from the file | `refuse-width-below-q8`, `refuse-width-missing` | (b) |
| non-finite divergences refused (NaN, ±inf) | `refuse-nonfinite-divergence` | (c) |
| signed-zero flip refused | `refuse-signed-zero-flip` | (c) |
| zero elements at ≥ 2 ULP | `refuse-beyond-one-ulp` | (c) |
| a strict majority of elements bit-identical | `refuse-systematic-shift` | (c) |
| an attribution computed by the lane, and true | `refuse-no-attribution`, `refuse-attribution-false` | D-16 3 |
| nothing judged | `judged-empty` | `pitfalls/40` |

Two guardrail decisions are mine (D-17 says the guardrails are engineering) and are worth naming because
a reviewer may disagree:

* **isolation is a strict-majority test, not a fitted percentage.** Acceptance requires
  `2 × (elements at exactly 1 ULP) < (elements judged)`. The two poles are far apart — the fixtures sit at
  2/16 (accept) and 16/16 (reject) — and the arm at 8/16 is rejected, which shows the line is drawn
  between "a few elements crossed a rounding boundary" and "the whole field shifted", not at a number
  chosen to fit.
* **a one-step overflow to +inf and a ±0 flip are refused**, even though both are exactly one ordered step.
  An overflow is a defect and the sign of a zero propagates. Stricter than the minimum; nothing asked for
  the looser reading.

Selftest verdict (`specification/evidence/ulp-policy-selftest-2026-10-26.log`, run under the venv):

```
BRANCHES declared=28 reached=28 unreached=none bogus_labels=none
RESULT PASS ulp_policy selftest checks=99 findings=0
```

Every arm the task asked for, with its verdict:

| arm | verdict | note |
|---|---|---|
| `POLICY isolated ±1 in a declared field at Q8 is ACCEPTED` | ok | 2 of 16 at exactly 1 ULP |
| `POLICY systematic ±1 on EVERY element is REJECTED (the mandatory tripwire)` | ok | the reason acceptance is safe |
| `POLICY half-the-field-at-1-ULP is REJECTED` | ok | isolation ≠ the ≥2 rule |
| `POLICY a single element at 2 ULP is REJECTED` | ok | |
| `POLICY NaN-vs-finite is REJECTED` | ok | and `ruler-<fmt>-nan-vs-adjacent-plus-inf`: the NaN pattern *one ordered step above +inf* is still refused |
| `POLICY a NaN payload move is REJECTED` | ok | not a rounding step |
| `POLICY Q4_K at exactly 1 ULP is REJECTED — and by the veto, not by accident` | ok | asserts the refusal *names the veto*; fixture is `blk.0.ffn_gate_exps.weight`, a real Q4_K tensor |
| `POLICY Q5_K / Q6_K at exactly 1 ULP is REJECTED` | ok | 5.5 and 6.5625 bpw |
| `POLICY F32 (32 bpw) accepts the same isolated ±1` | ok | the same census, the opposite verdict, width alone the difference |
| `POLICY a divergence in an undeclared field is REJECTED` | ok | "the count must stay zero", checked |
| `POLICY isolated ±1 on NEGATIVE values is ACCEPTED` / `systematic … REJECTED` | ok | bf16, f16 and f32 |
| `negatives-<fmt>-sign-flips-are-real-divergences` | ok | 6/6 pairs diverge, all ≥ 2 ULP |
| `dead-branch witness: the naive \`x<0\` form reports a vacuous zero` | ok | 0 of 6 vs the ruler's 6 — `pitfalls/30` §A firing on this fixture |
| `POLICY a one-step overflow to +inf is REJECTED` / `a signed-zero flip is REJECTED` | ok | |
| `POLICY an empty census is REJECTED` | ok | a check that cannot fail |
| `POLICY a missing width class is REJECTED` | ok | never defaults to permissive |
| ruler arms ×3 formats | ok | identical / 1 step / 1.0→2.0 = 2^mant / ±0 = 1 / 0→min subnormal / largest subnormal→smallest normal = 1 / largest finite→+inf = 1 / ±inf / NaN arms / negative step / across zero = 2·(bias<<mant)+1 |
| `a missing format source fails loudly (no typed fallback)` | ok | sabotage; no width appears in the message |
| `sabotage: a typed bf16 mantissa of 8 is caught` | ok | |
| `sabotage: an injected stale width-table row is caught` | ok | |
| `every declared branch was actually executed by this selftest` | ok | 28/28 reached; an unexecuted branch has been hoped for, not tested |

---

## 3. The first consumer: `prototype/moe_gather/`'s cross-arm check, per D-16 option A

`gate.py:cross_arm()` replaces `v.check("host arm and device arm agree word for word", nd == 0, …)`.

**The measurement that decided the shape** (`work/probe_ulp_hist.py`, over the committed words):
`gate` and `up` differ **0** times between the arms (0/9,216 each); `s` differs 206/9,216 and `raw`
6,425/36,864; the `raw` histogram runs to **3,840 f32 ULP** (case 44). So this divergence is *isolated*
but it is **not ±1 ULP**, and `ulp_policy.judge()` is therefore **deliberately not called** on `s`/`raw`:
±1 could not forgive it, and inventing a band to fit would be a band typed from what the build produces.
D-16 A is the authority; what is shared with D-17 is the **ruler and the census format**, not a tolerance.

Four checks replace one:

```
CHECK name=cross-arm: every declared-exact field is bit-identical between the arms ok=1 n=260096 0 differing words in 260096 words of declared-exact fields
CHECK name=cross-arm: nothing outside the declared counted cases moves at all        ok=1 n=0 0 differing words across 79 cases outside the 18 field_exact=gate_up cases
CHECK name=cross-arm: every accepted divergence is attributed by an independent third side ok=1 n=36 fields where only one arm differs from the pinned oracle: none
CHECK name=cross-arm: every accepted divergence is isolated, not a systematic shift  ok=1 n=36 fields where a majority of elements moved: none
```

Census format, per case and per counted field (guardrail d — accepted ≠ uncounted):

```
CENSUS format=f32 field=raw judged=2048 identical=1823 differing=225 categories[equal=1823 finite_step=225] ulp_histogram[1:43 2:105 3:1 4:45 6:1 8:13 10:2 12:1 16:3 32:8 64:1 256:2] max_ulp=256 subnormal_diverging=0 case=0 mode=MG_MODE_CORRECT differing_words_lane_rule=225 share=0.1099
ARMDIFF case=0 mode=MG_MODE_CORRECT field=raw verdict=counted-and-attributed attribution=expf route: glibc vs ROCm libdevice; host_vs_oracle=1257 device_vs_oracle=1258 arms_apart=225 isolated=1 strict_majority_identical=1823/2048
ARMDIFFSUM exact_fields=271 exact_words=260096 counted_fields=36 counted_words=46080 counted_differing=6631 exact_differing=0 nan_payload_pairs=0 worst_diverging_share=0.3604 (case 44 field raw)
ARMDIFFALL whole_buffer differing_words=6631 of 306176 (the number D-16 was decided on; it is reported, not asserted -- the assertion is now per declared field)
```

The attribution is **data, not prose**: `host_vs_oracle` and `device_vs_oracle` are recounted from the
pinned `expect.f32` on every run, and a field where only one arm disagrees with the oracle is a refusal
(mutation B in `work/negctl_crossarm.py` proves it). The `expf` glibc-vs-libdevice story from
`work/diag_armdiff.py` is the *explanation*; the third-side agreement is the *check*.

### RESULT before and after

| | line |
|---|---|
| before (committed, `3f1b977`, `prototype/moe_gather/evidence/moe-gather-device-2026-10-26.log:1137`, with the finding itself at `:1131` `CHECK name=host arm and device arm agree word for word ok=0 n=306176 differing words=6631 of 306176`) | `RESULT FAIL moe_gather device gate checks=908 findings=1` |
| after this turn (same log file, latest run — the wrapper truncates it at each run) | `RESULT PASS moe_gather device gate checks=911 findings=0` |

The device gate was run twice, both correctness-only, both under the lock held by `./gate` as sole
acquirer (`acquire-noblock`, `LOCKREF gate.py references to the lock tool must be 0: 0`), the second to
refresh the log after the `ARMDIFFSUM` fields were relabelled; the newest run is the one the code on disk
prints. Both carry `no timing taken` and a lock record, most recently
`WALL started=2026-10-25T17:06:17Z finished=2026-10-25T17:06:33Z (lock record only; no timing taken)`, with
`UTIL_BEFORE gpu use: 92%` / `UTIL_AFTER gpu use: 35%`. The pre-change `FAIL` line is quoted from git,
because `gate` overwrites its own evidence log at each run — see §6. Host arm unchanged and green before
and after: `RESULT PASS plan check (no device) checks=456 findings=0`, `RESULT PASS host selftest
checks=20 findings=0`.

### Negative controls on the new check, each witnessed firing (`work/negctl_crossarm.py`)

| mutation (in memory; the files are never touched) | check that went red |
|---|---|
| A one `gate` word of the device arm moved by 1 ULP | `every declared-exact field is bit-identical` |
| B device arm made to match the oracle in `s` | `every accepted divergence is attributed` |
| C every `s` element of one case shifted by 1 ULP | `every accepted divergence is isolated` |
| D one word outside the 18 declared cases moved | `nothing outside the declared counted cases moves at all` |

One harness change came with it: `prototype/moe_gather/gate` now runs `gate.py` under the venv
interpreter, so the consumer's ruler gets its second source (`crosscheck=numpy.finfo` in the `RULER`
line above) instead of printing a one-source derivation.

---

## 4. Equivalence with the four existing rulers (`ulp-policy-equiv-2026-10-26.log`)

Fixture populations are **derived**: all 65,536 bf16 images, and for f32 every exponent crossed with a
mantissa set naming the field edges (0, 1, 2, 3, half−1, half, half+1, max−1, max, /3, /5) with both signs
and ±1/±2 neighbourhoods — 13,824 images, 69,089 pairs; the harness asserts the corpus contains signed
zeros, the subnormal→normal edge, the 1.0→2.0 pair, infinities, NaN images and straddling pairs before it
trusts its own verdicts.

| ruler | pairs | agree | disagreement classes (count) | verdict |
|---|---|---|---|---|
| `tools/audit_published_quant.py:ulp_dist` | 69,089 | 68,799 | nan-arm (290) | **equivalent on every finite pair** |
| `prototype/moe_router/gate.py:bf16_ulpdist` | 327,548 | 326,147 | nan-arm (1,401) | **equivalent on every finite pair** (same transform) |
| `prototype/moe_router/gate.py:ulpdist_bits` | 69,089 | 55,026 | straddles-zero one-step-fewer (13,771), ±0 folded together (2), nan-arm (290) | **DIVERGES** |
| `speccheck/moe_ref.py:ulp_of` | 327,548 | 326,250 | subnormal/binade-edge miscount (1,280), returns 0.0 at ±0 and non-finite (18) | **not a step count** |

`ulpdist_bits`' map `a if a < 0x80000000 else 0x80000000 − a` folds +0 and −0 onto one key, so every pair
straddling zero is counted **one step short** — the direction matters: a band measured with it is one step
**laxer** across zero than the band it claims to be. `ulp_of` is a length, not a count: it mis-states the
step over the whole subnormal range (up to 128× at the smallest bf16 subnormal) and returns 0.0 at ±0.

`tools/moe_stats.py` carries a third copy of the typed width: `BF16_REL_WIDTH = 2.0 ** -9`, comment "8
explicit mantissa bits". With mant = 7 the half-ULP relative width is 2⁻⁸. **The constant has no reader in
that file**, so no number it ever printed was wrong; it is dead code carrying a wrong width.

---

## 5. Staged migration (patches generated, verified, NOT applied)

Five patches in `specification/evidence/ulp-policy-migrate-*.patch`, each generated from the file as it is
now, `patch -p1 --dry-run` verified, `ast.parse` verified, and all five **applied together to a mirror
tree and executed** (`ulp-policy-staged-patches-2026-10-26.log`):

| patch | target | applies | patched code runs | consequence the reviewer owns |
|---|---|---|---|---|
| `audit-ulp-dist` | `tools/audit_published_quant.py` | yes | `ulp_dist` agrees with the ruler on 3,997 sampled pairs; NaN pairs keep a numeric answer on purpose | none |
| `router-import` | `prototype/moe_router/gate.py` | yes | import only | none |
| `router-ulpdist-bits` | `prototype/moe_router/gate.py` | yes | agrees with the ruler; NaN answers `None` | **changes measurements**: straddling pairs read +1 step; the router's two bands were derived with the short ruler and must be re-derived |
| `moe-ref-ulp-of` | `speccheck/moe_ref.py` | yes | `ulp_of` identical to the typed-7 body on 65,282/65,282 bf16 images | prose fixed, limits documented. **The patch changes no number any caller sees**: the body computes the same expression with the width derived, and mant derives to 7. Its consumers are `selftest_moe.py:263` and `gdn_ref.py:1630` (injection step, `clean + step`), `gdn_ref.py:1117` (report field) and `moe_stats.py:227` (divides by it) |
| `moe-stats-rel-width` | `tools/moe_stats.py` | yes | 2⁻⁸ derived | dead constant; deleting it is arguably better |

`speccheck/` has a live worker in it; `speccheck/moe_ref.py` is untouched by this turn and the patch waits
until that lane is quiet.

---

## 6. NOT ESTABLISHED

* That the shared ruler is the ruler those four files should adopt. The harness proves where they agree
  and disagree; the router's bands are the one place the fold changes a verdict, and that is its lane's
  call.
* That D-17's ±1 acceptance is exercised by any real lane. Its first consumer is `tools/ulp_policy.py`'s
  own selftest. The gather lane's accepted divergence is *not* a ±1 case (3,840 f32 ULP at the worst), so
  **this machinery is not what turned `moe_gather` green — D-16 A was.** The policy's first real user is
  still ahead, and the Q4 veto has only been witnessed on synthetic fixtures at 1 ULP.
* The isolation threshold's discriminating power on real data. On the gather's worst field, 36.0 % of
  elements differ between the arms and still pass a strict-majority test. A defect that moves a large
  minority — not all, not a few — of a field would be accepted by it. Nothing measures where the boundary
  between "route" and "defect" actually sits for this transcendental.
* That "≥ Q8" means "bpw ≥ 8". Every type's bpw is derived, but the threshold itself is a reading of the
  user's words ("Q8 and above"). It is one constant, in one place, and it is the thing to change if that
  reading is wrong.
* Whether the Q4_0 width-table drift matters in practice — no Q4_0 tensor exists in the pinned file. It
  fails closed (the policy refuses to judge such a type), so it is latent, not live.
* That the isolation attribution survives a different transcendental. `expf` is what this lane measured;
  a lane whose counted fields cross a transcendental that *both* arms implement identically would show
  zero cross-arm divergence and would never exercise the path at all.
* The device gate was run **twice, both after the change** — there is no re-run of the pre-change build
  here, and no A/B and no second halogen-down window; a `halogen` worker cannot bring halogen down. The
  pre-change `RESULT FAIL … findings=1` line is the committed record from the previous lane, read out of
  git, not re-measured in this turn.
