# ULP-policy migration re-verified, and the two width rulers on Q4_0 — 2026-10-26

Two jobs, one file, because both are "a ruler was wrong and nothing said so".

* **Re-verification** of the five staged patches in this directory
  (`ulp-policy-migrate-*.patch`), after `git apply -p1`. Every claim the migration note makes is
  reproduced from the two sides of the tree, and each repaired defect is shown *changing a behaviour*.
  Driver: `specification/evidence/ulp_policy_migration_witness.py`; its run is
  `ulp-policy-migration-witness-2026-10-26.log`.
* **The Q4_0 disagreement** between the two width rulers, turned into a named check that fires on
  synthetic tensors and settled by arithmetic. Driver: `specification/evidence/q4_0_width_witness.py`;
  its run is `q4-0-width-witness-2026-10-26.log`, which is **FAIL — findings=1, and red is the correct
  state** until the stale row is corrected by the file's owner.

No device was opened and no clock read anywhere: **no timing taken**.

## 1. The three defects, each shown changing a behaviour

**Router `ulpdist_bits` (the sign-magnitude short count).** Over the derived 69,089-pair corpus the two
bodies disagree on **13,773 pairs** — 13,771 straddling pairs plus the two signed-zero images — and the
pre-migration digest reproduces the migration note's corpus exactly
(`{"n": 69089, "n_none": 0, "max": 4294967294}`), so this is the same measurement, not a new one. The
bit images, with the shared ruler beside them:

```
kind=zero_sign_flip  a=0x00000000(+0)  b=0x80000000(-0)                     old=0  new=1   delta=+1
kind=finite_step     a=0x00000001      b=0x80000001                          old=2  new=3   delta=+1
kind=finite_step     a=0x00000002      b=0x80000002                          old=4  new=5   delta=+1
kind=nan             a=0x80000000      b=0x7fffffff                          old=2147483647  new=None
```

`+0` and `-0` used to be the same key, so the ruler reported zero steps across the only place where
"how far" has a fixed answer; and every straddling pair was counted one step short, so a band measured
with it was **one step laxer across zero than it claimed to be** — which is exactly what
`work/oracle_floor.tsv`'s `exp_route_ulp=19` is a measurement *of*. The new body agrees with
`tools/ulp_policy.py` on all 69,089 pairs' finite entries (0 disagreements), and the 290 NaN pairs now
return `None`, which raises against a band instead of passing it.

**`moe_ref.ulp_of` (a length presented as a step count).** The patch does not touch `ulp_of`'s body — it
adds the counting ruler `ulpdist_bf16` beside it — so the honest evidence is two-sided: `ulp_of`'s output
over 2¹⁶/7 images is **byte-identical before and after** (same crc32), it mis-states the step at **256
images** (every subnormal and every binade edge; = the 1,280 miscounted *pair-occurrences* the pre-flight
harness counts, each image being the first element of 5 pairs) and reads 0 at ±0 and non-finite on 18
more; and **the ruler the patch adds counts every one of those edges as exactly 1 step**, including
largest-finite → ±inf, which `within()` then refuses. The defect the note names is gone; the length stays
a length. `gate.py`'s `ulp_of`-shaped band is unaffected because the router never called it.

**`moe_stats.BF16_REL_WIDTH`.** `0.001953125` (2⁻⁹) → `0.00390625` (2⁻⁸), ratio 2.0, mant derived = 7.
The constant has **no reader** anywhere in the repo (`occurrences of the name outside its own definition:
1`), so no published number moved — which is why the change is reported rather than celebrated.

**`audit_published_quant.ulp_dist` is behaviour-preserving**, and that is the right claim for it: identical
digest before and after, NaN arms still numeric (`n_none=0`), `_ruler` materialised on first call. It has
no callers, so this check exists to prove the delegation did not silently change the audit's meaning.

## 2. Nothing else moved, and one thing the patch got wrong

The scope check compares the **body** of every top-level definition (line numbers shift when a hunk is
inserted) against `git show HEAD:<path>`, and reports anything that changed outside the patches' own
`@@ +c,d @@` new-side ranges. It first read **FAIL, findings=2** when the router-lane repairs landed
undeclared, and reads PASS now that they are declared in `LANE_REPAIRS` — a declaration that matches no
changed definition is itself a finding, so the mechanism is falsifiable in both directions.

One of those declarations is a defect **in a staged patch**, not in the migration:
`ulp-policy-migrate-moe-stats-rel-width.patch` wrote `2.0 ** -(mant - 1)` = **2⁻⁶**, contradicting its own
comment ("Half an ULP of relative width is 2^-(mant+1)"). `tools/moe_stats.py` now reads `-(mant + 1)`, and
the witness still requires the value to differ from HEAD's typed 2⁻⁹, so a wrong-sign repair cannot pass
it quietly. The same off-by-two exists in the pre-flight harness's own print field
(`ulp_policy_equiv.py`'s `half_ulp = 2.0 ** -(mant - 1)`, labelled "2⁻⁸" and not used in any predicate);
that file is the staging lane's, so it is recorded here and not edited.

`tools/ulp_policy.py` is unchanged: `git status --porcelain -- tools/ulp_policy.py` is empty, and
`--selftest` still prints `RESULT PASS ulp_policy selftest checks=99 findings=0` /
`BRANCHES declared=28 reached=28 unreached=none bogus_labels=none` under the venv.

## 3. Q4_0: 8.5 bpw or 4.5 bpw, and what makes it a fact rather than an argument

`tools/gguf_width_table.py` prices Q4_0 at **34 bytes per 32 elements = 8.5 bpw**; `tools/ggml_block_geom.py`
derives **18 bytes per 32 = 4.5 bpw** from `engines/llama.cpp/ggml/src/ggml-common.h:195`. The pinned GGUF
contains **no Q4_0 tensor**, so `gguf_width_table.py`'s authority — the file-size identity
`sum(row bytes) + data_start == file size` — is vacuous for that row, and `ulp_policy.WidthClass` merely
refuses to price the type. Nothing *named* the disagreement. It now has a name and a witness.

```
WIDTHDRIFT type=Q4_0 elts_geom=32 elts_table=32 bytes_geom=18 bytes_table=34 bpw_geom=4.500000
  bpw_table=8.500000 struct=block_q4_0 static_assert='sizeof(ggml_half) + QK4_0 / 2'
  table_expression='sizeof(ggml_half) + QK' in_pinned_file=0
  note=LATENT: no tensor of this type in the pinned file ... this check is the only witness
ARITHMETIC elts=32 bytes=18 bpw=4.500000 static_assert='sizeof(ggml_half) + QK4_0 / 2' evaluates_to=18
ARITHMETIC members: ggml_half d = 2 B, uint8_t qs[QK4_0/2] = 32/2 = 16 B; total 18 B per 32 elts = 4.5 bpw
ARITHMETIC table_row=(32,34,'sizeof(ggml_half) + QK') does not evaluate against the pinned header:
  unknown symbol 'QK'
```

**Which source is right: `tools/ggml_block_geom.py`.** The arithmetic: `QK4_0 = 32`; the struct is
`ggml_half d` (2 B) + `uint8_t qs[QK4_0/2]` (16 B) = **18 B per 32 elements = 4.5 bpw**, which is also what
the header's own `static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2)` evaluates to. The
table's expression string is the same expression **with the `/2` dropped** — the two-nibbles-per-byte
packing — giving 2 + 32 = 34, and it does not even evaluate against the pinned header, which defines
`QK4_0` and not `QK`. The row belongs `(32, 18, "sizeof(ggml_half) + QK4_0/2")`. That edit belongs to
`tools/gguf_width_table.py`'s owner and is deliberately **not** made here.

**What makes it bite**, rather than argue: `specification/evidence/q4_0_width_witness.py` writes synthetic
GGUFs containing Q4_0 tensors — block counts derived from the geometry
(`[1, 2, 3, 4, 7, 8, 16, 32, 64, 128]`, from `block_elts("Q4_0") = 32`), each file's length set by the
*pinned struct* — and then runs `gguf_width_table.py`'s own `parse_gguf` and `family_bytes` over them with
each table. The identity **holds** under the derived geometry and **breaks** under the hand-typed row at
every size (`off_by = -16 × blocks`), so the tool's own witness now fires on Q4_0.

**Two findings the file-size witness could never have produced.** The check compares elts *and* bytes, not
only bpw, which caught a second stale row: **IQ4_NL** carries `(256, 144, "sizeof(ggml_half)*2 + QK_K/2")`
— the `block_iq4_xs` geometry — against `block_iq4_nl`'s own `(32, 18, "sizeof(ggml_half) + QK4_NL/2")`.
Its bpw is 4.5 either way, so a bpw-only comparison would have reported agreement on a row describing a
different struct. Both types are latent in the pinned file, which is the whole reason they survived.

**The D-17 Q4 veto, witnessed one width wider.** The pinned file has no Q4_0 tensor, so the veto had only
ever been seen firing on Q4_K/Q5_K/Q6_K fixtures. Here a 1-ULP census at a Q4_0 width is refused
(`vetoed_by_width=True`), the identical census at Q8_0 is accepted, and — while the two rulers disagree —
`WidthClass.from_type("Q4_0")` refuses to price the type at all. The corrected row is injected only for
that control and restored in a `finally`, with a check that it came back.

## 4. Reading the outputs

| line | meaning |
|---|---|
| `RESULT PASS ulp migration witness checks=24 findings=0` | all three defects verified changed, nothing else moved, each of the five patches moved at least one definition |
| `RESULT FAIL q4_0 width witness checks=29 findings=1` | **expected red**: the single finding is `two-width-rulers-agree-on-every-type`, naming `Q4_0` and `IQ4_NL`. Do not "fix" it by deleting the check or by aligning the two rulers to each other — one of them has to be corrected against the header |
| `RESULT PASS router P3 controls checks=14 findings=0` | the prediction-3 repairs' negative controls, including the pre-fix table being refused |
| `RESULT FAIL ulp_policy equivalence checks=13 findings=2` | **expected**, and the design of the pre-flight harness: its must-fire classes for the two repaired defects now come back empty |

Nothing here re-bands anything. The router lane's bands in `prototype/moe_router/work/oracle_floor.tsv`
were produced by `oracle_arm.py`'s own ruler and are untouched; §1 only states which way the old router
ruler was wrong, for whoever re-prices them.
