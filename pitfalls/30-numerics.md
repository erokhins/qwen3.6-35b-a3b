# 30 — numerics, bf16 and comparison-band traps

> **Provenance.** Copied and condensed on 2026-10-26 from the 0.8B project's
> (paths relative to `../qwen3.5-0.8b/`) `prototype/bf16_ulp.py`, `prototype/mlp_gemv/{floor_probe.py,results.md}`,
> `prototype/gdn_collapse/results.md`, `prototype/attn_wide/{a13_identity_metrics.py,A13-RESULT.md}`,
> `prototype/gdn_gated_norm/results.md`, `specification/ASSUMPTIONS.md` (A-18) and the numeric-format
> ruling in its `AGENTS.md`. **The bands themselves do not transfer** — only the method of sizing them.

## A BF16 ULP comparator written on a `uint16` view has a dead branch

`np.where(x < 0, …)` is **never true** for a `uint16` array, so every negative value collapses to +0.0 and
the gate passes vacuously. Correct form:

```python
vals = np.where(bits & 0x8000, bits ^ 0xFFFF, bits ^ 0x8000)   # not: np.where(x < 0, -x, x)
```

The 0.8B repo keeps a self-testing implementation of this comparator — copy the code, then run its
selftest before trusting any ULP figure produced by your own variant.

## `Tensor.to(torch.uint16)` casts; `.view(torch.uint16)` bitcasts

A fixture writer that confuses them writes garbage that reads like unwritten memory: float `0.195` → the
integer `0`, `-1.3` → `0xFFFF`, read back as BF16 `+0` and `NaN`. Symptom: a red device gate, a green CPU
mirror, and a kernel that was never at fault. Also: torch refuses `bf16.numpy()` — dump bits via
`view(torch.uint16)` and write the raw file yourself.

## Read a dump's dtype off its file size, not off the comparator's assumption

Engine logits rows were F32 (993,280 B for 248,320 elements). Read as BF16 they yield 496,640 "elements"
and `NaN` for cos / max_abs / max_rel — **and still print a plausible-looking `ndiff` of 82 %**. Correct:
divide the byte count by 4 and by 2, check which equals the model's vocab, and refuse to print any aggregate
that comes back NaN.

## A max-ULP band cannot see a whole-tensor 1-ULP error, and `ndiff` cannot see a chain

Measured on this family's own tensors:

- a BF16 subpartial inside a norm moves **2.05 % of elements by ≤1 ULP** and passes any max-ULP bound;
- a chained dot's *legal* `ndiff` is **29–47 %**, so `ndiff` there measures nothing;
- "max 1 ULP against the dump" is **not an achievable bound for a long dot**: reordering the reference's own
  F32 sum already moves ~1 element in 1e5 by more than 1 ULP, and always at an element that cancelled to
  ~1e-6 of the tensor max.

Correct division of labour: `ndiff` for stage-local near-exact ops (norms, residual adds, rope — their floor
is 1 ULP with **no** cancellation band), and cos / meanrel aggregates for chains. Any BF16 gate on a long dot
needs a **cancellation band plus aggregate metrics, both sized by a floor probe**, not by taste. Write the
floor probe first; the expert GEMV in this project is exactly the long-dot case.

## Two legal bf16 implementations agree only ~0.88 of the way on a long generation

Greedy token equality decays with depth on *any* bf16 implementation (the 0.8B project measured HF
sdpa-vs-eager at **0.8776** and its own spec at **0.9124** over 49 prompts × 24 tokens). So long-generation
token match is not a gate; short-fixture token equality and corpus-level aggregate gates are. When a
"byte-identical or die" admission dies on a tie, that is a *form gap*, not a bug — adjudicate it in three
parts: the flipped token must be the flag-off row's **runner-up**, the top1−top2 margin must be below **one
bf16 ULP of that row's top logit** (format-derived, not fitted), and who emitted it must be reported. Print
every margin, rank and per-line distribution so a reader can disagree.

## Byte-identity at short widths is not byte-identity at long context

A change bit-identical on five pinned geometries **and** at 4,096 lanes (including every free-run decision)
differed at 16,384 lanes: 1−cos 1.41e-05, argmax SAME, top-5 SAME, all 16,385 greedy tokens equal, one
printed max-logit differing by 1.5e-03 relative. So: state the width a byte-identity claim covers; at longer
widths judge with decision-level aggregates plus pooled 1−cos, and check the launch-geometry line-by-line so
coverage is disproved independently of the numbers. A bare `DIFFERS` at a width the short set never reached
cannot distinguish a re-associated summation from dropped work.

## A hoisted scalar constant is a different number, and a CPU mirror cannot see it

Writing `q̃ = qv * (rsqrtf(s) * SCALE)` where the shipped kernel computes `(qv * rsqrtf(s)) * SCALE` is a
different product tree: on device it moved **93,522 of 131,072 output elements while leaving the state
bit-exact**. The CPU mirror of the same recurrence could not see it, because the mirror reproduces the
*formula* and two sides that share a formula agree. What bit-identity is owed is the **source text's
association**, not the algebra — and that covers FMA contraction too. When a form claims bit-identity with a
shipped kernel, transcribe the shipped expressions character for character.

## A hook that reads `output` can pin the answer under the name of an intermediate

The MLP's activation is `down_proj`'s **input**; a hook that recorded the layer's output instead reproduced
real reference tensors and passed every self-check while being the wrong quantity. When pinning an
intermediate, assert *which tensor number it is* against the consumer's input, not against a name.

## Format changes: the gates are the decision procedure, with two exceptions

Any **16- or 32-bit** arithmetic change may be adopted if the existing gates stay green — a green gate set
*is* the decision (fp16 is 1+5+10 against bf16's 1+8+7, so inside its range it is *more* precise than the
format bands are calibrated to; the only new failure modes are overflow above 65504 and flush-to-zero below
~5.96e-08, which is a range census, not a band negotiation — the 0.8B repo's census over every pinned dump
found activations peaking at 82.5, logits 35.5, weights 0.17, nothing above 65500).

If a gate goes red, the test for widening is concrete, not a mood: **a widened band must still catch every
injection in the geometry-derived must-catch set.** Only a change that neither passes nor can be widened is a
user decision. Two things are never covered by this rule and must be asked: **int8/int4 quantisation** (its
error is orders of magnitude larger than a format change — this is `DECISIONS.md` D-01) and **widening a band
that a gate depends on for its discriminating power** — if the only way to make a switch pass is to loosen a
check that currently catches a real bug, that is reported, not routed around.

## Free-run parity depends on which index you align

Reference pins typically say `argmax(decode{s}.logits_last) == greedy_continuation[s+1]`, so a decode-kind
sample at step *s* pairs with `greedy[s+1]`, never `greedy[s]`; a REPL-style session's first generated token
pairs with `greedy[i]` instead. Aligned on the wrong index, a 128-token free run read 0/128 and a 64-step run
read 1/64 — both false. **Always state which alignment a comparison used.**

## array("f", [h << 16]) converts; it does not reinterpret

To turn a bf16 bit image `h` into the float it denotes, move bytes, not values:
`array("f", array("I", [h << 16]).tobytes())` -- **not** `array("f", [h << 16])`, which numerically converts the
integer and prints `1.01626675e+09` for the bits `0x3C93`, which denote `0.017944`. Found 2026-10-26 in a
shared-expert debug driver, where the bit columns were correct and only the value column lied.

That is the dangerous half of a half-correct printout: the column reviewers check agrees with the expectation,
and the absurd column is assumed to be a formatting choice. Same family as the `uint16` sign trap above -- a
comparator that reads plausible numbers out of the wrong interpretation is worse than one that raises, because
it produces evidence that survives review. Any debug print that shows both bits and a decoded value must derive
the value from the bytes, and it is worth a one-line self-check on a known pattern (`0x3F80` must print 1.0).
