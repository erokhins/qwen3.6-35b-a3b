# 40 — gates, controls and admissions: checks that cannot fail

> **Provenance.** Copied and condensed on 2026-10-26 from the 0.8B project's
> (paths relative to `../qwen3.5-0.8b/`) `engine/gate.py`,
> `prototype/gdn_collapse/run.py`, `prototype/attn_tile_model.py`, `prototype/mlp_gemv/run.py`,
> `prototype/mtp_accept/*`, `tools/mixed_bw.cpp` and its traps index. Every entry below cost that project
> a real wrong result before it was understood.

The umbrella law of this file: **a check whose input list can be empty, whose denominator is unnamed, or whose
reference is derived from the thing under test, cannot fail — and it will report success.**

## A driver whose parser matches nothing prints PASS

The first `gdn_collapse` driver had a regex that matched **zero** `CASE` lines; its per-case assertions ran
over an empty list and the driver printed `PASS` over a matrix it had never read. Any driver that parses
another program's output must **require the number of parsed records to equal the number of records printed**,
and a fixed matrix must assert its expected case count.

Related, same shape: a `key=value` parser whose value pattern is numeric (`(\w+)=(-?[\d.eE+-]+)`) silently
drops every word-valued field — it found no `census` in a line reading `census=ok` and printed "the engine
refused to bill this run", a refusal that never happened, from the tool whose job was to notice refusals. A
value pattern must cover the value domain (`[^\s]+`), and "I parsed nothing" must be reported separately from
"the thing said no". And anchor greps (`^DECWALL`, `^MTPSTAT`): a tool that prints a usage block when a pin is
missing will make a bare `grep DECWALL` match its own help text and report a rate for a run that never happened.

## A requirement written after the object it constrains is a note, not a forcing

`gate.py` raised its `--prefill-logits all` requirement **after** it assembled the argv, so the engine ran
`last` and wrote 1 of 1,024 pinned rows while the gate's own note claimed `all`. The row section then counted
`1023 missing` and continued into `check(not outside, "every judged prefill row is inside the floors")` —
which passed over the one row that existed. Two correct forms, both required:

1. decide such a requirement **before the argv exists** and assert it on the argv, not on the variable the
   forcing mutates;
2. write the verdict as `total == n_rows and not outside`, aborting a coverage gap **before any numerics
   section runs** — a verdict that does not name its denominator cannot fail. Then witness it by stubbing the
   predicate to always pass: if no case changes colour, the check was decoration.

## A check whose independent variable is not the variable under test cannot adjudicate it

Batched-vs-carried wave comparisons that pass the same `--attn-tile` to **all** arms hold the form identical
and vary only `--wave-rows`, so "run both sides against the same form" was already true and could not help.
Missing was *attribution*. The fix (`--scope form`) does not delete the comparison: it still runs, prints
`[DEFERRED-PASS]`/`[DEFERRED-FAIL]`, and lands in a JSON file beside the dumps so "does it fail identically on
both forms?" is a diff of two files rather than a reading of two logs. **Label and record such results; never
delete them.**

## A check whose reference side is derived from the thing under test cannot adjudicate it either

The first straddling comparison read the partial layout **once**, out of the tiled header, and applied it to
both sides — so it compared two aliased waves and blamed the fixed geometry. The frozen form must be modelled
with its own spelling, always. Companion rule: a default in a gate driver and the default in the binary under
test must move in one commit, or the gate silently judges a different binary than the one that ships.

## A negative control with an empty must-catch set is not a control

Re-injecting a known aliasing bug scored a clean **0 of 1,024** because the gate ran only the pinned spans —
and a one-pass row writes no partial, so it has no slot to steal. The control became meaningful only when
wide anchors and a synthetic boundary wave (rows of exactly 512/513/514/515 lanes) became **mandatory** cases.
Corollaries:

- must-catch **and** must-not-catch sets must be **derived from the geometry**, not hand-listed. A hand-listed
  set rots into "everything" the moment the fixture changes, and two structural blind spots were found that way
  (a score-only injection cannot move a row with exactly one lane — softmax of a singleton is 1.0 whatever
  arrives; a mask-widening injection cannot move a row that already attended the whole buffer).
- after writing any control, ask **what geometry makes it fire**, then check the must-not-catch side is also
  derived ("a row at or below the threshold structurally cannot be hit" is a fact about geometry, not a wish).
- a route **no fixture reaches** has no tests — including no tests of the gate that guards it. Make the
  unreachable route reachable in a gate before trusting any claim about it.

## An admission over a fixture list typed from memory is not an admission

Three gate admissions claimed "18/18 — nine canonical fixtures × both routes"; **seven of those nine names did
not exist**, the driver exited with `fixture X has no manifest`, and a harness doing `grep -c PASS` printed
`PASS=0` — which reads like a red result about the engine and is actually an error exit from the harness. The
total looked plausible because it was arithmetic on a list nobody enumerated. Correct: derive the population
(`for d in data/refs/*/; do [ -f $d/manifest.json ] && …`), assert `cases == |population|`, and treat any
`PASS=0` as an unclassified run until its own tail has been read. Companion: an **existing** fixture can still
be unadjudicable by a given invocation style — run the base binary, see the identical crash, and then report
it as a pre-existing invocation gap instead of dropping the case.

## Two self-consistent layouts joined by a branch are invisible to both sides

A tiled wave keyed its split partial by head-**in-group** while the core keyed by the **full** head, and the
combine step had one branch per layout: each form agreed with itself, every per-form check passed, and above
the threshold two lane blocks of one row wrote the same floats — half of every affected row was the other
group's answer, bit for bit. It was diagnosed as an **address**, not arithmetic: the mirror's own first
straddling print showed `cos 0.449` on ONE element group, not a gradient. Fix both ways round: **one address
expression shared by writer and reader**, and reach the route in a gate.

Related, and cheaper: a geometry fact that lives in two headers can be committed in only one of them, and no
single-file review sees the missing half — **HEAD did not compile** and a `static_assert` caught it at the
commit boundary. "Move such a fact in one commit" is necessary and not sufficient: **the sufficient form is a
build of the committed tree** (`git worktree add $d HEAD --detach` then build, ~17 s, or a single-TU
`hipcc -fsyntax-only`), because the working tree is exactly where the uncommitted half hides.

## Gates must fail for *something*

A gate that fails for **every** configuration catches nothing, and a reader will tabulate the plausible-looking
ms column anyway. That happened when a read checksum was silently 1/256 of the bytes read: every warm row
failed while its bandwidth looked entirely believable.

## Provenance and attribution rules for reports

- **A corpus-level read-back is not a per-file attribution.** A digest reported "routes read back: [hand,
  amortised]" and that was taken as proof all six log sets were read; three of the six were byte-identical
  copies of one arm, so a "six-arm" table carried 16 numbers belonging to one route. A summariser must state
  **how many files it parsed, per (route,width,role)**, and name the sets it actually has.
- **A counter incremented at two sites double-counts.** Only a check whose right-hand side comes from an
  *independent* source can see it (a verdict tally, the written file, the transcript lines) — and the
  disagreement must print `NO-RATE` instead of a number.
- **A table header must label every column.** Six numeric columns (2 routes × 3 widths) under three width
  labels were read as three widths twice; every number was internally consistent, so no total or ratio could
  catch it — only a cell-by-cell comparison against an independent measurement did.
- **A digest of a device instrument is an input/output pairing claim**; `wc -l` over the log does not tell you
  which log a number came from. Parse from the machine-readable line, never from a dump file that also carries
  other rows.
- **A required flag belongs on both arms of an A/B.** A wall A/B passed a pinning flag to one arm only and the
  base arm came out 300 ms faster on *prefill* before a single decode number was read. And a required route's
  real cost is measured by pricing it against the default on the **same** arm.
- **A run-wide flag pin adopted to satisfy a per-body requirement silently changes every other body** — and it
  was also costing accept: prefill throughput moved +37 % at 1,024 lanes and +72 % at 4,032 for a whole run, and
  the default carried it for a full ledger entry before anyone priced it. Enforce per-body; expect a scoping fix
  to be **numerically non-neutral elsewhere**; and use a **three-arm** A/B (new scoping, plain path, old pin),
  because a two-arm pair cannot distinguish "the tax was removed" from "the tax moved somewhere unmeasured".
- **A geometry that grows from 1 to N slots needs a case that reaches slot ≥ 1.** `ptr + slot * bytes` on a typed
  pointer is a scaled offset: at `k=1` the only slot is 0, the expression is identical to the correct one, every
  one-row gate passes, and the defect surfaces at the second slot — as a SIGSEGV inside libamdhip64's memcpy, one
  copy *after* the bad one, which is why allocation dumps and printed pointers all read clean. Put the arithmetic
  on `void *` (or index a `char *`) at the site.
