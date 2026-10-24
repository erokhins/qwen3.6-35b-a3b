# 60 — documentation, citations and self-containment

## The self-containment rule (this repo's own, added 2026-10-26)

> **You may *look at* another project on this box. You may not *reference* it.**
> Any fact, number, trap, band or protocol a worker needs in order to do a task correctly must be
> **copied into this repo** — with its numbers, its date, and a one-line provenance note. A pointer may
> accompany a copy; it may never replace one.

Why, in order of how it has actually hurt:

1. **A fresh session has only this repo.** Cross-repo cites are how a new context ends up reading three
   unrelated files before it can answer one question, or — worse — inventing the content it could not open.
2. **A cross-repo cite is unfalsifiable from inside the citing repo.** No checker here can see the edge, and a
   reader here cannot tell a real cite from a plausible fabricated one.
3. **The other repo is not ours to keep.** It may be archived, renamed or rebased; that project in particular
   *corrects figures in place* and marks the dead number, so text you quoted last month can be false today
   without any commit here changing.
4. **A line-number cite is a claim about a file's line count.** Insert one line anywhere above the target and
   every `path:NNN` pointing below it silently repoints at *a* line that still resolves. This is why prefer
   `path §label` over `path:NNN`, and why provenance lines name **files and dates**, not coordinates.

Concretely, in this repo: `pitfalls/*` and `devices/*` hold copied content and are the authority;
`reference-implementation.md` is a **reading list** for code shape and precedent, and is explicitly not where
a fact is stored. If you find yourself writing `../qwen3.5-0.8b/…` in prose to support a claim, copy the
supporting passage instead and add a provenance line.

## Line-number cites rot silently, and the rot check cannot see the commit that causes it

A checker whose baseline is `HEAD` compares the working tree against `HEAD` — so when the commit *itself* moves
a cited file, both sides move together and it reports **0 rotted**. Observed exactly: inserting one row at
`AGENTS.md:78` moved four `progress/` citations out from under their coordinates and the tool said clean.

The guard for touching a cited file is therefore three things, not one:

1. `wc -l` before and after (a one-newline slip shifts everything below and is invisible to the tool);
2. read each cited coordinate directly against the **previous revision** (`sed -n 'NNNp'` vs
   `git show <prev>:file`);
3. after repairing citations, do **not** trust the tool's verdict — the coordinate predicate is true both at the
   old number (text gone) and the new one (whatever the shift displaced), so the witness that adjudicates is
   per re-point: **`worktree:new == ref:old`**, plus a run with `--against HEAD` reading 0.

## A `§label` that names no file resolves only in a reader's memory

`engine/gate.py` carried seventeen `§A1…§A4` cites into an audit document without ever naming the file they lived
in — and the band they justify was enforced from those rows. Write `§N of <path>`, never a bare `§N`, when the
target lives in another file. Witness: a checker that reported 91 unnamed-label cites dropped to 74 when the
seventeen were fixed.

## A checker's clean bill measures the population it scanned

Two instances of the same failure: a doc checker that walked only `*.md` (so every citation inside `gate.py`,
a kernel header or a writer script was invisible), and a "missing paths" count of 96 of which 75 were scratch
directories recorded as history and were not defects at all. **Before trusting any count, ask what was walked
and what was filtered** — and when you extend a checker, expect the number to get *larger*, not smaller.

## Moving or pruning a document is two jobs, not one

(a) **Re-point the cites**, and (b) **keep every file that is cited by line the same length**. Both halves bit in
one pass: a census keyed by path silently shrank after 13 documents moved (`git diff --name-only` names only the
new side of a rename, `git show REF:newpath` is empty for the old path, and the citations left the population
without an error — the fix was an explicit rename map), and correcting a README headline had to be a
**line-count-neutral** rewrite because nine frozen citations pointed at specific lines in it. Prefer in-place
correction with a visible "retired" marker over deletion, exactly as the sibling repo does, so a quoted figure
that has been overtaken is named as overtaken rather than quietly removed.

## Numbers in prose need a home that survives the prose

Every load-bearing number in a markdown file should name its instrument and its window, or it will be quoted
across windows later and read as a regression. The pattern that worked in the sibling project: one live ledger
file whose section numbering is a declared compatibility requirement (so `§6` keeps meaning "what a number here
is worth" even as sections are added), plus dated frozen snapshots for history, and an explicit rule that
superseded numbers are named rather than deleted.
