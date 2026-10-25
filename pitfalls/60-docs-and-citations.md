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

## What a checker here can and cannot see

`tools/doc_ref_check.py --strict` is the in-repo checker. It walks every `*.md`, verifies that in-repo path
cites (`pitfalls/…`, `devices/…`, `tools/…`) resolve, fails on a markdown table row that does not end with
`|`, and prints how many line-form `path:NNN`, cross-repo `../qwen…` and bare-`§` cites exist — so the
population it examined stays visible instead of being implied by a zero. It exempts provenance blockquotes
(lines starting with `>`), which are allowed to name the sibling project.

What it cannot see, and never will: whether a citation's **meaning** survived an edit (only a human read of the
coordinate does that), anything outside `*.md` (gates and kernel headers carry cites too), or drift introduced
inside the commit being checked if the checker's baseline is `HEAD`. Hence the manual guard above: `wc -l`
before and after editing a cited file, plus a direct read of each cited coordinate against the previous
revision.

## A plan may name a file that does not exist yet — say so, in the exact words the checker reads

`PLAN.md` step 1.0 promises `tools/audit_published_quant.py` (planned), which is not written until the step runs, and
a plan must be writable before its own artifacts exist. `tools/doc_ref_check.py` therefore accepts such a
forward cite **only** when `(planned)` follows the cite immediately, and counts it into the info line so the
population is visible; unlabelled it stays `MISSING`. The label is deliberately narrow and deliberately not
a directory-wide exemption: a doc advertising a tool that does not exist is the same class of defect as a
gate that reports PASS over an empty list, and the escape hatch has to cost a word. Witnessed both colours
rather than trusted — `python3 tools/doc_ref_check.py --root /tmp/planwitness --min-files 1` over a scratch
file prints `MISSING` for the unlabelled cite and for a mislabelled `(plannedish)`, and accepts only the
labelled one (`labelled-planned forward refs: 1`). A checker exemption that was never seen to discriminate
is a hole with a name on it.

## An edit that reports success is not evidence the content reached a commit

**Discovered 2026-10-26 in this repo, in my own work, not a worker's.** A `DECISIONS.md` D-15 amendment was
written, the edit tool reported `Successfully replaced 1 block(s)`, I reported the amendment to the user, and
four commits later the block was **absent from the file and absent from every commit** (`git log -S
"Lone-worker refinement" -- DECISIONS.md` returns nothing, and it was not in the working tree either). The
substantive rule survived only because I had independently written it into `AGENTS.md`, which is what workers
actually read — so the failure was invisible except at the moment I went to append to the missing block.

**The rule: for a doc edit that carries a decision, the verification is `git show <commit>:<path> | grep -c
"<phrase>"`, not the edit tool's return value.** Three corollaries this repo already believed about gates, applied
to documentation:

- **A claim about a file is a claim about a commit.** "I recorded it in D-15" is a claim about `HEAD`; check it
  the way a gate checks a build.
- **Redundancy is what caught it.** The rule lived in two files, so losing one was survivable. A decision that
  lives in exactly one place has no witness, and `doc_ref_check.py` cannot help — the reference resolves fine,
  the *content* is what vanished.
- **Report what you verified, not what you did.** I told the user the amendment was recorded after making the
  edit; the honest sentence was "edited, pending commit". This is the same failure as a worker reporting a gate
  it never ran, one level up.
