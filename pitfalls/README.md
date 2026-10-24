# pitfalls/ — the trap library, self-contained

Every trap that binds work in this repo lives **here**, in this repo, with its numbers and its
correct form written out. Nothing in this folder requires reading another project.

## The rule that created this folder

> **An agent may *look at* `../qwen3.5-0.8b` or `../qwen3.8-flash-next-old` for code, style and
> precedent. This repo may not *depend on* them for facts.** Any claim a worker needs in order to
> do a task correctly is copied into this repo (with its numbers, its date and a provenance line),
> not cited as `../other-repo/…:138`. A pointer may accompany a copy; it may never replace one.

Why, stated once so it does not have to be rediscovered:

1. **A cross-repo cite is unverifiable from inside this repo.** A reader who cannot open the target
   cannot tell a real citation from a fabricated one, and no checker here can even see the edge.
2. **The other repo is not ours to keep.** It can be archived, renamed, rebased, or have the cited
   section rewritten in place — that repo corrects itself inside the same file and marks the dead
   figure, so a quoted line can become false without any commit here changing.
3. **Line-number cites rot silently.** The text at `path:138` keeps resolving to *a* line after any
   insertion, which is why "0 rotted" is not the same as "correct".
4. **A fresh context starts with nothing.** This repo is the entire working set of a new session; the
   sibling repo is not in its context window by construction.

Provenance convention: each file names where the material came from and the date it was copied, in
its header. That is a *record*, not a dependency — do not "tidy" it away, and do not add new
cross-repo cites to prose elsewhere.

## Index

| file | what it protects you from |
|---|---|
| `00-measurement.md` | taking a number that does not mean what it looks like (bimodal timings, sub-ms timers, contended windows, the two prefill rulers, the >240 GB/s rule and its exception) |
| `10-device-memory.md` | allocation and residency assumptions: the pool ceiling, silent `Managed` degradation, first-touch cost, staging rates, host-pinned vs GTT |
| `20-rates-and-occupancy.md` | quoting a rate against the wrong roof, occupancy from the wrong source, censuses that cannot fail, launch bills masquerading as body costs |
| `30-numerics.md` | bf16 comparison bugs that make a gate pass vacuously, dtype/dump mistakes, band sizing, byte-identity claims |
| `40-gates-and-admissions.md` | gates that cannot fail, controls that cannot fire, parse-count, derived populations, attribution |
| `50-harness-and-shell.md` | shell/harness failures that present as engine failures (argv limits, self-matching poll loops, per-thread "reductions", lock deadlocks) |
| `60-docs-and-citations.md` | citation and archive drift, and the self-containment rule itself |

## Adding a trap here

One row or one short section per trap, in the file whose topic it belongs to. Each entry must carry:
the **symptom** (what it looks like when it bites), the **mechanism** (why it happens), the
**correct form** (what to do instead), and — if the trap produced a number — the number that
falsified the wrong version. A trap without a number is a style opinion; move it to `AGENTS.md`.

Traps discovered in *this* project get added here directly, under a `## Discovered in this project`
heading, and must state which of them are still **predictions** rather than observed failures.
