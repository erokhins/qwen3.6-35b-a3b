# engine/HANDOFF-m1.md — M1 staging driver (`engine/stage_run.py`), handoff preserved

Moved out of `engine/HANDOFF.md` on 2026-10-26 so the M2 block-gate handoff and the M1 staging handoff
stop overwriting each other. Everything below this line is the M1 handoff **verbatim** — it is the text
that `engine/HANDOFF.md` had itself restored byte-for-byte from `git show HEAD:engine/HANDOFF.md`
after the M2 deadline checkpoint overwrote it. No word changed, no number changed. M2's own state is in
`engine/HANDOFF.md`; the two lanes share only `engine/loader.py` and the pinned artifact.

---

# `engine/stage_run.py` — HANDOFF (M1 fix: the device read-back must exist, count, and be falsifiable)

Supersedes the previous (staging-lane) version of this file. The finding implemented here is
`devices/evidence/device-window-2026-10-26c.md` §2: the first device run's `RESULT PASS` had
**no device digest anywhere in its evidence** (all 464 RUN records said
`dev=- staged=not-attempted` because `print_rows` ran before `stage_device`), one check answered
by construction (`device-path-was-exercised`, now deleted), `controls=0 fired=0`, and 31 checks
vs check-only's 34.

## What changed in the driver

* **Per-run read-back records.** `stage_device` fills `dev_sha` + `dev_read` on every staged row;
  `print_readbacks` prints one `READBACK idx=... readback_sha256=... expected_plan_sha256=...
  expected_header_sha256=... verdict=MATCH` record per plan run (464 = 463 stageable + 1
  draft-refused) into stdout and `--keep-evidence`. RUN lines print AFTER staging now, so `dev=`
  is real. Staging split into `stage_run_to_device` / `device_read_back` so a control can change
  the device buffer **between** H2D and D2H; the read-back hash covers exactly the bytes the D2H
  was asked for, and the verdict is a byte-count rule first (`device_read_back` docstring says
  why: a real `hipMemcpy` never short-copies without an error, so a short read-back can only ever
  be a short request).
* **Denominator enforced, derived.** `device_readback_count_eq_plan` requires digest count ==
  (plan runs − draft-sourced runs), counted from the plan's own rows, never typed;
  `device_readback_lines_reconciled` requires printed == parsed == one per plan run;
  `device_readback_digest_eq_file_digest` requires every stageable run's read-back digest to
  equal BOTH host digests; `run-rule-readback-short` / `run-rule-readback-digest` name the rules
  the controls must redden. A `--device` run with fewer digests FAILS, it cannot pass quietly
  (witnessed falsifiable by arm F of the stub harness: strip one row's digests → `FAIL`).
* **Falsifiable.** Two device-mode controls: `device-corrupt` (one byte flipped in the DEVICE
  buffer by a deliberate 1-byte H2D — the same four calls the real hip path makes — clean stage
  must match, corrupt must mismatch, re-stage must match again) and `device-short-d2h` (D2H
  returns length−1 bytes; the count rule reddens). Check `device_corruption_control_reddens`
  requires both to fire.
* **The 31-vs-34 inversion is structurally dead.** Device mode runs the 8 CPU controls plus the
  2 device ones (10/10 must fire), and
  `device_mode_checks_superset_of_check_only` shadow-evaluates the whole check-only roster in-process
  and FAILS if any name is missing (only carve-out: `device-path-not-exercised`, which `--device`
  makes self-contradictory). `device-path-was-exercised` is gone.
* **Evidence-record schema (device mode).** per run:
  `RUN ... plan=<sha> ref=<sha> dev=<sha> verdict=VERIFIED staged=STAGED ...` then
  `READBACK idx=N slab=S length=L dev_read=L readback_sha256=<sha> expected_plan_sha256=<sha>
  expected_header_sha256=<sha> verdict=MATCH` (draft row: `dev_read=0 readback_sha256=-
  verdict=refused-not-wired`); trailer `readback_printed/parsed/expected/done/match/refused`.
  The window-c grep (`grep -oE "dev_sha256|device_sha256|readback_sha256|dev_digest"`) now
  matches (468 hits against the stub full-plan evidence vs 0 for window c).
* **RESULT line** gained `readback=done/expected` (or `readback=off` in CPU modes).

## Where each mode stands (all green; before → after named-check counts)

| mode | before | after | controls |
|---|---|---|---|
| `--check-only` | 34 | **34** | 8 declared / 8 fired (unchanged) |
| `--selftest` | 33 | **33** | 8 / 8 |
| `--verify` (CPU, run alone) | 31 | **31** | 0 / 0 (no controls in verify mode) |
| `--device` (against the STUB hip) | 31 | **40** | 10 / 10 |

Verbatim lines: see `specification/evidence/stage-run-check-only-m1fix-2026-10-26.log`,
`specification/evidence/stage-run-stub-readback-2026-10-26.log` (+ its child evidence
`stage-run-stub-device-fullplan-2026-10-26.log`). The stub witness's every arm line states it ran
against the stub only — **the real hip path (real `hipMalloc`/`hipMemcpy`, real device-buffer
corruption) has still never been exercised**; every device check above was witnessed against the
Python stub.

## What the main agent must run in the next clean window

Same window rules as window-c (zero workers, halogen down, the machine-wide stream lock held by
you under your tag, blocking, foreground — `AGENTS.md §GPU-execution-is-serialized`):

```
/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python engine/stage_run.py --device \
    --verify --keep-evidence specification/evidence/stage-run-device-<date>.log
```

Expect `RESULT PASS mode=device checks=40 findings=0 controls=10 fired=10 run_set=464 plan=464
readback=463/463 verified=464 residue=0 uncovered=0 device=on`, 464 READBACK records (463 MATCH +
1 refused-not-wired), and `grep -cE "dev_sha256|device_sha256|readback_sha256|dev_digest"` over
the evidence returning 463 — the exact grep window-c §2 used and got 0 on. The two device
controls corrupt REAL device memory in this run, each with a clean re-stage immediately after; a
control that cannot restore exits the run red rather than leaving a slab wrong. CPU gates (any
time, no lock, no device): `--check-only`, `--selftest`, and
`/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python specification/evidence/stage-run-stub-device-2026-10-26.py`
(its full-plan child needs ~30 GiB free RAM and refuses under that; leave `SR_STUB_SKIP_CHILD`
unset for a witness-complete run).

**Run `stage_run.py` modes one at a time.** `ordering-log-append-witnessed` asserts each process's
ordering-log delta equals its own token grants; a second concurrent `stage_run` process (observed
with a CPU `--verify` overlapping the stub harness) makes it correctly red for both — the check is
real, but it assumes one process at a time. No timing taken.
