# `engine/stage_run.py` — HANDOFF (M1: execute the staging plan, certify it byte for byte)

Written by the implementation worker at ~minute 70 of the run, before the gate wiring. If you are the
main agent: §What to run in your window is the part you need; §Done and §Not done tell you whether the
lane finished.

## Status

**Code complete and CPU-green.** `--check-only` ends in

```
RESULT PASS mode=check-only checks=34 findings=0 controls=8 fired=8 run_set=464 plan=464
       verified=464 residue=0 uncovered=0 device=off ordering_log=242->243
```

transcript: `specification/evidence/stage-run-check-only-2026-10-26.log`.

## What exists

* `engine/stage_run.py` — one file, no new dependencies. Modes: `--plan-only`, `--verify`,
  `--selftest`, `--check-only`, `--device`, plus `--keep-evidence PATH`, `--fault NAME` (one control),
  `--cap BYTES`. An unknown flag is refused (`tag=unknown-argument`), no mode is refused
  (`tag=no-mode`), `--check-only --device` is refused.
* The plan is `loader.Loader.stage_runs()` verbatim — CHECK `plan-is-the-loaders-plan` compares the
  staged list against a freshly built `layout.ResidencyLayout.copy_plan()` field by field. No table,
  no plan, no width is re-derived or copied here.
* Two digests per run: the **plan lane** hashes `[run.file_off, +run.length)`; the **header lane**
  hashes the same bytes reached through the member tensors' own records
  (`loader.ruler`/`draft`, priced by `tools/gguf_width_table.py` while the plan's come from
  `tools/ggml_block_geom.py`). A run is `VERIFIED` only on equality **and** on span, membership,
  consecutiveness, per-tensor two-ruler agreement, the file bound, and no short read on either lane.
* Completeness: `residue` (bytes verified runs cover that no admitted tensor owns, plus any byte
  counted twice) and `uncovered` (admitted bytes no verified run covers), both derived by interval
  arithmetic over `layout`'s placements — `merge_ranges`/`intersect_len` here are ports of
  `engines/llama.cpp/src/llama-model-loader.cpp::merge_ranges`/`sub_ranges`. Printed every run; both
  are 0 on the whole plan.
* Device path (`--device` only): `hipMalloc` per slab, one `hipHostMalloc` staging buffer at the
  largest staged run's size, one bulk H2D per run, one bulk D2H read-back, a third sha256, `STAGED`
  only when all three digests agree. `weights.draft` is **refused**, obeying `loader.stage_draft`'s
  `NotWired("draft-requantizer-absent")`. Guarded in the function, not only the CLI:
  `stage_device(..., enabled=False)` raises `Refused(tag="device-flag-absent")`, and the device-guard
  control proves it while asserting `DEVICE_EXERCISED` did not move. `--check-only` goes red if the
  device path was exercised in-process.

## Controls (all eight fire; `--selftest` prints one `CONTROL` line each)

`short-read` (1 byte served short), `wrong-offset` (run read at `file_off+1`), `wrong-width` (one
member repriced F32→Q4_K **on the ruler side only**, so the plan keeps its span), `skipped-run`
(uncovered moves by exactly the dropped run's 1024 B), `corrupted-byte` (one byte flipped **in the
plan lane of the in-memory copy** — a corruption both lanes see is invisible to a digest comparison by
construction, so planting it in one lane is what proves the header walk is a real second read),
`duplicated-run` (residue moves by exactly the run's size; without it `residue=0` would never have
been seen non-zero), `window-guard` (a read outside every loaded window must be refused), and
`device-guard`. Injections land on copies; the artifact is never written.

## What the main agent must run, and what to expect

1. **CPU, any time, no lock, no device**:
   `/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python engine/stage_run.py --check-only
   --keep-evidence specification/evidence/stage-run-check-only-<date>.log`
   Expect 464 `RUN ... verdict=VERIFIED`, `residue=0 uncovered=0`, `controls=8 fired=8`, `device=off`.
   It reads the 22,134,528,992 B file twice through the audit reader and appends **one** line to
   `quantization/evidence/audit-ordering-2026-10-26.log` per process (the RESULT line reports the
   before/after count).
2. **The M1 device run — yours, in a clean window** (zero workers, halogen down, `gpu_lock` held by
   you, blocking, foreground):
   `... engine/stage_run.py --device --keep-evidence specification/evidence/stage-run-device-<date>.log`
   Expect, in order: 5 `hipMalloc`s (`weights.experts` 19,568,525,312 B; `.attention` 1,359,216,640;
   `.shared` 133,693,440; `.tables` 957,521,920; `.exact` 104,581,632 — **not** `weights.draft`), one
   pinned staging buffer at 540,344,320 B (the largest staged run, `output.weight`), then 463
   `RUN ... staged=STAGED` and one `staged=refused-not-wired`, then `STAGED slabs=5 ... refused=1`.
   Device footprint is 20,523,418,848 B of slabs + the staging buffer, inside the ≈32 GiB
   co-resident pool (`pitfalls/10-device-memory.md`); a refusal prints the hip error string, the
   `mem_info_gtt_used` and `MemAvailable` around the attempt, and exits non-zero — an instant refusal
   is the accounting limit, not memory pressure, and that distinction is in the line.
   This is also the answer to `quantization/residency-layout.md §7` item 2: it is **5** allocations of
   a lumpy distribution, not 712 equal ones, and not the 48 GiB single allocation that item 1 asks for.
   **Not yet executed by anyone** — the ctypes HIP path (`libamdhip64.so`, `hipMalloc`/`hipMemcpy`/
   `hipHostMalloc`) has only ever been read, never run, on this box. First execution may well find a
   bug; expect to fix it there, and treat the first green `staged=STAGED` line as untrusted until the
   log has been read.
3. **Gate**: `engine/gate.py` gained checks for this file (§README table). The expensive whole-plan
   arm is opt-in: `engine/gate.py --probe --stage-verify`; without `--stage-verify` that arm is
   `SKIPPED`, printed as such, not credited.

## Not done by this lane

* No device run (by instruction), so the `--device` branch is unexecuted.
* `engine/README.md` §Not established still says the plan "has never been executed"; the lane that
  runs `--device` should update it, along with the `--probe` bullet, which this file supersedes for
  staging (the probe samples 90 runs and reads 512 B windows; this executes all 464 and hashes every
  byte of them).
* The DECISIONS.md entry for the staging mechanism (agent-decided, must not wear the user's
  authority) — this lane's write boundary is `engine/` and `specification/evidence/`, so the text is
  in the worker's report and the main agent commits it. Suggested heading:
  `D-19 (agent, 2026-10-26): staging is one bulk copy per planned run, digested on both sides`.
