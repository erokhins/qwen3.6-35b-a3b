# Device window — 2026-10-26, fourth window (`window-2026-10-26d`) — **M1 established**

Run by the main agent, foreground and blocking, zero subagents active (`worker_lanes.sh`:
`tensorfold connections=0`, `halogen in_flight=0`, no `ALIVE` worker row), `tools/gpu_lock.sh` held under tag
`q36-main window-2026-10-26d m1-readback-device`, endpoint `tensorfold` (remote), engine stopped through
`tools/halogen_gate.sh down` (`stopped in 3s`, `VRAM Total Used Memory 891,719,680 B` before the run) and brought
back with `up`. One window, one purpose: execute M1's exit gate, which window `c` had shown was never executed.

## The run

```
RESULT PASS mode=device checks=40 findings=0 controls=10 fired=10 run_set=464 plan=464 readback=463/463 verified=464 residue=0 uncovered=0 device=on ordering_log=271->272
```

against real device memory, evidence at `specification/evidence/stage-run-device-readback-2026-10-26.log` and
per-run records at `work/device-window/c/stage-device-records-d.jsonl` (290 KB class, uncommitted by the same rule
that keeps `work/device-window/` out of the tree).

## Why this is a measurement and not a green word

Window `c`'s failure was that "verified" named nothing. Each leg below is a separate claim with its own evidence,
and the first one is specifically designed so that a fabricated implementation cannot pass it.

| claim | evidence | what would have caught a fake |
|---|---|---|
| a device digest exists per run | 466 digest-bearing lines, **463 distinct** `dev_sha256` values | a single constant digest written 463 times — `sort -u` would read 1, not 463 |
| it agrees with the artifact | 463 `verdict=MATCH`, 1 `readback_sha256=-` | comparing the digest against itself |
| the excluded run is not hidden | `draft:draft-requantizer-absent`, `length=1,689,281,536`, `dev_read=0` | typing 463 and never naming what was dropped |
| device bytes are load-bearing | `device-corrupt fired=RED`: one byte of the **device-side** buffer at run 343 (`weights.exact`, `slab_off+0`) flipped by a deliberate 1-byte H2D | a control that corrupts the artifact or the host copy, which the digest would never see |
| a short copy cannot hide | `device-short-d2h fired=RED`: run 343 read back **1023 of 1024 B**; the count rule reddens it, not the digest | trusting the return code, or trusting the digest alone |
| the device path is not silently skipped | `device-guard fired=RED`: `stage_device(enabled=False)` refused 71 verified rows with `tag=device-flag-absent`; device calls before=1 after=1 | a mode that "passes" while never touching the device |

`checks=40` in device mode against check-only's 34 — the superset rule from D-24 — is what prevents the
31-vs-34 inversion from window `c` recurring, where the "more rigorous" mode ran fewer checks.

## What is now established, and what is not

**Established.** Artifact-to-device staging for 23,812,820,480 B across 463 runs, with each run's device bytes
read back and sha256-compared to the digest the artifact region independently produces; a control that corrupts
real device memory reddens it; a short read-back is caught by a byte count that does not depend on the digest.
**M1's exit gate in `AGENTS.md` §GOAL ladder is met.** The engine can now be trusted to put the bytes it claims
into the device.

**Not established, named so it does not ride along on M1's coattails.**
- **The 464th run (MTP draft block)** is not staged: `loader.stage_draft` refuses draft-expert addressing
  (`draft-requantizer-absent`). That is D-05 territory, not a defect in M1.
- **Nothing has been timed.** Every number in this window is a byte count or a digest; the ladder's M4/M5 rungs
  (token stream, ≥ 48 t/s per D-22) are untouched, and no rate from any window is quotable beside another's ruler.
- **Device codegen**: no kernel of ours has executed on the iGPU. The gather, GDN and `lm_head` prototypes remain
  host-compiled, and the `GDN_SUMSQ_WIDE_F64`-versus-pairwise-tree question (D-23) is still unmeasured on `gfx1151`.

Restart cost this window: `down: stopped in 3s`; `up` appended to `evidence/halogen-restart.log` (fifth entry).
