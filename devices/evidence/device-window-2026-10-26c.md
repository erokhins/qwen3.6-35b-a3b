# Device window — 2026-10-26, third window (`window-2026-10-26c`)

Run by the main agent, foreground and blocking, with **zero subagents active** (verified: `tools/worker_lanes.sh`
listed only `HISTORICAL` rows and `tensorfold connections=0`), `tools/gpu_lock.sh` held under tag
`q36-main window-2026-10-26c m1-device-and-coresident`, session endpoint `tensorfold` (remote), so stopping the
local engine was not self-decapitation. Two halves, deliberately in this order: the **co-resident** half first,
because it requires the local engine to be resident, then the engine was stopped through `tools/halogen_gate.sh
down` for the clean half. Raw logs: `work/device-window/c/coresident-probe-full.log`,
`work/device-window/c/stage-device-records.jsonl`, `specification/evidence/stage-run-device-2026-10-26.log`.

Restart cost this window: `down: stopped in 3s`, `up: ready in 70s` — third and fourth entries in
`devices/evidence/halogen-restart.log`. The batch-not-oscillate rule is being honoured: this was one window
carrying both halves.

## 1. Co-resident residency — the D-11 deployment number, now measured

`DECISIONS.md` D-11 carries two budgets. The development bound is measured and lawful. The **deployment** bound
(how much one serving process may hold while the local engine is deliberately resident) has been
`NOT-ADJUDICABLE` all session because its bound was unmeasured. It is measured now.

Engine resident first: `rocm-smi --showmeminfo vram` → `VRAM Total Used Memory 1011572736 B` (≈0.94 GiB of the
2.00 GiB carve-out), `MemAvailable 98,638,832 kB` ≈ 94.06 GiB. Then
`work/device-window/mem_mode_probe <GiB> 2 2711882916` at 8/16/24/32 GiB, all four modes, `rc=0` at every size.
The probe's own sysfs columns (it globs `card<i>` itself; the card on this box is **card1**):

| size | `carve` vram+/gtt+ | `gtt` vram+/gtt+ | `man` vram+/gtt+ | `pin` vram+/gtt+ |
|---|---|---|---|---|
| 8 GiB | 0.00 / 1.00 | 0.01 / **8.00** | 0.00 / 0.00 | 0.01 / 0.00 |
| 16 GiB | 0.00 / 1.00 | 0.02 / **16.00** | 0.00 / 0.00 | 0.02 / 0.00 |
| 24 GiB | 0.00 / 1.00 | 0.02 / **24.00** | 0.00 / 0.00 | 0.03 / 0.00 |
| 32 GiB | 0.00 / 1.00 | 0.03 / **32.00** | 0.00 / 0.00 | 0.05 / 0.00 |

**What this adjudicates.** A single process took **32.00 GiB of GTT-backed staging to completion with the engine
resident and idle** — every requested byte accounted for in `mem_info_gtt_used`, no failure, no silent degradation
to a smaller grant. Policy A's resident requirement is 19.7848 GiB, so the deployment budget is **FITS with
margin**, and D-11's deployment bound moves from `NOT-ADJUDICABLE` to measured.

**What this is NOT — three limits, each of which is a way this number gets misused.**
1. **These are allocation facts, not bandwidth.** The `read GB/s` and `1st-run GB/s` columns were captured in the
   log and are deliberately **not quoted here**: the engine was resident, so this is not the clean-window ruler,
   and the first-run column spans 8.1 → 231.0 GB/s at the same size (page-fault/first-touch, i.e. noise). Quoting
   any figure from this half beside the ~240 GB/s ladder of window `b` would be the two-rulers error.
2. `hipMalloc` **refuses nothing** (window `b` §1: a 66.1337 GiB allocation succeeds and spills to GTT). So
   "32 GiB succeeded" is a statement about what the pool *accounts for*, not a ceiling — the ceiling question is
   bandwidth and it stays answered only by window `b`'s clean ladder.
3. Engine **idle**, not serving (`in_flight=0` before and `queued=0`). This is the deployment-resident case, not
   the deployment-while-serving case. The latter has still never been measured.

## 2. M1's device half — the pass is real but narrower than its own RESULT line

`engine/stage_run.py --device --verify --keep-evidence work/device-window/c/stage-device-records.jsonl`,
halogen down, pool held:

```
RESULT PASS mode=device checks=31 findings=0 controls=0 fired=0 run_set=464 plan=464 verified=464 residue=0 uncovered=0 device=on ordering_log=252->253
```

`device=on`, all 464 runs, 23,812,820,480 B, `residue=0 uncovered=0`, and the ordering log grew one line
(252 → 253) as the append protocol requires.

**M1 is still NOT established, and here is the check that let the line look bigger than it is**, verbatim from
the run:

```
CHECK name=device-path-was-exercised verdict=PASS n=1 --device asked for a copy and 1 device construction(s) happened
```

That question is `--device` was asked and one device construction happened. It is answered by construction — it
cannot come back RED for the thing anyone would care about. And the decisive absence: `grep -oE` for
`dev_sha256|device_sha256|readback_sha256|dev_digest` over the 290,067 B evidence file returns **zero** matches,
while its per-run records carry only `RUN STARTED / PLAN / SLAB` lines. So `verified=464` in device mode is the
**same host-side dual-sha256 as `--check-only`**, and `controls=0 fired=0` means none of the eight injection
controls were exercised on the device path either. `checks=31` vs 34 in check-only is the same story: the device
mode runs *fewer* checks, not more.

What is genuinely established: the artifact loads, the layout places 752 tensors, the plan is 464 runs /
23,812,820,480 B, and a HIP device context plus a staging copy can be constructed and torn down on this box with
the engine down, with digests matching for every byte as read from the artifact.
What is NOT established: that bytes written to the device are the bytes that come back — no per-run
`hipMemcpyDtoH` + sha256 against the expected digest exists yet, so "read-back digests match", which is M1's exit
gate in `AGENTS.md`, has never been executed.

Fix required, and it is specified so the next attempt cannot re-make this mistake: a per-run D2H read-back whose
sha256 is compared to the digest the file region produces, **count == 464 enforced** (a check that can produce no
output cannot fail), the digest written into the evidence record for every run, and one control that corrupts the
*device* buffer and must redden that comparison — without it the read-back is decorative even when it runs.

## 3. Harness defects caught rather than shipped — including my own

1. **My first co-resident attempt produced fabricated zeros and I caught it before it reached a facts file.** I
   sampled `/sys/class/drm/card0/device/mem_info_{vram,gtt}_used`; **card0 does not exist on this box** (the card
   is card1), so `cat` failed, awk received empty `-v` arguments, and printed `vram_delta_GiB=0,00` for every
   size — a measurement-shaped line with no measurement under it. Worse, my own `tail -2` had thrown away the
   probe's real `vram+GiB`/`gtt+GiB` columns, which is why the second attempt could use the instrument's own
   numbers instead of my parallel reads. Recorded here because "the numbers were wrong" is less useful than
   "the wrongness was structurally invisible": a failed read and a true zero printed identically.
2. `mem_mode_probe` had **no compiled binary** in `tools/` — only the `.cpp`; it lives at
   `work/device-window/mem_mode_probe` (deliberately uncommitted, per window `b`'s rule).
3. `--keep-evidence` **requires a value** (an output path); the first device attempt died with
   `REFUSED tag=missing-argument`. Good refusal, caught in 3 s, no device bytes touched.
4. `stage_run.py` rejects unknown flags outright (`--help` → `REFUSED tag=unknown-argument`), so its contract has
   to be read from `OPTS`/`MODES`/`FLAGS` rather than discovered — cheap to do, and the only way to avoid guessing.
