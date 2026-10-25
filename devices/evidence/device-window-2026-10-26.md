# Device window 2026-10-26 — the first measurements on this box for the engine

Taken by the main agent, blocking, in the foreground, with **zero active workers** (`worker_lanes.sh`:
`alive=0`) and **halogen down** (`down: stopped in 14s; flash_serve absent from rocm-smi --showpids`).
Session endpoint was `tensorfold` (`PI_PROVIDER=tensorfold`), so stopping halogen was not
self-decapitation. Everything here is a measurement, so nothing in this file was produced by a subagent.

Instrument: `/tmp/window_probe.py` for the load/ruler runs; `tools/halogen_gate.sh` for admission, the
stop and the timed restart (`devices/evidence/halogen-restart.log`).

## 1. The tokens/s ruler — llama.cpp, full offload, greedy

`llama-cli` @ `d327bf1` (tag `b11869`), model `models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
(22,134,528,992 B), `-ngl 99 -s 42 --temp 0 --no-warmup -st -v -n 128`, one fixed ~50-word prompt, seven
measured runs after a discarded warm run.

| quantity | value |
|---|---|
| offload | **41/41 layers to GPU** on every run (this is what makes the figure a GPU figure at all — see the smoke test, which ran 1/41) |
| generation | **min 44.9, median 49.1, max 51.4 t/s** over n=7 |
| prompt | 293–367 t/s (not a target; decode is) |
| context used | `n_ctx = 26214`, single sequence |

At policy A's **2.5252 GiB/token**, the median implies **124 GiB/s ≈ 133 GB/s** of weight traffic —
**≈63 % of the 210 GB/s reference point** this repo's roofline is written against. The roofline predicts
77 t/s at 210 GB/s; llama.cpp delivers 49.1. **That gap is our headroom, and ~51 t/s is the number the
engine has to beat** — not the 77, which is a physics ceiling and not an implementation target.

Dispersion is ±7 % across seven runs, which is bimodality-lite: reproducible enough to be a ruler, not
reproducible enough to be a single number. Quote the triple, never the max. The two short runs
(`-n 32`) read 35.5 and 51.3 t/s — short decodes are noisy at the low end, so `-n 128` is the minimum
worth reporting.

## 2. What the engine actually occupies — and which counter sees it

This is the D-11 row, and the interesting part is that **three plausible instruments were wrong about it**.

| instrument | reading | verdict |
|---|---|---|
| process `VmHWM` | **1.47 GiB** | **useless here.** llama.cpp `mmap`s the GGUF and the device reads it over GTT, so the weights never enter the process's page tables. RSS cannot see the footprint. |
| process `VmPeak` | 34.48 GiB | address space, not residency — 2× the file. Quotes a reservation, not a cost. |
| `mem_info_vram_used` | 1.68 GiB peak | **the 2 GiB dedicated BAR window**, not the pool. `mem_info_vram_total` = 2,147,483,648 B = 2.0 GiB, so this counter is structurally incapable of seeing a 21 GiB load. |
| `/sys/class/kfd/kfd/proc/<pid>/mem_info_vram_used` | **absent on this kernel** | per-process device memory is not exposed → NOT MEASURED by that route. |
| `mem_info_gtt_used` | **26.86–26.88 GiB** peak over 5 runs | **the instrument that works.** Spread 0.02 GiB (0.07 %). |
| host `MemAvailable` | 114.80 → **87.47 GiB** (Δ 27.33 GiB) | independent corroboration: agrees with the GTT figure within 0.5 GiB. |

**The number: ≈26.9 GiB of device-visible footprint** to run the shipping artifact at 41/41 offload with
halogen down. Two instruments, two code paths, same answer — which is why this one is allowed into the
docs where the other four are not.

Decomposition, bounded rather than asserted: the file is 20.624 GiB, so non-weight footprint (KV at
`n_ctx=26214`, compute buffers, scratch) is **≥ 6.24 GiB**. It is a lower bound, not a split: only the
experts a 128-token generation actually touches get faulted into the GTT, so weight pages counted are
≤ the file and the residue is correspondingly larger. A steady-state split needs the engine's own
allocator, which does not exist yet.

**Co-residency consequence:** 26.9 GiB against the ≈32 GiB co-resident budget (D-02) **fits** — with
~5 GiB of margin, not the ~12 GiB the modelled 19.78 GiB suggested. The measured footprint is **7.1 GiB
above** the policy-A weight model, and that gap is llama.cpp's KV plus compute buffers. Any budget
decision made from the modelled figure alone is 7 GiB optimistic.

## 3. Restart cost — the assumption D-13's batching rule was standing on

`tools/halogen_gate.sh`: **stop 14 s**, **ready 17 s** (`devices/evidence/halogen-restart.log`:
`2026-10-25T11:39:12Z start_to_ready=17s`), including a 60 s KV-flush grace allowance that did not bind.

`AGENTS.md` §"Batch, do not oscillate" was written while this was unknown and told us to take one window
per session. **A ~31 s round trip makes a window cheap.** The rule should become: one window per
*measurement campaign*, not per session — and `pitfalls/00-measurement.md`'s warning about reloading a
66.69 GB checkpoint between gates did not materialise at this scale.

## 4. Two things this window measured about its own harness

**The GPU lock did not span the window.** `release` reported *"releasing lock left by a dead holder …
pid=342295"*. `gpu_lock.sh` records the acquiring process's pid, and each tool call is its own process —
so the shell that acquired the lock exited, and for the rest of the window the lock was held by a corpse
and readable as `STALE` by anyone who asked. The window *was* exclusive, but because of the other three
preconditions (zero workers, halogen down, `rocm-smi --showpids` empty), **not because of the lock**. Any
other agent could have acquired it mid-measurement. Recorded in `pitfalls/00-measurement.md`.

**My own min-tracking read 0.00 GiB, and 0 is the vacuous value, not the safe one.** The first version
initialised the running minimum of `MemAvailable` to `0`, so no sample could ever be lower and the
"measurement" printed `0.00 GiB` while looking like a result. Same rule I hold workers to — a counter that
reads exactly 0 is unproven until it is witnessed non-zero where it must fire. Fixed to a `None` sentinel
and re-run: 87.47 GiB with 46 value-bearing samples.

## 5. Deliberately absent

No figure here was taken with halogen up, a worker alive, or a backgrounded run. No `t/s` from the
co-resident smoke test (`quantization/evidence/llama-cpp-smoke-gguf-2026-10-26.md`, 1/41 offload, halogen
resident) is quoted in any doc, and that raw capture stays in gitignored `tmp/evidence/`.

**Still UNMEASURED after this window:** the ≈55.5–56.5 GiB pool ceiling remains a D-02 *allocation-probe*
result, not a sysfs reading — `gtt_total` says 112.0 GiB and `vram_total` says 2.0 GiB, and neither is that
number, so do not "correct" D-02 with either. Per-process device memory (no KFD proc files), the engine's
own allocator split of the ≥6.24 GiB overhead, and everything requiring the engine to exist.
