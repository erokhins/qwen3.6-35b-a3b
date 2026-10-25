# 00 — measurement traps: what a number on this box is worth

> **Provenance.** Copied and condensed on 2026-10-26 from the 0.8B project's measurement record
> (paths below relative to `../qwen3.5-0.8b/`: its `AGENTS.md` traps index,
> `engine/performance-status.md` §6 and §11,
> `devices/strix-halo-memory-bandwidth.md`, `prototype/mlp_gemv/results.md`). The numbers below are
> measurements of **this box** and are true here; the sibling repo is not needed to use them.

## The protocol (violating this invalidates the result, not just its precision)

- **Blocking, foreground, zero subagents.** A foreground tool call suspends the agent session's own
  token generation, so the serving engine — which busy-polls this same iGPU — has nothing to emit for
  the duration. Backgrounded runs (`&`, `nohup`, `bg_wait`, `sleep`) keep that stream alive and make the
  measurement contend with itself. With the turn blocked and nothing else running, bandwidth-bound
  numbers repeat within **~2 %** of each other; that is the only condition under which a timing here is
  worth anything.
  **[2026-10-26 correction — read this before trusting the sentence above.** It was written when this
  session was served locally. Blocking the turn only silences a GPU-adjacent engine **if that engine is
  on this box**: the main session now runs on `tensorfold` (remote `192.168.10.27:8888`), so a
  foreground call silences nothing local, and the thing that must be silenced is halogen —
  §"A halogen stream is a device activity". The rest of the bullet stands: a backgrounded run still
  contends with itself.]**
- **`rocm-smi` / `gpu_lock.sh util` is not a gate.** Identical clean runs have read 24–91 % while the
  measured numbers repeated. Record `util` at acquire time for the log, then discount it.
- **Warm the pages first.** The first GPU pass over fresh pages runs at **~35 GB/s**, ~7× slower than
  the steady state. Every burst harness needs a warmup launch before the timed ones.
- **Best-of-N with the spread printed.** A median across a wide spread describes nothing: a re-take once
  printed 369.7 / 589.0 / 1208.9 ms at one width, whose median "discovered" an 18 % regression that did
  not exist (clean window: 368.0 [365.4, 371.1]). Print the spread beside the median and re-take any
  window whose spread is far outside ~2 % when the workload is bandwidth-bound.

## A halogen stream is a device activity, not a token stream (2026-10-26)

The test for "may this agent run while I measure?" is **where its `baseUrl` points**, and it is one
command away (`~/.pi/agent/models.json`, or `tools/halogen_gate.sh status`):

- **remote** (`tensorfold` → `http://192.168.10.27:8888/v1`) — its token generation costs this iGPU
  nothing, so it may think concurrently with other workers;
- **local** (`halogen` → `http://127.0.0.1:8731/v1`) — a container on this box. While it is up,
  `rocm-smi --showpids` charges `flash_serve` **22.62 GiB** (24 289 746 944 B) against the
  device-visible pool, and its polling loop keeps utilisation high whether or not anyone is streaming.

So the hazard has two parts, and they are not the same hazard:

| | cost | consequences |
|---|---|---|
| halogen **up and idle** | 22.62 GiB of the pool | a measurement runs in the ≈32 GiB pool, not the ≈55 GiB one — which changes what fits, and may change what a sweep can even allocate |
| halogen **up and streaming** | pool + a second client on the memory controller | the 0.8B project's clean-vs-contended spread: 2.61–3.59 ms against a documented 2.650 ± 0.02, **with the config ranking changing too** |

**The rule.** A timing window is: zero active workers, `tools/gpu_lock.sh` held, and **halogen down** —
`GPU_LOCK_TAG="$tag" tools/halogen_gate.sh down` … measure … `tools/halogen_gate.sh up`. Only the main
agent does this, blocking, and it batches a session's timings into one window rather than stopping per
gate (restart reloads a 66.69 GB checkpoint; cost unmeasured —
`devices/evidence/halogen-restart.log`). Never run it from a session whose `PI_PROVIDER` is `halogen`,
and never by hand with `~/halogen-models/stop.sh`, which skips the guards. Numbers and the admission
caveats: `devices/measured-facts.md` §"Two model endpoints, and where each one's GPU is".

**Correctness is not the same case.** `gpu_lock.sh` already records that correctness results survive
contention while timings do not, so a worker's short PASS/FAIL gate alongside a resident halogen is
allowed; a *number* is not. Do not let "correctness is fine" smuggle a timing into that window.

### The probe that has to happen before `docker pause` becomes policy

`tools/halogen_gate.sh pause` freezes the engine's processes — killing the polling loop — while
**keeping the 22.62 GiB mapped**. If contamination is compute, `pause` is a much cheaper second tier
than a full stop/start; if it is residency, `pause` buys nothing at all. Two hypotheses, one A/B, and
the repo does not get to guess. Main agent only, blocking, all arms inside **one** lock hold,
interleaved `A B C A B C …` (never `AAA BBB CCC` — a ruler re-taken in a new window is a different
ruler), 5 reps per arm, spread printed beside every median, on a **bandwidth-bound** harness (a
dispatch-bound one is bimodal at 158–269 % and will not adjudicate anything), plus a behavioural
canary so a silently-skipped arm cannot print a clean result:

| arm | state | what it separates |
|---|---|---|
| A | halogen down | the ruler |
| B | halogen up, idle (`admit` passes) | residency alone |
| C | halogen up, a stream running | residency + compute |
| D | halogen up, `pause` held | whether `pause` ≡ A |

Read the pattern, not the averages: **B ≈ A** ⇒ residency is innocent for bandwidth (still check the
pool separately with `tools/mem_mode_probe_hip.cpp`, because 22.62 GiB is charged either way); **B ≫ A**
⇒ residency binds and `pause` is useless; **C ≫ B** ⇒ compute is the story and `pause` is worth D.
A `pause` with a request in flight hangs that request to the API server's 3600 s queue timeout —
`pause` refuses on `requests_processing > 0`, and never end a session with the container paused.

## `hipEventElapsedTime` returns exactly 0.000 ms for a 2.1 ms kernel

Not only for sub-millisecond ones. A zero is a broken timer, not a fast kernel. The only timer to trust
for short kernels is a burst: N launches, one sync, `steady_clock`/N. (`hipEventElapsedTime` did exactly
this on a 2.1 ms kernel and shipped into a ledger line before it was caught.)

## Dispatch-bound timings are bimodal; bandwidth-bound ones are not

Launch/dispatch-bound measurements are bimodal on this box — best-of-N spreads of **158–269 %** at 512
threads have been seen, with a best that beat a historical *contended* number while being actually worse.
Consequences:

- quote dispatch-bound µs figures at one fixed thread count and state the spread, or characterise the
  bimodality before quoting them at all;
- **a paired A/B must take both arms inside one lock hold.** A re-take of a ruler in a new window read
  64 and 256 lanes as *slower* than the old table (708 vs 770 ms, 1,893 vs 1,971 ms) while a paired A/B
  inside a single window read the same two widths *faster* (83.26→80.91, 131.78→122.12 ms). The 5–10 %
  swing is the bimodality, not a regression. Quote pairs, state the spread, and let the long row carry
  the headline.
- **A/B against a possibly-stale binary needs a behavioural canary.** The builds here are not
  bit-reproducible, so `cmp` proves only "two builds". Any field the change is required to move must be
  read in the same breath as the measurement and the script must abort when it reads wrong on either side
  — every parity check compiled *into* the binary under test is blind to staleness, because a stale binary
  agrees with itself. And never discard the build log: it is often the only remaining signal.

## The repo has (and this repo will too) more than one ruler, and they are not comparable

Two prefill rulers coexist and differ by **1.22× at 4k and ~1.9× at 1,024 lanes**: a prompt-wave burst
figure and a whole-billed-prefill-window figure. Subtracting one from the other invents a regression.
State which instrument produced a number, and compare only pairs taken on one instrument inside one
window.

## "Above ~240 GB/s means skipped work" — true only for DRAM-resident working sets

Measured warm read bandwidth vs working-set size (best of 3, one lock hold, foreground):

| buffer | read GB/s | copy GB/s |
|---|---|---|
| 4 MiB | 467.2 | 675.2 |
| 8 MiB | 640.1 | 812.9 |
| 16 MiB | 784.0 | **887.0** |
| **32 MiB** | **876.7** | 325.7 |
| 64 MiB | 239.4 | 251.3 |
| 128 MiB | 242.2 | 230.0 |
| 1 GiB | 237.2 | 211.2 |

The plateau is a **cliff between 32 and 64 MiB**, not a decay: the big cache tier is ~32 MiB and reaches
876.7 GB/s. So: above ~240 GB/s is evidence of skipped work **unless** the working set is under ~32 MiB
and warm, in which case it is evidence of the big tier. Quoting the rule unqualified means "investigating"
a real result until you throw it away.

Anchors, measured: iGPU pure read **237–240 GB/s** (copy 210), H2D 76–82 / D2H 63–70, CPU-side torch
reduce 75–117 GB/s with large run-to-run variance (re-measure before quoting). The roofline denominator
used throughout is **237 GB/s**.

## A traffic demand divided by measured time is not a bandwidth wall

Three days of planning read one stage as bandwidth-bound because demanded-bytes ÷ measured-time landed at
225–248 GB/s against the 237 ceiling. A bit-identical instruction cut (116 issue slots per touch → 45) then
made it **1.521× faster with the demanded census unchanged**, which puts the implied rate at 345–381 GB/s —
above what the part can do. Demanded bytes are not DRAM bytes; L1/L2 absorb reuse. **Falsification before
funding a traffic or bandwidth route: cut instructions or cut traffic and see which one moves time, then
check whether the implied rate is even achievable.** A model fitted to a measurement also cannot falsify
itself — a byte model that fit three widths to within 5 % still described nothing.

## Occupancy/latency pricing beats instruction pricing for serial kernels

A route was priced bottom-up at 11.4 ms from 1,370 instructions/token/wavefront (one issue slot per SIMD
per cycle) and measured **112.04 ms — 9.8× above it**, because the grid realises ~1.6 *working* wavefronts
per CU (one warp per block drives the recurrence, three park at a barrier), so the cost is latency per
token, not instructions per token. `#pragma unroll 8` on the serial loops was bit-identical and moved the
time 112.8 → 113.4 ms, proving the loop body was not the stall. **Price a serial kernel in
cycles-per-dependency at its realised occupancy, or do not price it.** Same class: a chunked scan priced at
≥1.5× measured 5.9×.

## A counter that reads exactly 0 is vacuous

Two counters summed to **exactly 0.0 over a whole run** on a pass that provably touched DRAM. Read as "no
pressure", that is a false negative with a number attached. Before quoting a counter's zero, witness it
non-zero on a run where it must fire; if it still reads zero, report the counter as *unusable on this part*,
not as a result. Related: the same part's `SQ_WAVE_CYCLES/SQ_BUSY_CYCLES` ratio has read above the
architectural wave ceiling (41 against 32), so relative shares of one counter are usable and absolute
meanings are not until the denominator's set size is established.

## A `rocprofv3` pmc row is per (Dispatch_Id, Counter_Name)

`--time-waves` replays the wave **three times**, so a raw `sum(Counter_Value)` is 3× a pass and still reads
plausible. And matching on a bare kernel name fails — `Kernel_Name` carries the signature
(`attn_slim_wave(CoreArgs)`). Correct: dedup by `(Dispatch_Id, Counter_Name)`, match on the name prefix, and
divide by a pass count **derived** from a kernel whose per-pass site count is known.
