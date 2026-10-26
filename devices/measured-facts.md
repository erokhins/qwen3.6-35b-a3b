# Measured facts about this box

Self-contained copy: every number here was measured on **this** machine and is repeated here so nothing in this
repo has to reach into another project. Measurement discipline (when a number may be trusted at all) is in
`pitfalls/00-measurement.md`; allocation/residency consequences are in `pitfalls/10-device-memory.md`.

**Device:** Framework Desktop, AMD Ryzen AI Max+ 395 (Strix Halo, `gfx1151`), 128 GB LPDDR5x-8000 unified
memory, single NUMA node, `AMD Radeon 8060S Graphics`. Disk: 3.4 TiB volume, 2.4 TiB free as of 2026-10-26.

## Memory bandwidth (iGPU, warm, best-of-N)

| quantity | value | note |
|---|---|---|
| **iGPU pure read, DRAM-resident** | **237–240 GB/s** | the denominator used for every roofline in this repo |
| iGPU copy (read+write) | **210 GB/s** | 1 GiB buffers; 203 GB/s at 4 GiB |
| iGPU read, 16–32 MiB working set | **784–876 GB/s** | big cache tier; the plateau is a **cliff between 32 and 64 MiB** |
| iGPU copy, 8–16 MiB | 813–887 GB/s | same tier |
| H2D / D2H through the host | 76–82 / 63–70 GB/s | bounds any staging path |
| CPU-side read (torch reduce) | **75–117 GB/s** | high run-to-run variance; a CPU-resident engine is an order of magnitude below the GPU path |
| first GPU pass over fresh pages | **~35 GB/s** | ~7× slow — always warm before timing |
| `hipMallocManaged` at ~32 GiB | 51.6 GB/s **silently** | `hipMalloc` fails loudly instead |

Read vs copy matters: 237 GB/s is a *pure read* figure. A decode step also writes (state, logits), and whether
237 remains the right denominator for a mixed read/write step was left explicitly unresolved upstream. Treat a
mixed-traffic claim as unproven until a paired read/write probe is run here (the mixed-traffic probe `mixed_bw.cpp` was not copied;
`tools/bw_probe_hip.cpp` and `tools/mem_mode_probe_hip.cpp` were).

## Device geometry (from KFD and `rocminfo`, never from `hipGetDeviceProperties`)

| fact | value |
|---|---|
| Compute Units | **40** (HIP reports 20 — halved, do not use for roofs) |
| SIMDs per CU / max waves per SIMD / total SIMDs | 2 / 16 / 80 |
| max wavefronts per CU | 32 |
| LDS per CU | **64 KiB** |
| register file | 65,536 per **SIMD**, not per CU |
| clocks | up to 2.9 GHz (sustained is lower on a 120 W part sharing budget with 16 Zen 5 cores) |
| L2 | 2 MiB (+ ~32 MiB big tier on the I/O die, which `rocminfo` does not report) |

## Compute rates (measured, best sustained per class)

| class | T·FLOP/s | % of its own roof |
|---|---|---|
| scalar `v_fma_f32` | 7.30 | 49 % of 14.82 |
| `v_dot2_f32_bf16` | 23.28 | 79 % of 29.6 |
| `v_dot8_i32_i4` | 55.96 | 47 % of 118.8 |
| `v_wmma_f32_16x16x16_f16` | **52.63** | 89 % of 59.4 |
| library bf16 GEMM, 16-bit `C`, engine shapes (hipBLASLt) | **33.2–37.7** | the realistic ceiling for GEMM-shaped work |
| dense square GEMM, F32-`C` form | 4.7–5.9 | **do not compare a 16-bit path against this** — see `pitfalls/20-rates-and-occupancy.md` |

The engine's own kernels contain no matrix or dot-class instructions in the sibling project's build, so every
matrix rate above belongs to hipBLASLt's tuned kernels unless and until this engine grows its own.

## Device-visible memory

| test | result |
|---|---|
| one `hipMalloc` of 32 GiB | ok (231 GB/s) |
| one `hipMalloc` of 40 GiB | fails |
| 8 × 8 GiB | 4th ok, 5th fails |
| 24 × 2 GiB | 17 ok (34 GiB), 18th fails |
| `GPU_MAX_HEAP_SIZE=100`, `GPU_MAX_ALLOC_SIZE` | no effect |

**The device-visible pool is ≈55.5–56.5 GiB for the whole box, and `flash_serve` is inside it** — so
what one process can map is what is *left*, ≈32 GiB while halogen is resident. `totalGlobalMem`
reports ~110 GiB and is not what a process can use. Text weights are **66.1337 GiB bf16** with the MTP
draft block, **64.5604 GiB** without it (`quantization/evidence/bf16-census-adjudication-2026-10-26.md`,
0 bytes of unexplained residue; per-family cut via `tools/gguf_suffix_census.py`). This read
**65.72 GiB, cited to `tools/quant_scenarios.py --census`, until 2026-10-26 — and that tool is wrong by
−0.4138 GiB**, which is how a figure that matched no independent ruler stayed in circulation: it was
never cross-checked against the artifact's own tensor records, only against itself. So the pool binds
either way, and host-mapping does not escape it (it consumes it): see the next section.

### Residency above the pool — pinning consumes the pool, it does not raise it (D-02, measured 2026-10-26)

`tools/hostmap_probe.cpp` (this repo), run blocking with the GPU lock held; raw output in
`devices/evidence/d02-hostmap-*.txt` and `d02-two-process-*.txt`. Every bandwidth line carries an
independent checksum witness, so a number below cannot come from skipped work (`pitfalls/40`). Box
state: `flash_serve` (halogen, the engine serving this session) resident with gtt_used **23.53–23.66
GiB** of a 112 GiB GTT (boot param `amdgpu.gttsize=114688`), MemAvailable 86–93 GiB, system
`Mlocked` 144 kB, this shell's `Max locked memory` **15.45 GiB (soft and hard)** while
`flash_serve`'s is `unlimited`.

| test | result |
|---|---|
| `hipMalloc` 1 GiB, read | 190–193 GB/s (spread 15 %) — a *low* window, see `pitfalls/00` |
| `hipMalloc` 40 GiB | **out of memory** (re-confirms the ceiling in this window) |
| `hipHostMalloc` (pinned + mapped) 1 / 2 / 4 / 8 / 16 / 24 / **32** GiB | all granted, read at **195–225 GB/s**, checksum OK |
| `hipHostMalloc` 34 / 36 / 38 / 40 GiB | **refused, in 0.0 s** (`out of memory`), so it is an accounting limit, not paging pressure |
| `malloc` + `hipHostRegister` 1 GiB | ok, 213.6 GB/s; 40 GiB → `invalid argument` |
| **2 × 20 GiB pinned, same process** | 1st granted, **2nd refused** — the accounting is a total, not per mapping |
| **2 × 20 GiB pinned, two concurrent processes** | **1 granted, 1 refused** (`tools/pin_pair.sh`) — *this is what refutes a per-process cap*: each 20 GiB is well under 32 GiB, so a per-process ceiling would have granted both |
| 2 × 24 GiB pinned, two concurrent processes | 1 granted, 1 refused — same conclusion, larger size |
| `hipMalloc` 20 GiB + `hipHostMalloc` 24 GiB | device ok, pinned **refused**; with 20 + 12 GiB both are granted |
| device 20 GiB vs pinned 12 GiB, same window | 198.6 GB/s vs **215.8 GB/s** — pinned host reads at device speed or better |
| plain `malloc`, unregistered pointer passed to a kernel | **HSA memory fault**, queue error; the probe's checksum refused the resulting 1.08e6 GB/s as "NOT A BANDWIDTH RESULT" |

**Correction, same day.** The first version of this section read the ~32 GiB boundary as a *per-process*
limit. It is not: two separate processes each asking for 20 GiB could not coexist, and 20 GiB < 32 GiB,
so the cap cannot be per-process. Every observation fits one model instead —

> **a box-wide device-visible pool of ≈55.5–56.5 GiB, which `flash_serve`'s 23.5 GiB is part of.**
> 23.5 + 32 = 55.5 granted ✓ · 23.5 + 34 = 57.5 refused ✓ · 23.5 + 40 = 63.5 refused ✓ ·
> 23.5 + 20 + 20 = 63.5 refused ✓ · 23.5 + 20 + 12 = 55.5 granted ✓ · 23.5 + 20 + 24 = 67.5 refused ✓

Note the arithmetic is razor-thin at the top (55.5 granted, 57.5 refused), so the pool is somewhere in
55.5–57.5 GiB, and 56 GiB = **exactly half the 112 GiB GTT** is an obvious suspect but is *not*
established — `amdgpu.enforce_isolation` sharing among active clients would fit the same points, and
the test that separates them needs halogen stopped, which is not available from this session (it
serves this session). **[2026-10-26: that blocker is gone.** This session is now served by
`tensorfold`, a remote LAN box, so halogen *can* be stopped from here — §"Two model endpoints, and
where each one's GPU is" and `tools/halogen_gate.sh`. The test itself still has to be run, once the
other session using halogen is done with it.] Neither `RLIMIT_MEMLOCK` (15.45 GiB here, `unlimited` for halogen, and a 32 GiB
pin was granted anyway) nor system RAM nor GTT occupancy explains it.

Consequences, now stated against the corrected model:

1. **The engine's budget is `pool − co-resident footprint`, so it is an ops decision, not a constant.**
   ≈32 GiB while halogen is resident; **≈55 GiB if the engine owns the box.** That is a different
   project: Q8-throughout (35.16 GiB, corrected 2026-10-26 from 34.97) *fits* at 55 GiB, so 4-bit experts becomes a **speed** choice
   rather than a fit requirement (`DECISIONS.md` D-11 asks the question; `quantization/README.md`
   prices both budgets). bf16 (66.1337 GiB) fits in neither.
2. **Pinning does not escape the pool — it spends it.** Mapped host memory is charged to the same
   budget as device memory, and reads at ≥ device bandwidth (198.6 device vs 215.8 pinned GB/s in one
   window), so a two-tier layout is bandwidth-viable but buys no additional capacity.
3. **The endgame still needs more than this box.** For `Qwen3.8-Flash-Next` (232.9 GiB text bf16) even
   4-bit main weights are ≈58 GiB, above even the whole pool: it needs streaming with a real bandwidth
   cost, a pool raise (`gttsize`/`rmem`/BIOS, untested), or different hardware. Recorded so nobody
   re-derives it from hope.


### Measured engine footprint — the first one, 2026-10-26 window

What the engine actually occupies, measured rather than modelled. Main agent, blocking, zero workers, halogen
down; full record in `devices/evidence/device-window-2026-10-26.md`.

| instrument | reading | usable? |
|---|---|---|
| process `VmHWM` | 1.47 GiB | **no** — llama.cpp `mmap`s the GGUF and the device reads it over GTT, so the weights never enter the process page tables |
| process `VmPeak` | 34.48 GiB | no — address-space reservation, not residency |
| `mem_info_vram_used` | 1.68 GiB peak | **no** — this is the 2.0 GiB dedicated BAR (`mem_info_vram_total` = 2,147,483,648 B); structurally cannot see a 21 GiB load |
| KFD `/sys/class/kfd/kfd/proc/<pid>/` | **does not exist on this kernel** | per-process device memory NOT MEASURED by that route |
| `mem_info_gtt_used` | **26.86–26.88 GiB** (5 runs, spread 0.07 %) | **yes — this is the one that works** |
| host `MemAvailable` | 114.80 → 87.47 GiB (Δ 27.33) | yes — independent corroboration, agrees within 0.5 GiB |

**≈26.9 GiB** of device-visible footprint to run the shipping artifact at 41/41 layers offloaded, single
sequence, `n_ctx=26214`. The file is 20.624 GiB, so KV + compute buffers + scratch are **≥ 6.24 GiB** — a
lower bound, because only the experts a 128-token generation touches get faulted into the GTT.

Two consequences worth pinning. **Measured demand is 7.1 GiB above the modelled policy-A figure** (26.9 vs
19.78 GiB): a budget decision taken from the model alone is 7 GiB optimistic. And 26.9 GiB **does** fit the
≈32 GiB co-resident budget, but with ~5 GiB of margin rather than the ~12 GiB the model implied.

Do **not** reconcile these with the sysfs totals: `mem_info_gtt_total` = 112.0 GiB and
`mem_info_vram_total` = 2.0 GiB, and neither is the ≈55.5–56.5 GiB pool ceiling above, which stays a D-02
allocation-probe result. The counters live under `/sys/class/drm/card1/device/` on this box, not `card0`.

## Two model endpoints, and where each one's GPU is (measured 2026-10-26)

Pi on this box has two model endpoints, and they differ in the one way that decides measurement
discipline: **one is served by another machine, the other by this one.** Which endpoint an agent is
served by therefore says whether that agent's token generation touches the iGPU — which is the whole
basis of the concurrency policy in `AGENTS.md` §"Two endpoints, three classes of work".

| | `tensorfold/Qwen3.8-Flash-Next` | `halogen/halogen-qwen3.8-flash-next` |
|---|---|---|
| `baseUrl` | `http://192.168.10.27:8888/v1` — **a separate LAN serving host** | `http://127.0.0.1:8731/v1` — **this box** |
| what serves it | vLLM-style server on `.27` | Docker container `halogen-flash-server`, image `ghcr.io/peonist-ai/halogen-flash-server:0.18.9`; API `serve_api.py` :8731 → engine `flash_serve` :8730 |
| streams | `/health` → `max_batch_size: 5` | engine `--slots 4`; **treat as one stream at a time** (user's rule, 2026-10-26) |
| decode rate | `/health` live `decode_tokens_per_second` **81.0** at `connections: 0, waiting: 0`; the operator's working figure for a loaded stream is **~35 t/s** | `/metrics` `llamacpp:predicted_tokens_seconds` **58.5** under a live stream (reads **0** when idle) |
| cost to this iGPU | **none for token generation** — the bytes come over the network | `rocm-smi --showpids`: `flash_serve` holds **24 289 746 944 B = 22.62 GiB**, plus a polling loop that keeps utilisation high whether or not it is emitting |
| start / stop | n/a | `~/halogen-models/start.sh` (`docker run --rm`, `--device /dev/kfd --device /dev/dri`, `-v ~/halogen-models:/models:ro`, `-v ~/halogen-kvcache:/cache`) / `~/halogen-models/stop.sh` (`docker stop -t 60` then `rm`; the 60 s is the KV-tier flush). Wrapped, with guards, by `tools/halogen_gate.sh` |
| checkpoint | — | `qwen38-flash-next-v2.hgn` **66.69 GB**, 4.16 bpw, weights RAM-resident (`~/halogen-models/SETTINGS.md`) |

**Reading these counters correctly.** `llamacpp:requests_processing` and `/health`
`in_flight` / `queued` / `busy_for_s` are *right now* numbers: an idle server reads `0 / 0 / 0`, and
so does a server whose only client is thinking. `predicted_tokens_seconds` is an instantaneous rate
and reads `0` at idle, so a zero there is not evidence of anything. Use them to prove nothing is in
flight; do not use them to prove nobody is connected.

**The footprint has two readings, quote both.** The residency above the pool is 22.62 GiB as
`rocm-smi --showpids` charges it, and 23.53–23.66 GiB as `gtt_used` charged it in the D-02 runs above
(same box state, different counter). They bracket the same ~23 GiB; averaging them would be precision
neither instrument supports.

**What is NOT measured yet, and is load-bearing:**

1. **Restart cost.** `stop.sh` removes the container (`--rm`), so `up` reloads a 66.69 GB checkpoint
   from NVMe into pinned RAM; the KV/prompt tier in `~/halogen-kvcache` survives, so warm should beat
   cold. `tools/halogen_gate.sh up` times it and appends to
   `devices/evidence/halogen-restart.log`, which is empty as of writing. Until that log has entries,
   assume a restart is expensive and batch every timing of a session into **one** down-window rather
   than stopping per gate.
2. **What the contamination actually is** — compute (polling loop, GTT traffic) or mere residency. If
   compute, `docker pause` is a cheap second tier; if residency, it is not, because pause keeps the
   22.62 GiB mapped. Protocol in `pitfalls/00-measurement.md`; do not build policy on `pause` before
   that A/B exists.
3. **Admission is weaker than it looks.** `tools/halogen_gate.sh admit` proves *nothing is in flight*;
   a foreign session that is merely idle between turns is indistinguishable from a free server. The
   stream lock `/tmp/pi_halogen.lock` covers workers launched under this protocol, not another human's
   session — so the operator's word that halogen is free remains load-bearing.

### What this unlocks — D-11

The sentence above ("needs halogen stopped, which is not available from this session") was true when
it was written and is not true now: this session runs on `tensorfold` (`PI_PROVIDER=tensorfold`, which
`tools/halogen_gate.sh down` checks as a self-kill guard before it will touch anything). Stopping
halogen once, with the GPU lock held, is the only way to observe the pool with the engine owning the
box — the measurement `DECISIONS.md` D-11 is waiting on, and the one that decides whether
Q8-throughout (35.16 GiB) fits and 4-bit experts is a speed choice rather than a fit requirement.

## Roofline for the target model

Derived from `checkpoints/Qwen3.6-35B-A3B.config.json` ÷ the measured read bandwidth, by
`python3 tools/quant_scenarios.py --read 237 210 190`. The full policy table (residency and ceiling
for bf16 / Q4_K-experts variants / Q8-throughout) lives in `quantization/README.md`; this is the
shape of it:

| configuration | resident | GiB read per token | ceiling @237 GB/s |
|---|---|---|---|
| bf16 text weights | 66.1337 GiB ✗ above both budgets | 5.6083 | ~39 tok/s |
| Q8_0 throughout | 35.16 GiB — above the co-resident budget (~32), **fits if the engine owns the pool (~55)** | 3.0057 | ~73 tok/s |
| experts Q4_K + rest Q8_0 (policy A, **decided**) | 19.78 GiB ✓ fits either budget | 2.5252 (2.4653 target-only) | ~87 tok/s |
| experts Q4_K + attn Q8_0 + `lm_head` Q6_K (policy G) | 19.56 GiB ✓ | 2.4105 (2.3506 tgt) | ~92 tok/s |
| experts + attention both Q4_K (policy E) | 18.88 GiB ✓ | 1.7397 (1.6940 tgt) | ~127 tok/s |

Every row above is re-printed verbatim from the repaired `tools/quant_scenarios.py --read 237 210 190`
(2026-10-26), which derives bytes from the pinned artifact's own tensor records plus the draft block's
shapes and takes widths from `tools/ggml_block_geom.py`. Before that reprint, policy F read 2.811 /
~79, G read 19.36 / 2.216 / ~100 and E read 18.81 / 1.663 / ~133 — all priced from the census the
adjudication disproved. Every rate here is a **prediction** against the copy rates measured in
§Memory bandwidth, never a measurement, and the ~60 %-of-own-ceiling engine discount applies on top.

Two qualifications. Ceilings are ceilings: the sibling 0.8B engine reached ~60 % of its own, and the
bandwidth actually delivered in the 2026-10-26 window was 190–225 GB/s, so quote the 190 column as
the honest lower bound. And these are **bytes**, not time: attention time at 4k context is not the KV
byte count (KV is 10 layers × 2 KV heads × 256 dim × 2 B = 10 KiB/token, ~0.7 % of a bf16 step) —
long-context cost has to be measured, not inferred from this table.

Anchors for the arithmetic: policy A predicts 19.7848 GiB = 21.27 GB where the published GGUFs for this
architecture are 18.49 GB (unsloth Q4_K_M) and 19.77 GB (bartowski) — our figure is high by ~6–13 %
in the expected direction, because those quants are dynamic (some expert tensors below Q4_K) and keep
attention at Q4_K too. Details and sources: `quantization/evidence/`.

### The first ruler: llama.cpp at full offload — 2026-10-26

`llama-cli` @ `d327bf1`, the shipping Q4_K_M artifact, `-ngl 99 --temp 0 -s 42 --no-warmup -st -n 128`,
41/41 layers on GPU, seven measured runs after a discarded warm one:

- **generation 44.9 / 49.1 / 51.4 t/s** (min / median / max) — quote the triple, not the max
- prompt 293–367 t/s; dispersion ±7 %, which is a ruler and not a single number
- at 2.5252 GiB/token the median is **≈133 GB/s ≈ 63 % of the 210 GB/s** reference above

The roofline below predicts 77 t/s at 210 GB/s. llama.cpp delivers 49.1. **The implementation target is to
beat ~51 t/s; 77 is the physics ceiling, not the bar.** `-n 32` runs read 35.5 and 51.3 t/s — short decodes
are unreliable at the low end, so 128 tokens is the minimum worth reporting. Full record with the instrument
caveats: `devices/evidence/device-window-2026-10-26.md`.

## Re-measuring

Only after a driver, BIOS or memory-carve-out change. The probes that produced these numbers are copied into
`tools/` here: `bw_probe_hip.cpp` (size–bandwidth curve), `mem_mode_probe_hip.cpp` (allocation modes + pool
ceiling), `gemm_peak.cpp` + `gemm_common.h` (matrix rates), `kernel_occupancy_probe.cpp` (occupancy with canary),
`launch_overhead_hip.cpp` (launch cost), `bw_probe.py` (CPU side), `isa_probe.sh` (ISA/class inventory). Written
here: `hostmap_probe.cpp` (pool boundary, host pinning, checksum-witnessed reads) and `quant_scenarios.py`
(census + policy arithmetic, no device). Build with `hipcc -O2 -Wl,-rpath,/opt/rocm/lib`, take them blocking
with `tools/gpu_lock.sh` held, and record the split first: `cat /sys/class/drm/card*/device/mem_info_{vram,gtt}_{total,used}`.

## Copied from the sibling engine (2026-10-26)

Everything in this section is a **copy** of a measurement the sibling project (`../qwen3.5-0.8b`) took on **this**
machine. It lives here because a fact that lives only in that repo is a fact no session here can use
(`pitfalls/60-docs-and-citations.md` §"The self-containment rule"). Full provenance per row; the map of which
engine piece each row feeds is `planning/port-inventory-2026-10-26.md` (row ids `F-nn` below are its §9 ids).

**Two tags, and the difference matters.**

- `[SIB-BOX]` — measured on this silicon, and the quantity is model-independent (cache topology, dispatch cost,
  compiler behaviour, allocation API). Usable as a box fact; still **not** measured against Qwen3.6-35B-A3B
  tensors, so it is a design input and never an acceptance result.
- `[SIB-GEO]` — **sibling-model fact: measured on Qwen3.5-0.8B geometry, NOT measured on this box against
  Qwen3.6-35B-A3B.** An order-of-magnitude prior and a method; never a design constant here.

### A. Launch, dispatch and drain prices

| quantity | value | tag | provenance |
|---|---|---|---|
| async submit floor | **0.375 µs/launch** | `[SIB-BOX]` | `../qwen3.5-0.8b/prototype/launch_price/S1-GATEZERO.md` §2 shape-band table; instrument is `tools/launch_overhead_hip.cpp`, copied here |
| empty-kernel dispatch | **1.85 µs per launch-segment** (empty 48-launch burst, one sync) | `[SIB-BOX]` | same §2; instrument `../qwen3.5-0.8b/tools/submission_probe.cpp`, **not** copied here |
| serialized-sync gap | **5.9 µs per sync** (serialized minus burst = 1.80 ms over 305 syncs, identical work) | `[SIB-BOX]` | same §2, from `engine/stage_clock.cpp` rows |
| sync cost at an engine operating point | **1 sync 9.4798 ms / 305 syncs 11.0932 / 340 syncs 11.2855 ⇒ 5.29 µs per sync**, linear to 4 % | `[SIB-GEO]` (step shape is theirs) | `../qwen3.5-0.8b/engine/generation-limits.md` §4, taken 2026-10-21 at 4,032 lanes |
| **production launch price** | **3.10 µs**, measured by *deleting* 97 launches (**−0.3008 ms** at 4,032 lanes) | `[SIB-GEO]` shape, mechanism `[SIB-BOX]` | `generation-limits.md` §1 `cast` row, §6, §7 item (1), taken 2026-10-21 |
| a fitted slope at an empty body overstates | **3.9473 µs/launch** was printed as "S1 IS REAL" and withdrawn: body=tiny has nothing to overlap behind, and the merge gain (13.08 µs/launch) moved workgroups-per-launch (grid 271 vs 1918) with it | `[SIB-GEO]` | `../qwen3.5-0.8b/progress/2026-10-21/route-S1-segments-2026-10-21.md` §7 |
| small-op stage-line prices | `glue` **49 × 11.16 µs**, `copy` **19 × 10.7 µs**; a draft rig's `empty` arm **1.82 µs/launch** | `[SIB-GEO]` | `generation-limits.md` §1; `../qwen3.5-0.8b/prototype/mtp_draft/DRAFT-RESULT.md` |
| **the drain leg was the instrument, not the engine** | the shipping path defaults `time_stages = false`, so `syncs_per_step = 305` is the stage clock billing its own ruler; "any prize computed as 'segments × us_per_segment' is pricing the measurement, not the engine" | `[SIB-BOX]` conclusion | `generation-limits.md` §4; route-S1 §7 |

### B. Cache hierarchy and the residency API

| quantity | value | tag | provenance |
|---|---|---|---|
| fast tier identity | KFD node 1 cache **85, level 3, size 32768 KiB, line 64 B** = 32 MiB; `rocminfo` does not report it | `[SIB-BOX]` | `../qwen3.5-0.8b/prototype/prefetch_tier/P1-ARITHMETIC.md` §1 (`p1c_query`, no timing) |
| fast-tier read at 32 MiB | **876.7 GB/s**, cliff to **239.4** by 64 MiB | `[SIB-BOX]` | same §0 [Q3], from their bandwidth doc |
| fast-tier **copy** at 32 MiB | **325.7 GB/s** — the copy column is **not** the read column in this tier | `[SIB-BOX]` | same §0 [Q4]. **Load-bearing for a gather kernel: 876 does not mean 876 for read+write** |
| per-GPC L2 | **2 MiB** (KFD cache 84, level 2, line 128 B), agrees with the HIP attribute | `[SIB-BOX]` | same §0 [Q6] |
| `attr PersistingL2CacheMaxSize` | **2,097,152 B** — byte-identical to `L2CacheSize`, so the carve-out *is* the L2, not the tier | `[SIB-BOX]` | same §1 |
| `prop accessPolicyMaxWindowSize` | **0** — the runtime advertises no window it will accept | `[SIB-BOX]` | same §1 |
| `hipStreamSetAttribute(accessPolicyWindow)` | **rejected at 1, 2, 8, 16, 32 and 64 MiB — 0 of 6 accepted**; no `hipDeviceSetLimit` carve-out enumerator, no `hipCtxResetPersistingL2Cache` | `[SIB-BOX]` | same §1. Consequence: **hardware residency hints cannot address this tier; only an explicit early load can** |
| `hipMemPrefetchAsync` on a `hipMalloc` pointer | returns `no error`; nothing demonstrates it moved a byte — "Accepted is not performed" | `[SIB-BOX]` | same §1 |
| `prop concurrentKernels` | reads **1**, but a spinner+probe test observed real second-stream overlap **5/5**, with a same-stream negative control correctly reading **0** | `[SIB-BOX]` | same §1. The device property is not a usable answer to "can two kernels overlap here" |

### C. Per-block fetch capacity — launch shape, not launch count

| quantity | value | tag | provenance |
|---|---|---|---|
| iso-population diagonal (`pop = 1024`, equal bytes, equal total workgroups) | **7.20 ms** at L=16/G=64 → **311.37 ms** at L=1024/G=1 — **43×** | `[SIB-BOX]` mechanism, `[SIB-GEO]` absolute ms | `../qwen3.5-0.8b/prototype/wave_shape/ITEM2-RESULT.md` §2, window 2026-10-21 |
| the mechanism | at `G=1` a block must pull **1461 MiB / 1024 ≈ 1.43 MiB** with its own 256 threads "and one block cannot hold enough outstanding loads to approach the plateau" | `[SIB-BOX]` | same |
| where the plateau ends | `pop = 4096`: **6.98 / 6.91 / 7.33 ms** at L=16/32/64, then **11.03 → 76.85**; per-cell spreads **0.13–0.45 ms at G ≥ 32** vs **30–270 ms at G ≤ 4** | `[SIB-BOX]` | same |
| split-K on a one-row skinny-K body | loses at **2, 4, 7, 8, 14** ways (ship 0.0179 ms; `intra1/2/4/8` = 0.0179/0.0204/0.0251/0.0246; `two2…two14` = 0.0228…0.0369), because it re-stages the activation row 1,792 → 14,336 KiB against 7,168 KiB of weights; `intra8` also moved **≤ 48 ULP** | `[SIB-GEO]` | same §3 |

### D. Weight-walk pattern vs contiguity

| quantity | value | tag | provenance |
|---|---|---|---|
| engine per-thread walk ÷ contiguous sweep, same bytes and grid | **0.470** (gate/up), **0.472** (with the step's 62:1 writes), **0.629** (`down`) — transaction coalescing **100 % on both sides** | `[SIB-BOX]` pattern, `[SIB-GEO]` which kernel | `../qwen3.5-0.8b/devices/strix-halo-mixed-traffic.md` §Q9 Results, window `q9-mapping-locality` 2026-10-13 |
| LDS staging exonerated | `--stage 0` ⇒ `nostage/eng = 1.098` | `[SIB-GEO]` | same |
| the decisive row | `R_decisive` **140.5 GB/s = 0.598** of the contiguous reference **235.0 GB/s** (99.2 % of probe) | `[SIB-GEO]` | same |
| launch **order** is not the lever | identical 424-launch multiset over a 1549 MiB footprint: `B_blk` **183.3** vs `B_rr` **184.2 GB/s** = **1.005** | `[SIB-GEO]`, and it was their own footprint | same |
| the fix was a walk shape, and it was free | `interleave` → `contig` proven **bit-identical** (0 / 3584 `act`, 0 / 7168 `gate_pre+up_pre`, 0 / 1024 `out`, every layer), ISA witness `v4,v2,v4,v2,…` → `v17,v17,v17,…` | method, transfers | `../qwen3.5-0.8b/prototype/mlp_gemv/results.md` §Q10 |

### E. Cross-phase prefetch, measured

| quantity | value | tag | provenance |
|---|---|---|---|
| effect of an explicit prefetch on the use-site read | **+0.027 to +0.029 ms faster**, adjudicable wherever adjudicable | `[SIB-GEO]` (18 MiB target) | `../qwen3.5-0.8b/engine/generation-limits.md` §5 and §7, 2026-10-21 |
| the cliff | an 18 MiB state survives **7 MiB** of intervening traffic (**+0.0278**) and is gone by **21** (−0.0001 to +0.0008); a 7.875 MiB KV slice still survives 21 MiB (**+0.0138**) | `[SIB-GEO]` | same |
| verdict | best engine-shaped extrapolation **+0.0278 ms/step against a pre-registered 0.30 ms bar** → no engine change funded | `[SIB-GEO]` | same §7 |

### F. Compiler and occupancy

| quantity | value | tag | provenance |
|---|---|---|---|
| declared vs reserved LDS | `attn_core` declared **6,176 B** and **reserved 38,944 B**: LLVM promote-alloca-to-LDS moves `qreg[8]` and a `uint4` into LDS when the loop bound is a runtime value. At 64 KiB/CU that is 1 resident block | `[SIB-BOX]` compiler behaviour | `../qwen3.5-0.8b/engine/generation-limits.md` §3 |
| the fix | a spelling: `for (t=0;t<8;++t) if (t<nd)` — byte-identical across 126 stage comparisons and 21 cases, residency **1 → 4** blocks/CU, **−46 %** on the line, **−9.9 %** on the step | `[SIB-BOX]` technique | same |
| bound classification, and its precondition | traffic **27.7 %** of plateau and issue **5.9×** under its own roof ⇒ neither was the wall, so it was latency at occupancy; and at 128 lanes the launch is 8 blocks on 40 CUs, so neither residency nor population *can* bind | `[SIB-BOX]` method | same |

### G. The sibling engine's own step ledger `[SIB-GEO]`

| quantity | value | provenance |
|---|---|---|
| operating point | **4,032 lanes: 10.9357 ms/token = 91.4 tok/s**; **9.7855 ms = 102.2 tok/s** at 128 lanes; prefill 11,070 tok/s at 4,096 | `../qwen3.5-0.8b/engine/generation-limits.md` §0, paired windows 2026-10-21 |
| per-step census | **1532.3 MB read / 24.4 MB written** at **340 launches / 305 stage segments**; **243 / 184** after the cast fold; **6.47 ms** of unavoidable weight traffic inside the 10.936 ms step | same §0/§1 |
| family efficiency vs byte floors | `mlp` 70 %, `proj` 75 %, `lm_head` **104 %** ("AT THE WALL, cannot be faster"), `attn` 13 %, `gdn` 30 %, `cast` deleted; `host_tail` **0.7163 ms** — one blocking D2H of a 993 KB logits row plus CPU argmax, "on the critical path, not hidden" | same §1 |

### H. `lm_head`, and the batched head

| quantity | value | tag | provenance |
|---|---|---|---|
| their single-row head line | **2.1685 ms** for **509.6 MiB** = **246 GB/s demanded = 104 % of the wall** | `[SIB-GEO]` tied, H=1024 | `generation-limits.md` §1 |
| batched-head marginal | M=1 **2.1265 ms**, M=2 **2.1305** (**R = 1.0019**), M=4 **2.1538** (**1.0128**) | `[SIB-GEO]` | `../qwen3.5-0.8b/prototype/mtp_headbatch/HEADBATCH-RESULT.md`, 2026-10-22 |
| the witness behind it | `fetch_per_weight` **1.000000** on every M arm; calibration arms **2.000000** (`extra_pass`) and **1.25** (`extra_chunk`); `short_grid` **0.999968** with 16/32 poison holes | `[SIB-BOX]` instrument, transfers wholesale | same, and §2 of `HEADBATCH-HANDOVER.md` |
| plateau arithmetic for one table read | 485.0000 MiB / 237 GB/s = **2.1458 ms** (measured 1-row = 1.011× it); 2nd logits row **+0.19 %**; dots 0.509 → 1.017 GFLOP/pass = **3.2 % → 6.4 %** of the 7.30 TFLOP/s scalar roof | `[SIB-GEO]` inputs | `HEADBATCH-HANDOVER.md` §0 |
| `rows_per_block` sweep, all bit-identical | **1 → 192.7, 2 → 241.7, 4 → 242.1, 8 → 239.6, 16 → 239.0 GB/s** ("flat from 2 up") | `[SIB-GEO]` | `../qwen3.5-0.8b/prototype/lm_head_gemv/results.md` |
| fused argmax | bought **1 %** (2.099 vs 2.126 ms): "writing 1 MB of logits is nothing next to reading 485 MB" | `[SIB-GEO]`, conclusion transfers | same |
| second-row marginals on their shapes | `proj` **+7.6 %**, `mlp gateup` **+8 %**, `mlp down` **+76 %** ("the skinny `down_proj` is the worst marginal found"); `attn` measured *faster* for a second row and was refused as a result, because that rig sat at 13 % of its own byte floor | `[SIB-GEO]` | `../qwen3.5-0.8b/prototype/mtp_delta/D3-RESULT.md` |

### I. GDN

| quantity | value | tag | provenance |
|---|---|---|---|
| state size | **F32**, **1.000 MiB** per layer (16 × 128 × 128 × 4 B), **18.000 MiB** per request; conv window F32 **1.266 MiB** | `[SIB-GEO]` — ours is **60.0 MiB** (`derived here`, 30 × 32 × 128 × 128 × 4 = 62,914,560 B) | `../qwen3.5-0.8b/prototype/gdn_rec_isa/results.md` §8 |
| state traffic per launch | **2 read + 1 write**, confirmed in machine code as 256 state loads / 128 stores per token-step = 2 MiB read / 1 MiB written; re-read every token "and it cannot be anything else in this geometry" | `[SIB-GEO]`, form transfers | same §8 |
| state coldness | **+0.070 ms saturating at exactly 32 MiB** (`cold_0MiB -> 32 -> 512 -> 1536` = 0.1216 / 0.1917 / 0.1894 / 0.1922) — an L2-tier effect on an 18.9 MiB state, **not** a DRAM cliff | `[SIB-GEO]`, **does not transfer**: our 60 MiB state is 1.875× the tier | `../qwen3.5-0.8b/prototype/gdn_insitu/D4-DIAGNOSIS.md` §7 |
| whole-family warm launch | **0.0646 ms** for 18.0 MiB = **876 GB/s cache-resident = 370 % of the probe** (one layer `rec` alone **0.0064 ms**); the file itself retires its 108 MiB / 255 GB/s row as unreproducible from the current binary | `[SIB-GEO]` | `../qwen3.5-0.8b/prototype/gdn_decode/results.md` §Performance |
| launch geometry | `rec_kernel<<<NH=16, D=128>>>` = 16 blocks × 4 wavefronts = **64 wavefronts**, wave32, ≤1 per SIMD; prologue **323 of 1,523 slots (21 %)**, repeated 18× per token | `[SIB-GEO]` — ours is `derived here` 32 blocks = 128 wavefronts = **1.6 per SIMD** | `gdn_rec_isa/results.md` §1 |
| LDS-resident route | modelled **11.4 ms**, measured **112.04 ms** (**9.8×**) against pre-registered **63 ms** win / **84 ms** kill lines | `[SIB-GEO]` verdict, `[SIB-BOX]` lesson | `../qwen3.5-0.8b/prototype/gdn_collapse/results.md` §0/§5 |
| why it failed | **1.6 blocks per CU, one of whose four wavefronts does the recurrence**: "the cost is latency per token, not issue slots per token"; launches 18,432 → 36 removed traffic but bought nothing comparable | `[SIB-BOX]` mechanism | same §5 |
| launch merging inside the family | `split8` 0.1839 → `comb_all` **0.0644 ms**: merging 18 launches into 1 at identical bytes buys **0.12 ms** | `[SIB-GEO]` | `gdn_insitu/D4-DIAGNOSIS.md` §7 |

### J. The dense-MLP family (prior art for the shared expert) `[SIB-GEO]`

| quantity | value | provenance |
|---|---|---|
| their per-token MLP, cold | **2.650 ms = 199.4 GB/s = 84 % of the 237 probe** (rows/block 8), explicitly marked "provisional and due for a clean re-measure" because the sweep did not alternate the two kernels and validated only replica 0; a first run of the fixed method gave **2.594 ms** in a window their own file discounts | `../qwen3.5-0.8b/prototype/mlp_gemv/results.md` §Performance |
| their isolated "98 % floor" | **2.275 ms/token** on 528 MB `contig` — but that burst is one layer replicated 24× (**2 launches at 86,016 rows / 10,752 blocks**) while the engine ships **48 launches at 3,584 / 1,024 rows (448 / 128 blocks)**; the shape-matched arm reproduces it at **+1.0 %** | `../qwen3.5-0.8b/prototype/mlp_insitu/D2-DIAGNOSIS.md` §0(iv), §5 |
| in-situ decomposition of 1.0878 ms | `T_geo` **+0.7565 ms (69.5 %, launch shape)** + `T_serial` +0.2709 (24.9 %, instrument) + `T_pipe` +0.0484 (4.4 %) + memory-state residue **~0.06 ms (5.5 %)**; "D2 as written is dead, and no code should be funded against it" | same §5 |
| why the fat shape is unreachable | "layer *i*'s `act` must exist before its `down`, and `down` feeds the residual that layer *i+1* reads" — so the dependency chain forbids harvesting `T_geo` by merging | same §5 |
| per-layer MLP bytes | **22.02 MiB** per layer against a **2 MiB L2**, so a per-token launch pays that from DRAM per token | `mlp_gemv/results.md` §What runs and §P4 |
| warm vs cold config ranking | warm says rows/block 32 is 60 % better than 4; cold ranks all four configs within 4 % — "any of rows/block 4–16, 2.65–2.69 ms" | same §Performance |

### K. MTP draft economics `[SIB-GEO]`

| quantity | value | provenance |
|---|---|---|
| draft-step read ledger, witnessed | **16 distinct tensors / 549,465,088 B = 524.011 MiB**, of which the block is **39.0107 MiB** (15 tensors) ⇒ **92.6 % of a draft step's weight traffic is the head re-read**. Measured by running the reference drafter through a weights dict that records every key it indexes | `../qwen3.5-0.8b/prototype/mtp_gatezero/MTP-GATEZERO.md` §1.1. Our analogue, `derived here` from `checkpoints/Qwen3.6-35B-A3B.safetensors.index.json`, is ≈**95 %** bf16 / ≈**92 %** under policy A — same structure, different bytes |
| speedup identity | `speedup = (1 + a_eff) / ((1 + delta) * (1 + c))`, wins iff `a_eff > (1 + delta)(1 + c) − 1` | same §1 |
| draft block by depth | k1 **0.4371 ms**, k2 **0.7666** (ratio **1.754**), k4 **1.5039** (**3.441**); the block ran **1.83× its byte floor**; `empty` arm **1.82 µs/launch** ⇒ 14 launches ≈ 0.026 ms, so the launch floor is **overlapped, not exposed** | `../qwen3.5-0.8b/prototype/mtp_draft/DRAFT-RESULT.md` |
| `delta`, the verify-body cost | plan A **1.6454 ms = 0.1505 of a step → 1.231×**; pessimistic **5.1712 ms = 0.4729** (KILLED); two sequential 1-row bodies KILLED **structurally**; state rollback excluded deliberately at ~0.12 ms/round | `../qwen3.5-0.8b/prototype/mtp_delta/D3-RESULT.md` |


### An allocation is not a grant — the pool bound governs KFD grants, not `hipMalloc` (measured 2026-10-26, window `2026-10-26b`)

The bracket above (≈55.5–57.5 GiB) is **grant arithmetic**: 23.5 + 32 granted, 23.5 + 34 refused, 23.5 + 20 + 12
granted, 23.5 + 20 + 24 refused. It bounds what the KFD will *account* as device-visible memory. It is not a cap on
`hipMalloc`, and the distinction was measured rather than reasoned:

| quantity | value | instrument |
|---|---|---|
| largest `hipMalloc` that succeeded | **66.1337 GiB** — the whole bf16 model, draft block included | `tools/mem_mode_probe_hip.cpp`, 2 reps |
| where those pages came from | `gtt+` moved **66.13 GiB**, `vram+` moved ≤0.08 GiB | sysfs, sampled around the allocation |
| KFD grant ceiling | unchanged at **≈55.5–57.5 GiB** (not retested here — this probe never requested a 66 GiB grant) | grant/refusal arithmetic above |
| steady read of that 66 GiB buffer | **190.4 and 219.2 GB/s** — a spread across two runs, **not** quotable as one number | same probe |
| steady read, `pin` at same size | **237.1 GB/s** both runs | same probe |
| box identity | `visible device memory 110.0 GiB`, `carve-out total 2.00 GiB`, `GTT total 112.00 GiB` | probe's own print |

So `hipMalloc` does not get **refused** past the pool, it gets **slow**: it over-commits the 2.00 GiB carve-out into
GTT pages and pays in bandwidth (~190–219 against pinned 237) instead of in an error. Consequence for the project's
founding arithmetic: **bf16 was never blocked by capacity in this sense** — it can be allocated, and at 66.1337
GiB/token it runs at **~2.8–3.4 t/s**, roughly 28× too slow. D-01 (4-bit routed experts) therefore rests on
**bandwidth**, which is where it should have been stated all along; the decision is unaffected, the justification is
corrected. Full log: `devices/evidence/device-window-2026-10-26b.md`.

### The 60 MiB tier — placement, not kernel shape, decides the GDN state read (measured 2026-10-26, window `2026-10-26b`)

Device reads through HIP, five reps, steady-state GB/s. One layer's GDN state is **60.0 MiB**, which is where the
modes diverge most:

| buffer | `carve` (inside the reservation) | `gtt` (plain `hipMalloc`) | `man` | `pin` |
|---|---|---|---|---|
| **0.0586 GiB = 60.0 MiB** | **239.9** | 185.5 | 105.9 | 112.0 |
| 0.25 GiB | 242.0 | 228.1 | 219.8 | 203.9 |
| 1.00 GiB | 240.3 | 239.4 | 238.5 | 231.6 |
| 48.00 GiB | 240.2 | 236.3 | 237.7 | 237.0 |

`carve` never dips — 237.8 to 242.0 across four orders of magnitude of buffer size — which is why ~240 GB/s is the
denominator everywhere in this repo rather than a single sample. At 60.0 MiB specifically there is a real **~2.1×
tier**: reservation-resident 239.9 against host-backed 105.9–185.5. The 60 MiB `first-run` column is unusable (carve
read 2.3 GB/s there) and no first-run figure from this window is quotable.

### Roofline and restart cost, measured (2026-10-26, window `2026-10-26b`)

- **Policy-A decode cap ≈ 95 t/s** — 240 GB/s ÷ 2.5252 GiB/token, consistent with the probe's own `tok/s cap` column.
- **llama.cpp measured 53.4 / 53.4 / 54.1 t/s** (warm 53.5), `offloaded 41/41 layers` every run, prompt 330.8–360.3
  t/s → effective read ≈ **135 GB/s = 56 % of ceiling**, so **~1.8× headroom** exists in principle. Not a strict A/B
  against the morning's 44.9/49.1/51.4: that prompt was never recorded, this one is
  (`tools/device_window_ab_llama.sh`, 43 words, verbatim in `devices/evidence/device-window-2026-10-26b-llama-ab.log`).
  Best-case conditions: warm page cache, idle box, halogen down.
- **Full-offload residency**: `gtt_used_peak 25.89 GiB`, host `MemAvailable` 119.89 → **92.82 GiB** (the weights are
  host pages read over GTT), `VmPeak 33.79 GiB` (address space, not residency), `VmHWM 1.47 GiB`.
- **Halogen round trip ≈ 83 s** — `down: stopped in 4s` + `up: ready ... after 79s`, the second entry in
  `devices/evidence/halogen-restart.log`. "Batch, do not oscillate" still holds, but it is no longer standing on an
  assumption about reload cost.
- **Broken counter, recorded not skipped**: `tools/device_window_probe.py`'s `KFD per-process vram=UNAVAILABLE (no kfd
  proc files)` while `/sys/class/kfd/kfd/proc/` had entries during the same window — a check that prints nothing
  cannot fail, so AMDGPU sysfs is the load-bearing instrument until that reader is repaired.

## Co-resident residency and M1's device half — window `2026-10-26c` (main agent, engine down for the clean half)

- **D-11 deployment bound is now measured, not adjudicable-by-assertion**: one process took **32.00 GiB of
  GTT-backed staging to completion with the local engine resident and idle** (`mem_info_gtt_used` accounting
  exactly the request at 8/16/24/32 GiB, `rc=0` throughout). Policy A needs 19.7848 GiB resident, so the
  deployment budget is **FITS with margin**. Allocation fact only — the BW columns from that half are not quoted
  anywhere (engine resident; the first-run column spans 8.1 to 231.0 GB/s at one size, which is first-touch noise).
  Engine-**serving** co-residency is still unmeasured.
- **The card is card1 on this box; card0 does not exist.** A `cat` of a nonexistent sysfs file makes an awk delta
  print `0,00`, which is indistinguishable from a true zero unless you check the read. Always prefer the probe's
  own columns (it globs `card<i>`) over parallel reads written in a shell.
- **M1's device half ran and is still NOT established**: `RESULT PASS mode=device checks=31 findings=0
  verified=464 residue=0 uncovered=0 device=on`, yet zero device-side digest fields exist in the 464 records,
  `controls=0 fired=0`, and its only device check (`device-path-was-exercised`, `n=1 ... 1 device construction(s)
  happened`) is answered by construction. M1's exit gate — per-run D2H read-back sha256 vs the file digest,
  count enforced at 464, plus a control that corrupts the device buffer — has never been executed.
- Restart cost: `down: stopped in 3s`, `up: ready in 70s` (entries 3-4 of `evidence/halogen-restart.log`).
Provenance: main agent, third device window of 2026-10-26; full detail and the verbatim lines in
`evidence/device-window-2026-10-26c.md`.

## M1 established — window `2026-10-26d` (main agent, engine down, lock held, blocking)

- Artifact-to-device staging executed on real hardware: 463 of 464 plan runs staged, each read back with
  `hipMemcpyDtoH` and sha256-compared to the digest the artifact region independently produces.
  `RESULT PASS mode=device checks=40 findings=0 controls=10 fired=10 readback=463/463 verified=464 residue=0
  uncovered=0 device=on`; **463 DISTINCT device digests**, 463 MATCH, 1 refused.
- `AGENTS.md` ladder: **M1 met**. The two device controls corrupt/short-change REAL device memory and both fired
  RED (`device-corrupt`: 1 byte at run 343 `weights.exact` `slab_off+0`; `device-short-d2h`: 1023 of 1024 B, caught
  by the count rule and not the digest).
- Not established and not implied: the 464th run (MTP draft block, `draft-requantizer-absent`), any timing
  (M4/M5, >= 48 t/s per D-22), and any device codegen of our kernels.
Provenance: main agent, fourth device window of 2026-10-26; detail in `evidence/device-window-2026-10-26d.md`.
