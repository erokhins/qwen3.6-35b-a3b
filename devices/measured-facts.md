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
reports ~110 GiB and is not what a process can use. Text weights are **65.72 GiB bf16**
(`tools/quant_scenarios.py --census`), so the pool binds either way, and host-mapping does not
escape it (it consumes it): see the next section.

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
serves this session). Neither `RLIMIT_MEMLOCK` (15.45 GiB here, `unlimited` for halogen, and a 32 GiB
pin was granted anyway) nor system RAM nor GTT occupancy explains it.

Consequences, now stated against the corrected model:

1. **The engine's budget is `pool − co-resident footprint`, so it is an ops decision, not a constant.**
   ≈32 GiB while halogen is resident; **≈55 GiB if the engine owns the box.** That is a different
   project: Q8-throughout (34.97 GiB) *fits* at 55 GiB, so 4-bit experts becomes a **speed** choice
   rather than a fit requirement (`DECISIONS.md` D-11 asks the question; `quantization/README.md`
   prices both budgets). bf16 (65.72 GiB) fits in neither.
2. **Pinning does not escape the pool — it spends it.** Mapped host memory is charged to the same
   budget as device memory, and reads at ≥ device bandwidth (198.6 device vs 215.8 pinned GB/s in one
   window), so a two-tier layout is bandwidth-viable but buys no additional capacity.
3. **The endgame still needs more than this box.** For `Qwen3.8-Flash-Next` (232.9 GiB text bf16) even
   4-bit main weights are ≈58 GiB, above even the whole pool: it needs streaming with a real bandwidth
   cost, a pool raise (`gttsize`/`rmem`/BIOS, untested), or different hardware. Recorded so nobody
   re-derives it from hope.


## Roofline for the target model

Derived from `checkpoints/Qwen3.6-35B-A3B.config.json` ÷ the measured read bandwidth, by
`python3 tools/quant_scenarios.py --read 237 210 190`. The full policy table (residency and ceiling
for bf16 / Q4_K-experts variants / Q8-throughout) lives in `quantization/README.md`; this is the
shape of it:

| configuration | resident | GiB read per token | ceiling @237 GB/s |
|---|---|---|---|
| bf16 text weights | 65.72 GiB ✗ above both budgets | 5.194 | ~42 tok/s |
| Q8_0 throughout | 34.97 GiB — above the co-resident budget (~32), **fits if the engine owns the pool (~55)** | 2.811 | ~79 tok/s |
| experts Q4_K + rest Q8_0 (policy A, **decided**) | 19.59 GiB ✓ fits either budget | 2.331 | ~95 tok/s |
| experts Q4_K + attn Q8_0 + `lm_head` Q6_K (policy G) | 19.36 GiB ✓ | 2.216 | ~100 tok/s |
| experts + attention both Q4_K (policy E) | 18.81 GiB ✓ | 1.663 | ~133 tok/s |

Two qualifications. Ceilings are ceilings: the sibling 0.8B engine reached ~60 % of its own, and the
bandwidth actually delivered in the 2026-10-26 window was 190–225 GB/s, so quote the 190 column as
the honest lower bound. And these are **bytes**, not time: attention time at 4k context is not the KV
byte count (KV is 10 layers × 2 KV heads × 256 dim × 2 B = 10 KiB/token, ~0.7 % of a bf16 step) —
long-context cost has to be measured, not inferred from this table.

Anchors for the arithmetic: policy A predicts 19.59 GiB = 21.0 GB where the published GGUFs for this
architecture are 18.49 GB (unsloth Q4_K_M) and 19.77 GB (bartowski) — our figure is high by ~6–13 %
in the expected direction, because those quants are dynamic (some expert tensors below Q4_K) and keep
attention at Q4_K too. Details and sources: `quantization/evidence/`.

## Re-measuring

Only after a driver, BIOS or memory-carve-out change. The probes that produced these numbers are copied into
`tools/` here: `bw_probe_hip.cpp` (size–bandwidth curve), `mem_mode_probe_hip.cpp` (allocation modes + pool
ceiling), `gemm_peak.cpp` + `gemm_common.h` (matrix rates), `kernel_occupancy_probe.cpp` (occupancy with canary),
`launch_overhead_hip.cpp` (launch cost), `bw_probe.py` (CPU side), `isa_probe.sh` (ISA/class inventory). Written
here: `hostmap_probe.cpp` (pool boundary, host pinning, checksum-witnessed reads) and `quant_scenarios.py`
(census + policy arithmetic, no device). Build with `hipcc -O2 -Wl,-rpath,/opt/rocm/lib`, take them blocking
with `tools/gpu_lock.sh` held, and record the split first: `cat /sys/class/drm/card*/device/mem_info_{vram,gtt}_{total,used}`.
