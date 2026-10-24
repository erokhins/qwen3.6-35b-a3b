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

**~32–34 GiB per process.** `totalGlobalMem` reports ~110 GiB and is not what a process can use. This is the
binding constraint of this project — bf16 text weights are 64.6 GiB — and the escape hatch (host-mapped buffers
above the pool) is untested: `DECISIONS.md` D-02.

## Roofline for the target model (derived here from `config.json` ÷ measured 237 GB/s)

| configuration | GiB read per token | ceiling |
|---|---|---|
| bf16 text weights, 4k context | 5.49 | **~43 tok/s** |
| Q8_0-resident (32.3 GiB resident — does not fit) | 2.74 | ~86 tok/s |
| Q4_K_M-resident (~16–21 GiB) | 1.37 | **~173 tok/s** (realistically 60–70 % of this) |
| 4,032 lanes of KV on top of the short-context row | +0.04 GiB | KV is 10 layers × 2 KV heads × 256 dim × 2 B = **10 KiB/token**, i.e. ~0.7 % of the bf16 per-token read. Note this is *bytes stored*, not attention cost — attention time at long context is dominated by other traffic and must be measured, not inferred from this row |

Sanity anchors for the derivation: derived Q8 = 32.3 GiB vs public `Q8_0.gguf` = 35 GB; derived Q4 = 16.1 GiB
vs the quoted 20.4–20.8 GB Q4_K_M. The 0.8B sibling's equivalent row (1.565 GB/token → 151 tok/s) is the same
formula, and its engine reached 91.4 tok/s marginal at 4k with MTP on top — so treat the ceilings as ceilings
and expect ~60–70 % of them from a real engine.

## Re-measuring

Only after a driver, BIOS or memory-carve-out change. The probes that produced these numbers are copied into
`tools/` here: `bw_probe_hip.cpp` (size–bandwidth curve), `mem_mode_probe_hip.cpp` (allocation modes + pool
ceiling), `gemm_peak.cpp` + `gemm_common.h` (matrix rates), `kernel_occupancy_probe.cpp` (occupancy with canary),
`launch_overhead_hip.cpp` (launch cost), `bw_probe.py` (CPU side), `isa_probe.sh` (ISA/class inventory). Build
with `hipcc -O2 -Wl,-rpath,/opt/rocm/lib`, take them blocking with `tools/gpu_lock.sh` held, and record the split
first: `cat /sys/class/drm/card*/device/mem_info_{vram,gtt}_{total,used}`.
