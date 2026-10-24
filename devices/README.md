# devices/ — measured facts about this box (measured elsewhere, reused here)

Nothing in this project should re-measure the platform. The measurements were made in
`../qwen3.5-0.8b/devices/` on the same machine (Framework Desktop, AMD Ryzen AI Max+ 395,
Strix Halo `gfx1151`, 128 GB LPDDR5x-8000 unified memory, single NUMA node) and their probes
live in `../qwen3.5-0.8b/tools/`. Read the source file before quoting; correct-in-place marks
matter (that repo has retired several early figures by editing the same document).

| quantity | value | source (read it, don't re-measure) |
|---|---|---|
| iGPU pure-read bandwidth | **~237–240 GB/s** (210 GB/s for a copy; the standing denominator for every roofline here is 237) | `../qwen3.5-0.8b/devices/strix-halo-memory-bandwidth.md` §size–bandwidth curve |
| CPU-side bandwidth | **75–117 GB/s** — a CPU-resident decode caps an order of magnitude below the GPU path | same file |
| device-visible memory pool | **~32–34 GiB per process**, granularity-independent; 40 GiB `hipMalloc` fails; `GPU_MAX_HEAP_SIZE=100` does not move it | `../qwen3.5-0.8b/devices/strix-halo-unified-memory.md` §"The two real limits" |
| first touch on fresh pages | **~35 GB/s** (7× slow) — warm before timing, always | same file |
| `hipMallocManaged` near the ceiling | degrades to ~52 GB/s **silently**; `hipMalloc` fails loudly → use `hipMalloc` + explicit `hipMemcpy` for weights | same file |
| host-pinned vs GTT read speed | same speed ⇒ residency choices are about address space and control, not throughput | same file, §"What this does not buy" |
| staged host→device copy | 54–58 GB/s including destination first touch, 76–82 GB/s steady | same file, §"The recipe the engine will use" |
| instruction-class ceilings | scalar `v_fma_f32` 7.30 → `v_dot2_f32_bf16` 23.28 → WMMA 52.63 TFLOP/s — always compare a rate against the ceiling of its own class | `../qwen3.5-0.8b/prototype/issue_rate_probe/results.md` §1 |
| CU / SIMD / LDS facts | 40 CUs, `simd_per_cu 2`, `simd_count 80`, LDS 64 KiB — read from **KFD and `rocminfo`**, never from `hipGetDeviceProperties` on this SKU (it reports a halved CU count and a 2× wavefront ceiling) | `../qwen3.5-0.8b/tools/kernel_occupancy_probe.cpp`, `prototype/attn_wmma/evidence/` |
| compute notes, WMMA signed-zero caveat, mixed read/write traffic | `../qwen3.5-0.8b/devices/{strix-halo-compute.md,strix-halo-wmma-signed-zero.md,strix-halo-mixed-traffic.md}` | reuse |
| official spec sheet | `../qwen3.5-0.8b/devices/strix-halo-official-specs.md` | for vendor numbers only |

**When a re-take is legitimate:** after a driver, BIOS or memory-carve-out change, using the
checklist at the end of `strix-halo-unified-memory.md` (`mem_info_{vram,gtt}`, `mem_mode_probe`,
pool-ceiling probe, `free -g`). Re-taking a number just to confirm it is a budget leak.

**Measurement protocol for this box** (whatever is being measured): `tools/gpu_lock.sh` acquired
by TAG, zero subagents active, foreground/blocking only, warm pages first, medians with their
spread, and both sides of an A/B inside one lock hold. Full version:
`../qwen3.5-0.8b/engine/performance-status.md` §6.
