# 20 — rates, occupancy and censuses

> **Provenance.** Copied and condensed on 2026-10-26 from the 0.8B project's
> (paths relative to `../qwen3.5-0.8b/`) `prototype/issue_rate_probe/results.md`,
> `devices/strix-halo-compute.md`,
> `prototype/attn_percost/{a11_read_vgpr.py,a11_alloc_check.py}`, `prototype/attn_wide/A12-WHY-NOT-FASTER.md`,
> `prototype/gdn_walk/BET.md`, `prototype/gdn_collapse/results.md` and its traps index. All figures are
> measurements on this box. Probes are copied into `tools/` here.

## A rate must be compared against the roof of its own instruction class

Measured sustained rate per class, 40 CU × 64 FP32 lanes × 2.894 GHz:

| class | instruction | T·FLOP/s | % of its own roof |
|---|---|---|---|
| scalar FP32 | `v_fma_f32` (i.e. `v_fmac_f32_e32` in emitted ISA) | **7.30** | 49 % of 14.82 |
| packed FP16 | `v_pk_fma_f16` | 17.66 | 60 % |
| dot, f16 | `v_dot2_f32_f16` | 22.59 | 76 % |
| dot, bf16 | `v_dot2_f32_bf16` | **23.28** | 79 % of 29.6 |
| dot4 i8 | `v_dot4_i32_i8` | 28.75 | 49 % |
| dot8 i4 | `v_dot8_i32_i4` | 55.96 | 47 % |
| WMMA f16 | `v_wmma_f32_16x16x16_f16` | **52.63** | 89 % of 59.4 |
| library bf16 GEMM with a 16-bit `C`, at engine shapes | hipBLASLt tuned kernels | **33.2–37.7** | — |
| dense vector roof (bf16/fp16 at 1:1) | — | 14.82 | — |

Three failures this table exists to prevent:

1. Judging a scalar-`v_fma_f32` kernel against a matrix roof overstated a gap by 3× and nearly funded a
   route on the wrong premise.
2. **Check who wrote the kernel you are comparing against.** The 0.8B engine's own 40 kernels contain zero
   matrix or dot-class instructions, so every matrix rate in that ledger belongs to hipBLASLt, whose tuned
   kernels are unreadable CCO containers.
3. A retired figure can still be quoted. A WMMA route was once priced at 11.2× against **4.696 TFLOP/s**,
   which the source file had already retired in a section titled *"Second correction, same day: every rate
   above is an F32-`C` rate, and it is 6–8× below the same call"*. The real prize was 1.4–1.6×. This repo's
   rule: check a number's file for a correction section before letting it size a route.

## Device geometry: read it from KFD and `rocminfo`, never from HIP on this SKU

`hipGetDeviceProperties` on this part reports `multiProcessorCount = 20` (half the machine),
`maxThreadsPerMultiProcessor = 2048` (= 64 wavefronts against an architectural 32) and
`sharedMemPerMultiprocessor = 65536`. Use instead:

| fact | source | value |
|---|---|---|
| CUs | `rocminfo` (`Compute Unit:` for the gfx1151 agent) | **40** |
| SIMDs per CU, max waves per SIMD, total SIMDs | `/sys/devices/virtual/kfd/kfd/topology/nodes/1/properties` | 2 / 16 / 80 |
| LDS per CU | same KFD properties, `lds_size_in_kb` | 64 KiB |
| max wavefronts per CU | `rocminfo` `MaxWavesPerCU` | 32 |
| register file | **per SIMD, not per CU** | 65,536 registers per SIMD |

The register-file point cost a design doc an argument from a limit that was never binding: 256 threads ×
256 VGPR is a *per-thread-ceiling* number, and dividing 65,536 "per CU" made 3 resident blocks look
impossible. Measurement (padding declared LDS 20,480 → 32,768 costs +3.36 % at 3,072 lanes and +4.42 % at
16,384 with disjoint per-rep ranges) requires 3 resident blocks, which is only consistent with a per-SIMD
file. **A capacity from the driver's occupancy query is not a resource a launch owns**: realised residency
is `min(residency from LDS/registers, blocks available per CU)`.

## A compiler-reported occupancy is a constant, not a measurement

clang's `-Rpass-analysis=kernel-resource-usage` prints `Occupancy [waves/SIMD]: 16` for every kernel in the
translation unit regardless of LDS or VGPR. Any claim resting on "occupancy unchanged at 16 waves/SIMD" is
resting on a check that cannot fail. Use the occupancy probe (`tools/kernel_occupancy_probe.cpp`, copied
here) and **note its canary pattern**: it proves the query distinguishes a 64 KiB-LDS stub (1 block) from a
0-LDS stub (16) from the kernel under test (5 blocks = 20 wavefronts/CU of the 32 allowed). A probe with no
canary that must fail is a decoration.

## Register counts come from the code object, not from the LLVM note block

`.vgpr_count` in the note block is LLVM's own count and note entries are **not in definition order**, so a
parser that pairs a kernel name with the next `.vgpr_count` reports another kernel's value — that is how a
wrong "185 VGPR / 0 spill" entered a repo and matched a driver query that agreed only because it read the
same build's metadata. The field the driver reserves is **`.amdhsa_next_free_vgpr`** in the `.amdhsa_kernel`
block (241 → 248 at granularity 8), and that reproduces the pmc `VGPR_Count` exactly for three known
kernels. Related: a `__shared__` pad array written only under a condition the compiler cannot prove false
is still **dead** (nothing reads it, allocation eliminated) — so a "padding experiment" that pads an unused
array measures the same binary twice.

## A launch bill is not a body price

A static launch-record sum priced a batched 2-row verify body at **0.1505 of a step**; measured, the same
body cost **+0.72 to +1.98** steps, because no launch record says "this body is forced onto the route that
walks the weights once per row". The bill was low for the wrong reason and the fix was a route, not a
rewrite. **Price a body by differencing two device arms at the same width; when a carried bill and a
measurement disagree by an order of magnitude, the bill is the finding.**

The same class, one axis over: **read workgroups per launch on both sides of an A/B before believing the
ratio.** A "wide" attention block won **+25.1 %** in a rig's single 2,048-workgroup launch and **0 %** in the
engine's 32-workgroup launches; per-body device timestamps read ratio 1.0001 across 31 paired bodies. The
block covered 32 query positions with no lane split, so a 512-row body emitted 32 workgroups on a 40-CU
machine — ≤1 block/CU on 32 CUs, 8 wavefronts/CU against the 9 its predecessor sustained.

And a third: a sweep timed at a width the shipping path never issues is a fact about the library, not the
engine. Derive the call widths the engine can actually emit and aggregate at those; keep the others only as
a labelled bound.

## Census and counter traps

- **A flat per-kernel ISA census divides by nothing**, so it bills once-per-launch overhead once per token.
  Kernarg loads, guard chains and the `tokens<=0` path run per launch; splitting per-launch from per-token
  using LLVM's own loop annotations turned **+156 slots/token** into **−47**. Always ask a slot figure which
  denominator it used.
- **A census must name the files it could read.** "0 of the 55 `gfx1151` code objects contain a matrix
  instruction" was taken over the `*.hsaco` fallbacks a disassembler parses and missed the tuned kernels in
  CCO containers it refuses. Write "0 of the readable 55" until "0 of 55" is actually true. The same failure
  one level over: citing vendored code that is inside a preprocessor guard the target platform does not
  enable — print the guard, and check which side of it the target falls on.
- **Before writing a probe or census tool, list the existing ones.** Three census tools already existed for
  one kernel family and a fourth was being written; it would have printed a confident wrong number, because
  (a) clang emits function bodies under a bare column-0 label, not `.globl`, so a `.globl` slicer reports a
  57-slot stub as the kernel; (b) basic-block labels carry `; in Loop:` comments, so a `^(\.L\w+):$` pattern
  matches nothing and a "widest backward branch" heuristic bills an inner unrolled strip (190 slots with
  **zero FMAs** for a loop that must hold 256); (c) the RDNA3 mnemonics are `v_fmac_f32_e32`, `ds_load_b128`,
  `ds_bpermute_b32`, `s_delay_alu`, `s_waitcnt` — not the naive table's names.
- **A device selftest must verify bytes touched, not arithmetic.** A geometry selftest printed `ok=1` for
  exactly the configurations whose kernel then read 62× past its own buffer and faulted, because the solver's
  `rounds` and the kernel's `rounds` meant different things. Re-derive the kernel's index arithmetic on the
  host from the same struct the launch is built from, and check an exact read checksum, write-landing
  read-back, and poison guard bands. Run each device case in its own forked child: one faulting kernel
  poisons the context and every later case silently never ran.
- **A read-back checksum that is published per thread is not a block reduction.** `sums[blockIdx.x] =
  per_thread_acc` writes the same address from every thread, so a reader tabulated 1/256 of the bytes it had
  read while the ms and GB/s looked entirely plausible. Reduce across the block, and derive the expected
  value from the plan rather than from the kernel so the check cannot pass by construction. Same lesson
  mirrored: a write side verified by reading back a buffer pre-filled with the value the writes store proves
  nothing — pre-fill it with a value writes never store.
