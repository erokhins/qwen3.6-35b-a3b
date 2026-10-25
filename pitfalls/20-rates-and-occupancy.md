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

## Copied from the sibling engine (2026-10-26)

Method-level traps lifted out of `../qwen3.5-0.8b` on 2026-10-26, with their numbers and dates, because they
still bite on this box at this model's widths. Map of which engine piece each came from:
`planning/port-inventory-2026-10-26.md` §6 and §9 (`T-n` ids below are its ids). Tags as in
`devices/measured-facts.md` §"Copied from the sibling engine (2026-10-26)": `[SIB-BOX]` = this silicon,
model-independent; `[SIB-GEO]` = sibling-model fact, measured on Qwen3.5-0.8B geometry, NOT measured on this
box against Qwen3.6-35B-A3B.

### An instruction-slot bill is only valid where there is concurrency to hide the latency

The 0.8B project priced a GDN "keep the state resident in LDS" route bottom-up at **11.4 ms** (1,370 instructions
per token per useful wavefront → 1,370 cycles/token → 318 µs/site × 36 sites), cross-checked as a rate against
the fp32 vector row; the device returned **112.04 ms — 9.8× the model**, and 2.8× even their own pessimistic
figure. The geometry that the bill had recorded but not priced: "the grid is 64 blocks over 40 CUs, so about
**1.6 blocks per CU, one of whose four wavefronts does the recurrence**. An instruction-count model assumes an
instruction is issued per cycle per SIMD; at 1.6 *working* wavefronts per CU there is nothing else in the machine
to schedule while a lane waits on a dependent load, so **the cost is latency per token, not issue slots per
token**." `[SIB-GEO]` measurement (2026-10-14 window, adjudicated against pre-registered 63 ms / 84 ms lines),
`[SIB-BOX]` lesson. §5 of `../qwen3.5-0.8b/prototype/gdn_collapse/results.md`.

**The rule this repo adopts: every slot figure in a census must be printed next to the launch's
wavefronts-per-SIMD.** A census that reports slots/token without saying whether the launch has the concurrency to
hide a dependent load is reporting a lower bound it will not earn. (Related to §"A launch bill is not a body
price" above, which is the same failure on the launch axis; and our `pitfalls/00` §"Occupancy/latency pricing
beats instruction pricing for serial kernels" is the same failure on the bound axis.)

### Price a launch by deleting launches

Three launch/sync prices exist in the sibling ledger and they are not interchangeable: **0.375 µs/launch** async
submit floor, **1.85 µs per launch-segment** empty-kernel dispatch, **5.9 µs per sync** serialized-drain gap
(`[SIB-BOX]`, `../qwen3.5-0.8b/prototype/launch_price/S1-GATEZERO.md` §2). The number that actually described the
running engine came from **deletion**: folding all 97 `cast` launches into their consumers moved the wall
**−0.3008 ms at 4,032 lanes = 3.10 µs per launch** (`[SIB-GEO]` step, `[SIB-BOX]` mechanism, 2026-10-21,
`../qwen3.5-0.8b/engine/generation-limits.md` §1/§7).

And the two ways the cheap way of measuring it lied, both recorded in that repo at a cost:

1. A fitted ladder at an empty body printed `FIT ok` and `3.9473 us/launch -> S1 IS REAL`, and was withdrawn:
   the ladder ran at `body=tiny` (5.3 MiB, 340 launches) which is "precisely the regime the brief warned would
   overstate", because nothing can overlap behind an empty body (§7 of
   `../qwen3.5-0.8b/progress/2026-10-21/route-S1-segments-2026-10-21.md`).
2. The merge arm compared 340 launches at `grid_max=271` against 48 at `grid_max=1918`, i.e. it moved
   **workgroups-per-launch along with launch count** — "a confound, not a result". Same §7. Our `pitfalls/20`
   already had "read workgroups per launch on both sides of an A/B"; this is the same rule earned a second time.
3. Third, and worse than either: a large part of the measured *sync* leg was the instrument. The engine's shipping
   path never drains per segment, so a prize computed as `segments × µs/segment` (their `mlp` 0.2709 ms, `gdn`
   ~0.19–0.21 ms, the ledger's `segments × 7.2 us`) priced the stage clock, not the engine. Those prizes were
   **withdrawn, not renegotiated** (`[SIB-GEO]` amounts, `[SIB-BOX]` conclusion — `generation-limits.md` §4).

### Blocks per launch has a plateau, then a cliff — and the cliff is fetch capacity

Holding total bytes **and** total workgroups constant and moving only the split between launches and
blocks-per-launch moved time **43×**: at `pop = 1024`, **7.20 ms (L=16, G=64) → 311.37 ms (L=1024, G=1)**.
"That is not launch cost (1008 extra launches × 3.10 µs = 3.1 ms, not 304 ms) and it is not noise. The mechanism
is per-block fetch capacity: at `G=1` a single block must pull 1461 MiB / 1024 ≈ **1.43 MiB** with its own 256
threads and nothing else in flight." The curve is a plateau then a cliff (`pop = 4096`: **6.98 / 6.91 / 7.33** ms
at L=16/32/64, then **11.03 → 76.85**), and the fixed-`G` rows say where the plateau ends: spreads of
**0.13–0.45 ms at G ≥ 32** against **30–270 ms at G ≤ 4**. `[SIB-GEO]` absolute times (taken 2026-10-21),
`[SIB-BOX]` mechanism. §2 of `../qwen3.5-0.8b/prototype/wave_shape/ITEM2-RESULT.md`.

Two consequences for a gather kernel: **do not trade blocks-per-launch away to reduce launch count**, and when an
A/B changes launch count, verify workgroups-per-launch on both sides or the ratio is about the shape, not the
launches. Note also that their own engine sat on the plateau and could not reach the cliff — the fat launch would
have required a cross-layer merge their residual chain forbids — so the cliff is only a hazard for a kernel that
voluntarily makes each block cover a huge byte range with few blocks in flight.

### A kernel can reserve far more LDS than it declares, because of a runtime trip count

`attn_core` **declared 6,176 B** of `__shared__` and **reserved 38,944 B**: "LLVM promote-alloca-to-LDS moves
`qreg[8]` and the K-row `uint4` into LDS when the loop bound is the runtime `nd`. At 64 KiB LDS/CU that is 1
resident block per CU — 8 of 32 wavefronts." The fix was a spelling — `for (t=0;t<8;++t) if (t<nd)` —
byte-identical across 126 stage comparisons and 21 cases, residency **1 → 4** blocks/CU, **−46 %** on the family
line and **−9.9 %** on the step. `[SIB-BOX]` compiler behaviour (gfx1151, clang as shipped by ROCm here);
`[SIB-GEO]` the milliseconds. §3 of `../qwen3.5-0.8b/engine/generation-limits.md`, 2026-10-21.

Three rules ride along, all of them ours already and all of them sharpened by this: (1) an occupancy claim comes
from the probe (`tools/kernel_occupancy_probe.cpp`), never from `__shared__` text or a compiler note; (2) bound
classification is arithmetic before it is intuition — there the traffic was **27.7 %** of the plateau and the
issue rate **5.9×** under its own roof, so neither could be the wall, which left latency at occupancy; and (3)
**check that the lever's shape can bind at the reported width at all** — at 128 lanes their launch was 8 blocks on
40 CUs, where neither residency nor population can change anything, and the line did not move when the fix
landed, exactly as predicted.

### A probe's own summary column can be a mislabeled counter

`issue_rate_probe`'s binary printed `mac_per_lane_clk` numerically equal to `icpcb`: it was reporting
**warp-instructions per CU per clock labelled MACs per lane per clock**, which is why `irp_f32_8` claimed **172 %
of a 1-MAC ceiling** and `irp_mix_8` claimed **248 %**. The table anyone quotes is recomputed from geometry
instead: `MACs per launch = warps × 32 × iters × ops_per_iter(isa) × macs_per_lane_per_op`, divided by best
seconds and by **2560 lanes × 2.894 GHz** (40 CU × 64 lanes — the KFD geometry in the section above about
reading geometry from KFD, not from HIP). §0 of `../qwen3.5-0.8b/prototype/issue_rate_probe/results.md`,
`[SIB-BOX]`, taken 2026-10-16 with the lock held. Two adjacent disciplines from the same run: the leg whose byte
witness printed `checksum=FAIL` was **excluded** rather than footnoted, and a leg with no byte account at all got
no row; and their full-occupancy timing recipe is `blocks=320 threads=128` → 1,280 waves = **32 wavefronts/CU on
40 CUs**.

### A noise floor is per-comparison, not a global absolute

Two sibling failures, both fixed by refusing rather than relaxing. (1) Their shape matrix's adjudication used a
single noise floor defined as the largest **absolute** within-arm spread, which let one **311 ms** cell make a
7 ms comparison unadjudicable (floor **0.3071 ms ≥ the 0.30 ms WIN band** → `ADJUDICATION-FAILED`, and the band
was **not** moved); the fix was to judge each comparison against the spread of **the two arms it uses**. (2) A
per-**stratum** bimodality cap no arm could pass — the warm leg of a large single read repeats **2.0–2.5 % apart
at REPS=5 and at REPS=12 alike** — discarded almost every stratum until it was scoped per comparison. Separately,
a re-take refused bimodally at **21.8 %** (0.0285 vs 0.0347 ms across two passes) and the pre-registration said
re-take, not average. `[SIB-BOX]` (this box's dispatch-bound bimodality, which our `pitfalls/00` already names),
measured 2026-10-21. §1 of `../qwen3.5-0.8b/prototype/wave_shape/ITEM2-RESULT.md` and the corrections log of
`generation-limits.md` §6.

### An above-probe GB/s is explained by tiling past the cache — and a row no knob can reproduce is retired

The 0.8B's own `>240 GB/s means skipped work` rule (ours, in `pitfalls/00-measurement.md`) needed a clause: their
GDN whole-family launch read **18.0 MiB at 876 GB/s**, which was real and cache-resident. "The way to tell is to
tile the working set past any plausible cache and re-measure … Without that check it looked exactly like the
`lm_head_gemv` grid bug." `[SIB-GEO]`, 2026-10-11 re-take. And the same file retires its own companion row —
108 MiB at 255 GB/s — because "the parser takes `mode|dir|out|iters|layers` … and has no tiling knob — so 255
GB/s is quoted from a build that no longer exists". **A number whose instrument cannot re-produce it is retired,
not annotated.** §Performance and the re-take note of
`../qwen3.5-0.8b/prototype/gdn_decode/results.md`.

### Split-K buys parallelism twice: it re-stages the activation, and it re-associates the sum

Split-K on their one-row `down_proj` **lost at every setting tested** — 2, 4, 7, 8 and 14 ways, ship **0.0179 ms**
against **0.0204 / 0.0251 / 0.0246** for `intra2/4/8` and **0.0228 … 0.0369** for the two-pass forms — "because a
split re-stages the activation row **1,792 → 14,336 KiB against 7,168 KiB of weights**". The numerics rode
along: `intra1` was `ndiff 0`, but `intra2/4/8` moved **≤ 2 / ≤ 4 / ≤ 48 ULP**, so a split-K that ever wins also
changes the addition order. `[SIB-GEO]`, 2026-10-21. §3 of
`../qwen3.5-0.8b/prototype/wave_shape/ITEM2-RESULT.md`. For this repo: price and band a split in the same arm, and
check the activation-vs-weight staging ratio first — at our widths the shared expert's activation row is 8 KiB
F32 (`derived here`) against a 6 MiB expert weight, which is a more favourable ratio than theirs, so the verdict
must be re-measured rather than inherited.
