# Port inventory — prior art copied out of `../qwen3.5-0.8b` for the five unbuilt pieces

Written 2026-10-26. **CPU-side synthesis only: no device context was opened and no timing was taken.** Every
millisecond, GB/s and µs figure here is **copied** from a measurement the sibling project took, with its path
and the geometry it was taken on. Where a number is derived here rather than copied, the arithmetic is printed
inline and labelled `derived here`.

The rule being executed is `AGENTS.md` §"Self-containment: look elsewhere, but write it here" and
`pitfalls/60-docs-and-citations.md` §"The self-containment rule": we may **look at** `../qwen3.5-0.8b`, we may
not **reference** it. A pointer may accompany a copy; it may never replace one. The copies landed in
`devices/measured-facts.md` §"Copied from the sibling engine (2026-10-26)", `pitfalls/20-rates-and-occupancy.md`
§"Copied from the sibling engine (2026-10-26)", `pitfalls/50-harness-and-shell.md` §"Copied from the sibling
engine (2026-10-26)", and `reference-implementation.md` §"Reuse table, second pass". This file is the map.

## 0. Tag vocabulary (two classes, because one tag would lie)

The task's tag — *"sibling-model fact: measured on Qwen3.5-0.8B geometry, NOT measured on this box against
Qwen3.6-35B-A3B"* — is right for most of what is here and wrong for a minority of it, so two tags are used and
every ledger row in §9 carries exactly one. Putting the geometry tag on a cache-size measurement would
understate it; dropping it from a bandwidth row is the failure mode this task exists to prevent.

| tag | meaning | how it may be used |
|---|---|---|
| `[SIB-BOX]` | measured on **this** silicon inside the sibling project, and the quantity is model-independent (cache topology, launch/dispatch cost, compiler behaviour, allocation API, instruction issue rates) | a box fact. Still **not** measured against Qwen3.6-35B-A3B tensors, so it is a design input, never an acceptance result |
| `[SIB-GEO]` | **sibling-model fact: measured on Qwen3.5-0.8B geometry, NOT measured on this box against Qwen3.6-35B-A3B** | an order-of-magnitude prior and a method. Never a design constant here; re-derive at our widths |

Sibling geometry, for every comparison below (`../qwen3.5-0.8b/README.md` §target table): H=1024, 24 layers
(18 GDN + 6 full-attn), GDN **16 V heads / 16 QK heads**, head dim 128, conv 4, MLP I=3584, tied head. Ours
(`checkpoints/Qwen3.6-35B-A3B.config.json`): H=2048, 40 layers (30 GDN + 10 full-attn), GDN **16 k-heads / 32
v-heads** — **the k:v head ratio differs (1:1 there, 1:2 here)** — d=128, conv 4, `moe_intermediate_size: 512`,
`shared_expert_intermediate_size: 512`, untied head, 256 experts / 8 routed / 1 shared. The recurring derived
constants:

```
our lm_head bf16   248320 × 2048 × 2 B          = 1,017,118,720 B = 970.0 MiB = 0.9475 GiB    derived here
our lm_head Q8_0   508,559,360 elts × 1.0625 B  =   540,344,320 B = 515.3 MiB (34 B / 32 elts; 64 blocks/row, no pad)
their tied table   bf16                          =   508,559,360 B = 485.0 MiB    [SIB-GEO] same vocab, H=1024
our GDN state/req  30 × 32 × 128 × 128 × 4 B    =  62,914,560 B =  60.0 MiB       derived here
their GDN state    18 × 16 × 128 × 128 × 4 B    =  18,874,368 B =  18.0 MiB       [SIB-GEO] (§8 of ../qwen3.5-0.8b/prototype/gdn_rec_isa/results.md)
```

That Q8_0 row is the most useful arithmetic in this file: **our policy-A `lm_head` read (515.3 MiB) is 1.06×
their bf16 tied table (485.0 MiB)**, so their measured single-row head line is a direct prior for ours even
though our table has twice the parameters at half the bytes.

## 1. Piece 1 — the shared expert (dense MLP: gate/up/down, I=512, H=2048, plus `shared_expert_gate`)

**Sibling prior art:** `prototype/mlp_gemv/{results.md,floor_probe.py,ulp_probe.py,run.py}`,
`prototype/mlp_insitu/D2-DIAGNOSIS.md`, `prototype/proj_gemv/results.md`,
`prototype/gemm_algo_sweep/{README.md,results.md}`, `prototype/cast_fold/ITEM1-HANDOVER.md`,
`prototype/glue_copy_fold/GLUECOPY-HANDOVER.md`.

### What it established (quoted)

- **The kernel shape that gates.** `mlp_gateup<1024>` fuses gate **and** up into one launch (14.68 MiB of
  weights) because the SwiGLU epilogue needs both dots' results; `mlp_down<3584>` is the second (7.34 MiB).
  "One wavefront per output row, `uint4` (8 × BF16) per lane, F32 accumulator, wave-shuffle reduction,
  `warpSize` used everywhere so wave32 (this part) and wave64 (Navi) both work. Row loop is
  `#pragma unroll`-free and index-strided." Fusion is not optional: "§07.4's launch floor (1.85 µs × ~200
  launches/token) says fusion is the lever, not a micro-optimisation." §What runs of
  `../qwen3.5-0.8b/prototype/mlp_gemv/results.md`.
- **The epilogue rounding policy is a rule, not a detail.** `act = BF16( BF16( silu_F32( BF16(gate) ) ) × BF16(up) )`
  — three roundings. "Computing the epilogue entirely in F32 — the tempting, 'mathematically prettier' reading —
  moves **27 % of `act` by 1 ULP**." Two sub-traps: HF's order is `x * (1/(1+exp(-x)))`, and the algebraically
  identical `x / (1+exp(-x))` "cost a 2 ULP outlier at a *large* element"; and an F32 `exp` 1 ULP from libm
  "flips the BF16 silu result roughly once per few thousand elements". Hence their `act` ≤ 2 ULP band.
  §The SwiGLU rounding policy of that file. **This transfers as arithmetic**: same SiLU, same three casts, same
  HF `Qwen3_5MoeMLP` path our README cites.
- **Warm ranking is not cold ranking.** Warm says rows/block 32 is 60 % better than rows/block 4 (0.0291 vs
  0.0331 ms, 504.2 vs 443.6 GB/s); cold — weights replicated 6× so consecutive touches are 84 MiB apart, "far
  past the ~20 MB iGPU cache" — ranks the same configs "within 4 % of each other", and "Treat the honest
  statement as 'any of rows/block 4–16, 2.65–2.69 ms' rather than a winner." Their own headline is marked
  "**That headline is provisional and due for a clean re-measure**" because the sweep did not **alternate** the
  two kernels inside a burst and validated only replica 0. §Performance of that file.
- **The in-situ penalty was not memory.** Decomposition of their 1.0878 ms: `T_geo = R0 − R0f` = **+0.7565 ms =
  69.5 % — launch shape**; `T_serial = R1 − R0` = **+0.2709 ms = 24.9 %** (an instrument, per §4 of
  `../qwen3.5-0.8b/engine/generation-limits.md`); `T_pipe` = +0.0484 ms = 4.4 %; "residue for memory state
  **~0.06 ms** | 5.5 %". Verdict: "**D2 as written is dead, and no code should be funded against it.**" §5 of
  `../qwen3.5-0.8b/prototype/mlp_insitu/D2-DIAGNOSIS.md`. `[SIB-GEO]` in magnitude, `[SIB-BOX]` in mechanism.
- **Where the shape term lives, and why it is not a free prize.** "the shipped shape `down_proj` runs 1,024 rows
  across 128 blocks on 40 CUs — roughly three blocks per CU — so a one-token MLP is
  **under-parallelised per launch**"; but "The fat shape merges 24 layers' MLP into two launches, which the
  decode step cannot do: layer *i*'s `act` must exist before its `down`, and `down` feeds the residual that
  layer *i+1* reads." Same §5. Later priced and closed: **split-K on the one-row `down_proj` loses at 2, 4, 7, 8
  and 14 ways** (ship 0.0179 ms/call; `intra1/2/4/8` = 0.0179 / 0.0204 / 0.0251 / 0.0246 ms; `two2…two14`
  0.0228 … 0.0369) "because a split re-stages the activation row 1,792 → 14,336 KiB against 7,168 KiB of
  weights" — and `intra8` moved **≤ 48 ULP**, i.e. split-K re-associates. §3 of
  `../qwen3.5-0.8b/prototype/wave_shape/ITEM2-RESULT.md`.
- **The weight-walk *shape* costs a factor of two at 100 % coalescing.** `gbs(eng)/gbs(plain)` = **0.470**
  (gate/up), **0.472** (gate/up with the step's 62:1 writes), **0.629** (`down`), "all three inside the ≤ 0.70
  'confirmed' band … with transaction coalescing measured at 100 % on both sides and LDS staging exonerated
  (`--stage 0` ⇒ `nostage/eng = 1.098`)"; the decisive row `R_decisive / A_read_512m = **0.598**`, and "it
  arrives at half the rate. Contiguous streaming is not free on this box; a strided two-stream walk with a row
  tail is a memory-system pattern, and the pattern is the bottleneck." §Q9 Results of
  `../qwen3.5-0.8b/devices/strix-halo-mixed-traffic.md`. The named cause was the **two-stream row interleave**
  ("a warp's live addresses alternate between two separate 7.2 MiB allocations every instruction"), and the fix
  (`--mapping contig`: walk the gate row end to end, then the up row) was proven bit-identical:
  "`act` interleave vs contig | 0 / 3584 differ | 0 at every layer", ditto `gate_pre`+`up_pre` (0 / 7168) and
  `out` (0 / 1024), with the ISA witness `v4,v2,v4,v2,…` → `v17,v17,v17,…`. §Q10 of
  `../qwen3.5-0.8b/prototype/mlp_gemv/results.md`.
- **What the published 98 % floor actually measured.** "`mlp` isolated burst, 528 MB, `contig` |
  **2.275 ms/token** (98 % of probe)" — but that burst row is "one layer's tensor **replicated 24x and
  concatenated**: 2 launches, gateup 86,016 rows / 10,752 blocks, down 24,576 / 3,072" while "The engine's
  shipped walk is 48 launches at 3,584 / 1,024 rows (448 / 128 blocks)"; the shape-matched arm reproduces the
  floor at **+1.0 %**. D2-DIAGNOSIS §0(iv) and §5.
- **Token-batched (prefill) form, and what makes it safe.** "Same `v = lane; v += ws` walk over the same
  `uint4` words, same 8-element inner loop, same `reduce_row` ladder, same three roundings. Only the
  accumulator became an array", and the gate requires identity: "**bit-identity vs k=1** (`act`+`out`, 5
  tokens) | **23 040 elements, 0 differ**". "Shared memory, not registers, is the bound … `TPB=4` (16 KiB) at
  `IN=1024`, `TPB=2` (28 KiB) at `IN=3584`. Both stay under the 48 KiB static limit and leave two blocks
  resident per CU." §P4 of the mlp_gemv file.
- **Price only shapes the engine emits.** "a sweep timed at a width the shipping path never issues is a fact
  about the library, not the engine. Derive the call widths the engine can actually emit and aggregate at
  those; keep the others only as a labelled bound." §"What this says about the 1.4–1.6× headroom" of
  `../qwen3.5-0.8b/prototype/gemm_algo_sweep/results.md`; the derived-must-catch rule is §5a of its README
  ("The must-catch set is derived from the object, not from arithmetic"). Their library-route headroom over
  hipBLASLt's default was **1.4–1.6×**.
- **Deletion beats folding when the widening is exact — and folding is not always a win.** "Widening BF16→F32
  is exact, so all 6 sites fold into their consumers and the step saves **0.3008 ms** — which is also the answer
  to whether a launch costs anything: **3.10 us per launch**, a production cost, not an instrument artefact"
  (§1 `cast` row, `../qwen3.5-0.8b/engine/generation-limits.md`); but the 49 `glue` launches were **not**
  folded, because riding a consumer would cost "394 MB/token of extra reads to ride a consumer, 10.9× their
  0.152 ms prize" (§7).

### What transfers

1. Kernel shape wholesale: one wavefront per output row, 16 B per lane, F32 accumulator, wave-shuffle
   reduction, `warpSize`-general, unroll-free strided row loop; gate+up fused with the epilogue inside one
   launch; **contiguous** per-stream walks, never an interleaved two-stream row loop.
2. The SwiGLU cast policy verbatim (three roundings; HF's `x * (1/(1+exp(-x)))`) as a rule enforced by a
   `--negative f32act` injection that must localise to `act` + the chain and nowhere else.
3. Gate structure: F1 (dots independently) / F2 (epilogue alone) / F3 (`down` alone, fed the **reference**
   activation, so a `down` bug cannot hide behind propagation) / F4 (the chain, aggregates only) + six
   injections (`half`, `tail`, `trunc`, `rowswap`, `preswap`, `f32act`), each caught where the bug is **and
   nowhere else**; `--mode down` *skips* the epilogue injections rather than reporting them as failures.
4. Floor-probe method: measure the reference's own reordering error, size the band off the worst offender's
   scale, and **report band usage** ("the band exempts exactly the cancellations and nothing else").
5. Cold-bandwidth method: replicate weights past any plausible cache, **alternate** the kernels touching them
   inside one burst, validate the **last** replica, and check that no row exceeds the probe unexplained.
6. A `--mapping`-style switch ships with (a) an ISA witness that it is not inert and (b) a bit-identity stage
   that proves it is free; and a flag that does not apply **refuses** rather than being ignored
   ("`--mode down` refuses `--mapping` rather than ignoring it … a flag that silently did nothing on half the
   shapes would make a null A/B indistinguishable from a mis-typed one").
7. Pricing discipline: price only emitted shapes; prefer a deletion A/B to a fitted slope; check whether a
   merge is *legal* (dependency chain) before pricing it.

### What must NOT transfer

- Every millisecond, GB/s, µs and % above. Their MLP is `H=1024, I=3584` (I/H = 3.5); ours is `H=2048, I=512`
  (I/H = **0.25**). Row counts, K lengths, blocks-per-launch and LDS tiles all move, so 2.650 ms, 2.275 ms,
  0.0331 ms, the rows/block ranking and the split-K verdict are priors only.
- Their bands (`ndiff ≤ 2 %`, `cos ≥ 0.9999`, `meanrel ≤ 5e-3`, cancellation band `3e-5`, `act` ≤ 2 ULP) —
  bought against **their** pins at K=3584/1024. Ours come from our floor probe at K=2048 and K=512.
- Their `rows/block = 8` default and the 6× replication factor (chosen against their cache span and 24-layer
  byte budget).
- hipBLASLt's 1.4–1.6×: our expert body will be a custom gather, and the widths worth sweeping are
  `[E,1024,2048]` / `[E,2048,512]`.

### What must be derived fresh here

- **Row-count parallelism inverts between the two families.** `derived here`: our gate/up row count is
  2I = **512** rows per expert (theirs 3,584), so at rows/block 8 that is **64 blocks** on 40 CUs —
  under-parallelised — while our `down` is **2,048** rows → 256 blocks (their `down` had 1,024/128). Their
  "under-parallelised at `down_proj`" finding lands on the *other* matrix here. Grouping helps the small one:
  8 experts × 512 gate rows = 4,096 rows → 512 blocks at rpb 8, on the plateau of §6 item 1. Confirm by shape
  row, not by timing.
- **The token-batched LDS tile.** `derived here`: their tile is `TPB×IN` floats; ours is `TPB×2048×4 B`
  = `TPB × 8 KiB`, so `TPB=4` is 32 KiB (fits the 48 KiB static limit) but **`TPB=8` is 64 KiB and does not**.
  Their `TPB=4 / TPB=2` pair does not transfer; pick ours from our LDS budget (64 KiB/CU,
  `devices/measured-facts.md` §Device geometry).
- **`shared_expert_gate` has no analog in their dense MLP.** `model.language_model.layers.N.mlp.shared_expert_gate.weight`
  is present for every layer in `checkpoints/Qwen3.6-35B-A3B.safetensors.index.json` (and in the draft block),
  so the shared expert is gated per token on top of gate/up/down. Its exact form must be read from
  `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`, not guessed from `mlp_gemv`.
- **Where the shared expert sits relative to the routed gather** (own launch, or folded into the grouped one)
  follows from the 64-block figure above and is a design decision this repo has not made.

## 2. Piece 2 — GDN linear attention (16 k-heads / 32 v-heads, d=128, k_conv=4)

**Sibling prior art:** `prototype/gdn_decode/{results.md,gdn_decode_kernels.h,check.py}`,
`prototype/gdn_chunk/`, `prototype/gdn_collapse/results.md`, `prototype/gdn_insitu/D4-DIAGNOSIS.md`,
`prototype/gdn_rec_isa/{results.md,traffic_census.py,isa_census.py}`, `prototype/gdn_walk/`,
`prototype/gdn_gated_norm/results.md`.

### What it established (quoted)

- **The kernel shape, verbatim.** "One block of 128 threads per value head; thread `j` owns column `j` of `S`
  end to end, so every access is consecutive across the warp and the `r` reduction is thread-local. Pass A
  reads `S` to form `r`; pass B re-reads `S`, writes `γS + k̃⊗Δv`, and accumulates `o` from the stored value.
  `k̃`, `q̃` staged in shared (1 KiB each); two block reductions for the L2 norms. q/k/v arrive bf16-rounded
  (the reference casts before the recurrence) and everything else is F32." §Shape of the kernel of
  `../qwen3.5-0.8b/prototype/gdn_decode/results.md`. Traffic contract, confirmed in machine code: "**2 read + 1
  write**" state passes per launch — "§9's volume check reads 256 state loads and 128 state stores per
  token-step, i.e. exactly 2 MiB read and 1 MiB written" — and the state "is re-read every token — and it
  cannot be anything else in this geometry". §8 of `../qwen3.5-0.8b/prototype/gdn_rec_isa/results.md`.
- **This op is latency-bound, and an instruction bill was wrong by 9.8×.** The LDS-resident "collapse" route
  was bit-identical and **dead**: priced bottom-up at **11.4 ms**, device returned **112.04 ms** against a
  **63 ms** win line and **84 ms** kill line. "the grid is 64 blocks over 40 CUs, so about **1.6 blocks per CU,
  one of whose four wavefronts does the recurrence**. An instruction-count model assumes an instruction is
  issued per cycle per SIMD; at 1.6 *working* wavefronts per CU there is nothing else in the machine to
  schedule while a lane waits on a dependent load, so the cost is latency per token, not issue slots per
  token." §5 of `../qwen3.5-0.8b/prototype/gdn_collapse/results.md`. Companion: the prologue is "323 of 1,523
  slots (21 %) and it repeats 18 times per token", and "the recurrence inside a head is serial
  (`s = s*decay + k*v`, then `o` needs `s`), so there is no second token to pipeline against" (§1 of
  `../qwen3.5-0.8b/prototype/gdn_rec_isa/results.md`).
- **The algebraically prettier `o` is numerically worse.** "`o = γ(q̃ᵀS) + (q̃·k̃)Δv` (one pass over `S`, no
  re-read) measures **1.5e-3** against the reference because the two terms cancel, while accumulating from the
  stored `S` in the reference's order measures **4.5e-8** — for free, since pass B re-reads `S` anyway."
  Finding 3 of the gdn_decode file. Delta-rule algebra, not geometry — this one transfers outright.
- **Dump dtype masqueraded as arithmetic.** "HF's recurrent/chunk delta-rule functions cast back to
  `initial_dtype` before returning, so `o_pre_norm` on disk is bf16 stored as fp32. Comparing an F32 kernel
  against it shows meanrel **1.45e-3** — which is almost exactly bf16's own quantization step … Round to bf16
  and the kernel is **bit-exact except for 1-elem-in-2048 ties worth exactly 1 ULP**."
  Same file, finding 2. Also finding 1: their spec's conv formula had a stray term that "double-counts the
  current token", found by the per-op gate and fixed in the spec with the measurement cited.
- **The conv carry rule and its gate consequence.** A tile starting at token `t > 0` has 3 columns of history
  that "are the carried **pre-conv** `raw` values, not zeros", and: "**A window bug is invisible in `u`, so the
  gate compares window contents.** … Any gate that checks only `u` passes both"; plus "**A tile-indexing bug is
  invisible at `tokens == 1`.** `GC_TAP_STRIDE` reads the same buffer as if it were HF's channel-major
  `[CH][T]`; at one token both indexings are the same address, so the bug is undetectable on a decode step and
  moves every row of a wave." §P6 of that file. The positive side of the same gate: "window written by that
  wave vs `raw[T−3..T−1]` | **bitexact** (contents, not a fingerprint)" and the injected `zero_window` "moves
  **rows {0,1,2} of wave B and no other row**".
- **State residency is a tier effect that saturates.** "state coldness costs **+0.070 ms and then stops** — it
  saturates at 32 MiB, which is the L2 tier the 18.9 MiB state lives in. Not a DRAM cliff", with the row
  "`cold_0MiB -> 32 -> 512 -> 1536` | 0.1216 / 0.1917 / 0.1894 / 0.1922", and in the same table
  "`split8` vs `comb_all` | 0.1839 vs **0.0644** | merging 18 launches into 1 at identical bytes buys
  **0.12 ms**". §7 of `../qwen3.5-0.8b/prototype/gdn_insitu/D4-DIAGNOSIS.md`. From the other side, prefetching
  an 18 MiB state makes the use-site read "+0.027 to +0.029 ms" faster and "survives 7 MiB of intervention, is
  gone by 21" (§5 and §7 of `../qwen3.5-0.8b/engine/generation-limits.md`).
- **A census needs a same-code control, both directions.** "a copy of the shipped kernel (`rec0_kernel`, its own
  `red[32]`, same 1,152 B LDS, same launch geometry) censuses **IDENTICAL** to `rec_kernel` (1,523 vs 1,523
  slots, category for category), and the copy compiled in a *different* TU … Had that row failed, none of §2–§4
  would be usable." §1 of the gdn_rec_isa file. Its launch geometry, corrected in place: "`rec_kernel<<<NH=16,
  D=128>>>` … 16 blocks × 4 wavefronts = **64 wavefronts**", wave32, "so 64 wavefronts is ≤1 per SIMD on this
  part either way"; the same paragraph records that a line cite there had drifted past the end of the file it
  cited and that `tools/doc_ref_check.py` caught it.
- **The state census, with a witness per row.** state dtype **F32**; "one layer's state 16 heads × 128 × 128 ×
  4 B = **1.000 MiB**"; "per request, 18 GDN layers | **18.000 MiB**"; conv window **F32** 1.266 MiB/request
  (and the note that the brief's 0.63 MiB was a **retired** BF16 cell); "token loop | a **host** loop around
  the launch". §8 of that file — and the census tool "is itself the gate that re-derives `state.h` against
  §00.5, so a spec edit that renames a shape fails the census instead of quietly disagreeing with it".

### What transfers

1. The recurrence kernel shape and its traffic contract (block per value head, column ownership, two-pass over
   `S`, 2 read + 1 write, bf16-rounded q/k/v into an F32 recurrence). The recurrence itself is identical:
   same d=128, same conv width 4, same `mamba_ssm_dtype: float32`.
2. The two-pass / no-closed-form rule for `o`, with the 1.5e-3-vs-4.5e-8 reason quoted so nobody "simplifies"
   it away.
3. The conv-window rule (pre-conv carry of `k_conv−1` columns) **and its gate shape**: compare the window as
   bytes, test a split-wave interior boundary, require an injection that moves only the boundary rows.
4. Latency-at-occupancy as the default bound hypothesis for this op, plus the pre-registration discipline that
   killed route A (win/kill lines written before the build, honoured after).
5. The census-control pattern (identical copy in the same TU **and** in another TU) for every ISA bill here.
6. Dump-dtype hygiene, and the shape of their `beta`/`g` two-level gate: "against torch on the same input bytes
   (1e−6 band, the sharp one) and against the pin (bounded, 5e−3)" because a BF16-valued pin "charges the
   dump's own rounding to the kernel".

### What must NOT transfer

- **The collapse/hoist/unroll kernel families and their wave geometry** — `AGENTS.md` §"Traps that do NOT
  transfer". Their LDS-resident geometry was priced against a 16-head launch and their own worktable.
- **The 18.0 MiB residency conclusion.** `derived here`: our state is **60.0 MiB** (30 × 32 × 128 × 128 × 4 B
  = 62,914,560 B) = **1.875× the 32 MiB fast tier** in `devices/measured-facts.md` §Memory bandwidth. Their
  "+0.070 ms and it saturates at 32 MiB because the state lives in the tier" **cannot** be assumed here: our
  state does not fit the tier, so the benefit either vanishes or applies per-layer to a slice. Fresh
  measurement, not a port.
- **The 876 GB/s cache-resident whole-family row** (`gdn_decode` 0.0646 ms for 18.0 MiB = "876 GB/s
  cache-resident = 370 % of the probe"), and their own file's companion row at "108 MiB (6× replicas) | 1.344
  ms | 255 GB/s", which that file itself retires: "the parser … has no tiling knob — so 255 GB/s is quoted from
  a build that no longer exists". A number with no knob that can re-produce it is retired, not reused.
- **GDN `o`/state bands** (their ≤1 ULP on every element of all 18 layers, their 1e-6/1e-8 mirrors). Our floor
  probe, our pins.
- **Their `MAXSC` wave rules and `--split` geometry**; `generation-limits` §7 shows the flagship `--split 18`
  prize was "both unbuilt and 6× smaller than claimed".

### What must be derived fresh here

- **The k:v head split (16 vs 32) is a real geometry change.** Their 16 V / 16 QK heads make head index ==
  k-head index; here two value heads share one key head. Every `head → (k, q)` index expression must be
  re-derived (the usual GQA grouping is `v_head / 2`, to be confirmed against
  `modeling_qwen3_5_moe.py`), the projection strip widths change (`derived here`: key strip 16×128 = 2048,
  value strip 32×128 = 4096), and the grid doubles: `derived here` **32 blocks × 4 wavefronts = 128 wavefronts
  over 80 SIMDs = 1.6 wavefronts/SIMD**, versus their 64/80 = 0.8. Their "64 wavefronts is ≤1 per SIMD either
  way" is a consequence of *their* head count; ours is a different point on the latency-hiding curve, so route
  A's "nothing else in the machine to schedule" premise must be re-checked before it is reused as an argument.
- **Whether the LDS-resident route is dead here for the same reason.** `derived here`: one head's state tile is
  128×128×4 B = **65,536 B = exactly the 64 KiB LDS/CU**, leaving nothing for the `k̃`/`q̃` staging (1 KiB each).
  The arithmetic is identical here (same d, same F32 state), so the qualitative verdict transfers; the 112 ms
  does not.
- **Our GDN byte/name census.** `tools/bf16_census.py` gives GDN = 1.885 GiB text-side, and
  `tools/PROVENANCE.md` §`gguf_suffix_census.py` records the trap: this artifact spells GDN `attn_qkv` /
  `attn_gate` / `ssm_out`, **not** `in_proj` / `linear_attn`, so any copied name-based classifier must be
  rewritten, not reused. `tools/residency_budget.py` (not `state.h`) is the existing instrument for the 60 MiB
  state-pool arithmetic.

## 3. Piece 3 — the MTP draft (1 layer, in scope per `DECISIONS.md` D-05)

**Sibling prior art:** `prototype/mtp_gatezero/MTP-GATEZERO.md`, `prototype/mtp_draft/{DRAFT-PRICE.md,DRAFT-RESULT.md}`,
`prototype/mtp_verify/{VERIFY-HANDOVER.md,CAUSE.md,CROSSROW.md,FIX-ZERO.md}`,
`prototype/mtp_accept/{ACCEPT-RESULT.md,M3-RESULT.md,DEFAULT-ON.md}`, `prototype/mtp_delta/D3-RESULT.md`,
`prototype/mtp_headbatch/` (see also §4), `prototype/mtp_llama/`, `engine/mtp_*.cpp`.

### What it established (quoted)

- **The economics, as a formula with named terms.** `speedup = (1 + a_eff) / ((1 + delta) * (1 + c))`, winning
  iff `a_eff > (1 + delta)(1 + c) − 1`, where `delta` is "what **verifying two rows per step** costs over
  verifying one" and "Unmeasured, so it is not assumed away; §4 prints the largest `delta` each route
  survives." §1 of `../qwen3.5-0.8b/prototype/mtp_gatezero/MTP-GATEZERO.md`.
- **Draft traffic is head traffic — witnessed, not arithmetic.** Their draft step reads "**16 distinct
  tensors** | **549,465,088 B = 524.011 MiB**", of which the block is "40,905,728 B = 39.0107 MiB" (15
  tensors): "so **92.6 % of a draft step's weight traffic is the tied head re-read**, 7.4 % is the block
  everyone has been calling '39 MiB, trivial'." Measured by "a recorder inside a live `MTPDraft.draft`" —
  the drafter run through a weights dict that records every key it indexes — with "the must-not-touch witness
  passed: every tensor read is `mtp.*` or the tied table — nothing from the 24 target layers". §1.1 of that file.
- **Draft cost is linear in depth, and the launch floor is overlapped.** block-k1 **0.4371 ms**, k2 **0.7666**
  (ratio **1.754**), k4 **1.5039** (ratio 3.441); "Depth is linear, so a deeper draft is a serial chain and not
  a re-read of the same weights"; "the `empty` arm reads **1.82 µs per launch** … so 14 launches ≈ 0.026 ms,
  already inside the block number"; the block ran **1.83× its byte floor**, "which is the same place the
  engine's own GEMV families live (proj 75 %, mlp 70 % of floor), not a defect in the block"; and the
  pre-registered instruction it produced: "**fund a staging pass, do not fund deeper k.**" §DRAFT result of
  `../qwen3.5-0.8b/prototype/mtp_draft/DRAFT-RESULT.md`.
- **`delta` lives or dies on the verify body, not the accept rate.** plan A `delta` = **1.6454 ms = 0.1505 of a
  step → 1.231×**; pessimistic = **5.1712 ms = 0.4729 → KILLED**; and the third arm, **two sequential 1-row
  bodies** at **10.9357 ms = 1.000 of a step**, was "**KILLED**, structurally". Rig marginals: `proj_gemv`
  second row **+7.6 %**, `mlp --mode
  gateup` **+8 %**, `mlp --mode down` **+76 %** ("the skinny `down_proj` is the worst marginal found"), and
  `attn_decode` **second row measured *faster*** — which the harness refused to believe: "the rig is at 13 % of
  its own byte floor, so bytes cannot price it". §D3 result of `../qwen3.5-0.8b/prototype/mtp_delta/D3-RESULT.md`.
  Also excluded deliberately: "state checkpoint/rollback, ~0.12 ms per round, which is a *draft-round* cost,
  not a verify-row cost".
- **A restricted head is safe in the direction that matters.** "Restriction can only lose accepts, never buy a
  wrong one, because the accept test is the target's own argmax." §1.2 of the gate-zero file. Its reason for
  existing: argmax fusion was "Rejected without measuring. The traffic is reading the table to *have* logits;
  fusing the argmax removes host work, not 509 MB."
- **What "lossless" is owed before a rate is printed.** alignment proved per fixture
  (`argmax(logits_last) == greedy[0]`, `argmax(decode{s}.logits_last) == greedy[s+1]`,
  `decode{s}.input_ids == greedy[s]`) and "a fixture whose pairing fails is dropped, not guessed at"; and the
  fixture trap that motivated it — "a candidate-set question needs a valid `(x, logits)` pair and bare dump
  names are last-writer-wins (§10.5, cos 0.245), so both studies below *prove their pairing* before printing a
  rate". Same §1.2/§2. Our `pitfalls/30` §"Free-run parity depends on which index you align" is the same rule.
- **The engine already had the batched-body machinery.** "`decode_step.h:411-424` already runs `n`-row
  layer-major bodies for one sequence (the P8 finding), so the machinery is there; what does not exist is a
  path that submits two candidate rows over a populated KV." D3-RESULT.

### What transfers

1. The `speedup` template with its terms named, the rule that `delta` must be measured by a rig **before** a
   route is funded, and the rule that a route whose *shape* is wrong cannot be rescued by a good accept rate.
2. Witness-then-price: census the tensors *touched* by running the reference drafter with a recording weights
   dict, before pricing any kernel. Copy the instrument shape, not the byte totals.
3. Pre-registered WIN/NARROW/KILL bands fixed before any millisecond, plus a pre-registered **instruction** to
   the next increment so the measurement does work rather than only adjudicating.
4. The restricted-head safety argument (verbatim — it is a proof about the accept test) and the rejection of
   argmax fusion on the same grounds.
5. Fixture-pairing proofs and the drop-don't-guess rule.
6. The depth-linearity question and the arm that answers it (identical bytes, varying depth, plus an `empty`
   launch-floor arm whose unit is **µs/launch**).
7. The llama.cpp cross-check route (`prototype/mtp_llama/LLAMA-HOW.md` records the harness traps —
   `--log-verbose`, `-p` not `-f`, locale decimal commas). Our own pinned `engines/llama.cpp` @ `d327bf1f`
   already has native MTP for this architecture (`tools/build_llama.sh`).

### What must NOT transfer

- **Accept rates.** `AGENTS.md` names them. Concretely refused: `a = 611/932 = 0.6556`, `a = 0.6641` on engine
  bits, the carried `0.7329`, the by-position 0.588 → 0.808 profile, and the kill lines built from them. Our
  draft block is a **different network** (below), so no rate carries.
- **`survival(k)` and the "decision room" figures** (median rank 19, p90 1,047, p99 17,584; survival 47.98 %
  → 95.45 % over k=16→4096; top1−top2 median 6.500 and "1.26 % of rows have a relative gap under one ULP").
  Accept-rate-shaped empirical facts about their 396 pairs. The **method** transfers (strict vs tie-inclusive
  ranks, publish the population behind the count, compare the gap against one bf16 ULP of the top logit —
  already our `pitfalls/30` adjudication rule); the numbers do not.
- **Every millisecond and ratio** (`delta` 1.6454/5.1712 ms, block 0.4371/0.7666/1.5039 ms, +7.6 %/+8 %/+76 %,
  39.0107 MiB, 524.011 MiB, 1.231×) — their step (10.9357 ms), their tied table, their draft block.
- **Their "39 MiB block is 7.4 %" framing.** `derived here`: our draft block is **not** a small dense
  appendage. `checkpoints/Qwen3.6-35B-A3B.safetensors.index.json` carries **19** `mtp.*` tensors including
  `mtp.layers.0.mlp.experts.gate_up_proj` + `.down_proj` (a **full 256-expert stack, 1.500 GiB bf16** per
  `README.md` §Expert storage), `mtp.layers.0.mlp.gate.weight`, `mtp.layers.0.mlp.shared_expert*` +
  `shared_expert_gate`, and a **full** `self_attn` (q/k/v/o + q_norm/k_norm), plus `mtp.fc.weight`,
  `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`, `mtp.norm`. Ours is a full attention + MoE layer.

### What must be derived fresh here

- **Our draft-step byte ledger, from our config.** `derived here`, bf16 per draft token: 8 routed experts ×
  6.0 MiB = 48.0 + shared expert ≈ 6.0 + router/norms ≈ 54 MiB; draft `self_attn` ≈ 18 MiB; `mtp.fc` ≈ 16 MiB;
  `lm_head` **970.0 MiB** → head share ≈ **95 %**. Under policy A (experts Q4_K at 4.5 bpw → 8 experts
  ≈ 13.5 MiB; attention/fc/shared at Q8_0 ≈ 30 MiB; `lm_head` Q8_0 515.3 MiB) → head share ≈ **92 %**. So the
  *structural* claim (draft traffic is head traffic) reproduces independently at our widths — which is the kind
  of claim worth carrying — while their byte totals do not.
- **The drafter's input path.** `mtp_use_dedicated_embeddings: false` and `mtp.fc.weight` (concat of previous-token
  embedding + hidden → 2048) mean the input construction differs from their `speccheck/mtp.py`; read it from
  `modeling_qwen3_5_moe.py`.
- **Depth economics at our step size**, and **whether the second row is cheap here**: the "+76 % on skinny
  `down_proj`" warning points straight at our `down_proj` `[2048, 512]` (K=512 — skinnier reduction than their
  3,584), so our batched verify body's MLP marginal is likely *worse* than their +76 %. Measure with our own
  `--tokens 1 vs 2` arms before pricing any `delta`.

## 4. Piece 4 — `lm_head` batching against an untied 248,320 × 2048 table (prediction 5)

**Sibling prior art:** `prototype/lm_head_gemv/{results.md,floor_probe.py,run.py}`,
`prototype/mtp_headbatch/{HEADBATCH-HANDOVER.md,HEADBATCH-RESULT.md,MTP-IMPL.md,hb_kernels.h}`,
`engine/kernels_lmhead.cpp`, `engine/logits_floor.py`.

### What it established (quoted)

- **A batched head is genuinely batched, and that was *witnessed*, not assumed.** M=1 **2.1265 ms**, M=2
  **2.1305** (R = **1.0019**), M=4 **2.1538** (R = **1.0128**); "`fetch_per_weight = 1.000000` on every M arm,
  with the calibration arms reading **2.000000** (`extra_pass`) and **1.25** (`extra_chunk`) and `short_grid`
  opening 16/32 poison holes — so 'the table is read once for every row served' is a witnessed byte fact, not
  an arithmetic assumption." §HEADBATCH result of
  `../qwen3.5-0.8b/prototype/mtp_headbatch/HEADBATCH-RESULT.md`. The witness is a **real block reduction**
  ("64-bit warp butterfly, one publish per warp to `wsum/wraw/wcnt`, `__syncthreads`, then warp 0 folds the
  ≤ 32 warp partials"), the host expectation "deliberately does not mirror the kernel's lane striding: mirroring
  it would make the check pass by construction", slots are per block **per pass** with **no atomics** (integer
  addition is associative), the guard bands are pre-filled with a NaN payload (`0x7FC00001`) "a value writes
  never store", and the witness and timed instantiations must be bit-identical ("Otherwise the instrument would
  be the subject"). §2 of `HEADBATCH-HANDOVER.md`.
- **The plateau arithmetic that predicted it.** `table once 485.0000 MiB / 237 GB/s = 2.1458 ms (measured 1-row
  pass 2.1685 = 1.011 of it)`; `+ 2nd logits row 0.9475 MiB = +0.0042 ms (+0.19 %)`; `dots M=1 → M=2 0.509 →
  1.017 GFLOP/pass = 235 → 469 GFLOP/s against the quoted scalar v_fma_f32 roof 7.30 TFLOP/s … 3.2 % → 6.4 % of
  that roof`; `LDS per block M × 4 KiB; at rpb 8 both M=1 and M=2 stay thread-bound at 8 blocks/CU (64 KiB/8 KiB
  = 8 = 2048 threads/256), so no occupancy term at M=2`. §0 of `HEADBATCH-HANDOVER.md`. Pre-registered with
  dissolve-before-renegotiate lines (at **R ≥ 1.862** a batched head stops beating calling the head twice;
  `R_KILL = 2.2279`), and the reason a band exists: "it sizes the prize, it does not gate the route".
- **The kernel shape.** "One warp per vocab row; `h` staged once into 4 KiB of shared memory; 32 lanes × 8 bf16
  = 256 elements per step so `H = 1024` is four steps of 16 B loads, each warp instruction touching 512
  contiguous bytes; fp32 accumulation with a 5-step shuffle reduction; one `atomicMax` per row for the fused
  mode (float→order-preserving-uint, 20 bits of index, since 248320 < 2²⁰)." §Shape of the kernel of
  `../qwen3.5-0.8b/prototype/lm_head_gemv/results.md`. `rows_per_block` sweep, all bit-identical (`ndiff 0`):
  **1 → 192.7, 2 → 241.7, 4 → 242.1, 8 → 239.6, 16 → 239.0 GB/s** — "Flat from 2 up".
- **Fusing the argmax is not a bandwidth win.** "What the fused argmax bought: **1 %** (2.099 vs 2.126 ms) …
  writing 1 MB of logits is nothing next to reading 485 MB. The real reasons to keep the fused form are that
  greedy decoding never needs the logits (A-17), it avoids a 1 MB host copy per token, and it skips the
  host-side argmax; not bandwidth. That correction belongs in §07.4". Same file.
- **A per-element logits rule, derived over all 248,320 rows.** Eight legal addition orders — one of them the
  kernel's own warp pattern emulated from source ("lane *l* takes uint4 chunks *l*, *l*+32, *l*+64, *l*+96,
  eight sequential `acc +=` inside each, then the `__shfl_down_sync` butterfly at 16, 8, 4, 2, 1") — scored with
  the shared comparator: "every legal order lands at **n_bad 0**, with ndiff **50–73 elements (0.020–0.029 %** of
  the vocabulary) and max ULP up to 12 — all of that ULP inside the cancellation band, the widest band any legal
  order would pass at being **1.01e-05**." Four assertions, "because they fail differently": ULP outside the
  band; an ndiff percentage ("a vector sitting 1 ULP off *everywhere* has `n_bad 0` and would sail through the
  first check"); zeros and NaNs **named separately**; and the top-5 set. `--selftest` proves it bites: a
  row-swap and a dropped last term both walked the old aggregate-only gate ("`argmax =`, cos 1.000000, meanrel
  0.00141, top5 5/5" and "cos 0.999420 … **inside** the 5 % band"). §gate-audit G1 of that file.
- **The impossible-number trap.** "The first `--rows` sweep printed 415–710 GB/s, above the 256 GB/s physical
  peak of LPDDR5x-8000. Cause: the kernel used the compile-time `ROWS_PER_BLOCK` while the host launched with
  the runtime value, so `--rows 2` covered only a quarter of the vocab rows — less work, so 'faster'."
  Same file.

### What transfers

1. Kernel shape with the width swap: warp per vocab row, one staged `h`, 16 B per lane per step, wave-shuffle
   reduction, fused-argmax `atomicMax` idiom (order-preserving float→uint, 20-bit index — still valid, our vocab
   is the same **248,320**).
2. The whole **traffic-witness** apparatus: `fetch_per_weight`, calibration arms at exactly 2.0 and 1.25,
   poison-valued guards, per-pass slots without atomics, a host expectation that deliberately does not mirror
   the kernel's stride, and witness-kernel ≡ timed-kernel bit identity. This is the most portable instrument in
   the sibling tree, and prediction 5 is exactly where it is needed.
3. The pre-registered-band shape: a stand / narrower-claim / not-free triple with a named dissolution line above
   which the requirement *dissolves* rather than being renegotiated.
4. The four-way logits gate (band, ndiff %, zero/NaN named apart, top-5) plus the two-injection selftest that
   proves an aggregate-only gate was blind.
5. "Fused argmax is host work, not bandwidth" — ours writes the same logits row (`derived here`: 248,320 × 4 B
   = **993,280 B**) against a 515.3 MiB (policy A) read, so expect the same ~1 % and do not re-litigate it.

### What must NOT transfer

- The numbers: 2.1265 / 2.1305 / 2.1538 / 2.1685 ms, the R bands (1.10 / 1.35 / 1.862 / 2.2279),
  `rows_per_block` ranking, 242.1 GB/s, the 1 % fusion figure, and `ndiff 50–73 / widest-passing band
  1.01e-05` — all measured at **H=1024 with a tied table**. Our floor probe re-derives them at H=2048 and at
  Q8_0 dequant.
- **The tied-table premise.** Ours is **untied** (`tie_word_embeddings: false`), so the row-gather and the full
  GEMV read a different allocation from the embedding table — which is `AGENTS.md` prediction 5, and their
  `hb_kernels.h` (which `#include`s the shipped tied header) never exercised it. Batch/gather identity must be
  re-proved here, not inherited.
- Any band, and any `engine/logits_floor.py` constant.

### What must be derived fresh here

- **LDS becomes the binding limit at our width.** `derived here`: a staged F32 row is `2048 × 4 B = 8 KiB`, so at
  rpb 8 / 256 threads one block needs `M × 8 KiB`; `64 KiB / 8 KiB = 8` blocks/CU at **M=1** (exactly the thread
  bound, no penalty) but `64 KiB / 16 KiB = **4** blocks/CU at M=2 — **half** the thread bound, where they had
  "no occupancy term at M=2". Either stage the row as **bf16** (4 KiB) or keep batched rows in registers; settle
  this before measuring any R, because an LDS-clipped M=2 arm would read as "the batch is not free".
- **The untied read**: batched row-gather over `lm_head.weight` while embeddings stay untouched; a gather-vs-GEMV
  A/B at identical vocabulary; and the policy-A byte figure (515.3 MiB `derived here`, `DECISIONS.md` D-10), not
  the bf16 970.0.
- **Our floor probe at K=2048**: eight-order emulation, `n_bad`/`ndiff`/band-usage columns, two injections, at
  our width. Their `1.01e-05` measures *their* cancellation profile.
- **What `M` even is here** — an output of the MTP design (§3), not an input.

## 5. Piece 5 — engine assembly (decode step, host side, gate shape)

**Sibling prior art:** `engine/{decode_step.cpp,decode_run.cpp,gate.py,chat.py,build.sh,kernels_attn.cpp,stage_clock.cpp}`, `engine/generation-limits.md`.

### What it established (quoted)

- **The host-side invariant, stated as an invariant.** Header of `../qwen3.5-0.8b/engine/decode_step.cpp`:
  "no arithmetic of its own — not one add, norm, gate or dot happens here. Every value a step produces comes
  from a kernel that has its own gate against the pinned reference dumps. **no host round-trip inside the
  token** … the step is submitted as one stream of launches on the default stream with a single sync at the
  logits … `submit_token_body` contains no readback at all, which is what lets `engine/prefill.cpp` queue a
  whole wave of them before waiting … no dtype conversion written by hand. Every widening goes through
  `kernels_cast.cpp` so the rounding is one named thing (RNE, same as torch's) instead of eighteen local casts."
  And the logits site: "One `lm_head_full`, **ONE blocking D2H (the sync of the token)**, and §03.4's argmax."
- **What that one sync costs, and what it is not.** `host_tail` = **0.7163 ms** = "one blocking D2H of the whole
  993 KB logits row plus §03.4's argmax (`decode_step.cpp:1177`), sampling and bookkeeping. **On the critical
  path, not hidden**: greedy needs the token id before the next step can be submitted, so this is serial per
  token." §1 of `../qwen3.5-0.8b/engine/generation-limits.md`. `[SIB-GEO]` as a total, but `derived here` the
  vocab-shaped part: our logits row is the same 993,280 B, so their split (≈14–16 µs of D2H at the 63–70 GB/s
  D2H figure in `devices/measured-facts.md`, the rest CPU argmax over 248,320 F32) applies here nearly unchanged.
- **The launch-plan drift guard.** "`planned_launch_counts()` below derives the n = 1 row of that table, and
  `gate.py` **re-derives every row from §07 in Python and fails on a mismatch**, so this comment and the code
  cannot drift apart quietly" — with the billing rule "Each collapsed family bills the SAME 49/114/48/18/18/18
  launches for a body of 295 tokens as for a body of one — **one launch per op per body, whatever its width**",
  and one *named* exception (`proj` under route D1, `pack_proj = (n == 1) && packed`, so a one-row body pays 48
  where a wide body pays 114). Same header.
- **The step's real shape at its own operating point.** 4,032 lanes, **10.9357 ms/token = 91.4 tok/s** (9.7855 ms
  = 102.2 at 128 lanes); per-step census **1532.3 MB read / 24.4 MB written at 340 launches and 305 stage-clock
  segments** before the cast fold, **243 / 184** after; "Divided by the plateau that is **6.47 ms of unavoidable
  weight traffic** inside a 10.936 ms step." §0 of that file.
- **The instrument-artifact result worth more than any win.** Sync leg, one variable: "**1 sync 9.4798 ms,
  305 syncs 11.0932, 340 syncs 11.2855**, i.e. **5.29 us per sync**, linear to 4 %" — "**And the shipping path
  pays none of it** … `syncs_per_step=305` is a property of the **instrument** … any prize computed as 'segments
  × us_per_segment' is pricing the measurement, not the engine." §4. The closure, with its own correction: "**S1's
  sync leg is closed.** … **S1's launch leg is unpriced, not refuted** — and the fitter's headline
  `3.9473 us/launch -> S1 IS REAL` must not be read as funding it, for two named reasons. The slope ladder is
  measured at body=tiny (5.3 MiB, 340 launches, nothing to overlap behind) … and the merge gain (13.08 us per
  launch removed) … moves workgroups-per-launch along with launch count … a confound, not a result." §7 of
  `../qwen3.5-0.8b/progress/2026-10-21/route-S1-segments-2026-10-21.md`.
- **How a launch *was* priced honestly: deletion.** Folding 97 `cast` launches out of the step moved the wall
  **−0.3008 ms = 3.10 µs per launch**; from the corrections log: "a gate-zero slope that printed 'S1 IS REAL
  (3.95 us/launch)' from a regime with nothing to overlap behind it (the honest number came from deleting
  launches instead: 3.10)". §6/§7 of `generation-limits.md`.
- **A compiler artifact wearing a hardware story.** "`attn_core` declared **6,176 B** of `__shared__` and
  **reserved 38,944 B**, because LLVM promote-alloca-to-LDS moves `qreg[8]` and the K-row `uint4` into LDS when
  the loop bound is the runtime `nd`. At 64 KiB LDS/CU that is 1 resident block per CU — 8 of 32 wavefronts … The
  fix was a spelling: `for (t=0;t<8;++t) if (t<nd)`, byte-identical across 126 stage comparisons and 21 cases,
  residency 1 → 4 blocks/CU, **−46 % on the line, −9.9 % on the step**." And the bound classification: "traffic
  was 27.7 % of the plateau (not a wall), issue was 5.9× under its own roof (not a wall), so it was **latency at
  occupancy**"; plus the control that says when the lever cannot apply: "At 128 lanes the launch is 8 blocks on 40
  CUs: one round at either residency, so neither residency nor population levers can bind". §3.
- **Decode attention is not a matmul.** "the decode step is one row, and the wave forms only route prefill
  (`kernels_attn.cpp:627`: `if (rows == 1) attn_core<<<...>>>`). Decode attention is a GEMV-shaped read, not a
  matmul; matrix units do not help a one-row walk." §5.
- **Build discipline.** `engine/build.sh`: `FLAGS=(-O3 -Wno-unused-command-line-argument -I. -Iengine -Wl,-rpath,/opt/rocm/lib)`,
  with "-O3 always, -Wl,-rpath,/opt/rocm/lib always" and a separate `-O0 -g` debug line. Already recorded here
  (`reference-implementation.md` §HIP build pattern); never time an `-O0` build.

### What transfers

1. The four host-side invariants wholesale: no arithmetic in the step, **one** blocking D2H per token, no
   hand-written casts, body/finish split so a wave queues before any readback. This is the direct prior art for
   `AGENTS.md` prediction 1 — the router must be scored and selected **on device**, because the step's budget
   already contains the one sync it is allowed.
2. Launch-plan double derivation (engine's own table vs the gate re-deriving every row in Python and failing on
   mismatch) — the cheapest anti-drift mechanism in the sibling tree, and exactly what a MoE bill (router +
   gather + shared expert × 40 layers + draft) needs.
3. Stage-clock semantics: per-family rows valid as an **ordering** and as byte attribution, inflated by ~5.3 µs
   per segment as a **wall** figure; `--time-stages` off by default on the shipping path.
4. Bound-classification method: compare traffic to the plateau and issue to its own roof before naming a bound;
   and check that a lever's shape *can* bind at the reported width.
5. Gate structure worth imitating (recorded here because my write boundary for traps is `pitfalls/20`/`50`): one
   `BAND`/`CASCADE_BAND` table with a source string per band; a route allowance must name the **route** it was
   priced on and refuse a mismatch ("the format also records no **route**, so cross-route pricing reddens
   `hand", §7); and an allowance expressed as cos/meanrel **cannot adjudicate a per-element `n_bad`** check —
   which was the actual `>MAXSC` dispute (§0 finding 1 of
   `../qwen3.5-0.8b/prototype/maxsc_allowance/MAXSC-HANDOVER.md`). File these under `pitfalls/40` when someone
   touches gates.
6. `engine/chat.py` as prior art for our token path, **with the stop rule re-derived**: `README.md` §target pins
   our `eos_token_id: 248044` and forbids carrying their other rule over (refer to ids by number only).

### What must NOT transfer

- Their launch/segment counts (340/305, 243/184), family table, 10.9357 ms, 91.4 tok/s, 1532.3 MB/step — 24
  dense layers, no MoE, no router.
- Their `MAXSC` wave geometry, `--split` caps, attn-tile wave rules and every `BAND` key (`AGENTS.md`), and the
  `segments × 7.2 µs` subtraction or any `T_serial`-shaped prize.
- The `attn_core` **fix text** as a port: the lesson (promote-alloca-to-LDS under a runtime trip count; prove
  residency with the probe) transfers; their `qreg[8]` spelling belongs to a 6-head tile.
- Anything from their `progress/` journals beyond the routes index.

### What must be derived fresh here

- **Our host-sync census.** `AGENTS.md` prediction 1 asks for host-sync count per decode step. Their answer is
  "one"; ours must be one *plus zero for the router*, and the router design must be checked against that before
  the first gather kernel is written (host-side top-k via `nonzero`-style selection is what the prediction warns of).
- **Our launch bill**: 40 layers × (router + top-k + one grouped gather + shared expert + GDN/attention) plus the
  draft block, written as a table before the first kernel and double-derived as in (2).
- **Whether `--time-stages` is worth building here at all.** Its cost is ~5.3 µs × segments of *reported* time the
  engine does not pay; its benefit (family ordering) is obtainable from a burst arm. Their own route died on the
  difference. An open design choice, not a given.
- **Our mixed read/write denominator.** `devices/measured-facts.md` already records that the mixed-traffic probe
  was **not** copied and that the mixed-step denominator question was left unresolved; §Q9's order test
  (`B_blk` 183.3 vs `B_rr` 184.2 GB/s, ratio **1.005**) refutes inter-kernel launch order as a lever, so the
  remaining candidate is the walk inside a kernel.

## 6. Retrospective: what the unread launch/rate work says about `moe_gather` before it graduates

`prototype/q4k_dequant`, `prototype/moe_router` and `prototype/moe_gather` were written without reading
`prototype/launch_price`, `issue_rate_probe`, `maxsc_allowance`, `prefetch_tier`, `attn_tile_model.py`,
`weight_names.py`, `bf16_ulp.py`. Re-read 2026-10-26 for this. Six lessons bear on `moe_gather`; none requires a
rewrite, and four are cheap to check with gates we already have.

1. **Grouped-launch shape has a plateau and a cliff, and the cliff is per-block fetch capacity, not launch
   count.** At fixed bytes and fixed *total* workgroups, moving work between launches and blocks-per-launch
   moves time **43×**: `L=16/G=64 → 7.20 ms` … `L=1024/G=1 → 311.37 ms`; "at `G=1` a single block must pull
   1461 MiB / 1024 ≈ **1.43 MiB** with its own 256 threads and nothing else in flight, and one block cannot
   hold enough outstanding loads to approach the plateau"; "spreads of 0.13–0.45 ms at `G ≥ 32` against 30–270
   ms at `G ≤ 4`". §2 of `../qwen3.5-0.8b/prototype/wave_shape/ITEM2-RESULT.md`. **For `moe_gather`:** keep
   blocks-per-launch on the plateau (their evidence puts the edge below `G ≈ 32` on a 40-CU part) and do not
   trade blocks-per-launch away to reduce launch count. `derived here`: 8 experts × 512 gate/up rows = 4,096 rows
   → 512 blocks at rows/block 8 (plateau); a per-expert launch (64 blocks) is only 2× above the edge. One shape
   row in the gate, not a timing.
2. **A launch bill is not a body price, and an instruction-slot bill is not a time.** Our `pitfalls/20` already
   carries the launch-record case (a static sum priced a 2-row verify body at 0.1505 of a step; two device arms
   measured +0.72 to +1.98). The new sibling datum is the **9.8×** slot-model gap (11.4 ms modelled, 112.04 ms
   measured) caused by pricing issue slots where the machine pays latency at ~1.6 working wavefronts per CU.
   **For `moe_gather`:** every slot figure from its ISA census must state wavefronts-per-SIMD beside it, because
   the census is only valid where there is concurrency to hide a dependent load. Our 8-expert grouped launch has
   plenty; a per-tile or per-expert variant may not.
3. **Price a launch by deleting launches, and refuse a slope fitted at an empty body.** Their ladder
   (§2 of `../qwen3.5-0.8b/prototype/launch_price/S1-GATEZERO.md`): async submit floor **0.375 µs/launch**,
   empty-kernel dispatch **1.85 µs per launch-segment**, serialized-sync gap **5.9 µs per sync** — then two
   corrections: the 305-segment leg was the *instrument*, and the honest production price was **3.10 µs per
   launch** from deleting 97. **For `moe_gather`:** any "N launches/token saved by grouping" claim must come
   from an arm that actually removes launches at fixed bytes **and** fixed workgroups-per-launch; that is the
   only arm their history did not falsify.
4. **Prefetching has no hardware hook on this box, and the tier is small.** `hipStreamSetAttribute(accessPolicyWindow)`
   was rejected "at 1, 2, 8, 16, 32 and 64 MiB — **0 of 6 accepted**"; `accessPolicyMaxWindowSize = 0`;
   `PersistingL2CacheMaxSize = 2,097,152 B`, byte-identical to `L2CacheSize`; "so every P1 arm must be **explicit
   prefetch**, a real load issued early. No gate should be written against `hipStreamSetAttribute`,
   `hipDeviceSetCacheConfig` or a `hipDeviceSetLimit` carve-out" (§1 of
   `../qwen3.5-0.8b/prototype/prefetch_tier/P1-ARITHMETIC.md`). Payoff ceiling measured: **+0.027 to +0.029 ms**
   per staged read, gone by 21 MiB of intervening traffic, against a 0.30 ms bar. Two adjacent facts matter more
   here: `concurrentKernels == 1` is **not** a usable answer (spinner-and-probe saw real overlap 5/5, with a
   same-stream negative control reading 0), and `hipMemPrefetchAsync` on a `hipMalloc` pointer "returns `no
   error`" while nothing demonstrates it moved a byte — "Accepted is not performed." **For `moe_gather`:** a
   design that assumes a residency hint ("keep the hot experts in L2") is dead before measurement; and `derived
   here` the expert working set is 8 × 6 MiB = **48 MiB bf16** per token per layer (≈13.5 MiB at Q4_K), so the
   bf16 case does not fit the 32 MiB tier even if a hint existed.
5. **A negative control must be derived from the launch geometry, with a non-empty must-catch set on both
   sides.** `attn_tile_model.py`'s `neg_expect` derives must/must-not from geometry rather than hand-listing it,
   because "a negative that cannot fail is worse than none, and one that catches everything measures nothing",
   and records three blindness results that came out of failures: "`readpast` cannot move a row that already
   attended `cap` lanes … the bug is invisible by construction, not by weakness"; "No score-only bug can move a
   row with exactly ONE lane: softmax of a singleton is 1.0 whatever the score arrives as"; "`partalias` cannot
   move a row the staging fits: a one-pass row finishes inside its own block and writes NO partial, so there is
   no slot to steal". **For `moe_gather`** the generating geometry is *(expert list, tile, lanes per row, partial
   slots per split)*, so the minimum set is: read a non-selected expert; top-k index off by one; a duplicated
   expert id in the list; a weight row landing in the neighbour's slot; a short grid leaving poison holes; an
   extra pass making `fetch_per_weight` 2.0 — and the structurally-blind one, **an expert whose tile fits one
   block cannot expose a partial-slot alias**, which must be declared blind-by-proof rather than scored.
6. **One module owns the name mapping, and a lookup must refuse rather than normalise.** Their
   `weight_names.py` exists because the store is keyed on the **normalised** name while every prototype reader
   opens the **raw** checkpoint key, and `at()` "deliberately refuses instead of normalising what it is handed …
   a lookup that tolerated both spellings would answer for tensors the store does not hold, and would turn a
   naming mistake into a silent pointer to the wrong blob"; its selftest reads the shard *index*, not the header,
   "because the header parser lives in `loader.cpp` behind a `load()` that hipMallocs and workers do not touch the
   device". **For `moe_gather`** this bites harder here: `tools/PROVENANCE.md` §`gguf_suffix_census.py` records
   that an early name-based bucketing of this artifact's 733 records "mislabelled the family entirely while still
   getting the total right", and the expert tensors are rank-3 fused. The gather's plan checker needs **one**
   module mapping HF name ↔ GGUF name ↔ resident blob, asserting the two lists are the same length and refusing
   a name it cannot resolve; the `MAP_HF_GGUF` map in `tools/bf16_census.py` is the existing seed.

One more, from `maxsc_allowance`, which is about gates rather than kernels and belongs in `pitfalls/40` (outside
my write boundary, so recorded here): an allowance expressed as cos/meanrel **cannot adjudicate** a per-element
`n_bad` check — "a route allowance is a cos/meanrel rule and cannot adjudicate `n_bad` at all" (§0 finding 1) —
and an allowance must name the **route** it was priced on, because cross-route pricing reddens a run for the
wrong reason (§7 of `../qwen3.5-0.8b/engine/generation-limits.md`).

## 7. The non-transfer list as actually applied (refused, and why)

| refused | because |
|---|---|
| every numerics band, floor, `cancel_rel`, `ndiff` % and `cos`/`meanrel` cap in the sibling tree | bought against **their** pins at their K widths (`AGENTS.md` §"Traps that do NOT transfer"). Copied instead: the floor-probe method, `worst_off_rel` as the calibration quantity, the four-assertion logits gate shape, the F1–F4 isolation split |
| `MAXSC` wave geometry, `--split` caps, attn-tile wave rules | their geometry (8 Q / 2 KV heads, 18 GDN layers) and their wave planner; and their own §7 of `generation-limits.md` shows the flagship `--split 18` prize was "both unbuilt and 6× smaller than claimed" |
| GDN collapse / hoist / unroll kernel variants | do-not-transfer list, and the LDS-resident premise fails on identical arithmetic here (128×128×4 B = 64 KiB = the whole LDS), so the variants could not be ported even if wanted |
| **MTP accept rates** (`0.6556`, `0.6641`, carried `0.7329`, by-position `0.588 → 0.808`) | do-not-transfer list, and our draft block is a different network (full-attention + 256-expert MoE, 19 `mtp.*` tensors), so no rate can be carried across it |
| `survival(k)` (47.98 % → 95.45 %), median rank 19, `top1−top2` median 6.500, `1.26 % of rows under one ULP` | accept-rate-shaped empirical facts about their 396 pairs and their corpus; the **method** (strict vs tie-inclusive ranks, publish the population, gap-vs-1-ULP) is copied instead |
| `delta` (1.6454 / 5.1712 ms), block-k timings, second-row marginals, `39.0107 MiB`, `524.011 MiB`, `1.231×` | their step size and tied head; the head-dominance **claim** was re-derived at our widths (§3) rather than the totals imported |
| their `lm_head` R-bands (1.10 / 1.35 / 1.862 / 2.2279) and `rows_per_block` ranking | route-specific economics for their step, and the **tied** premise fails here (prediction 5) |
| their `T_geo` / `T_serial` / `T_pipe` milliseconds | their 24-layer chain and 48-launch MLP; the **decomposition method** and the "merging is forbidden by the dependency chain" argument transfer |
| MLP/GEMV millisecond headlines (2.650 / 2.275 / 2.1685 / 0.0646 ms …) and the 18.0 MiB residency row | `[SIB-GEO]` by definition; the residency row additionally fails on arithmetic here (60.0 MiB vs a 32 MiB tier) |
| `1.85 µs` as *the* launch price | it is an **empty-kernel dispatch** figure. The production price from deletion is **3.10 µs**, and the fitted 3.95 µs slope came from a regime with nothing to overlap. All three are copied **with their regimes attached**, because the number without the regime is what caused their own wrong route |
| a mixed read/write denominator | their `mixed_bw.cpp` was **not** copied (recorded in `devices/measured-facts.md`), and §Q9's order test refutes inter-kernel locality as a lever, so this repo still owes its own probe |
| their instruction-rate table | already copied on 2026-10-26 into `pitfalls/20-rates-and-occupancy.md` and `devices/measured-facts.md` §Compute rates — re-copying would create a second spelling of the same numbers |

## 8. Third-ruler verdict: `../qwen3.5-0.8b/prototype/bf16_ulp.py` vs `tools/ulp_policy.py`

**Confirmed as the origin.** Our `pitfalls/30` §"A BF16 ULP comparator written on a `uint16` view has a dead
branch" **is** that file: its provenance header names `prototype/bf16_ulp.py`, and its "correct form"
(`np.where(bits & 0x8000, bits ^ 0xFFFF, bits ^ 0x8000)`) is `monotone_key` there byte for byte.
`tools/ulp_policy.py` `Ruler.order_key` implements the same total order a third way (`(~b & mask)` / `b | sign`);
all three agree on sign placement, so the ordering itself is not in dispute and needs no reconciliation.

**It does contain rulers our equivalence audit should absorb — three, in priority order.**

1. **`band_usage` → `worst_off_rel`** ("the largest rel among offenders: **the tightest band these numbers would
   pass at**"). `tools/ulp_policy.py` has **no scale-relative notion at all** (no cancellation band, no per-element
   magnitude over tensor max), yet its acceptance rule (c) demands "**zero** elements at ≥ 2 ULP". `pitfalls/30`
   already records that "reordering the reference's own F32 sum already moves ~1 element in 1e5 by more than 1
   ULP, and always at an element that cancelled to ~1e-6 of the tensor max". **So D-17's isolation test and
   `pitfalls/30` are in tension as written**: a legal long-dot reassociation would be refused and escalated. Either
   the judged field set provably excludes long-dot outputs, or the band — with its own floor-probe provenance,
   never a hand-set tolerance — belongs in the ruler. Flagged for the supervisor, not fixed here: `tools/ulp_policy.py`
   is outside my write boundary and D-17 is the user's decision.
2. **`zero_nan` as an absolute census over a single tensor.** Our `Census` counts zeros/NaNs only *between* a pair
   (`classify(a, b)`), so "the kernel never wrote" is visible only if the pin happens to differ from zero. Their
   reason for the separate count: "a ULP distance cannot tell these two failures apart: an output of +0 against a
   pin of -2.39 is ~49 000 ULP, and so is almost any other value that collapsed to nothing". Cheap to add. Their
   pattern rules — which our ruler already matches — are worth pinning in the audit's fixture set: ±0 both count as
   zero, a **denormal is not a zero** ("a BF16 denormal is exactly what a wrong-dtype weight read produces"), and
   `selftest` requires `zero_nan == (2, 2)` over `+0, −0, denormal, NaN, −NaN, ±inf, two finite`.
3. **`bits_from_key`** — the inverse key, "so a probe can step a value exactly one representable ULP". Our policy
   requires fixtures that exercise the 1-ULP and ≥2-ULP classes, and `pitfalls/30` records two ways to get that
   wrong: `torch.nextafter` must be called on **bf16** tensors (in F32 the neighbour rounds straight back), and
   −min-subnormal → +0 is **2** ULP, not 1, because −0 and +0 are distinct patterns. An explicit
   `step_ulps(value, n)` in the shared ruler removes the need for each gate to rediscover both.

Also copied as a caution, not a ruler: their measured failure magnitude — the wrong comparator "claimed that
reordering the reference's summation moved **33 %** of the elements of a dot by more than one ULP, which is four
orders of magnitude from the truth and would have made every gate that used it unfalsifiable". Our `pitfalls/30`
states the trap without a magnitude; the magnitude is what makes it a scare story worth retelling.

**One more trap from the same family, for `pitfalls/30` (outside my write boundary, so recorded here):** a gate
that ranks logits read out of a `logits_storage = bf16` fixture must decode by the **declared dtype**, not treat
the uint16 payload as a number — "uint16 order is not float order for negatives, so reading them as numbers
produced a confident survival table over garbage" (§2 of `../qwen3.5-0.8b/prototype/mtp_gatezero/MTP-GATEZERO.md`).
It is the same root cause as the dead-branch trap (integer order ≠ float order) but in a *ranking* site rather
than a *comparison* site, which is why the existing entry does not cover it.

## 9. Fact ledger: what was copied, its tag, and where it landed

Every row is checkable against the sibling file named. "Landed" is the section added on 2026-10-26; a row that
landed in this file only was deliberately **not** promoted to a binding doc (reason in the row or in §7).
`mf` = `devices/measured-facts.md` §"Copied from the sibling engine (2026-10-26)"; `p20` =
`pitfalls/20-rates-and-occupancy.md` §"Copied from the sibling engine (2026-10-26)"; `p50` =
`pitfalls/50-harness-and-shell.md` §"Copied from the sibling engine (2026-10-26)"; `here` = this file.

### 9.1 Box facts — launch, dispatch, sync `[SIB-BOX]`

| id | fact | sibling source § | landed |
|---|---|---|---|
| F-01 | async submit floor **0.375 µs/launch** (`tools/launch_overhead_hip.cpp`, copied here) | `launch_price/S1-GATEZERO.md` §2 | mf-A, p20-T2 |
| F-02 | empty-kernel dispatch **1.85 µs per launch-segment** (empty 48-launch burst, one sync) | same §2 | mf-A, p20-T2 |
| F-03 | serialized-sync gap **5.9 µs per sync** = 1.80 ms over 305 syncs for identical work | same §2 | mf-A, p20-T2 |
| F-04 | rig leg at the engine operating point: **1 sync 9.4798 ms / 305 syncs 11.0932 / 340 syncs 11.2855 ⇒ 5.29 µs per sync, linear to 4 %** | `generation-limits.md` §4 | mf-A, p20-T2 |
| F-05 | production launch price **3.10 µs**, measured by deleting 97 launches (**−0.3008 ms**); the earlier fitted **3.95 µs/launch** was at `body=tiny` and its merge gain (13.08 µs) confounded workgroups-per-launch | `generation-limits.md` §1/§4/§6; `route-S1-segments-2026-10-21.md` §7 | mf-A, p20-T2 |
| F-06 | small-op stage-line prices: `glue` **49 × 11.16 µs**, `copy` **19 × 10.7 µs**; `empty` draft arm **1.82 µs/launch** | `generation-limits.md` §1; `mtp_draft/DRAFT-RESULT.md` | mf-A |
| F-07 | the sync half of that cost is an **instrument** property: the shipping path defaults `time_stages=false`, so `syncs_per_step=305` is the stage clock billing its own ruler; "any prize computed as 'segments × us_per_segment' is pricing the measurement, not the engine" | `generation-limits.md` §4; route-S1 §7 | p20-T2 |

### 9.2 Box facts — cache hierarchy and the residency API `[SIB-BOX]`

| id | fact | sibling source § | landed |
|---|---|---|---|
| F-08 | fast tier: KFD node-1 cache 85, **level 3, size 32768 KiB, line 64 B**; read at 32 MiB **876.7 GB/s**, copy at 32 MiB **325.7 GB/s**, cliff to **239.4** by 64 MiB | `prefetch_tier/P1-ARITHMETIC.md` §0 [Q3][Q4][Q5] | mf-B |
| F-09 | per-GPC L2 **2 MiB** (KFD cache 84, level 2, line 128 B), agreeing with the HIP attribute | same [Q6] | mf-B |
| F-10 | `PersistingL2CacheMaxSize` = **2,097,152 B**, byte-identical to `L2CacheSize` — the carve-out is the L2, not the tier | `P1-ARITHMETIC.md` §1 | mf-B, p20-T4 |
| F-11 | `accessPolicyMaxWindowSize = 0` and `hipStreamSetAttribute(accessPolicyWindow)` rejected at **1, 2, 8, 16, 32, 64 MiB — 0 of 6 accepted**; no `hipDeviceSetLimit` carve-out enumerator exists | same §1 | mf-B, p20-T4 |
| F-12 | `hipMemPrefetchAsync` on a `hipMalloc` pointer returns `no error` — "Accepted is not performed" | same §1 | mf-B |
| F-13 | `concurrentKernels == 1` is **not** a usable answer: spinner+probe saw real second-stream overlap **5/5**, same-stream negative control read **0** | same §1 | mf-B |
| F-14 | measured cross-phase prefetch effect **+0.027 to +0.029 ms** per staged read; an 18 MiB target survives **7 MiB** of intervening traffic (+0.0278) and is gone by **21** (−0.0001 to +0.0008); a 7.875 MiB KV slice survives 21 MiB (+0.0138); bar 0.30 ms ⇒ no engine change funded | `generation-limits.md` §5/§7 | mf-E |

### 9.3 Box facts — per-block fetch capacity and launch shape `[SIB-BOX]`

| id | fact | sibling source § | landed |
|---|---|---|---|
| F-15 | iso-population diagonal at `pop=1024`: **7.20 ms (L=16,G=64) → 311.37 ms (L=1024,G=1)** — **43×** at equal bytes and equal total workgroups | `wave_shape/ITEM2-RESULT.md` §2 | mf-C, p20-T3 |
| F-16 | the mechanism: at `G=1` one block must pull **1.43 MiB** with 256 threads and cannot hold enough outstanding loads | same | mf-C, p20-T3 |
| F-17 | where the plateau ends: `pop=4096` reads **6.98 / 6.91 / 7.33** ms at L=16/32/64 then **11.03 → 76.85**; per-cell spreads **0.13–0.45 ms at G ≥ 32** vs **30–270 ms at G ≤ 4** | same | mf-C, p20-T3 |
| F-18 | split-K on a one-row skinny-K body loses at **2, 4, 7, 8, 14** ways (ship 0.0179 ms; `intra1/2/4/8` 0.0179/0.0204/0.0251/0.0246; `two2…two14` 0.0228…0.0369) and re-associates (**≤ 48 ULP**) | same §3 | mf-C, p20-T8 |

### 9.4 Box facts — weight-walk pattern vs contiguity `[SIB-BOX]`

| id | fact | sibling source § | landed |
|---|---|---|---|
| F-19 | engine per-thread walk ÷ contiguous sweep over the same bytes and grid: **0.470** (gate/up), **0.472** (with 62:1 writes), **0.629** (`down`), at **100 %** transaction coalescing on both sides | `strix-halo-mixed-traffic.md` §Q9 Results | mf-D |
| F-20 | LDS staging exonerated: `nostage/eng = 1.098`; decisive row `R_decisive` **140.5 GB/s** = **0.598** of the contiguous reference **235.0 GB/s** | same | mf-D |
| F-21 | launch **order** is not the lever: identical 424-launch multiset over a 1549 MiB footprint, `B_blk` **183.3** vs `B_rr` **184.2** GB/s = ratio **1.005** | same | mf-D |
| F-22 | the fix was a walk-shape change (`interleave` → `contig`), proven **bit-identical** (0 / 3584 `act`, 0 / 7168 `gate_pre+up_pre`, 0 / 1024 `out` at every layer) with an ISA witness (`v4,v2,…` → `v17,v17,…`) | `mlp_gemv/results.md` §Q10 | mf-D, here §1 |
| F-23 | a strided two-stream walk "is a memory-system pattern, and the pattern is the bottleneck" — the in-step MLP residue reproduced at ~60 % of probe from mapping alone | `strix-halo-mixed-traffic.md` §Q9 Results | mf-D, p20-T3 |

### 9.5 Compiler and occupancy `[SIB-BOX]`

| id | fact | sibling source § | landed |
|---|---|---|---|
| F-24 | `attn_core` **declared 6,176 B** of `__shared__` and **reserved 38,944 B**: LLVM promote-alloca-to-LDS moves arrays into LDS when the loop bound is a runtime value | `generation-limits.md` §3 | p20-T4 |
| F-25 | the fix was a spelling (`for (t=0;t<8;++t) if (t<nd)`): byte-identical across 126 stage comparisons and 21 cases, residency **1 → 4** blocks/CU, **−46 %** on the line, **−9.9 %** on the step | same | p20-T4 |
| F-26 | bound classification, and its precondition: traffic **27.7 %** of plateau and issue **5.9×** under its own roof ⇒ neither was the wall; and at 128 lanes the launch is 8 blocks on 40 CUs so neither residency nor population *can* bind | same | p20-T4 |

### 9.6 Sibling-engine step ledger `[SIB-GEO]`

| id | fact | sibling source § | landed |
|---|---|---|---|
| F-27 | operating point **4,032 lanes: 10.9357 ms/token = 91.4 tok/s** (9.7855 ms = 102.2 at 128 lanes); prefill 11,070 tok/s at 4,096 | `generation-limits.md` §0 | mf-G |
| F-28 | per-step census **1532.3 MB read / 24.4 MB written** at **340 launches / 305 segments**, **243 / 184** after the cast fold; **6.47 ms** of unavoidable weight traffic | same §0/§1 | mf-G |
| F-29 | family efficiency against byte floors: `mlp` 70 %, `proj` 75 %, `lm_head` **104 %** ("AT THE WALL, cannot be faster"), `attn` 13 %, `gdn` 30 %, `host_tail` **0.7163 ms** serial per token, `cast` deleted | same §1 | mf-G |
| F-30 | MLP in-situ decomposition: `T_geo` **+0.7565 ms (69.5 %, launch shape)**, `T_serial` +0.2709 (24.9 %), `T_pipe` +0.0484 (4.4 %), memory-state residue **~0.06 ms** | `mlp_insitu/D2-DIAGNOSIS.md` §5 | mf-J |
| F-31 | the "98 % floor" was a different shape: replicated-24× burst = 2 launches at 86,016/10,752 blocks vs the engine's 48 launches at 3,584/1,024 rows (448/128 blocks); shape-matched arm reproduces at **+1.0 %** | same §0(iv)/§5 | mf-J |
| F-32 | their MLP headline **2.650 ms/token = 199.4 GB/s = 84 % of the 237 probe** (rows/block 8, cold), marked provisional, with the alternation defect named | `mlp_gemv/results.md` §Performance | mf-J |
| F-33 | per-layer MLP bytes **22.02 MiB**, i.e. 22 MiB per layer against a **2 MiB L2**, so a per-token launch pays that from DRAM per token | same §P4 | mf-J |

### 9.7 `lm_head` and the batched head `[SIB-GEO]`

| id | fact | sibling source § | landed |
|---|---|---|---|
| F-34 | single-row head line **2.1685 ms** for **509.6 MiB** = **246 GB/s demanded = 104 % of the wall** | `generation-limits.md` §1 | mf-H |
| F-35 | batched-head R: M=1 **2.1265**, M=2 **2.1305** (**1.0019**), M=4 **2.1538** (**1.0128**); `fetch_per_weight` **1.000000** on every M arm, calibration **2.000000** / **1.25**, `short_grid` **0.999968** with 16/32 poison holes | `mtp_headbatch/HEADBATCH-RESULT.md` | mf-H |
| F-36 | the plateau arithmetic behind it: table once 485.0000 MiB / 237 = **2.1458 ms** (measured 1-row 1.011×); 2nd logits row **+0.19 %**; dots 0.509 → 1.017 GFLOP/pass = **3.2 % → 6.4 %** of the 7.30 TFLOP/s scalar roof; LDS `M × 4 KiB` at rpb 8 stayed thread-bound at 8 blocks/CU | `HEADBATCH-HANDOVER.md` §0 | mf-H, here §4 |
| F-37 | `rows_per_block` sweep, bit-identical: **1 → 192.7, 2 → 241.7, 4 → 242.1, 8 → 239.6, 16 → 239.0 GB/s** | `lm_head_gemv/results.md` | mf-H |
| F-38 | fused argmax bought **1 %** (2.099 vs 2.126 ms): "writing 1 MB of logits is nothing next to reading 485 MB" | same | mf-H |
| F-39 | second-row marginals on their shapes: `proj` **+7.6 %**, `mlp gateup` **+8 %**, `mlp down` **+76 %**; and `attn` measured *faster* for a second row, refused because the rig sat at 13 % of its own byte floor | `mtp_delta/D3-RESULT.md` | mf-H |

### 9.8 GDN — `[SIB-GEO]` unless the row says otherwise

| id | fact | sibling source § | landed |
|---|---|---|---|
| F-40 | state **F32**, **1.000 MiB** per layer (16 × 128 × 128 × 4 B), **18.000 MiB** per request; conv window F32 1.266 MiB/request | `gdn_rec_isa/results.md` §8 | mf-I |
| F-41 | **2 read + 1 write** state passes per launch, independently confirmed in machine code as 256 state loads / 128 stores per token-step = exactly 2 MiB read / 1 MiB written | same §8 | mf-I |
| F-42 | state is re-read every token "and it cannot be anything else in this geometry"; the token loop is a host loop around the launch | same §8 | mf-I |
| F-43 | state coldness **+0.070 ms saturating at exactly 32 MiB** (`cold_0MiB -> 32 -> 512 -> 1536` = 0.1216 / 0.1917 / 0.1894 / 0.1922) — an L2-tier effect, not a DRAM cliff | `gdn_insitu/D4-DIAGNOSIS.md` §7 | mf-I |
| F-44 | whole-family warm launch **0.0646 ms** for 18.0 MiB = **876 GB/s cache-resident = 370 % of the probe**; one layer `rec` alone **0.0064 ms**; the 108 MiB row (**255 GB/s**) retired by that file itself as unreproducible | `gdn_decode/results.md` §Performance | mf-I |
| F-45 | launch geometry `rec_kernel<<<NH=16, D=128>>>` = 16 blocks × 4 wavefronts = **64 wavefronts**, wave32, ≤1 per SIMD; prologue **323 of 1,523 slots (21 %)**, repeated 18× per token | `gdn_rec_isa/results.md` §1 | mf-I |
| F-46 | LDS-resident route: modelled **11.4 ms**, measured **112.04 ms** (**9.8×**) against pre-registered **63 ms** win / **84 ms** kill lines | `gdn_collapse/results.md` §0/§5 | mf-I, p20-T1 |
| F-47 | why: **1.6 blocks per CU, one of whose four wavefronts does the recurrence**; "the cost is latency per token, not issue slots per token"; and launches 18,432 → 36 removed 0.12 ms-class traffic but bought nothing comparable | same §5 | p20-T1, mf-I |
| F-48 | merging 18 launches into 1 at identical bytes: `split8` 0.1839 → `comb_all` **0.0644 ms** = **0.12 ms** | `gdn_insitu/D4-DIAGNOSIS.md` §7 | mf-I |
| F-49 | conv window: carried **pre-conv** raw values (not zeros); a window bug is invisible in the convolved output, so the gate compares window **bytes**; a tile-indexing bug is invisible at `tokens == 1` | `gdn_decode/results.md` §P6 | here §2 |
| F-50 | the two-pass `o` rule: closed form measures **1.5e-3** vs **4.5e-8** for the reference order, because the two terms cancel | same finding 3 | here §2 |

### 9.9 MTP `[SIB-GEO]`

| id | fact | sibling source § | landed |
|---|---|---|---|
| F-51 | draft-step read ledger: **16 distinct tensors / 549,465,088 B = 524.011 MiB**, of which the block is **39.0107 MiB** (15 tensors) → **92.6 %** of draft traffic is the head | `mtp_gatezero/MTP-GATEZERO.md` §1.1 | mf-K |
| F-52 | speedup identity `speedup = (1 + a_eff) / ((1 + delta) * (1 + c))`, wins iff `a_eff > (1 + delta)(1 + c) − 1` | same §1 | here §3 |
| F-53 | draft block cost by depth: k1 **0.4371 ms**, k2 **0.7666** (ratio **1.754**), k4 **1.5039** (**3.441**); block ran **1.83× its byte floor**; empty arm **1.82 µs/launch** ⇒ 14 launches ≈ 0.026 ms, i.e. launch floor **overlapped, not exposed** | `mtp_draft/DRAFT-RESULT.md` | mf-K |
| F-54 | `delta`: plan A **1.6454 ms = 0.1505 of a step (1.231×)**; pessimistic **5.1712 ms = 0.4729** (KILLED); two sequential 1-row bodies KILLED structurally; rollback excluded at ~0.12 ms/round | `mtp_delta/D3-RESULT.md` | mf-K |
| F-55 | restricted-candidate-set head is safe in the direction that matters: "Restriction can only lose accepts, never buy a wrong one, because the accept test is the target's own argmax"; argmax fusion rejected unprompted because fusion removes host work, not the table read | `MTP-GATEZERO.md` §1.2 | here §3 |

### 9.10 Method traps copied into the pitfall files

| id | trap | sibling source § | landed |
|---|---|---|---|
| T-1 | an instruction-slot bill is only valid where there is concurrency to hide dependent loads (9.8× case) | `gdn_collapse/results.md` §5 | p20-T1 |
| T-2 | price a launch by **deleting** launches; reject slopes fitted at an empty body and merge gains that move workgroups-per-launch | `generation-limits.md` §4/§6, route-S1 §7 | p20-T2 |
| T-3 | blocks-per-launch has a plateau and a cliff (43× at equal bytes/workgroups) | `wave_shape/ITEM2-RESULT.md` §2 | p20-T3 |
| T-4 | declared vs **reserved** LDS (promote-alloca-to-LDS under a runtime trip count); classify a bound by arithmetic; check the shape can bind | `generation-limits.md` §3 | p20-T4 |
| T-5 | a probe's own summary column can be a mislabeled counter (`mac_per_lane_clk` ≡ `icpcb`, claiming 172 % / 248 % of a 1-MAC ceiling); recompute from geometry (**40 CU × 64 lanes × 2.894 GHz**); drop legs whose byte witness failed; occupancy recipe `blocks=320 threads=128` = 32 waves/CU | `issue_rate_probe/results.md` §0/§1 | p20-T5 |
| T-6 | a noise floor is per-comparison, not a global absolute: 0.3071 ms floor ≥ 0.30 band adjudication-failed; a **311 ms** cell made a 7 ms comparison unadjudicable; a per-stratum bimodality cap no arm could pass (2.0–2.5 % at REPS=5 *and* 12); re-take ≠ average | `wave_shape/ITEM2-RESULT.md` §1, `generation-limits.md` corrections log | p20-T6 |
| T-7 | above-probe GB/s: tile the working set past any plausible cache to explain it — and retire a row no knob in the binary can reproduce | `gdn_decode/results.md` finding 4 + re-take note | p20-T7 |
| T-8 | split-K re-stages the activation per split **and** re-associates (≤ 48 ULP): price and band it in one arm | `wave_shape/ITEM2-RESULT.md` §3 | p20-T8 |
| U-1 | MB written where MiB was meant: `lm_head` reads **509.552 MB = 485.947 MiB**; the rig was 4.6 % heavier than the census it reproduced | `launch_price/S1-GATEZERO.md` §4 | p50-U1 |
| U-2 | two units on visually identical lines: the draft `empty` arm is **µs/launch** while block arms are ms; read as ms it is a 40× error | `mtp_draft/DRAFT-RESULT.md` | p50-U2 |
| U-3 | a fitter with **no argument handling** accepted `--evidence DIR` by silence and printed bands it had never compared; the paired lesson: the regex bug (`pass 1` vs `pass1`) was only visible because the replacement refuses an empty parse | same, finding 1/3 | p50-U3 |
| U-4 | a runner that deleted its own first pass (`rm -f "$dir"/*.txt` per phase) still looked green | `mtp_delta/D3-RESULT.md` | p50-U4 |
| U-5 | a trailer counting **distinct cells** where the fitter counts **records** made a complete 54-record file refuse itself | `wave_shape/ITEM2-RESULT.md` §1 | p50-U5 |
| U-6 | a verdict computed twice: a silently inverted `chk.ok` printed `caught=0/8` over a green run; one verdict per case, computed once | `launch_price/S1-GATEZERO.md` §4 | p50-U6 |
| U-7 | sweeps and negative tests must **load their own weights** (they inherited the last layer's files, so a sweep would rank configs against the wrong matrix and the negatives were vacuous) | `mlp_gemv/results.md` §Traps | p50-U7 |
| U-8 | a compile-time constant used by the kernel while the host sweeps the runtime value ⇒ **415–710 GB/s**, above the physical peak | `lm_head_gemv/results.md` | p50-U8 |
| U-9 | a control that is not expressible on the rig exits 1 and a `len(...) >= 2` guard silently drops the de-confounder; record the drop, do not silence it | `mtp_delta/D3-RESULT.md` | p50-U9 |
| U-10 | a flag whose premise is invalid must **refuse**: `--geometry fat` refuses `blob72`/`shipped72` ("the contiguity premise is checked, not assumed"); `--mode down` refuses `--mapping` | `mlp_insitu/D2-DIAGNOSIS.md` §4 table; `mlp_gemv/results.md` §Q10 | p50-U10 |
| U-11 | a probe that released its spinner immediately after launching raced itself: "that was a race, not a flaky machine"; release only after the probe signals completion | `prefetch_tier/P1-ARITHMETIC.md` §1 | p50-U11 |
| U-12 | witness architecture for a launch-accounting rig: per-(launch, block) slot with sentinel, closed-form invertible read fill `word(p) = K·(2p+1)` (K odd, asserted invertible mod 2⁶⁴), writes indexed by offset **inside their own launch**, chunk arithmetic **booked not inferred**, one forked child per control, `NO-TIMING` refusals, a grep that fails a correctness log containing `ms_median\|GB/s`, and `--pages cold` refused outright | `launch_price/S1-GATEZERO.md` §3–§4 | p50-U12 |
| U-13 | two spellings of one weight name (raw checkpoint key vs normalised store key) need **one** Python mapping module, a lookup that **refuses rather than normalises**, and a selftest that reads the shard *index* because the header parser hipMallocs | `weight_names.py` docstring | p50-U13 |

**Count: 55 numbered facts (F-01…F-55) plus 21 method traps (T-1…T-8, U-1…U-13) = 76 copied items.** Every
F-row carries its sibling `path §label`, what it was measured on, and its tag; several rows carry a **split tag**
(`[SIB-BOX]` mechanism, `[SIB-GEO]` magnitude) because that is the honest description — a per-block fetch-capacity
mechanism is a box fact while the milliseconds it was observed through are theirs. A row with a split tag must be
cited with both halves; quoting only the number and dropping the qualifier is the failure this section exists to
prevent.

## 10. `NOT ESTABLISHED:` — what this harvest could not settle

1. **`NOT ESTABLISHED:` our own mixed read/write denominator.** Their `mixed_bw.cpp` was never copied
   (`devices/measured-facts.md` §Memory bandwidth), and §Q9 shows launch *order* is not the lever
   (ratio 1.005), so the remaining candidate is the walk inside one kernel. A paired read/write probe here is
   still owed.
2. **`NOT ESTABLISHED:` the price of *our* launch and drain pattern.** Every launch/sync price above is a
   sibling-engine measurement on 24 dense layers; our step has a router, a 40-layer expert gather and a draft
   block. Only a deletion-style A/B on our own step can price it.
3. **`NOT ESTABLISHED:` whether a 60.0 MiB GDN state gets any tier benefit.** Their residency result is an 18.0
   MiB state inside a 32 MiB tier; ours is 1.875× the tier. Nobody has measured the 60 MiB case on any box.
4. **`NOT ESTABLISHED:` our GDN k-head → v-head mapping.** `derived here` that two value heads share one key
   head, but the exact index rule (and its effect on the conv strip and `A_log`/`b` widths) must be read from
   `modeling_qwen3_5_moe.py`, not inferred from their 1:1 kernel.
5. **`NOT ESTABLISHED:` the shared expert's gate form.** `shared_expert_gate.weight` exists per layer and in the
   draft block; whether it is a scalar sigmoid or per-channel is unread in this repo and changes the kernel's
   epilogue.
6. **`NOT ESTABLISHED:` any accept rate, survival rate, or MTP depth for this model.** Refused by `AGENTS.md`
   and by §7; our draft block is a different network.
7. **`NOT ESTABLISHED:` whether D-17's ±1-ULP acceptance can survive a legal long-dot reassociation.** §8
   finding 1 is a real tension between `tools/ulp_policy.py` (c) and `pitfalls/30`; resolving it is a user
   decision (band in the ruler, or an explicit field-set exclusion), not a code preference.
8. **`NOT ESTABLISHED:` the router's host-sync cost.** `AGENTS.md` prediction 1 is still a prediction; their
   engine's "one blocking D2H per token" is prior art, not a measurement of anything here.
9. **`NOT ESTABLISHED:` whether the 32 MiB tier is even reachable at all under policy A residency**, since the
   pool figures in `devices/measured-facts.md` and the tier are different counters and the tier's behaviour
   under a 20 GiB working set has never been probed.
10. **`NOT ESTABLISHED:` any figure in this file is a measurement of Qwen3.6-35B-A3B.** Nothing here was run on
    a device; every number is a copy (with tag) or an arithmetic derivation from
    `checkpoints/Qwen3.6-35B-A3B.config.json` and the pinned index.

