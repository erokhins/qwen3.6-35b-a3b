# `prototype/q4k_dequant` — device dequant gate, first green run

**Verdict: PASS.** 30 checks, 0 findings. The iGPU (gfx1151) dequantizes Q4_K and Q8_0 blocks
**bit-identically** to this repo's audited CPU reader `tools/gguf_reader.py`, and all nine injected
misreadings fire only where they belong.

Artifact names carry the project date (2026-10-26), matching `quantization/evidence/`; the box clock
at run time read `2026-10-25T06:14:00Z`.

## Device-discipline record (AGENTS.md §"GPU execution is serialized…")

| field | value |
|---|---|
| command line | `./gate` → `python3 prototype/q4k_dequant/gate.py` → `q4k_dequant --plan work/units.tsv --cases work/cases.tsv --slab work/slab.raw --out work/out.f32 --report work/report.tsv --expect-units 1147 --expect-cases 4052` |
| wall time (lock record input, not a measurement) | started `2026-10-25T06:22:11Z`, finished `2026-10-25T06:22:12Z` |
| `tools/gpu_lock.sh util` at the time | before `gpu use: 99%`, after `gpu use: 78%` |
| lock | acquired by the `gate` wrapper only, tag `q36 W6 q4k_dequant device gate`, released on EXIT; `LOCKREF gate.py references to the lock tool must be 0: 0` |
| concurrent tenants | `util` was 99% and `/tmp/pi_halogen.lock` was held by another session's worker for the whole run. A bit-exact PASS/FAIL comparison is insensitive to contention; no timing was taken, so neither fact weakens this verdict. |
| timing | **no timing taken** — no timer in `gate.py`, no `ms`/`GB/s`/`tok/s` anywhere in this directory, no warm loop, no burst, no best-of-N |
| full log | `evidence/gate-device-2026-10-26.log` (30 `CHECK` records, `CHECKS ran=30 failed=0`, `RESULT PASS device`) |
| the two host-only arms, same build | `./build.sh` → `BUILD ok`; `./gate --selftest` → `CHECKS ran=15 failed=0`, no lock taken |

The whole device run is seconds: 10 launches, 4052 cases, 513,824 output floats (1,962 KiB) and an
80,625-byte slab. Nothing here came near the ~32 GiB co-resident pool ceiling, and nothing near the
21 GiB file was allocated — only 972 blocks' bytes are ever copied to the device.

## What was gated

Derived, never hand-listed: the sample comes from the GGUF's own 733 tensor-info records and is
reconciled against a census derived from `checkpoints/Qwen3.6-35B-A3B.config.json`
(`gate.py §derived_census`, `§reconcile_census`).

```
GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf bytes=22134528992 n_tensors=733 data_start=10990048 align=32
GEOMETRY attn_output_gate=True full_attention_interval=4 head_dim=256 hidden_size=2048
         moe_intermediate_size=512 num_attention_heads=16 num_experts=256 num_hidden_layers=40 vocab=248320
PLAN units=1147 cases=4052 launches=10 out_floats=513824 slab_bytes=80625
PLAN units_by_kind_cls Q4_K/real=189 Q4_K/synth=4 Q4_K/support=175 Q8_0/real=775 Q8_0/synth=4
PLAN cases_by_expect ALT=2081 IMMUNE=107 MATCH=1147 MISMATCH=717
PLAN tensors_sampled=42 blocks_sampled=972 support_blocks=175
```

13 of 13 census families sampled, including the four tensors the brief names by hand
(`blk.0.ffn_gate_exps.weight`, `blk.1.ffn_up_exps.weight`, `blk.0.ssm_out.weight`,
`token_embd.weight`) — required *by rule* inside their family's even-spaced selection, asserted
present rather than appended (`gate.py §select_samples`). Every block index is inside its own
tensor's block count (972 checked). The per-tensor table is
`evidence/q4k-dequant-sample-2026-10-26.md`.

**Sample cut, recorded:** the default sample was 1501 units; `--max-units 1400` cut
`tensors_per_family 3→2`, giving 1147 units. The cut was in the *sample*, not in an assertion — the
must-catch set, the must-not-catch set and the bit-exactness claims are unchanged by it, and the cut
prints itself: `CUT reason=max_units=1400 units=1501 action=tensors_per_family 3->2`.

## Verdict detail

| check | n | result |
|---|---|---|
| `expect-MATCH` — device == reader's decode of the same bytes, bit-for-bit | 1147 | ok=1147/1147 |
| `expect-MISMATCH` — injected misreading differs from the correct decode | 717 | ok=717/717 |
| `expect-ALT` — equals the reader's decode of the bytes the mode *should* have read | 2081 | ok=2081/2081 |
| `expect-IMMUNE` — mode provably cannot change this block | 107 | ok=107/107 |
| `every-injection-mode-fires` | 9 | modes_used=9 silent=none |
| `driver-bad-column-agrees` (driver's `bad_vs_mode0` vs Python's own count) | 4052 | mismatches=0 |
| `deqcells-out_sum` (driver cell sums vs Python's) | 22 | outside allowance=0 |
| `cell-kind-purity` — Q4_K-only modes on Q8_0 units or the reverse | 22 | **leaks=none (0)** |
| `no-unexpected-nan` — device NaN where the reader's word is finite, control/immune rows | 145,632 elts | 0 |
| `witness-comparator-flip`, `comparator-rules-witnessed`, `witness-parsers` | 1 / 5 / 2 | all green |
| host arm: `host-probe-clean`, `host-probe-corrupt-rejects`, `host-probe-f16-exhaustive`, `host-probe-f16-corrupt-rejects`, `comparator-nan-rule` | 23 / 1 / 65,536 / 1 / 2 | all green |
| `driver-selftest` | 6 | lines=6 checks=6 failures=0 |
| `driver-plan-parse`, `driver-planchannel`, `driver-out-file`, `driver-result` | 2 / 6348 / 1 / 1 | all green |

`RESULT PASS device plan units=1147 cases=4052 cells=22 out_floats=513824`, and the parsers refuse an
empty cell table (`empty-cell-parse-is-refused` in the `--selftest` roster, 3 checks forced red on an
empty parse).

### Must-catch, by mode (cases that fired; derived from geometry, not hand-listed)

| mode | fired | what it is |
|---|---|---|
| `q4k_as_q4nl` | 193 | Q4_K walked as 8 × 18-byte Q4_NL blocks (6-bit sub-block scales *and* the min term gone) |
| `q4k_drop_min` | 175 | `value = d*sc*q`, the `- dmin*mg` term omitted |
| `q4k_naive_sc_min` | 175 | the 96 scale/min bits read as 16 contiguous 6-bit fields |
| `q4k_subblock_rot3` | 174 | element `e` given sub-block `((e>>5)+3)&7`'s scale **and** min |
| `q4k_block_off_1` | 185 | block index off by one inside its tensor group |
| `q4k_byte_shift_1` | 175 | tensor data start one byte late |
| `q4k_xpose_expert` | 167 | `[E,N,K]` expert stack indexed as if the expert axis were trailing |
| `q8_0_block_off_1` | 777 | as above, Q8_0 |
| `q8_0_byte_shift_1` | 777 | as above, Q8_0 |

### Must-not-catch

107 `IMMUNE` cases passed with 0 mismatches — the all-zero Q4_K/Q8_0 blocks, byte-identical adjacent
blocks, flat-scale blocks under a sub-block rotation, and the expert-transpose fixed points
(`blk` 0 of an expert tensor maps to itself). The immunity is *derived from the block bytes and the
format's arithmetic* (`gate.py §immune_scale_mode`), never from a hand-written list, and the
`fixture-signed-zero-discriminates` check proves the all-zero fixtures are not "zeros agreeing with
zeros": the correct decode is +0.0 everywhere while the `as_q4nl` misread yields −0.0, a different
word. **Leaks: 0** — `cell-kind-purity` and the plan-side `plan-mode-kind-pairing` (4052 cases) both
report none, and an inapplicable (kind, mode) pair writes NaN in every slot the lane owns rather than
a number.

## Why the comparison is bit-exact, and what that rests on

`gate.py` compares the 4 bytes of `struct.pack('<f', v)` from the reader's dequant against the
device's `out.f32`; the kernel does all arithmetic in plain f32. Two CPU-only facts, established
before any device work, carry that claim:

1. **The f32 route is not an approximation here.** Over 4,000 real Q4_K blocks from
   `blk.0.ffn_gate_exps.weight` and `blk.1.ffn_up_exps.weight` — 1,024,000 elements — the f32 route
   and the reader's double-then-round-once route produced **zero** differing bit patterns. The reason
   is arithmetic, not luck: a Q4_K value is `d*(sc*q) - dmin*mg` with at most 11 + 6 + 4 = 21
   significand bits, and Q8_0 is `d*q` with 11 + 8, both inside f32's 24.
2. **The sub-block index is `e >> 5`.** Witnessed on a synthetic block (`d=1.0`, `dmin=0`, one
   6-bit field = 63, all nibbles zero) which produced exactly one non-zero element: index **33**,
   value **945.0** — i.e. chunk index `e>>5`, equivalent to `2*(e>>6) + ((e>>5)&1)`, and *not* the
   ggml `make_qx_quants` printing order.

**One equality rule, three implementations.** `q4kd_word_eq` in `q4k_dequant_kernels.h` is used by the
device kernel's counters, by `host_probe.cpp`, and mirrored in Python by `gate.py §word_eq`. It is
strict bit equality on finite values — including the sign of a zero, so +0.0 ≠ −0.0 — with exactly one
exception: two NaNs are equal whatever their sign or payload, because neither route *defines* those
bits for a NaN product (CPython folds `nan * -1.0` into `-nan`; an f32 multiply keeps the first
operand). That exception is gated, not assumed: `comparator-nan-rule` re-emits every probe row with
the sign bit of every expected NaN word flipped (64 words in 2 rows) and requires the probe's counts to
come back identical to the clean run, while `host-probe-f16-corrupt-rejects` keeps the strict side
alive. `q4kd_f16` canonicalizes NaN to ±0x7FC00000 to match `struct.unpack('<e')` — the reader's route
— and the exhaustive `f16.tsv` arm checks all 65,536 half patterns against the reader's own answers
(rows parsed must equal rows written, and coverage of NaN / denormal / inf / zero must each be
non-zero).

97 of 4052 cases carry NaN in the device output; the reader produces NaN for those same real bytes, so
that is a fact about the file, not a verdict — and none of them sit on a control or immune row.

## What bring-up found (and which control found it)

Every line here is a defect that a green first impression would have shipped. Listed with the check
that caught it, because that is the part worth keeping.

| # | defect | caught by |
|---|---|---|
| 1 | `q8_0_decode_lane`'s leak guard read `mode <= Q4K_XPOSE_EXPERT`, which includes mode 0 — the **control row** for Q8_0 was NaN-filled | `driver-selftest` `q8_0_hand` |
| 2 | The Q4_K leak path NaN-filled `dst[0..7]` instead of the lane's scattered slots `64g+32h+lane`, so an unwritten element could read as a match | `driver-selftest` `leak_guard` (strengthened to all 256 slots after a memset) |
| 3 | `q4kd_f16` kept the sNaN payload where the reader's `<e` collapses to the canonical quiet NaN — invisible in 13 hand cases, it moved a whole 32-element sub-block of the `as_q4nl` control | `host-probe-clean` (`n_bad=32` on rows 7 and 12); then an exhaustive 65,536-pattern f16 arm was added |
| 4 | `pack_scale_min` put `mg[j]>>4` into the high nibble of `q[j+4]` instead of `mg[j]&15`, so half the synthetic fixtures were packing the wrong mins | `build_q4k`'s round-trip assert through `gr.qk_scale_min` |
| 5 | The kernel's `atomicAdd` for `csum`/`cbad`/`cnan` was guarded by `threadIdx.x == 0`, throwing away every partial but thread 0's: 32 differing elements reported as 1, and a cell sum that was only element 0 | `driver-bad-column-agrees`, `deqcells-out_sum` |
| 6 | After (5), `cnan` became a per-thread count where the report column is a per-case flag | `driver-bad-column-agrees` (97 cases) → `atomicMax` |
| 7 | Python's neighbour rule wrapped the last unit of a group to **itself** while the kernel wrapped to the group's front — 40 last-block-of-tensor cases were classified `IMMUNE` and the off-by-one was blind at exactly the tensor boundary | `expect-IMMUNE` (107/147) |
| 8 | The driver never wrote `out.f32` at all | `read_out` FileNotFoundError → new `driver-out-file` check and an `OUT file=… floats=… bytes=…` line |
| 9 | The driver read only the units file; the cases file was never opened, so it ran a plan with 1147 units and 0 cases while still printing `PLANCHECK` | `driver-plan-parse` count equality (0 ≠ 4052) |
| 10 | The derived census forgot `attn_output_gate: true`, which doubles `attn_q`'s rows (8192, not 4096) | `reconcile_census` refusing the file |
| 11 | Four hand-worked hex constants in the driver's self-test were wrong (−63, 882, 945, 1023·2⁻²⁴) while the kernel was right | `driver-selftest` — the check failing on its own expectation rather than on the code |
| 12 | `no-nan-in-output` rested on a false premise (that NaN can only come from a leak); real bytes here do decode to NaN | `no-unexpected-nan` replaced it: NaN is unexpected only on control/immune rows, where the reader's word is finite. No band was widened — the four expectation classes were already green. |
| 13 | `--check-only` could report `driver-plan-parse` green on a driver that had printed FAILs | fixed with the count-equality fields; the roster check now also refuses a missing/extra/duplicated check name |

## Reproduce

```sh
cd prototype/q4k_dequant
./build.sh          # hipcc -x hip --offload-arch=gfx1151 -O3, objects in build/, -Wl,-rpath,/opt/rocm/lib
./gate --selftest   # host only: no device, no lock, 15 checks
./gate --check-only # plan build + driver structural verification: no device, no lock
./gate              # the device gate; takes the lock, ~seconds, PASS/FAIL only
```

`./gate --selftest` and `./gate --check-only` deliberately do **not** take the lock — there is no
device work in them, so the device stays free for the main agent. `gate.py` contains zero references
to the lock tool (the wrapper prints the `grep -c` result as `LOCKREF … : 0` at every run), the
wrapper owns acquire/release with a `trap … EXIT`, refuses a live foreign holder instead of queueing
(`RESULT FAIL not-run lock-held`), and the driver's stdout is redirected to `work/driver.log` and
parsed from the file — never piped, so its exit code survives.

Binaries and `work/` are gitignored by exact path; the directory itself is tracked.

## Not claimed

* No performance claim of any kind. No timing was taken; `util` before/after is contention, not a
  result, and the wall-clock pair exists because AGENTS.md asks a device run to carry a lock record.
* This gates **dequantization**, not the MoE block: no gather, no router, no expert accumulation. The
  f32-exactness result is about these two formats' arithmetic, not about matmul association.
* `out_sum` agreement uses an accumulator allowance as a *copy witness* on a sum the gate does not
  otherwise consult; every value that matters is compared as 4 raw bytes.
* The audit marker is verified, not written: `gate.py §require_audit_marker` mirrors
  `tools/audit_published_quant.py §require_selftest` and refuses to run if
  `quantization/evidence/audit-selftest-2026-10-26.json` is stale relative to the reader or the audit.
  It does not append to step 1's ordering log.
