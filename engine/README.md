# `engine/` — the deliverable, certified at the CPU layer

Status: **certified for what this directory claims, and nothing beyond it.** The gate is
`engine/gate.py`; its last run is `engine/evidence/gate-run-2026-10-26.log` and it ends in
`RESULT PASS checks=27 findings=0 notes=41 arms=40`. Until that file existed this directory was parked,
because `loader.py` was missing and a directory whose deliverable is absent cannot be certified.

What "certified" means here is narrow and it is worth naming before anyone quotes it: the artifact is
identified, policy A's resident set is derived from the artifact rather than asserted, every byte of it
is addressed, both memory budgets are printed with their own evidence state, and every refusal that
this layer is supposed to make has been witnessed firing on the real artifact. What it does **not**
mean is in §Not established, and that list is the part to read first if you are about to build on top.

Run it (this is the whole gate; it takes no device lock, reads the whole GGUF once for the digest, and
prints no rate figures):

```
/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python engine/gate.py \
    --keep-evidence engine/evidence/gate-run-2026-10-26.log
```

## The artifact this layer is about

| field | value |
|---|---|
| file | `models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (the published artifact this repo inherited; the pin is what makes it ours) |
| size | 22,134,528,992 B |
| sha256 | `ac0e2c1189e0…` — full digest pinned in `checkpoints/Qwen3.6-35B-A3B.gguf-pin.json`, with two independent passes recorded in `quantization/evidence/audit-hash-2026-10-26.json`; a fresh full-file hash reproduces all three (gate arm `loader-sha256`) |
| `general.alignment` | 32 |
| `general.tensor_count` | 733, and 733 parsed |
| `general.tensor_offset` (`data_start`) | 10,990,048 = ceil(10,990,035 / 32) × 32 |
| data section | 22,123,538,944 B = Σ row_bytes, residue **0 B** |
| widths as stored | Q4_K×80, Q5_K×37, Q8_0×251, Q6_K×4, F32×361 |
| draft block | **not in this file**: `models/w1-mtp-extract/mtp.safetensors`, 19 BF16 tensors, 1,689,281,536 B of tensor payload |
| geometry | 40 layers (10 `full_attention`, 30 `linear_attention`), E=256 experts, 8 selected, H=2048, I=512, twoI=1024, V=248320, `lm_head` untied |

## The four claims, and what backs each one

**1. The width veto refuses by name and is never coerced (D-17).**
`Loader.admit(name, type)` answers a tensor's own ggml type against a floor derived from the family
widths policy A declares (`quantization/README.md §The policies, priced`) — derived through
`tools/ggml_block_geom.py §bpw_of`, never a typed literal, so a future table edit moves the floor and this
check moves with it. D-17 is the reason it refuses rather than adapts: *“The width veto is the user's and
is structural, not a tunable.”*

```
routed_experts>=4.5000  gdn_linear_attn>=8.5000  full_attn>=8.5000  shared_experts>=8.5000
embed>=8.5000  lm_head>=8.5000  routers>=16.0000  norms_misc>=16.0000
```

Three refusals were witnessed on tensors that exist in the real file, each naming the tensor:

```
$ engine/loader.py --fault wrong_width
  [FAIL] width-veto-admits-artifact  refused 1 (first: [width-veto] blk.0.attn_qkv.weight is family
        'gdn_linear_attn', whose policy-A floor is 8.5000 bpw (Q8_0), but the file stores it Q4_K at
        4.5000 bpw. Refusing by name: this loader does not ...)
$ engine/loader.py --fault exact_role_quantized
  [FAIL] width-veto-admits-artifact  refused 1 (first: [width-veto] blk.0.ffn_gate_inp.weight is a
        never-quantized role (blk.N.ffn_gate_inp.weight) but the file stores it Q8_0 at 8.5000 bpw;
        the router, the norms, conv1d, A_log and dt_bias stay exact and are never coerced to fit a slab)
$ engine/loader.py --fault lm_head_waiver_withdrawn
  [FAIL] width-veto-admits-artifact  refused 1 (first: [width-veto] output.weight is family 'lm_head',
        whose policy-A floor is 8.5000 bpw (Q8_0), but the file stores it Q6_K at 6.5625 bpw ...)
```

The first two relabel a real tensor so the veto has something to bite; the third is the interesting one,
because it touches nothing: withdraw the waiver and the veto fires on `output.weight` **exactly as it is
stored**, Q6_K against the Q8_0 floor. That is the proof that the waiver is load-bearing rather than
decorative — and `WIDTH_WAIVERS` is checked for being exercised, so a waiver list that nothing hits is
itself a finding. The waiver is not this file's decision: policy A stores `lm_head` at Q8_0
(D-01a/D-10), the published artifact carries it at Q6_K — that is policy G's row,
`quantization/README.md` "(not taken, D-10)" — so the loader admits the file as shipped, names the
deviation, and attributes it in bytes: `output.weight` is 123,166,720 B **below** policy A. Waivers are
the only way a tensor enters outside its floor, and every one is named, cited and priced.

**2. The two budgets are printed separately, each with its own verdict and evidence state (D-11).**
This was a live defect in the parked `layout.py`: it printed `verdict=FITS co-resident` against a bound
of `>=55.50 GiB` whose own evidence line said `unmeasured`. `BUDGETS` now carries
`bound_measured`/`evidence`/`evidence_note` and its own verdict function, and `verdict()` refuses to
manufacture a fit claim from an unmeasured bound:

```
[as-stored] budget=development
[as-stored] budget-bound=31.84 GiB (pessimistic end of a 31.84-33.88 bracket)
[as-stored] budget-bound-evidence=measured
[as-stored] verdict=FITS co-resident (<= the pessimistic bound)
[as-stored] budget=deployment
[as-stored] budget-bound=≥55.50 GiB
[as-stored] budget-bound-evidence=unmeasured
[as-stored] verdict=NOT-ADJUDICABLE (this budget's bound is unmeasured, so no fit claim is lawful)
```

Six blocks are printed — `as-stored` and `policy-A` against both budgets, plus the modelled-state view —
each announcing how many `key=value` lines it is about to print so a reader can catch a parser that drops
one (`residency-claims-parse-equals-printed`: 68 promised, 68 parsed). The gate re-reads those lines and
enforces the pairing in **both** directions: an unmeasured bound may not carry a verdict, and a measured
bound may not have one withheld (`no-verdict-for-unmeasured`). The conflation can be re-injected
(`engine/layout.py --fault deployment_bound_claimed_measured`) and the gate requires that it exits **0**
with a green-looking table — the fault has to be *caught*, not crashed, or a future "fix" could be a crash
instead of a refusal.

**3. Everything material is re-derived by a second parser, not trusted from `artifact.py`.**
`engine/loader.py` re-tokenizes the header itself (reusing `gguf_reader.Cur`, the ruler's own tokenizer,
but keeping the 8-byte tensor offset that `tools/gguf_width_table.py` discards) and prices blocks with
`gguf_width_table.BLK` instead of `ggml_block_geom`. Two tokenizers, two byte tables, one file:

| re-derived | loader | the module it is checked against |
|---|---|---|
| records | 733 gguf + 19 draft, 752 names, all distinct | `artifact.py` census |
| per-family bytes | identical on all 5 types | `tools/gguf_width_table.py` |
| `data_start` | 10,990,048 = ceil(header_end+13 / 32) × 32, agreed by four independent readers | the file |
| file-size identity | Σ row_bytes 22,123,538,944 + 10,990,048 = 22,134,528,992, residue 0 B | the file size |
| residency set | 733 names derived from the safetensors index × `config.json`, 40 layers × role set, 120 routed records × 256 experts | the artifact |
| Q6_K `down_proj` layers | **[34, 38, 39]**, Q5_K×37, re-derived and compared to the pin's clause | `checkpoints/…gguf-pin.json`, `layout.py` |
| residency totals | 19.7848 GiB policy A = 20,740,587,816 B main + 503,143,072 B draft | `layout.py`, `residency_budget.py` |
| expert slicing | per-expert range = base + e·(row/E), byte-for-byte equal to `layout.expert_slice` for 120 routed records × 5 sampled indices, plus 480 out-of-range refusals | `layout.py` |
| width delta | 22,123,538,944 − 20,740,587,816 = 1,382,951,128 B, fully attributed: never-quantized +56,985,816, `down_proj` above policy +1,449,132,032, `lm_head` waiver −123,166,720, unexplained **0 B** | itself |

Every population here is derived from geometry — all 733 records, all 40 layers, all 256 experts, all
752 placements. The expert-slice check is the one place that samples rather than enumerates (5 indices
per routed record, `0 … 255`, chosen to include both bounds and both ends of the selection window), and
it says so on its own output line rather than pretending to be exhaustive; the 480 refusals that come
with it are what make the bound a witnessed bound instead of an assumption. The staging plan is checked
against the ruler's byte totals rather than against itself: 464 copy runs, 23,812,820,480 B, every run
inside the file it reads from.

**4. Every refusal this layer must make has been seen to fire.** 32 injections, one per process, each
required to turn a *named* check red — the list is derived from the three modules' own `FAULTS` tables,
so an injection added without saying what catches it fails the gate. Full mapping in
`engine/gate.py §EXPECT`; the witnessed pairs:

| injection | the one check it must turn red |
|---|---|
| `loader --fault wrong_width` | `width-veto-admits-artifact` |
| `loader --fault exact_role_quantized` | `width-veto-admits-artifact` |
| `loader --fault lm_head_waiver_withdrawn` | `width-veto-admits-artifact` |
| `loader --fault missing_tensor` | `residency-set-complete` |
| `loader --fault offset_past_eof` | `offsets-in-range` |
| `loader --fault ranges_overlap` | `file-ranges-disjoint` |
| `loader --fault data_start_mismatch` | `data-start-arithmetic` |
| `loader --fault file_truncated` | `final-range-complete` |
| `loader --fault sha_mismatch` | `sha-vs-pin` |
| `loader --fault q6k_extra_layer` | `q6k-layer-set` |
| `loader --fault expert_axis_tampered` | `expert-slices-addressable` |
| `loader --fault unattributed_width` | `width-delta-attributed` |
| `loader --fault run_past_file` | `staging-plan-matches-ruler` |
| `loader --fault draft_staged_anyway` | `draft-requantize-not-wired` |
| `loader --fault budgets_conflated` | `budget-lines-separate` |
| `layout --fault drop_a_slab_tensor` | `slab-coverage` |
| `layout --fault slab_overlap` | `no-overlap` |
| `layout --fault run_length_short` | `run-coverage` |
| `layout --fault read_class_mis_tag` | `read-class-coverage` |
| `layout --fault expert_stride_fused` | `expert-slices` |
| `layout --fault policy_width_swap` | `policy-a-vs-quant_scenarios` |
| `layout --fault deployment_bound_claimed_measured` | the gate's own bound-vs-verdict rule |
| `layout --align 4096 --fault pad_unaccounted` | `slab-coverage` |
| `artifact --fault short_parse` / `add_draft_tensor` | `record-count` |
| `artifact --fault rename_gdn_tensor` / `hf_spelling_in_gguf` | `name-set` |
| `artifact --fault down_proj_group_width` / `lm_head_width` | `width-census` |
| `artifact --fault stale_q5k_bpw` | `row-bytes-vs-ruler` |
| `artifact --fault offset_shift_1` | `data-section-coverage` |
| `artifact --fault draft_from_gguf` | `draft-block` |

Two of those need a word. **`pad_unaccounted` is vacuous at the file's own alignment**: `alignment=32`
and every Q4_K/Q5_K/Q8_0 row is already a multiple of 32, so total padding is 0 B and unaccounted padding
cannot exist. The gate proves the padding really is 0 B and then runs the injection at `--align 4096`,
which is the alignment a device allocator will actually use — that is where the loss shows up, and saying
so is the difference between a witness and a formality. **`deployment_bound_claimed_measured` must exit
0**, as above.

## What `loader.py` hands a kernel lane

```python
from loader import Loader
L = Loader()                       # header only; opens no data-section byte unless asked
h = L.handle("blk.0.ffn_gate_exps.weight")
#   {name, slab, slab_off, length, file_off, type_name, family, read_class, policy_type,
#    n_elts, dims, source}
r = L.expert_range("blk.0.ffn_gate_exps.weight", 7)   # one expert of 256, a byte range
```

`slab`/`slab_off` come from `layout.py`'s placement, `file_off` from the record, `read_class` from
`layout.py`'s coverage closure, so a kernel lane knows whether it is `read_in_full`, `resident_pinned`
or `per_token_gathered` and can refuse a width it was not written for (`type_name`). The expert axis is
named, not guessed: gguf puts the expert axis last (`dims[-1]`) and safetensors first (`dims[0]`), which is
`AGENTS.md §Predictions to falsify early in this project` item 3, and the reason `layout.py`'s `expert-slices`
closure — which tiles a row by `row/E` and cannot see the axis — needs a byte-for-byte cross-check rather
than a stride formula. `expert_range` therefore compares against `layout.expert_slice` before returning
anything, and refuses an index outside `[0, E)` **by name**.

### Stubbed: the two paths that raise instead of guessing

Two paths are deliberately **not** built, and they raise rather than guess:

* `stage_draft()` raises `NotWired("draft-requantizer-absent")`. The draft block is 19 BF16 safetensors
  tensors, 1,689,281,536 B; policy A wants 503,143,072 B of it. Nothing in this repo converts
  BF16-safetensors to ggml block-quantized blocks, and the gap — 1,186,138,464 B, 1.1047 GiB — is the
  price of writing one. A stub that returned plausible bytes would put unquantized draft weights on the
  device and a gate that measured them would look green; `draft_staged_anyway` proves the refusal fires.
* `device_slabs()` raises `NotWired("device-allocator-absent")`. The recipe (`hipMalloc` + bulk copy,
  464 runs) is in `pitfalls/10-device-memory.md`, the allocator is not written, and the **allocation count
  has never been probed on this box** — `quantization/residency-layout.md §7` item 2 is still open.

## Not established

Read this before building on the directory. None of it is a gate failure; all of it is a claim this
directory does not make.

* **No GPU work in the gate itself.** The gate does no allocation, copy, kernel or residency
  measurement; `engine/`'s only device path is `stage_run.py`'s `--device` branch, which refuses by
  name without the flag, owns no lock, and whose first hardware run is still pending (its first
  attempt was invalidated by `devices/evidence/device-window-2026-10-26c.md` §2; the replacement
  command is in `engine/HANDOFF.md`). The gate never invokes the stream lock and prints no rate
  figures, and it checks that nobody has added either.
* **The data section has been read — by the probe arm, and it crashed first.** This bullet used to read
  "No data-section bytes were read in this certification run", which was true when it was written: the
  arm had never executed and the gate printed `SKIPPED`, never `PASS`. It was run on 2026-10-26 with the
  audit selftest token earned by
  `tools/audit_published_quant.py --selftest` (whose grant appends a line to
  `quantization/evidence/audit-ordering-2026-10-26.log`), and the first execution died with
  `AttributeError: 'TensorRecord' object has no attribute 'slab_off'` — a tensor's slab coordinate lives
  on its *placement* (`layout.py §Placement.__slots__`), not on the record. Fixed by resolving
  name → (slab_off, row) through the slabs. It now reports
  `staging probe: marker selftest passed 2026-10-25T23:20:50Z; 464 runs planned, 90 sampled, 180 preads,
  92160 B read, 0 short`, and the gate carries `probe-reads-real-bytes PASS n=180` at
  `RESULT PASS checks=28 findings=0 arms=41`. Bytes are read and **counted, never interpreted**, and no
  timing is taken. The certification command is therefore
  `engine/gate.py --probe --keep-evidence engine/evidence/gate-run-2026-10-26.log`; without `--probe` the
  arm is `SKIPPED` and the count is 27, which is an honest state, not a failure.
* **What the probe still does not establish.** It reads a 512 B window at each end of 90 sampled runs —
  not the 464 — and it never *executes* the staging plan, so `read_file_range`'s arithmetic is now
  witnessed against real bytes at the sampled positions while the fault arms remain the only witness for
  its refusals. Real block *sizes* are validated against the real file by the file-size identity
  (residue 0 B), not only by a block table.
* **The staging plan is executed on the CPU side; the device read-back has never run on
  hardware.** 464 runs / 23,812,820,480 B are now staged and dual-digested by
  `engine/stage_run.py --check-only` (`RESULT PASS mode=check-only checks=34 findings=0
  controls=8 fired=8 device=off`; evidence `specification/evidence/stage-run-check-only-*.log`),
  and `--device` now writes a per-run `READBACK idx=... readback_sha256=...` record with a
  plan-derived denominator plus the two device controls — but **the real-hip run of that is the
  next clean window's** (window-c §2 replaced the first attempt's hollow PASS). What staging
  actually costs at peak is therefore still unconfirmed, and so is the allocation-*count* question
  §7 item 2 was written for. And the shape that would be executed is *not* the shape §7 item 2
  proposes to probe: this plan is
  `weights.experts` 19,568,525,312 B in 120 runs, `weights.tables`
  957,521,920 B in 2, `weights.attention` 1,359,216,640 B in 90, `weights.shared` 133,693,440 B in 120,
  `weights.exact` 104,581,632 B in 131 runs averaging 0.76 MiB, and the draft as a single 1,689,281,536 B
  run — lumpy, not 712 equal ~28 MiB allocations. Whoever opens that window should probe **this**
  distribution; `layout.py` prints it, so the probe costs no new arithmetic to specify.
* **Nothing about numerics.** No logits, no bands, no reference dumps, no chaos floor. A correctly
  addressed Q4_K block is not a correctly computing MoE block; `speccheck/` remains the oracle and the
  MoE block itself is `prototype/moe_gather/`'s job, gated separately.
* **The deployment budget is still unmeasured.** Fixing the *reporting* of that fact is not the same as
  measuring it: the 48 GiB single-`hipMalloc` probe (D-02 follow-up, `residency-layout.md §7` item 1) is
  still queued, and until it runs no fit claim on that budget is lawful from anywhere, including here.
* **`engine/layout.py` prices state from `tools/residency_budget.py`; it does not yet model** the 4-bit
  gather's working set or any per-layer scratch a real MoE kernel will want.

## What was changed in the parked pair, and why

`artifact.py` and `layout.py` were parked as a *pair with both selftests green*. They were not: the
selftest exited 1 with `KeyError: 'draft_kv'`, and the D-11 conflation described above was live. Four
edits, each the smallest one that makes the pair certifiable — no rewrite:

1. `layout.py::_allowance_modelled_state` — read `b.kv_bytes` and `b.draft_kv_bytes` separately instead
   of a folded `draft_kv` key that `Budget.row()` never returns. A missing byte total is worse than a
   different one, so the fold was the bug, not the missing key.
2. `layout.py::BUDGETS`/`verdict` — the D-11 fix described above, plus `--fault`, so it is citable.
3. `artifact.py::ArtifactError.__init__` — documented as carrying `.tag` and never set it; three lines
   so a refusal names its own check, which is what makes the nine artifact injections assertable.
4. `artifact.py::main` / `layout.py::main` — a `--fault` flag each, and `layout.py`'s `CLAIM` headers and
   trailers, so every refusal is citable as a command line and every claim block can be re-parsed.

## Reading order

`quantization/README.md` (policy A, the widths) → `quantization/residency-layout.md` (the layout this
module re-derives) → `DECISIONS.md` D-01, D-01a, D-10, D-11, D-15, D-17 → `engine/artifact.py` (identify)
→ `engine/layout.py` (place and price) → `engine/loader.py` (bind bytes to placement) → this file.
Traps that bind every file here: `pitfalls/40-gates-and-admissions.md`, `pitfalls/10-device-memory.md`,
`pitfalls/30-numerics.md`.
