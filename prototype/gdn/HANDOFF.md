# prototype/gdn — HANDOFF (living file, written early on purpose)

Written ~minute 38 of the lane. This file is updated as things execute; it is not a summary
written at the end, because the previous lane at this slice died at the 90-minute deadline with
three headers and no executable verdict. Checkpoint log is at the bottom.

## Inventory of the dead lane's three headers (step 0)

| file | verdict | why |
|---|---|---|
| `gdn_kernels.h` | KEPT, with four fixes | the arithmetic transcription is correct and matches `speccheck/gdn_ref.py` `scan_chunked`/`gated_norm`/`out_proj` line for line; the geometry constants match the pinned manifest (h_k=16, h_v=32, d_k=d_v=128, C=64); its dump-orientation comments match `data/dumps/2026-10-26-gdn` |
| `gdn_case.h` | KEPT, with renames | the item decomposition and the derived per-stage item denominators are right; its stage NAMES were internal item names (`enter`,`cum`,`pair`,…) and not the oracle's, so the emitted-stage table is rewritten to the seven `gdn_ref` stage names |
| `gdn_arm.h` | REWRITTEN | it did not compile: `GdnArgs` had no `read_field_dummy`; it called `gdn_fault_id`, `gdn_field_id`, `gdn_field_name`, `gdn_echo_hash`, `gdn_bind`, `gdn_case_run_device`, `c->field_bad_total` — none of which exist in any of the three files. There was no driver, no build, no gate, so nothing in it had ever been executed |

The four fixes to `gdn_kernels.h`, each of which is a real defect and not cosmetics:

1. `gdn_item_vnew` and `gdn_item_core` read the recurrent state as `S + col*GDN_D_V` and dotted
   over `d_k`. In the `[d_k, d_v]` layout that is `S^T`, i.e. the oracle's own
   `state_stored_transposed` injection. It is invisible in slice 1 only because the initial state
   is zero — which is exactly the kind of latent wrongness that survives into slice 2. Fixed to a
   strided read of `S[k, col]`.
2. §04.7's product was not rounded to the storage dtype. `gated_norm` computes
   `weight(bf16) * x.to(bf16)` → the PRODUCT is bf16, then it is multiplied by the f32 gate. The
   header multiplied in f32. Fixed to `bf16_rt(w * bf16_rt(x̂))`.
3. The q_hat/k_hat STAGE was never emitted (only per-head scratch). The oracle's `q_hat`/`k_hat`
   stages are `[B,h_v,T,d]` f32 in the scan dtype; the header had no output for them, so two of
   the seven banded stages could not have been compared at all.
4. The in-kernel read-sabotage ladder (`gdn_sabotage_*`, `field_bad`) is removed: the perturbation
   belongs in the driver, in the bytes handed to the arm, the way `gdn_ref.py`'s own sabotage
   witnesses perturb pinned tensors in memory after `read_dump_tensor` has hashed them. It also
   lets the driver derive the expected moved-count exactly instead of counting it inside the arm.

## What runs

    ./build.sh                                   g++ host twin + hipcc device TU (CPU work, no device touched)
    ./build.sh --broken                          ALSO the bf16-truncation control build (the witness)
    ./gate.py                                    the whole gate; CPU only, no device, no lock, no timer
    ./gate.py --only three-arm|band|exact|controls|witness|oracle
    ./bf16_vectors.py                            derives the self-test's bf16 table from torch

## What passes (executed 2026-10-26; full log in `evidence/gdn-slice1-gate-2026-10-26.log`)

* `SELFTEST PASS` on BOTH compiled arms, 21 checks each.
* Compiled pair at zero tolerance: `out.bin` 35,143,680 B and `echo.bin` 3,163,136 B, `ndiff_bytes=0`.
  The hip arm announced `arm=hip_hostpath hip_calls=0`; gate refuses it otherwise.
* Band verdicts, `band_check_row` only, against the pinned dump, `band_digest = 27a68c7e82c9aa05…`:
  **12 of 14 rows ACCEPTED.** `k_hat, scan_out, core_attn, state, normed, y` ACCEPTED on both
  blocks. **`q_hat` REJECTED on both blocks: `max_abs = 2.980e-08 > licensed 2.900e-08`** — one f32
  ULP over a band whose own `mean_rel` (7.0e-08 vs licensed 1.16e-07) and `1-cos` (2.4e-14 vs
  1.6e-13) sit comfortably inside. See "open" below; the band was NOT touched.
* Exact read/decode checks: **16 of 16 BIT_IDENTICAL** (7 echoed fields per block at zero tolerance,
  plus the 8,388,608-word `out_weight` positional fingerprint per block).
* Controls: `stub` 14/14 rows red on the derived count, `crossed` 14/14 red on the derived count,
  a 2-arm agreement REFUSED, and the `--broken` control build (bf16 truncation) FIREs the
  self-test and turns 8 of 14 stage rows red — every new check was witnessed failing.
* The oracle CLI arm: `speccheck/gdn_ref.py --cross-form` exited 0 with `CROSS-FORM PASS` and the
  same band digest, in this lane, in this window (`evidence/gdn-cross-form-in-lane-2026-10-26.txt`).
* `pad_nonzero=0` and `state_nonzero=0` on every case: §04b.2's zero padding and slice 1's zero
  initial state are counted facts of the run, not assumptions.

## The `norm_weight` witness — what it actually showed (`gate.py --only witness`)

The accepted case reproduces the oracle's own number **exactly**: blending this kernel's honest
`normed` toward its `norm_weight_f32` `normed` at 1/2**2 is ACCEPTED by `band_check_row` with
**190 of 131,072 stored elements still changed** (meanrel 1.944e-06 against a licensed 3.941e-06,
maxabs 3.906e-03 against 3.125e-02) — the same row `speccheck/evidence/gdn-cross-form-2026-10-26.txt`
records. So the hole is real and it is reproduced here rather than cited.

But the brief's expected pairing does **not exist** at this field, and that is a finding, not a
shortfall in the check. Perturbing `norm_weight` in memory (bf16 LSB flip on 1 word in 2**rung):
at rung 7 — a single word out of 128 — the exact check goes RED (1 word moved, exactly the derived
count) and `band_check_row` on `normed` also REJECTS (meanrel 9.324e-06 > 3.941e-06). Every rung
0..7 is RED on both sides. A `normed` band cannot hide a *byte* change to the weight because one
weight word owns 1,024 output elements; what it hides is the *width* defect at k=2, which changes
190 elements and whose weight bytes are honest. The two checks are therefore **complementary, not
nested**: byte-exactness is strictly more sensitive on the read path (1 word of 128), the band is
the only thing that sees the arithmetic defect — and on the accepted case the exact check is
GREEN, which is reported rather than hidden.

## Open (the only thing standing between this lane and a green gate)

`q_hat`'s `max_abs` is one f32 ULP over the licensed value. Suspected cause, measured not assumed:
§04.4's l2norm takes a sum of 128 squares, and torch's `.sum(-1)` is a blocked reduction while
`gdn_kernels.h` declares an ascending single-accumulator reduction (the declaration that makes the
two compiled arms agree bit for bit). `1/sqrt` vs `torch.rsqrt` was already removed as a cause:
switching to a double-precision reciprocal square root took max_abs from 3.725e-08 to 2.980e-08 and
no further. The three ways out are all someone else's call, and none of them is tightening or
widening a band from here:
  1. reproduce torch's reduction order for that one sum (and then this arm's spread has to be
     re-measured, because the arm's spelling is what the band's `max_abs` describes);
  2. re-run the oracle's `--floor-probe` with a non-torch-reduction arm so the band has a record for
     this class of implementation (only `--floor-probe` writes the band; this lane never ran it);
  3. accept `q_hat` as a documented out-of-band stage for slice 1 and revisit it when the engine's
     real `l2norm` kernel exists.

## Checkpoint log (continued)

- min 68 — `gate.py` in progress.
- min 78 — first verdict: 12/14 banded rows ACCEPTED, `q_hat` one ULP over. ASan build used to find
  a heap overflow (the scratch formula undercounted `kcd` by 262,144 words — a silent overrun that
  no band would ever have shown).
- min 85 — exact checks 16/16; controls all firing; the accepted `norm_weight` case reproduced at
  exactly the oracle's recorded numbers; oracle CLI arm `CROSS-FORM PASS`.
- min 90 — evidence copied to `evidence/`, 1.2 GB of regenerable `work/*.bin` deleted, halogen
  stream lock released. Nothing staged; nothing outside `prototype/gdn/` touched.

## Checkpoint log

- min 38 — inventory done; band digest read through `load_band()` = `27a68c7e82c9aa05…`, matches
  the authoritative value; the seven banded stages confirmed present in the band file.
- min 52 — `build.sh` builds both arms clean (`g++ -O2 -ffp-contract=off` host twin, `hipcc
  -x hip --offload-arch=gfx1151 -O3` device TU). No device touched; compile is CPU work.
- min 66 — `--selftest` on both arms: **PASS, 21 checks each**. Getting there found four real
  defects, all in the dead lane's never-executed code:
  * `gdn_bf16_of_f32` tested `((u>>23) & 0x1ff) == 0x1ff`, which can never fire for a **positive**
    float (u>>23 ≤ 0xff), so a positive signalling NaN was converted to **+inf**. Mask is `0xff`.
    This is the exact corruption §04b.3 exists to prevent, arriving through the cast.
  * the bf16 expectation table was hand-typed and four rows were wrong (65504 → 0x4780 not 0x477e;
    1e-8 → 0x322c not 0x3243; a NaN round-trip asserted identity where quieting is correct; the
    hand solve residual had a sign flip). `bf16_vectors.py` now DERIVES the table from torch and
    checks the C rule against torch over 200k random f32 patterns: **0 mismatches**.
  * `gdn_item_cum` did `(float)g_decay[i]` — an integer-to-float conversion of an f32 *bit image*
    instead of a bitcast. Now goes through `gdn_load`.
  * the emitted-stage table (`gdn_stage_name`) is asserted against the oracle's seven names inside
    the self-test, because a rename there silently detaches every band verdict from its stage.
- min 68 — `gate.py` (the driver) in progress. Nothing has produced a numerics verdict yet; that is
  the next thing.
