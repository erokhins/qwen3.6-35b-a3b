# Session status — 2026-10-26, after M2's device arm went green

Written by the main agent at the point where the gather vertical is delegated
(`c03f05a0`, halogen). Nothing here needs a device to read. A fresh session can
resume from this file plus `AGENTS.md` §GOAL and `engine/HANDOFF.md`.

## Where the ladder stands

| rung | state | evidence |
|---|---|---|
| M1 staging | **ESTABLISHED** | `devices/evidence/device-window-2026-10-26d.md` — 464 runs, 463 distinct device read-back digests, 463 `MATCH` + 1 named draft refusal, 10/10 controls fired |
| M2 block gate (CPU) | **QUALIFIED**, ceiling is honest | `engine/block_gate.py` — `checks=17 red=0 controls=11 fired=11`, two `block_output` rows `referent=absent` |
| M2 device arm (router + shared expert) | **GREEN** | window `f`: `RESULT mode=device checks=32 findings=0 controls=10 fired=10 device=on launches=7` |
| M2 gather vertical | in flight | lane `c03f05a0`; `engine/block_device.py` had **0** gather references before it |
| M2 GDN device slice | not started | deliberately hard-stopped, queued after gather |
| M3–M6 | not started | 40 blocks + `lm_head`, token stream, **≥ 48 t/s**, MTP |

**No timing exists anywhere in this repo's engine work.** Every device window is
correctness-only. D-22's bar (≥ 48 t/s = 90 % of llama.cpp's measured 53.4) has
never been approached, let alone measured.

## What was proved today, in one line each

- **D-11's deployment bound is measured, not assumed**: one process took 32.00 GiB
  GTT-backed staging with the engine resident and idle, so policy A's 19.7848 GiB
  fits with margin (`device-window-2026-10-26c.md`).
- **`q_hat` was a kernel defect, not a band problem** — the licence value was a fold of
  its own probe; fixed with a pairwise f32 tree, licence untouched (`b58a6e7`, D-23).
- **Prediction 4 is confirmed in the checkpoint's bytes**: HF casts renormalised top-k
  weights to the logits dtype *before* the expert multiply; `layer0.routed_sum` refuses
  the f32 carry (`ndiff 1326` vs `0`) (`890fb88`).
- **The `wpre` host/device difference is documented `exp()` scope, not codegen**: per-slot
  ULP `1,1,1,0,0,0,2,1`, max 2, inside a bound of 9 taken from §R5 *before* measuring,
  with a rejection control that reddens at 10 (`1a037f5`, `1d434dd`).
- **Our kernels run on the iGPU and agree with the pins**: `router_logits`, `topk_ids`,
  `topk_weights` `ACCEPTED ndiff=0` from the hipcc arm (`a2d23d2`, `1d434dd`).

## Four times the obvious wrong party was not the wrong party

Worth keeping as an epistemic note, because each of these would have produced a
confident wrong commit:

1. `q_hat` looked like a band too tight; the kernel's ascending f32 accumulator was wrong.
2. The `carried` one-liner in the GDN pad witness: applying the lane's fix as written
   would have reclassified all 8 non-carried cases — reading the caller was required.
3. `policy_a/topk_weights` nearly got filed as "unlicensed"; it was prediction 4 firing.
4. `wpre` between compilers nearly got deleted from a claimed-exactness field; that would
   have been a relaxation smuggled in as a fix.

The recurring move was **attribute before adjudicating**, and the recurring failure mode
was a verdict line that could print `PASS` without ever having counted anything.

## Two self-caused errors that became rules

- **Window `e` ran unlocked.** I chained `gpu_lock.sh acquire`, then `halogen_gate.sh down`
  piped into `tail -1`, then the run; a pipeline reports `tail`'s status, so the refusal was invisible, and
  separately I had left a halogen stream claim alive past its worker's death. Log kept as
  `specification/evidence/blockdevice-device-2026-10-26-VOID-no-lock.log`. Fix:
  `engine/work/window_f.sh` asserts preconditions as data, matches `gpu_lock status`
  against the expected tag (`status` prints `HELD: someone else` and still exits 0), and
  reads engine-down from `rocm-smi`, not from a message. Claims are now minted with a life
  past the lane deadline and refreshed, not released when remembered.
- **`--device` was a flag that parsed my own log file.** It printed device-sounding verdicts
  while allocating nothing. Renamed to `--staged-evidence` (a PRECONDITION), `--device` now
  refuses with `live-device-path-absent` unless a real device act happens, and
  `device-claims-come-from-device-only` guards it (`48701cb`, `890fb88`).

## The one thing waiting on the user

The pinned manifest contains **no layer-0 residual stream**, so the two
`stage=block_output` rows can never be licensed and M2's exit gate can never print `PASS`
— only `QUALIFIED`. Creating or replacing a pinned fixture is reserved to the user
(D-18). Options on the table, asked and unanswered:

- **(a) recommended** — authorize extending the dumper to emit layer-0 residual-in/out;
  M3 needs those records anyway, so the cost is shared.
- (b) accept a permanent `QUALIFIED` for M2.
- (c) name another authoritative referent.

This blocks M2's *wording*, not progress: gather, GDN, M3 and the numerics all proceed.

## Next actions, in order

1. Verify and commit the gather vertical (expect: two arms, gate-up stride/offset swap
   control shown reddening, read census with a non-zero witness, per-path `legs=`,
   `band_digest=27a68c7e82c9aa05`, and an explicit `ACCEPTED` vs plumbing-only count).
2. Delegate the GDN device slice as its own vertical.
3. Take a device window for the gather/GDN device legs — **only** with zero active workers,
   `gpu_lock` held with a declared tag, halogen down, main agent blocking in the foreground.
4. M3: 40 blocks + `lm_head`, pinned logits, greedy argmax exact.
5. Timings (M4/M5) only after 1–4, in a clean window, batched — one halogen-down window per
   session until `devices/evidence/halogen-restart.log` proves restarts are cheap.

## Housekeeping state

- `python3 tools/doc_ref_check.py --strict` → `findings: 0`.
- Working tree clean apart from untracked `engine/work/` (gitignored).
- Halogen UP; `gpu_lock` free; stream claim held for lane `c03f05a0` (pid `1010920`).
- Device is **card1** — a failed `cat` of a `card0` path once made me print fabricated
  `vram_delta_GiB=0,00`.
