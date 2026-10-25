# Paused for operator maintenance — 2026-10-26 (box clock 2026-056)

Operator asked for a **safe pause**: no new work started, no new workers launched. This file records
the exact state so the next turn resumes without guessing. Everything here was read from the box at
pause time, not remembered.

## Locks and processes at pause

| thing | state | consequence |
|---|---|---|
| `tools/gpu_lock.sh status` | **free** | no device measurement in progress; none is owed right now (the window was taken and recorded in `devices/evidence/device-window-2026-10-26.md`) |
| `/tmp/pi_halogen.lock` | **not held** | no worker is claiming the local stream |
| halogen serve (`127.0.0.1:8731/health`) | **UP** | the container is resident and holds **22.62 GiB** of the device-visible pool. Left up deliberately — stopping it is a user-visible action and the operator did not ask for it. **If maintenance touches the GPU, this must come down first, and only via `tools/halogen_gate.sh down` with the GPU lock declared** — never a bare `stop.sh` |
| subagent W18 `992083de` | **ALIVE** (`evts=661`, `age=74 s`, `pending-message+tf-stream`) | `tensorfold` worker building `prototype/moe_gather/`. Untouched: no pending supervisor request, and steering a live stream spends the fifth-slot margin. **It may write files and may fire a correctness-only device gate under `gpu_lock`.** |
| main session | `tensorfold` | idle, nothing in flight |

**If maintenance needs a quiet GPU:** stopping W18 is the only device-adjacent mover left besides
halogen, and it is the operator's call, not an inference of mine. W18's work is on disk, so stopping
it is recoverable by resume rather than a loss.

## Committed before pausing

- `4901b84` — `speccheck/results.md` (W16), supervisor-verified, preamble corrected to separate
  quoted from derived figures.
- this commit — W19's `ssm_a` compounding bound (`specification/evidence/`) plus
  `tools/redact_template_tokens.py`.

## Landed and VERIFIED (committed here)

- **W19 `1da144b5`** — `specification/evidence/ssm-a-ulp-compounding-2026-10-26.md` + `ssm_a_ulp_bound.py`.
  Supervisor re-ran the selftest: **12/12 PASS**, including `zero` returning exactly `0.0` (not
  "below epsilon"), `thousand-ulp` at **1000.12×** the one-ulp figure (monotonicity in δ is the
  property under test), and a **tripwire proving the O(N) summation refuses `N=1025`** — the exact
  runaway that killed the first attempt at this task. Bound `B(N,α)` sup is **ρ, N-independent**
  (`1.189e-07` at N = 4,096 / 32,768 / 262,144), at `α ≈ 1/N`.
- **`tools/redact_template_tokens.py`** — new, selftest **6/6 PASS** including a negative control and
  an idempotence arm. Exists because W17's final report was **cut off mid-sentence by a literal
  template token** and the artifact on disk holds only the fragment: a chat renderer's output cannot
  be logged unprotected. `chat_render.py` output must be piped through it.

## Landed but UNVERIFIED — deliberately left out of the commit

W17's four ports sit untracked in `tools/` and **have not been checked**: `weights_report.py`,
`chat_render.py`, `check_dump_reproduction.py`, `dead_after_return.py`. Not committed because
committing unverified code is how a future session ends up trusting a tool that reads the wrong
tensor names — the failure this repo has already been bitten by. `tools/PROVENANCE.md` was **not**
updated by W17 either, so these ports are currently unprovenanced as well as unverified.

**Do not run `chat_render.py` without the redactor.** Its render includes an empty think block
(a genuine finding for fixture work, still open: the sibling's `enable_thinking=False` convention
may not apply to this template).

## Open when the pause ends

1. Verify W17's four ports independently (witness commands + the negative controls it claims), then
   commit with PROVENANCE rows.
2. Receive/verify W18's `prototype/moe_gather/` when it reports.
3. **`quantization/evidence/audit-ordering-2026-10-26.log` carries an unexplained 11-line addition**
   (+11, no deletions). Evidence files are historical records; a bare append is neither a correction
   nor a dated corrigendum. Attribute it before committing anything, and decide whether to keep it.
4. D-15 still says the next **lone** worker goes to `halogen`; `speccheck/results.md` is no longer
   that job since W16 finished it.
