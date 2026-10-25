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
| subagent W18 `992083de` | **stopped cleanly at a safe checkpoint** (steered, `subagent.steer.delivered`, ended its turn) | landed as draft commit `13040de` — `prototype/moe_gather/` with `HANDOFF.md`. Device half never compiled, `gate.py`/`gate`/`oracle_arm.py`/`results.md` do not exist, so the lane is **not** a deliverable. No device work was run during the pause. |
| main session | `tensorfold` | idle, nothing in flight |

**Nothing of mine is running now.** The only device-adjacent mover left is halogen itself (up, 22.62 GiB
resident), and stopping it is the operator's call — `tools/halogen_gate.sh down` with the GPU lock
declared, never a bare `stop.sh`.

If steering had not delivered, the recorded fallback was: kill worker PID `347676`, the queued steer flips
to delivered, then resume with "continue". It delivered, so the fallback stayed unused — but the pattern is
the same one that recovered W14, and it is worth knowing the two discriminator signals: whether the
worker is blocked inside one long tool call, and whether `events.jsonl` shows `subagent.steer.delivered`.

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

0. **Verify the safetensors header-offset claim, before anything that trusts prediction 3.**
   `prototype/moe_gather/HANDOFF.md` §9 asserts `prototype/moe_router/work/probe15.py` memmapped at
   `header_json_len + data_offsets[0]` instead of `8 + header_json_len + data_offsets[0]` — four bf16
   elements off — which would mean the router lane's recorded `P3 UNRESOLVED` was an artefact of its own
   probe, not a property of the dump, and that pinned `layer0.expert_raw` in fact reproduces bit-exactly
   under the corrected offset. **This is unverified by me on purpose**: it overturns a committed
   conclusion and deserves a proper witness run, not one squeezed in before a maintenance window. CPU-only,
   so runnable any time. If confirmed, `prototype/moe_router/results.md` gets a dated corrigendum
   (corrections are appended, never silently patched) and P3 becomes resolvable.
1. Verify W17's four ports independently (witness commands + the negative controls it claims), then
   commit with PROVENANCE rows.
2. Resume `prototype/moe_gather/` from `HANDOFF.md` §7 first (the harness defects — a `RESULT PASS` that
   does not require the saboteur arms is not a gate) and then §8 order.
3. **`quantization/evidence/audit-ordering-2026-10-26.log` carries an unexplained 11-line addition**
   (+11, no deletions). Evidence files are historical records; a bare append is neither a correction
   nor a dated corrigendum. Attribute it before committing anything, and decide whether to keep it.
4. D-15 still says the next **lone** worker goes to `halogen`; `speccheck/results.md` is no longer
   that job since W16 finished it.
