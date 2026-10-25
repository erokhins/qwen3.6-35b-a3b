# 50 — harness, shell and build traps

> **Provenance.** Copied and condensed on 2026-10-26 from the 0.8B project's traps index
> (paths relative to `../qwen3.5-0.8b/`),
> `prototype/*/build.sh`, `engine/build.sh`, `tools/gpu_lock.sh`, `tools/longctx_check.sh` and
> `prototype/attn_wide/*` harnesses.

## One argv slot caps at 131,072 bytes, and `getconf ARG_MAX` is not the limit you hit

`MAX_ARG_STRLEN` bounds a **single** argv field. A 32,768-token prompt passed as `--ids "$(…)"` is 191,588 B
in one field, so `exec` fails with `E2BIG` → **shell exit 126, "Argument list too long", in 0 seconds**, before
your binary is reached. It presents exactly like "the engine crashes at 32k". 16,384 ids (95,796 B) passes, so
the bug bites only the widest case a check exists to test — and the ~0 s wall time is the tell that nothing ran.

Correct, twice over: any tool that must ingest a long list takes it **by file**, and the harness asserts the
field count it wrote (`awk -F',' '{print NF}'`), not the nominal width in the filename (fixture builders round,
so a 32,000-token fixture is legitimately 31,825).

Companion: a `--selftest` that writes fixtures into the directory a real run reads from **silently feeds the
real run the fixture input** — 1,024 ids to a run labelled 8,192, producing three reps of a plausible fake table.

## A comma-separated file parser and a space-joined fixture

`--ids-file` splitting on commas means a space-joined fixture reads as **one token** — and the run *succeeds*:
rc 0, every echo present, `RESULT` printed, wall time ~1 s instead of ~4 s. A whole timing window came out as
"23 ms at 16,384 lanes"; only a `form=` canary aborted it, because a one-token prefill never prints the line the
canary greps. Write fixtures comma-separated and assert `PREFILL tokens=` against the count read off the file.

## A poll loop that `pgrep -f`s a script named in its own command line matches itself

…and therefore never breaks. Wait on a sentinel the job writes (a log line, a `touch` marker) instead.

## Self-locking gate drivers deadlock silently

A driver that acquires the GPU lock and is called by a caller already holding it parks with **no CPU work, no
GPU work, no message, no exit** — which looks like a hung box. One such harness sat 30 minutes and died on a tool
timeout; afterwards `gpu_lock.sh status` read `STALE (holder pid dead)`. Check before acquiring anything:
`grep -c gpu_lock <script>`. Hold the lock only around scripts that do not self-acquire, and release before
invoking one that does.

Worse than the hang: the deadlocked attempt still printed a **plausible table** (three reps at 4,096 lanes
reading 369.7 / 589.0 / 1208.9 ms — a median that "discovered" an 18 % regression that did not exist). A median
across a 3.3× spread describes nothing: print the spread, and re-take a window whose spread is far outside the
box's normal ~2 % for that workload.

This repo ships `tools/gpu_lock.sh` unchanged and it locks **`/tmp/pi_gpu.lock`, machine-wide on purpose** —
so a worker here and a worker in the sibling project exclude each other correctly. Do not make it repo-local.

## `n + ($0 ~ /a/ && $0 ~ /b/)` adds nothing in this box's awk

The boolean does not coerce in that position; the total silently stays 0, and a counter that can only be 0
cannot fail. Count with an `if` and an explicit `n++`.

## Two arms that differ in anything but the thing under test price the difference

Recorded in full in `20-rates-and-occupancy.md` (launch population) and `40-gates-and-admissions.md`
(required flags). The cheap version: before believing any kernel A/B ratio, read workgroups per launch on both
sides, and assert a behavioural canary that the change is required to move.

## Device faults present as hangs, not crashes

A staged launch plan has **read** pointers as well as write pointers. If only the write half is populated, the
row dereferences a null base while the host sits in a stream sync busy-polling a faulted queue: ~10 cores busy,
no output, no exit, for as long as nobody looks at the dump. The dump names both halves (`kernel: X(...)` and
`faulting addr = …`), so the address identifies the row. Consequence for planning: **a path that only exists for
rows > 1 is invisible to every one-row gate the tree has.**

## Build and toolchain facts that are not optional

- HIP builds need **`-Wl,-rpath,/opt/rocm/lib`**; builds use `-O3`; **never time an `-O0` build**.
- The `.venv` torch on this box is **CPU-only**. CUDA-specific bf16/bitcast intrinsics differ on ROCm.
- `__shfl_down_sync`'s mask is **64-bit** here (a 32-bit mask silently drops half the butterfly).
- `hipcc` may not be bit-reproducible on this box, so "two builds" ≠ "the change is present" — see the canary
  rule in `00-measurement.md`.
- Grep for the marker the tool prints first: a baseline line printed *after* the device `RESULT` line will
  swallow it if you pipe through `tail -1`. Grep for `RESULT`, not for `ms`.
- **Piping a gate through `tail` swallows its exit code**, so `gate --strict | tail -2 && git commit` commits a
  red gate: the pipeline's status is `tail`'s, which is 0 even when the gate printed `findings: 1`. This bit
  the commit that *added* the label rule below — the checker had gone red on the very doc describing the rule.
  Correct: run the gate un-piped and check `$?`, or `set -o pipefail`, or redirect to a file and read it.
  Same family as the `RESULT` line above, one level up: a filter that can hide the verdict cannot gate on it.
- A `.gitignore` line like `prototype/**/<op>` (meant for the built binary) matches the op **directory** too, so
  the whole `prototype/<op>/` sits untracked and only `README.md` gets committed. Ignore the binary by exact path
  `prototype/<op>/<op>` and check `git status --ignored` after adding an op.
- `reference_dump.py --verify` once compared every file in a fixture directory, so any fixture that had been
  **priced** (an allowance JSON written beside the pins) could never reproduce — it exited 1 on 826 byte-identical
  tensors. A carried file is **named and skipped**, never silently ignored.

## Parsing a container format

- **A GGUF `BOOL` metadata value is ONE byte, not eight** (gguf-py types it as `np.bool_`). Reading it as
  `int64` desyncs the whole metadata walk by seven bytes per bool, and the symptom is not "bad bool" — the
  next string length comes back as ~8.8e18 and the file looks corrupt. Cost two runs and one fetched range
  to find. Two general lessons from it: (a) when hand-writing a binary parser, take the widths from the
  spec or from the reference implementation's reader, not from what the type name suggests; (b) make the
  parser **fail loudly on desync** — `tools/gguf_header_types.py` refuses to print a partial census and
  grows its fetch range instead, because a truncated width-policy table reads exactly like a complete one.
- **A header-only read of a 19 GiB file is the right instrument for "what quantization is this?"** A
  K-quant is a policy over tensors, so only the per-tensor types in the file's own tensor-info array
  answer it; prose in quantize tools and model cards describes intent, not the produced artifact. Range
  request, parse, stop (`tools/gguf_header_types.py`).

## Never print chat-template special tokens literally

In replies, commit messages, gate output, logs or docs. Every agent on this box is served by a checkpoint whose
stop list is its own eos id, so a literal end-of-turn token **ends that message where it stands**: in prose the
tail is silently dropped; inside a tool call the call is lost and the session restarts on the same call, looping.
Refer to these tokens by id. Two agents were cut off this way in the sibling project, one of them a worker brief
that lost its second half. Same rule as an engine requirement: a prompt fixture legitimately *contains* the
end-of-turn id, so a stop-token check is generation-phase only and must never be applied while feeding a prompt.

## A live worker is not a reachable worker

`worker_lanes.sh` said **ALIVE**, `status.json` said **`running`**, and both were true while the worker could
not hear a word I said. Observed 2026-10-26 (W14, a CPU-only numerical derivation): one script
(`tmp/scratch_bound2.py`, a closed-form bound wrapped in an O(N) explicit summation, called from inside a
1,281-point sweep — about 10¹⁰ interpreted iterations) held ~100 % CPU for **48 minutes**, produced **no**
declarable file, and sat behind **two steers that were never read**. It was not wedged; it would eventually
have finished. The failure was that I could not reach it, and my usual liveness checks could not see that.

**The discriminator is not elapsed time.** A recurrence walked to `N = 262,144` is legitimately minutes of
CPU, and "it's been a while" is not evidence. Three things are:

- **the actual command**, from `ps -o pid,etime,time,args` — `ETIME` ≈ `TIME` means one CPU-bound call, not
  a hang between calls;
- **whether the declared deliverables exist on disk.** A worker that has been "thinking" for 45 minutes with
  an empty output directory is not close to done, it is lost somewhere;
- **the steering state string.** `"state": "queued"` means the message exists and the worker has not seen it;
  a worker blocked in an in-flight tool call **cannot consume steering at all**. `queued` → `delivered` is
  the only proof an instruction arrived.

**Recovery when there is no command controller.** `command.status` / `command.cancel` answered *"No command
controller is available for this child"* for an async worker, so the precise lever is the PID: `kill <pid>`
of the runaway script makes the bash call return non-zero, the worker resumes, and the queued steer flips to
`delivered` in the same minute. Verified in that order: `48:38` → killed → `state: running`, steering
`delivered`.

**Kill by PID, never by pattern.** The same `ps` listing contained halogen's `serve_api.py` (the inference
server, 22.6 GiB resident) and an unrelated project's venv python. A `pkill -f python3` or a kill-by-name
there is an outage, not a recovery — and it is exactly the kind of command that succeeds and then explains
itself afterwards. Same rule as the self-matching `pgrep` above: name the process, don't describe it.
