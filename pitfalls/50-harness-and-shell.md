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

## CC Safety Net blocks are not suggestions — and they kill workers mid-run

This box runs a command safety net that **refuses** several ordinary shell forms. It is not advisory: a
worker that hits one dies mid-task and must be resumed, and whatever it had queued in that call is gone.
Rediscovered the hard way on 2026-10-26 when W17 died on a heredoc. The forms, the rule the net cites, and
the house workaround:

| blocked form | rule | do this instead |
|---|---|---|
| `python3 - <<PY`, any heredoc into an interpreter | `analysis.unsupported-heredoc` | **`write` the script to a file, then run the file.** Never pipe a heredoc to an interpreter. |
| `python3 -c "…"` containing a loop or comprehension | `analysis.dynamic-shell-source` | same: file, then run |
| `bash -c "…"` / any dynamically assembled command | `analysis.dynamic-shell-source` | literal command strings; put the logic in a script file |
| `rm -rf $DIR`, recursive-force on an expanding target | rm safety rule | literal paths, or don't delete |
| `git checkout -- <path>` (discard working-tree changes) | `git.checkout-double-dash` | `git stash push -- <paths>` — recoverable, which is the whole point |
| `pgrep -f <pattern>` that can match your own shell | (silent wrong answer, not a block) | name a PID |

Two things worth internalising. **The workaround is always the same shape**: the net objects to *interpreter
source it cannot inspect*, so anything with logic in it becomes a file — which incidentally makes it
reviewable, re-runnable and citable. And **a plain shell loop is fine** (`for f in …; do ./prog "$f"; done`);
the objector is inline interpreter source, not iteration.

Related, though not a block: `awk` here is **mawk 1.3.4**, which does not implement POSIX `{n}` intervals, so
a regex like `x{8}` silently matches nothing — and a parse returning zero records looks exactly like a clean
run (`pitfalls/40-gates-and-admissions.md`, empty-parse-PASS). Prove a regex matched something before
believing a zero.

**Supervisor duty:** brief workers on this list up front. It is cheaper to say "write files, don't heredoc"
than to resume a lane that burned twenty minutes and died on the last step.

## An order-free digest of values is blind to a uniform rescale

**Discovered 2026-10-26 in the `prototype/lm_head/` lane**, by trying to sabotage its own kernel and watching
the digest not move. An XOR of value bits — the obvious order-free digest, commutative and cheap — **cannot see
a uniform power-of-two rescale**: `d = 0.5` and `d = 1.0` over the same quants both give
`xorb = 0xFF400000` (measured, scratch `tmp/worker-lm-head/probe9_literals.cpp`). Every scale-multiplying bug in
that kernel would have walked straight past a gate whose value digest was XOR alone.

**The rule: state what each digest provably detects, and give it a witness that it moves.** XOR-of-bits detects
*multiplicity* — which values are present and how often — so it is a fine companion and a poor sole judge. The
lane added `vsum`, an integer sum of the value bits, which a uniform rescale must change. Two corollaries:

- A digest that survives a bug is not a weak test of that bug, it is **no test of it** — and the failure is
  invisible, because the digest still prints a plausible hex word. Ask "what transformation leaves this
  unchanged?" before trusting any summary, exactly as with `sum == target`.
- The float analogue has the same shape: an absolute-error check is blind to a systematic *relative* error, and
  a relative check can cancel a uniform scale. When a gate quotes a band, say which of the two it is.

## hipcc compiles the TU twice, so `atomicAdd` inside `__host__ __device__` is a hard error

**Measured 2026-10-26** by the same lane. A single-source header shared between `g++` and `hipcc` fails the
**host** pass with `reference to __device__ function 'atomicAdd' in __host__ __device__ function`, because
`hipcc` compiles the translation unit once for the host and once for the device. Guard the device-only call:

```c
#if defined(__HIP_DEVICE_COMPILE__)
    atomicAdd(...);
#endif
```

`__HIP_DEVICE_COMPILE__` is defined in the gfx1151 pass and not in the host pass (verified, scratch
`tmp/worker-lm-head/probe6_guard.hip`). General form: in a shared header, "compiles under hipcc" and "compiles
under g++" are two compilations of one text, so any name of a `__device__` intrinsic needs a guard — and the
error points at the host half, which reads like the host compiler is broken while the device half is fine.

## Piping a gate through `head` kills it mid-run and destroys the evidence it was writing

**Done by the main agent, 2026-10-26, not by a worker.** While verifying a lane I ran `./gate 2>&1 | grep -E
"^RESULT" | head -6`. `head` exits as soon as it has six lines, the pipe closes, `gate` gets `SIGPIPE` and dies
mid-run — and because **`gate` truncates its own evidence log at startup**, the file left on disk was a partial
from the killed process with no `RESULT` line in it. The worker's complete 911-check device log was gone. The
follow-up `grep '^RESULT' evidence/…log` printing *nothing* then looked like the lane had fabricated its verdict,
when the actual cause was my own pipe four commands earlier.

**Rules:**
- **Never put a consumer that exits early in front of a process that writes the evidence file.** Run it to
  completion (`./gate > /tmp/run.out 2>&1`), then grep the file. `head` exits early and sends `SIGPIPE`; `tail
  -n` reads to EOF and does not.
- **A tool that truncates its own log must write a start sentinel** — `RUN STARTED pid=… argv=…` before the
  work — so a truncated log is *identifiable as truncated*. Without it, "killed halfway", "never ran" and
  "lied about running" are all the same empty file, which is the same defect as a counter seeded at 0.
- **Recovering from this is only legitimate because it was a correctness gate.** I regenerated the log by
  running the gate again, and that is admissible for PASS/FAIL with no timing taken. It would **not** be
  admissible for a measurement: a re-run under live workers is not the run whose number you quoted, and the
  only honest output in that case is "the measurement is lost, it must be retaken in an exclusive window".
- And the reflex this cost: when evidence goes missing, **suspect your own last five commands before suspecting
  the author of the file.**

## Copied from the sibling engine (2026-10-26)

Harness, rig and build traps lifted out of `../qwen3.5-0.8b` on 2026-10-26, with their numbers and dates. All of
them are **method-level**: they are about how an instrument is written, so they bite here exactly as they bit
there. `U-n` ids map to `planning/port-inventory-2026-10-26.md` §9.10.

- **A `MB` typed where `MiB` was meant makes a rig silently heavier than the census it reproduces** (U-1). Their
  engine arm was **4.6 %** heavier because `lm_head` reads **509.552 MB = 485.947 MiB** and one side spelled it
  as MiB. The rig's own invariant abort caught it; the fix keeps the two spellings distinct and makes the refusal
  print both numbers. §4 of `../qwen3.5-0.8b/prototype/launch_price/S1-GATEZERO.md`. This repo has the same
  collision available in three places (GiB read per token, GGUF byte offsets, `n_ctx` lanes), so a bytes helper
  that prints both forms is cheaper than the audit.
- **Two units on visually identical lines is a 40× error** (U-2). The draft rig printed `ms` for block arms and
  **`µs/launch`** for its `empty` arm on lines that looked the same; "Read as ms it looks like an empty chain
  costing 1.82 ms — 40× the launch floor it exists to measure — and would have been quoted as a finding about
  launch cost." §"Three defects found while taking this" of
  `../qwen3.5-0.8b/prototype/mtp_draft/DRAFT-RESULT.md`. Rule: the unit is part of the line, so print it on every
  line, and make the reader's regex refuse a line whose unit it does not recognise.
- **A tool with no argument handling accepts any flag by silence** (U-3). `price_block.py` had *no* argument
  handling, yet the handover command passed it `--evidence DIR`; "The flag was accepted by silence and ignored, so
  the tool printed bands it had never compared to a number — the empty-parse failure class exactly." After the
  fix, its first refusal (`parsed arms [] lack block-k1/block-k2`) is what made a **second** bug visible: the
  runner's regex matched `pass 1` where the log prints `pass1`. "a refusal path is what makes a parser bug
  *visible* instead of silent." Same file, findings 1 and 3.
- **A runner that deletes its own first pass still looks green** (U-4). Each phase began with `rm -f "$dir"/*.txt`,
  so pass 2 erased pass 1's evidence before anyone read it, and the first failure message pointed at a file that
  no longer existed. Fix: clear only `$dir/$phase-*.txt`. §"Two instrument defects found while taking it" of
  `../qwen3.5-0.8b/prototype/mtp_delta/D3-RESULT.md`.
- **A record-count trailer must count what the checker counts** (U-5). `sweep.sh`'s `CELLS-PRINTED` trailer counted
  *distinct cells* while `fit_shape.py`'s count check compares *records*, so a complete **54-record** file refused
  itself. Fix: the trailer counts records. §1 of `../qwen3.5-0.8b/prototype/wave_shape/ITEM2-RESULT.md`. This is
  our own "a check that can produce no output cannot fail" rule seen from the other side: a count that measures a
  different population than the checker's also cannot fail — it can only refuse correct work.
- **Compute one verdict per case, once** (U-6). A kind-1 tally that silently inverted `chk.ok` printed
  `caught=0/8` over a fully green run; the fix is "one verdict per case, computed once".
  §4 of `../qwen3.5-0.8b/prototype/launch_price/S1-GATEZERO.md`.
- **A sweep or a negative test must load its own weights** (U-7). Both inherited the last layer's weight files from
  the main loop, so "the sweep would have ranked configs against the wrong matrix and the negatives were
  vacuous". §Traps of `../qwen3.5-0.8b/prototype/mlp_gemv/results.md`. Same class here: a fixture set derived from
  the artifact's own tensor records must be re-read per arm, or the must-catch set is inherited and vacuous.
- **A compile-time constant the kernel uses while the host sweeps a runtime value prints impossible numbers**
  (U-8). Their first `--rows` sweep printed **415–710 GB/s**, above the 256 GB/s physical peak, because the kernel
  used `ROWS_PER_BLOCK` at compile time while the host launched the runtime value, so `--rows 2` "covered only a
  quarter of the vocab rows — less work, so 'faster'". The permanent fix passes the value as an argument **and**
  compares each config's output against the reference instead of trusting the timer.
  `../qwen3.5-0.8b/prototype/lm_head_gemv/results.md` §Two traps.
- **A control that is not expressible on the rig must not be silently droppable** (U-9). An occupancy control
  enumerated a non-instantiated shape combination, exited 1 at its probe, and the caller's `len(...) >= 2` guard
  meant the de-confounder it existed to provide simply vanished — cost noted in the runner, "Recorded in the
  runner, not silently." `../qwen3.5-0.8b/prototype/mtp_delta/D3-RESULT.md`.
- **A flag whose premise does not hold must refuse, not reinterpret** (U-10). Two instances: `--geometry fat`
  refuses `--layout blob72`/`shipped72` because "the contiguity premise is checked, not assumed", and `--mode down`
  refuses `--mapping` rather than ignoring it, because "a flag that silently did nothing on half the shapes would
  make a null A/B indistinguishable from a mis-typed one". §4's table of
  `../qwen3.5-0.8b/prototype/mlp_insitu/D2-DIAGNOSIS.md`; §Q10 of `../qwen3.5-0.8b/prototype/mlp_gemv/results.md`.
- **A probe that races its own fixture produces a flaky-looking machine** (U-11). The concurrency probe initially
  released its spinner immediately after launching the detector, and results flipped between runs: "that was a
  race, not a flaky machine, and the protocol now releases only after the probe signals completion". §1 of
  `../qwen3.5-0.8b/prototype/prefetch_tier/P1-ARITHMETIC.md`.
- **The witness architecture worth copying whole for any launch-accounting rig** (U-12).
  `../qwen3.5-0.8b/prototype/launch_price/S1-GATEZERO.md` §3, all of it device-found, none of it guessed:
  each block writes `{read-sum, read-count, write-sum, write-count, launch tag}` into **its own slot**, so a slot
  left at the sentinel is a block that never ran; the read fill has a closed form `word(p) = K·(2p+1)` over 8-byte
  elements with `K` odd, so the host predicts any range's sum exactly and **invertibility mod 2⁶⁴ is asserted**
  ("the first value tried was not invertible and the assertion caught it"); written values are indexed by offset
  **inside their own launch**, not by destination, so an aliased launch is visible only to the byte-content check
  and that is the point; chunk arithmetic is **booked, not inferred from `gridDim.x`** — deriving it from the grid
  made every surviving block enlarge its own work, so a "one dropped block" control moved 64 records instead of 1;
  **"one element short" has to be short in the block's own chunk** or two controls do nothing while printing
  `must=0/2`; each control runs in **its own forked child** (one faulting kernel poisons the context and every
  later case silently never ran); and a timed row that has not earned its number prints `NO-TIMING reason=…`
  rather than a number, with the gate grepping its own correctness log for `ms_median|GB/s` and failing if one
  appears, and `--pages cold` refused outright (first touch runs ~7× slow).
- **Two spellings of one weight name need exactly one owner, and the lookup must refuse rather than helpfully
  normalise** (U-13). Their store is keyed on the **normalised** name while every prototype reader opens the **raw**
  checkpoint key; `at()` deliberately aborts — "a lookup that tolerated both spellings would answer for tensors the
  store does not hold, and would turn a naming mistake into a silent pointer to the wrong blob" — so the fix
  belongs on the producer side, and the producer is Python. Its selftest reads `model.safetensors.index.json`
  rather than the shard header "because the header parser lives in `loader.cpp` behind a `load()` that hipMallocs
  and workers do not touch the device". Docstring of `../qwen3.5-0.8b/prototype/weight_names.py`. This repo's
  version has an extra axis (HF name ↔ GGUF name ↔ resident blob, over rank-3 fused expert tensors), and
  `tools/PROVENANCE.md` §`gguf_suffix_census.py` already records what a hand-written name classifier did here:
  it "mislabelled the family entirely while still getting the total right".

## A lane whose remaining work is executing code for the first time gets its own run

Five lanes hit their run deadline on 2026-10-26, and the moments of death had a common shape: the remaining
work was either "run 2,000 lines that have never been executed" or "assemble the evidence files". Neither is
slow, both are exactly where the interesting failures appear, and both had been scheduled as the tail of a run
that already spent its budget building. The fix is scheduling, not faster workers:

- a build-only run ends when the binaries compile warning-free and the driver selftest is green;
- a run-only run has one deliverable, execute the existing code and fix what breaks, and is forbidden from
  writing new code, new files, or the wrapper;
- evidence assembly and the device gate are their own run.

Two corollaries. Commit the checkpoint before relaunching, so a death costs the report and not the work:
lm_head, gdn and shared_expert were each well ahead of their 113-, 123- and 19-byte final reports, and none of
that work survives an uncommitted tree. And a killed worker cannot release a lock it was holding, so the main
agent must verify the holder pid is dead and clear the lock itself -- a stale /tmp/pi_halogen.lock had stayed,
tools/halogen_gate.sh admit would have refused every measurement window for no reason at all.

## Never stage a live lane's files: a commit you certify must contain only work you certified

Found 2026-10-26, in the **main session**, against its own rule. A commit certifying a `speccheck` slice was staged
with an explicit path list that *also* named another lane's directory — whose worker was alive and mid-write at that
moment. The commit therefore certified files nobody had reviewed, and the review would have been impossible anyway:
a half-written kernel is not a reviewable artefact.

Two rules follow, and both are cheap:

- **Stage by path only from directories no worker owns.** Before any commit, `git status --short` and ask who last
  wrote each staged path. A worker's directory is off-limits while it is alive, even if its files look complete and
  even if they are genuinely modified.
- **A commit message may not fix a staging error by disclosing it.** Writing "these files are here only because they
  were already staged and are not certified" is a confession, not a control: the bytes are in history either way, and
  the next reader will find them attached to a verdict they do not belong to. Correct the commit instead —
  `git reset --soft HEAD~1`, then `git restore --staged <path>` (which leaves the worker's bytes on disk untouched),
  then re-commit. Verify with `git show --stat`, not with the message you intended.

The reason this earns a trap entry rather than a scold is the **asymmetry**: the mistake is invisible in the commit
message and visible only in the file list. So the only check that works is the one run against the file list.
