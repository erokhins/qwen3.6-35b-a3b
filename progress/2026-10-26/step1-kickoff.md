# 2026-10-26 — step 1 kickoff: fetch, spec, llama.cpp, engine start (delegated)

**Box clock note.** `date` on this box reads `2026-10-25T00:40Z`; every document in this repo is
dated **2026-10-26** (the convention the seed commits established). Dates in this repo follow the repo
convention, not the box clock. Do not "correct" them.

## What the operator asked for (this session)

1. start the download and do the audit,
2. author the model specification,
3. get llama.cpp working on the downloaded artifact,
4. start the engine for the chosen quantization (policy A), step by step,
5. do the heavy lifting with subagents; the main agent guides and enforces the subagent rules.

## Endpoint / device state at kickoff (main agent, read-only probes)

- `PI_PROVIDER=tensorfold`, `PI_MODEL=Qwen3.8-Flash-Next` — **this session is served remotely**
  (`192.168.10.27:8888`, `/health` → `max_batch_size: 5`), so worker token generation costs this iGPU
  nothing. Per D-13 the main session holds 1 of 5 streams; workers may hold 3 more, and the 5th stays
  closed as the stability margin.
- `halogen` container `halogen-flash-server` is **up and idle** (`in_flight 0`, `queued 0`,
  `requests_proc 0`), resident **24,289,746,944 B = 22.62 GiB** — so the box is *not*
  measurement-clean while it runs. No timing is being taken in this phase.
- `tools/gpu_lock.sh status` → `free`. It stayed free for the whole phase: **no worker in this phase
  touched the iGPU** (all four tasks are CPU-only by instruction).
- Disk: 2.4 TiB free on `/home`; RAM 123 GiB total, 86 GiB available.

## Acquisition (D-03 / `PLAN.md` step 1) — DONE by the main agent, 2026-10-26

Both fetches ran to completion before any worker was launched (`hf` = `huggingface-hub` via pipx;
unauthenticated, so rate-limited but complete):

| artifact | where | bytes on disk |
|---|---|---|
| `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | `models/unsloth-q4km/` | 21 GiB (≈20.7 GiB weights expected) |
| `model-00001-of-00026.safetensors` | `models/qwen36-src/` | 3,996,199,712 |
| `model-00002-of-00026.safetensors` | `models/qwen36-src/` | 1,284,907,696 |
| `model-00025-of-00026.safetensors` | `models/qwen36-src/` | 3,832,888,256 |
| `model-00026-of-00026.safetensors` | `models/qwen36-src/` | 2,231,416,848 |

Shard selection derived, not guessed: `python3 tools/mtp_shard_cost.py --layers 0,1,39` → 51 tensors in
**4 of 26** shards (≈10.30 GiB), and `--mtp` → 19 tensors in shards 25–26. The layer-39 audit input
lands in shard 26, which is why `--layers 0,1,39` and `--mtp` share two shards.
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` confirmed present in `unsloth/Qwen3.6-35B-A3B-GGUF` at revision
`a483e9e6cbd595906af30beda3187c2663a1118c` (HF API listing, 2026-10-26).

**Sequencing deviation, recorded deliberately.** `PLAN.md` puts step 1.0 (write `tools/audit_published_quant.py` (planned) — being written by W1 now)
*before* either fetch command, and its stated reason is design
contamination: "so the gate is designed from the spec rather than from the data it judges". The
operator asked for the download to start immediately, so the fetch overlapped the gate's authoring.
What that rule actually protects is preserved by construction rather than by ordering: the gate author
writes from `checkpoints/*.json`, the committed index and `ggml-common.h`; it may parse the GGUF
**header** (names/types/shapes/offsets — the structure pass) but must not read a single **tensor data
byte** until its own `--selftest` on synthetic blocks passes, and its report must state which of the
two happened first. The audit is worthless if that ordering is not honoured, so it is a reported
field, not an assumption.

## llama.cpp pin

`engines/llama.cpp` (gitignored) cloned by the **main agent** at
**`d327bf1f0780f87b8c01b367b61853b0c291d7c4`** (`b11869`, `jinja : fix select and selectattr argument
handling (#31059)`), depth-1, from `github.com/ggml-org/llama.cpp`. It maps
`LLM_ARCH_QWEN35MOE → llama_model_qwen35moe` (`src/llama-model.cpp:339-340`), so the architecture is
supported by this version. Cloned by the main agent *before* dispatch specifically so the audit
worker, the reader worker and the build worker all read the **same** `ggml/src/ggml-common.h` — three
workers fetching "the latest llama.cpp" would be three versions, which is the two-rulers problem
rebuilt at the source level. The 0.8B project's HIP checkout (`797de63`, `b11716`) was **not** reused:
D-12 requires our own copy with the version pinned here.

## The main agent's own ruler, taken before any worker reported

`python3 tools/gguf_header_types.py models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` → exit 0,
**all 733 tensor-info records parsed, 733 of 733 matched** the family table. Headings, as the
comparison target for W1's structure pass (1.3) — a worker's audit table is only checkable if the
main agent holds a second reading of the same bytes:

| family | stored type × count | GiB |
|---|---|---|
| routed experts gate/up | Q4_K ×80 | 11.250 |
| routed experts down | Q5_K ×37, Q6_K ×3 | 7.047 |
| `lm_head` | Q6_K ×1 | 0.389 |
| token_embd | Q8_0 ×1 | 0.503 |
| shared expert | Q8_0 ×120 | 0.125 |
| router `ffn_gate_inp` | F32 ×80 | 0.078 |
| GDN `attn_qkv` / `attn_gate` / `ssm_out` | Q8_0 ×30 each | 0.498 / 0.249 / 0.249 |
| full-attn q/k/v/output | Q8_0 ×10 each | 0.166 / 0.010 / 0.010 / 0.083 |
| GDN conv1d / a-b-gate-beta-A_log-dt_bias / norms | F32 ×30 / ×150 / ×101 | 0.004 / 0.015 / 0.001 |
| **total** | | **20.676 GiB = 22,201,133,568 B** |

Also confirmed here, independently: layer indices present **0..39 (n=40)** and
**`tensors whose name contains 'mtp': 0`** — the published file carries no draft block, which is the
fact `PLAN.md` step 1 was built around, and the reason llama.cpp alone can never witness the MTP path.
This matches `quantization/evidence/q4km-published-width-map-2026-10-26.md` family-for-family, so the
committed evidence is not stale and the file we downloaded is the file that evidence describes.

## Device-window plan (main agent, later — NOT yet taken)

`devices/evidence/halogen-restart.log` is still **empty**, so per `AGENTS.md` ("batch, do not
oscillate") the default is one down-window per session. But the same log being empty means the
restart cost is unmeasured, and the D-13 protocol says that measurement is owed. So the first window
is planned to be the **cheapest thing that pays for itself**: with zero workers active, `gpu_lock`
held and halogen taken down through `tools/halogen_gate.sh` only —

1. the llama.cpp smoke test on the shipping GGUF (a *correctness* run: does the pinned build load
   `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, does it emit a continuation, does it stop on its own eos id 248044),
2. llama.cpp's own tokens/s on this box — **the ruler every later engine number is compared against**
   (D-12 "perf ruler"), which is why it is worth a window of its own rather than waiting for the engine,
3. the restart-cost measurement itself (`tools/halogen_gate.sh up` prints wall time and appends to the
   evidence log), plus the D-11 ≈55 GiB allocation row while halogen is genuinely down,

run blocking, in the foreground, by the main agent. Everything W1–W4 produced stays CPU-only, so no
number in their reports needs any of this. If W3's build fails, this window does not happen.

## Server restart, three workers killed, all four lanes re-armed (same day)

The LAN inference server was restarted by the operator mid-phase. `tensorfold` came back healthy
(`/health` → `status: ok`, `max_batch_size: 5`), but W2, W3 and W4 died on it with `Connection error`.
W1 was on `halogen` (the local container, never served this session) and survived, finished its audit,
and asked twice for a gate decision while I was busy — both escalations timed out unanswered, which is
on me: a worker held at a decision it is not allowed to make is a worker doing nothing.

**What survived is not nothing.** The workers wrote to disk continuously, so the restart cost reasoning,
not files: W1 delivered the whole audit; W2 had `specification/derive.py` (810 lines) + README; W3 had a
**complete HIP build** (`build-hip/bin/{llama-cli,llama-quantize}` + 26 artifacts, `arch=gfx1151` in the
build log at 02:09:09 and 02:24:41) and `tools/build_llama.sh`; W4 had `speccheck/moe_ref.py` (657),
`cross_check_hf.py` (417) and the HF capability probe. Each resume therefore began with *inventory what
you already wrote, do not restart from zero* — the failure mode of a resumed worker is a second
half-finished copy of its own output.

### W1's audit result, and the two things I checked before ruling

Verdicts: **1.1 PASS, 1.2 PASS, 1.3 PASS, 1.4 FAIL (one 1-ulp near miss), 1.5a RUN, 1.5b PASS,
1.5c PASS, 1.6 PASS**; sha256 `ac0e2c11…31a61` twice and equal to the HF LFS etag; 150/150 injection
cases correct with 0 leaks; ordering evidence shows `--selftest` PASS at `01:31:24Z` **before** the first
tensor-data read at `01:42:30Z`, and `--exact` was *refused* at 01:42:26Z because the tool files had
changed since the PASS — the marker stores both tools' sha256, so the design-before-bytes rule bites
rather than being promised. The sequencing deviation I recorded at kickoff is thus closed on its own
terms: the gate was written from the spec, and the bytes were unread until it passed.

I verified two of its claims with my own ruler instead of taking the report (`tmp/verify_gguf_bytes.py`,
which reuses the proven GGUF tokenizer but swaps in block sizes transcribed from the pinned
`ggml-common.h`, then asserts `sum(row_bytes) + data_start == file size`):

| W1's claim | my independent reading | verdict |
|---|---|---|
| PLAN's MTP figure (41,943,040 elts / 79.9 MiB) is wrong | raw shard headers: **19 tensors, 844,640,768 elts, 1,611.02 MiB bf16**; `gate_up_proj [256,1024,2048]` = 536,870,912 and `down_proj [256,2048,512]` = 268,435,456 = 95.34 % of it | **CONFIRMED** — the draft block carries its own 256-expert MoE stack, which `README.md`'s 61.50 GiB census already implied and `PLAN.md` had contradicted |
| the published GGUF carries no draft block | 0 tensors matching `mtp`/`nextn` across all 733 records | **CONFIRMED** (also matches the header read taken at kickoff) |
| `Q5_K` is 5.5 bpw, our tools say 5.5625 | `static_assert` in `ggml-common.h` fixes `block_q5_K` at `2·half + K_SCALE_SIZE(12) + QK_K/2(128) + QK_K/8(32)` = **176 B / 256 = 5.5 bpw**; file-size identity holds at 176 B with Q4_K 144 B, Q6_K 210 B, Q8_0 34 B/32, F32 4 B | **CONFIRMED, and Q5_K is the only wrong row** — my re-derivation agrees with the tools on Q4_K 4.5, Q6_K 6.5625 and Q8_0 8.5 |
| drift ≈232 MiB over the 37 Q5_K tensors | my identity gives a Q5_K family of **6.3594 GiB** and a drift of **74.0 MiB** | **NOT RECONCILED** — same identity, so one of us slipped; sent back to W1 to show the working, with instructions not to adopt my number unchallenged |

Total stored weights **22,123,538,944 B = 20.6042 GiB** (the tools' width map printed 20.676 GiB, the
expected direction for a too-high Q5_K bpw). `general.file_type = 15`, `quantization_version = 2`,
`arch = qwen35moe`, `expert_count 256`, `expert_used_count 8`, both `expert_feed_forward_length` and
`expert_shared_feed_forward_length` = 512, `head_count 16` / `head_count_kv 2`.

**Policy A does not move.** No policy row in `tools/quant_scenarios.py` uses Q5_K (they use
Q4_K/Q8_0/bf16/IQ4_XS/MXFP4, with Q6_K only for `lm_head` in E and G), so 19.59 GiB resident and
2.331 GiB/token stand; only the published-file width-map GiB column is affected. That sentence exists
because a correct-looking bpw fix applied to the shipping arithmetic is how a green table becomes a
wrong one.

### The gate decision, and why it is not the forbidden kind

W1 asked whether a single 1-ulp deviation on `ssm_a` blocks step 1. Ruled: **accept, as a route-derived
predicate rather than a tolerance** — `DECISIONS.md` **D-14**, `PLAN.md` §"Amendment to 1.4". The
load-bearing fact is that **HF never stores `ssm_a`**: the reference computes `A = -exp(A_log.float())`
at init, so there is no stored authority to be bit-exact against, and our engine reads `ssm_a` from the
file and never recomputes `exp`. Copied families keep zero tolerance; only the documented transcendental
route gets 1 ulp, with the two reference routes required to agree with *each other* (a reference
disagreement FAILS, it never becomes "within tolerance") and must-catch **grown** to ±2 ulp, sign flip,
element permutation and a wrong-`A_log` read. Escalated to the user for ratification rather than assumed,
and reversible with one flag, because `AGENTS.md` makes "what counts as a band" the user's question.

### Board after re-arming

| lane | endpoint | state | on resume |
|---|---|---|---|
| W1 `6b2b7fb4` | `halogen` | revived with the D-14 ruling | re-run 1.4 as 1.4a/1.4b, patch the Q5_K bpw *derived+asserted*, reconcile the 74 MiB vs 232 MiB drift, refresh summary and pin |
| W2 `3cf4d0ea` | `tensorfold` | revived | inventory `specification/`, then finish it carrying the corrected MTP numbers |
| W3 `26240c65` | `tensorfold` | revived | **do not rebuild** — prove the HIP backend is really wired in (`ldd bin/llama-cli` showed no `libggml-hip` in the chain, which is the silent-CPU-fallback shape and would make every later cross-check meaningless), then the four D-12 verdicts |
| W4 `3d702c27` | `tensorfold` | revived | inventory `speccheck/`, then `tools/dump_reference.py` (planned), the manifest, `selftest_moe.py`, `results.md` |

Still 3 × `tensorfold` + 1 × `halogen` = the D-13 ceiling, main holding 1 of 5 LAN streams, the 5th
slot closed. `/tmp/pi_halogen.lock` has been held continuously across W1's completion and revival, since
the rule is the worker's whole life and a gap is exactly when a down-window would admit.
`tools/gpu_lock.sh status` has stayed `free` throughout: **no device run in this entire phase**,
including W3's compile. Every resume brief repeated *no inference, no model load, no timing of any kind*.


## Worker board as first dispatched (D-13: ≤3 tensorfold + ≤1 halogen, session holds 1 of 5 LAN streams)

| id | endpoint | task | device rights |
|---|---|---|---|
| W1 | `halogen` | `tools/audit_published_quant.py` (planned) then run steps 1.1–1.6 | none (CPU-only) |
| W2 | `tensorfold` | `specification/` — math-only spec of the text path | none (CPU-only) |
| W3 | `tensorfold` | HIP build of the pinned llama.cpp + provenance + build recipe | none (build only; no inference, `-ngl 0` ceiling) |
| W4 | `tensorfold` | CPU oracle env + `speccheck/` MoE reference + pinned dumps | none (torch CPU) |

`/tmp/pi_halogen.lock/holder` was claimed as
`35b: W1 audit-published-quant (halogen worker)` before W1 launched and is released when W1 stops —
that is what lets `tools/halogen_gate.sh admit` refuse a down-window while a halogen worker lives.

**Nothing in this phase is a measurement.** No bandwidth figure, no tokens/s, no latency appears in
this file or in any worker report from this phase. Every device number in this project is taken by the
main agent, with zero workers active, `gpu_lock` held and halogen down, in one batched window
(`AGENTS.md` §"Two endpoints, three classes of work", `pitfalls/00-measurement.md`).

## The 74-vs-232 MiB dispute, settled by a tool rather than by memory

W1 reported ~232 MiB of Q5_K drift; my witness said 74.0 MiB. Same identity, so one of us had slipped,
and the number was heading into a committed evidence file. It also arrived at the wrong moment: W1's
run logged **one auto-compaction failure** (`"Auto-compaction failed: Summarization failed: generation
hit the token cap and the summary is incomplete"`) — the only lane with one, and unsurprisingly: it is
a ~2 h, 691k-token, 226-tool conversation I had revived, the largest context in the fleet. The other
three lanes logged zero.

So the ruling was neither "trust the worker" nor "trust me" — it was **make the number come from a
tool**. `tools/gguf_width_table.py` (new, committed) derives every printed byte from the file's own
length (`sum(row_bytes) + data_start == file size`), re-derives the K-quant block sizes from
`ggml-common.h`'s own `static_assert` expressions instead of trusting a transcription, and refuses to
print on an unmapped type, a partial record count, or an empty table. Its `--selftest` **injects** wrong
block sizes and requires the identity to go False — a witness that has never fired is not a witness —
and it checks that each injected type actually occurs in the file, so a fault injected into an absent
type cannot count as a pass.

```
python3 tools/gguf_width_table.py --selftest     # 3/3 injected byte-table faults FIRED (Q5_K, Q4_K, Q8_0), each type present
python3 tools/gguf_width_table.py models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --check-header
  ok: Q4_K = 144 B/256 = 4.5000 bpw  from `2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2`
  ok: Q5_K = 176 B/256 = 5.5000 bpw  from `2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 + QK_K/8`
  ok: Q6_K = 210 B/256 = 6.5625 bpw  from `sizeof(ggml_half) + QK_K / 16 + 3*QK_K/4`
  WITNESS  sum(row_bytes) + data_start == file size : True      (22,134,528,992 B)
```

**Adjudicated: 74.0 MiB, not 232 MiB.** The 37 Q5_K tensors hold 6,828,326,912 B at 176 B/256 and
would hold 6,905,921,536 B at 178 B/256 — a difference of 77,594,624 B = **74.0 MiB**, and setting Q5_K
to 178 flips the witness to False. Three independent routes agree (the file-size identity, the header's
own `static_assert`, the arithmetic), so the evidence file carries 74.0 MiB unless W1 produces a working
that beats all three — invited, not forbidden.

Two things worth keeping from making it. My first `--check-header` matched nothing because of a
`%`-precedence slip (`block_4_kK`), and its message was *"cannot verify, refusing to assume"* — which
reads like rigour while actually being a checker that had stopped checking. A no-output branch that
prints something plausible is the same trap as an empty parse reporting PASS, which is why the fixed
version also asserts it read the block constants at all (`{K_SCALE_SIZE: 12, QK8_0: 32, QK_K: 256}`)
and calls the derivation vacuous otherwise.

And the lane was **not** idle when it looked idle. `halogen_gate.sh status` read `in_flight: 1`,
`requests_proc: 1`, `busy_for_s: 26.2`, and `tensorfold` read `connections: 3` — all four workers
streaming, exactly at the D-13 ceiling. A "no activity for 300 s" nudge into a live stream adds a
second stream to a server that advertises five slots and that I hand four on purpose, so the counters,
not the silence, decided it: **no nudge, no interrupt**. A worker that is thinking is not a worker that
is stuck, and the difference is observable rather than guessable.

## W3 delivered — and corrected me; and a citation inside a user decision record is now known-false

W3 came back with the llama.cpp lane finished and my own alarm disproved. My "no `libggml-hip` in the
chain" was **an artifact of my own command**: `build-hip/ggml/libggml-hip.so` does not exist, and
`readelf -d` on a missing path prints nothing at all — which is exactly what "no NEEDED lines" looks
like. The lib lives at `build-hip/bin/libggml-hip.so.0.27.0` (72,910,832 B), `libggml.so.0.27.0` NEEDs
it, and the answer to "is the HIP backend really wired in" came as a **census rather than a sample**:
`.hip_fatbin` is one 56,112,304 B section and all **141** carved code objects report
`amdgcn-amd-amdhsa--gfx1151` under `llvm-readobj --notes`, 0 anything else — with the note that
`--file-headers` *cannot* answer that question, and that `llvm-offload-device-info` would have passed on
a CPU-only build because a gfx1151 GPU is attached. Two of its own first-version gate checks were
vacuous for exactly that reason, and it replaced them rather than leaving them green. I re-ran its gate
independently: `tools/build_llama.sh --check` → **17/17 GATE GREEN**. Committed as `8576744`.

**The finding that is not mine to act on.** `quantization/evidence/llama-cpp-tensor-policy-2026-10-26.md`
records that llama.cpp forces the latent/gate tensor categories to ≥ Q8_0, and **`DECISIONS.md` §D-01a
cites that rule as support for the user's decision to put attention/GDN at Q8_0**; `quantization/README.md`
repeats it. At this commit the rule is `if (arch == LLM_ARCH_MUSE_SPARK && …)` — for `Qwen35MoE`,
llama.cpp raises nothing, so one leg of the reasoning recorded next to a user decision is false. The
**decision stands** on its other, independent legs (unsloth's per-tensor sweep on this exact
architecture family found `attn_*` the most sensitive class and `ssm_out` the worst single tensor, and
Q8_0 on 1.975 GiB is cheap), but a decision must not keep standing on a citation that does not exist,
and the record of a user's choice is the user's to amend. Escalated, not edited; W3 likewise refused to
touch it.

Same verdict batch: the `llama-quantize` per-tensor flags D-12 rests on are **CONFIRMED**, with two
traps absent from our docs — overrides are `std::regex_search` **substring** matches, so
`--tensor-type attn_q=q8_0` also rewrites `attn_qkv` and `attn_q_norm`; and they are **silently ignored
unless the base type is quantized**, so an `F16`/`BF16`/`COPY` base makes them a no-op. Those matter
because they are the difference between producing a policy-A artifact and producing a plausible-looking
wrong one. Also **Q6_K is not K-quant-shaped** (8-bit scales, no min, single `d`), and **expert gate/up
fusion is opt-in and off by default** upstream, so our own conversion emits *split*
`ffn_gate_exps`/`ffn_up_exps` while the published file may be fused — the reader must accept both, as
llama.cpp's loader does.

## W2's third death: the lane was the problem, so a fourth resume was not the answer

W2 failed again. The signature is diagnostic rather than random: **15 `terminated` errors with
`totalTokens.output = 0`** — generations killed before producing anything — while W3/W4 logged zero such
errors and W1 one compaction failure. The lanes that died were the **revived** ones, whose context had
been accumulating for hours; the fresh lanes were fine. Resuming that conversation a fourth time with
the same brief would have reproduced the failure, which makes it the wrong supervisory move, not a
patient one. Both replacements launched **fresh**, with the on-disk state as their starting point and a
work-shape rule aimed at the actual mechanism:

| lane | role | brief shape |
|---|---|---|
| W5 `a7e9a248` | replaces W2's spec lane | `derive.py` **frozen**; one file per `write`; **≤250 lines per file**, split rather than grow; report every 2 files. Its predecessor spent 2.5 h on a 42 KB derivation script and shipped zero spec files, so the deliverable order is enforced, not suggested. |
| W6 `ae846f77` | the engine lane, previously queued behind the cap | `prototype/q4k_dequant`: device Q4_K + Q8_0 dequant, **bit-exact** against `tools/gguf_reader.py`, already witnessed (40/40 synthetic, 150/150 injections, 0 leaks). Correctness-only, <60 s, under `gpu_lock`, PASS/FAIL, no timings. |

The engine lane started **now** rather than after the spec because the dequant kernel is the one piece
whose oracle already exists — it needs `ggml-common.h` and the audited CPU reader, neither of which
waits on the spec or on W4's dumps. `PLAN.md` step 2 names it first for the same reason.

## `tools/worker_lanes.sh`: the quiet-lane question, answerable without guessing

Two "no observed activity for 300 s" alarms fired today and **both were healthy workers** — a
`thinking high` lane emitting nothing while it composes a large file reads identically to a stall, and
the tempting response (`steer`/`interrupt`) injects into a live stream and spends the fifth-slot margin.
So the judgement is a command now rather than a call: endpoint counters are the authority, a lane is
ALIVE only on its own recent event or an unclosed `message_start` with its endpoint streaming, and
anything else is UNKNOWN — never ALIVE.

Its first version was wrong in the instructive way: it counted "some file in the repo changed" as
evidence of life, which marked **10-hour-old corpses from earlier sessions ALIVE** because other workers
are always writing — a check that cannot report STALE is not a check. Dropping that unattributable
signal fixed it, and `bash tools/worker_lanes.sh 1` confirms the STALE branch is reachable, so the
verdict is witnessed rather than hoped for. It currently separates my 4 live lanes from 5 real corpses
and 10 historical ones.

## Step 1 closed: the audit's own verdicts, re-run by me rather than adopted

W1 finished the D-14 work and I re-ran the gate myself instead of reading its summary — `--selftest`
PASS, and `--exact` reproduces its numbers exactly: **1.4a 25 cases / 0 outside** at zero tolerance,
**1.4b 64 elements → 63 bit-exact + 1 within-route + 0 outside + 0 reference disagreements**, the summary
line reading `27 = 26 bit-exact + 1 within-route (a within-route case is NOT a bit-exact case)`, and the
zero-tolerance `FAIL on blk.1.ssm_a` printed beside the PASS-with-finding rather than under it. Its own
selftest caught a live bug in the new predicate — an inverted bit mapping made "within one step" mean
"within 3.8", a bound that could not fail — which is why the spacing is now witnessed against
hand-worked binary neighbours (`2**-23` on [1,2), `2**-24` on [0.5,1), `2**-22` on [2,4)) instead of
trusted.

On the drift it adopted my number and kept its own on the record as a correction rather than deleting
it: `38,797,312 blocks × 2 B = 77,594,624 B = 74.00 MiB`, which I re-derived independently; its earlier
~232 MiB is unreproducible from any correct ruler (nearest misread: putting the 0.0625 bpw delta over
the whole-model denominator). It then went further than I asked and made the widths **derived from
`ggml-common.h` at run time** (`tools/ggml_block_geom.py`, each struct checked against its own
`static_assert`) instead of retyped, re-auditing every family: exactly one wrong row in eight, the rest
already right — which is the shape that survives a spot-check and the reason two independent rulers now
agree on this file. Policy A is unmoved and says so in the tool's own output: **19.59 GiB resident /
2.331 GiB/token**, because no policy row spends Q5_K.

**And an error of mine, caught because I re-read my own file after the fix landed.** Once
`gguf_header_types.py` and `quant_scenarios.py` were corrected, `tools/gguf_width_table.py` was still
printing "tools say 5.5625" next to Q5_K — an accusation that had outlived the bug it described. An
outdated claim in a validator is worse than no claim, because the validator is the thing people trust.
The constant is now empty with the history recorded in a comment, and `--selftest` still fires 3/3
because the negative control re-derives the failure from an injected stale row rather than from the
removed table. Lesson worth keeping: **when a bug gets fixed, grep for the places that were written to
catch it.**

I also released `/tmp/pi_halogen.lock` on W1's completion — and the first attempt silently failed,
because I ran `rmdir` on a directory still holding `holder`. `halogen_gate.sh status` printed the stale
claim and only then did I notice; the retry removes the file first and now reads `stream claim: none`,
so a down-window can admit. A lock released by a command that failed is a lock still held.

## The census dispute: I was wrong, and nearly made it worse

`tools/quant_scenarios.py`'s 65.7198 GiB — the figure in `AGENTS.md`, `README.md`, `PLAN.md` and D-11,
the one whose excess over the pool *forced* 4-bit experts — was contradicted by the artifact itself.
I built my own ruler from the GGUF's tensor records: **64.5604 GiB**. Three rulers, three answers, two
of them tools. W10 adjudicated it to **0 bytes of residue**: the authoritative text figure is
**66.1337 GiB** with the draft block, and my 64.5604 is the *same element set minus the 19 `mtp.*`
tensors* — which I had already witnessed during step 1's audit (the published GGUF carries zero draft
tensors) and then left out of my own arithmetic.

**I measured the wrong quantity and defended it.** The closure is elements first: 34,660,610,688
(artifact) + 844,640,768 (draft) = 35,505,251,456 → 66.1337 GiB, and adding vision's 446,571,248 elts
gives 71,903,645,408 B = the shard index's own `total_size` exactly. The tool's −0.4138 GiB decomposes
to the element: `ssm_out` omitted (+251,658,240), `mtp.fc` omitted (+8,388,608), phantom `3h×3h`
(−37,748,736), phantom `+2*h`×30 (−122,880, which the first census pass missed), 86 real h-norms vs its
84 (+4,096). My recomputation reproduced every one of those numbers independently.

**Then I nearly introduced a new error while fixing old ones.** Editing `README.md`'s family breakdown,
I "corrected" routed experts from 61.500 to 60.0000 — the artifact's own sum. Wrong: 61.500 was right,
because it counts the draft block's own 1.500 GiB MoE stack. Caught only because I checked whether the
original was actually wrong before replacing it. The real errors were GDN (1.416 → 1.885, the omitted
`ssm_out`) and 0.0704 of phantoms sitting in "norms and misc". Rule worth keeping: **a number I intend
to "fix" has to be falsified as wrong first; agreeing with my own ruler is not falsification.**

**What moved, and what I refused to invent.** Policy A: resident 19.59 → **19.7848 GiB**, per-token
2.331 → **2.5252** (2.4653 target-only), so the roofline prediction 95/84/76 → **87/77/70**. Policy F
resident 34.97 → 35.16. The fit verdicts do not move, and neither does any width decision — the
control arm is what makes that trustworthy: pricing policy A from the *broken* census reproduces
19.59/2.331 exactly, so all the movement is census, not width rule.

Only rows A and F were adjudicated, so **rows B, C, D, E, G and the other rate cells are marked ⚠** in
both tables rather than quietly left as if current or hand-derived by me: they are expected to move
~+1 % resident and +6–8 % per-token (the omitted `ssm_out` is read every token), and the instruction
in the doc is to re-print them from the repaired tool so the docs and the tool cannot drift apart. An
invented 96 tok/s and a stale 96 tok/s look identical to a reader, which is why the ⚠ matters more than
the number. `AGENTS.md`, `README.md`, `quantization/README.md` and `devices/measured-facts.md` are
corrected with provenance; **`DECISIONS.md` is not touched** — D-11 quotes 65.72 and 34.97, and while
the conclusion is unaffected (all three candidate figures exceed the pool), that file is the user's.
The journal's own earlier entries still assert 19.59/2.331 and are deliberately left as written: a
journal records what was believed when, and gets corrected by a later entry, never by editing.

## llama.cpp drives the pinned artifact: the runtime half of the GPU question closes

With the board clear I took the part of the device window that does not need the inference server down.
`llama-cli` @ `d327bf1` loaded `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, and the two lines that matter are
`system_info: … | ROCm : NO_VMM = 1 | …` and `load_tensors: offloaded 1/41 layers to GPU` — 24 tokens
generated greedy at seed 42, answering `France`. W3's build audit had proved device code was *present*
(141/141 `gfx1151` code objects, 56 MB fatbin); a census cannot prove the loader *reaches* it, and this
run does. Written up with its witnesses in
`quantization/evidence/llama-cpp-smoke-gguf-2026-10-26.md`.

Two findings, neither about speed. **`tokenizer.ggml.eos_token_id u32 = 248046`** — witnessed from the
artifact as llama.cpp reads it, corroborating one side of the A-04 disagreement with `text_config`'s
248044; A-04 stays open because `generation_config.json` is still not on disk, so our decode loop takes
`STOP_IDS` as a named input instead of defaulting to either. And **llama.cpp guesses a reasoning format
for this arch** (`chat_format: peg-native`, `reasoning_format: deepseek`, generation prompt ending in the
think-boundary token, `preserved_tokens [248058, 248059, 248068, 248069]`) — its guess about a checkpoint
whose own config is ambiguous, which our engine must decide deliberately rather than inherit, and which
touches A-20 in the MTP path.

**Every rate from this run is discarded on purpose.** llama.cpp prints its own timings; halogen was
resident and 40 of 41 layers ran on CPU, so the figure is contaminated *and* unrepresentative. The raw
capture sits in gitignored `tmp/evidence/` so it cannot leak into a commit, and the evidence file says so
in its own text rather than just omitting the number — a reader who finds the log later needs to know why
it is not quoted.

Two harness traps worth keeping, both of which produced a **false green** before I caught them. `llama-cli
-p …` alone enters interactive mode: it read empty stdin, printed an empty completion, and **exited 0** —
so a smoke test gated on exit code passes on a run that generated nothing, and the completion text is the
actual evidence. And `--no-cnv` does not exist in this build (`invalid argument`, exit 1); the flag here is
`-st/--single-turn`. Separately, without `-v` the load prints no backend or offload lines at all, so a
silent smoke test cannot see the one line that distinguishes a GPU build from a CPU fallback — the exact
failure the whole build audit was chasing.
