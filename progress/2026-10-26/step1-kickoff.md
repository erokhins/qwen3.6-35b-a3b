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
