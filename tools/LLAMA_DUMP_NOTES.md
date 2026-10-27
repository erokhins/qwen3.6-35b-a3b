# Reading an instrumented llama.cpp dump

**What this is.** A second implementation's per-layer activations, dumped from the vendored
`engines/llama.cpp` at the pinned commit `d327bf1f0780f87b8c01b367b61853b0c291d7c4` (`b11869`), read
off the **CPU** build while it runs the pinned Q4_K_M GGUF. It exists so the main agent can ask "what
number does another correct implementation get at this point in the block" without needing the
safetensors the CPU oracle path needs.

**What this is not.** Not the oracle, and not a band. `speccheck/` is the oracle and the bands come
from a floor probe. Nothing here may be used to widen a tolerance: this is a **quantized**
implementation (Q4_K_M weights, f32 activations) and its agreement with ours is a cross-check, not a
proof. Where this file says what a llama.cpp tensor "is", that is a hypothesis for the main agent to
adjudicate, never a verdict — see §Stage mapping.

**Device discipline.** The build is CPU-only and gated as such, the run passes `-ngl 0`, the run
script contains no lockdir path and no acquire/release call, and nothing in this lane prints a rate,
a duration or any other measure of how long something took. Every number below is a count, a byte
length or a digest.

## Files

| path | what |
|---|---|
| `tools/llama-instrument.patch` | the whole instrumentation, 4 files: new `src/llama-dump.h`, `src/llama-dump.cpp`; modified `src/llama-context.cpp` (arm before `graph_compute()`, close after it, in `process_ubatch()`), `src/CMakeLists.txt` (one source added) |
| `tools/llama-dump-expected.h` | the generated name sets. Generated, not typed; never hand-edited |
| `tools/llama_dump_expected_gen.py` | regenerates the header **from the vendored source**: the `cb()` call sites in `src/models/qwen35moe.cpp` give the activation names, the `LLM_TENSOR_*` name table in `src/llama-arch.cpp` filtered to what `qwen35moe.cpp::load_arch_tensors` asks for gives the weight families |
| `tools/llama_instrument_apply.sh` | `apply` / `build` / `restore` / `regen` / `status` |
| `tools/llama_dump_run.sh` | one instrumented run: `<tag> <prompt>` |
| `tools/llama_dump_verify.py` | `self` / `determinism` / `prompt-change` |
| `tools/llama_dump_probe.py` | read-only numeric probe over dumped rows: range/sum for a named tensor in a graph, per-block digest counts, and per-block digest equality between two manifests. A human aid for "are these the bytes the name claims", not a gate |
| `tools/llama_dump_provenance.py` | writes `<dump-dir>/provenance.json` after the run |
| `data/dumps/2026-10-27-llama/<tag>/` | the `.bin` files plus `provenance.json`. Append-only: the driver refuses a non-empty target |
| `data/dumps/2026-10-27-llama/superseded-postcompute-walk/` | the three runs of the **first** version of the instrument, which read tensors after the compute returned. Kept, not deleted, because they are the witness that the aliasing check in §Why the capture is a per-node callback can fire: `verify.py self` on those manifests reports 149 aliasing groups, on the current ones 0 |
| `data/dumps/2026-10-27-llama/logs/<tag>.log` | the instrument's own stderr: one `TENSOR` line per graph node (15276 per run), one `CENSUS` line and two `CENSUS-NAMES` lines per `(graph, block)` (164 and 328), plus `CONFIG`, `GGUF`, `CHECKS` and `DONE` |
| `checkpoints/llama-dump-manifest-2026-10-27-<tag>.json` | write-once index of everything |

## Sequence to reproduce a run

```
WORK=/tmp/llama-instrument-vendored tools/llama_instrument_apply.sh apply   # patch + generated header
WORK=/tmp/llama-instrument-vendored tools/llama_instrument_apply.sh build   # CPU-only build + gate
WORK=/tmp/llama-instrument-vendored tools/llama_dump_run.sh run-a "The capital of France is"
tools/llama_dump_verify.py self          checkpoints/llama-dump-manifest-2026-10-27-run-a.json
tools/llama_dump_verify.py determinism   checkpoints/...-run-a.json checkpoints/...-run-b.json
tools/llama_dump_verify.py prompt-change checkpoints/...-run-a.json checkpoints/...-run-c.json
WORK=/tmp/llama-instrument-vendored tools/llama_instrument_apply.sh restore # leaves the tree pristine
```

`apply` refuses unless the checkout is at the pinned commit and pristine, so instrumentation cannot
land on top of other instrumentation. `build` refuses unless `apply` ran. `regen` rewrites the patch
from the instrumented checkout **and proves it applies to a clean clone** before keeping it; the
vendored index is never touched. `restore` reverse-applies and leaves `git status` empty — run it
when you are done, or the next person who builds llama.cpp gets our hooks without asking for them.

### The CPU-only gate (what `build` actually checks, 8 checks)

binary exists · `ldd` contains zero ROCm/HIP libraries · zero `not found` deps ·
`compile_commands.json` has zero `offload-arch` lines and zero device sources ·
`CMakeCache.txt` reads `GGML_HIP:BOOL=OFF` **and** `GGML_CUDA:BOOL=OFF` · no `libggml-hip*`,
`*.hip_fatbin` or device `.co` in the build dir · `llama-cli --version` reports `d327bf1` ·
`llama_dump_after_graph` is present in `libllama`. It prints `checks ran: 8 / expected 8` — the
count is asserted, so a gate that silently stopped running checks fails instead of passing.

Build directory is outside the repo (`$WORK/build-cpu`) and the command the script runs is:

```
cmake -S engines/llama.cpp -B $WORK/build-cpu -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
      -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_OPENMP=ON -DBUILD_SHARED_LIBS=ON \
      -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_CXX_FLAGS="-DLLAMA_DUMP_GIT_COMMIT=\"<pinned commit>\""
cmake --build $WORK/build-cpu --target llama-cli --parallel $JOBS
```

`CMAKE_EXPORT_COMPILE_COMMANDS=ON` is load-bearing: the gate reads `compile_commands.json` to prove no
compile carries an `offload-arch` flag. `BUILD_SHARED_LIBS=ON` is why the gate can ask `ldd` about the
binary and look for `llama_dump_after_graph` in `libllama`. `-DLLAMA_DUMP_GIT_COMMIT` is how the build
carries the commit the instrumentation was applied at.

The run is `-t 8` with `GGML_OPENMP=ON`, and the determinism control still comes back `differing=0` —
determinism here is a **measured property of this build**, not something a flag guarantees. If it ever
breaks, the levers are `-t 1` first and `-DGGML_OPENMP=OFF` second, and the answer belongs in
`DECISIONS.md` rather than in a flag quietly flipped.

## What a run captures

`llama-cli` computes more graphs than "the prompt". A run with
`-m <gguf> -p <prompt> -n 4 -ngl 0 -c 64 -b 64 -ub 32 -t 8 --temp 0 -s 42 --no-display-prompt
--no-warmup -st -co off` produces, in order:

| graph | n_tokens | what it is |
|---|---|---|
| 0 | 2 | llama-cli's own init graph, token ids `[0, 0]` — **not the prompt** |
| 1 | 11 | the prompt prefill, chat-template wrapped: `[248045, 846, 198, 760, 6511, 314, 9338, 369, 248046, 198, 248045]` (ids, not literals — AGENTS.md §the special-token rule) |
| 2 | 4 | the assistant-header prefill: `[74455, 198, 248068, 198]` |
| 3 | 1 | the first decode step |

`MAX_GRAPHS` (default 4, also passed as `-n`) is the hard cap; `-n` is set to the same number so the
graphs exist before the dumper stops. `-c 64` is enough context for a 11-token prompt plus 4 generated
tokens, and `-s 42 --temp 0 -co off` (continuous batching off) keeps the sequence of graphs single and
repeatable. The `graphs` list in the manifest is the authority on which graph is which; it carries
`n_tokens`, `n_seqs`, `n_seq_tokens`, `n_pos` and the token ids of every captured graph.

**Two consequences you have to hold in mind**: the prefill is the *template-wrapped* prompt, so if
you feed the same ids to our engine, take them from `graphs[1].token_ids` and not from the prompt
string; and graph 0's contents are prompt-independent, so a prompt-change control shows graph 0
identical between runs and graph 0 must not be counted as evidence of anything.

## Why the capture is a per-node callback, not a post-compute walk

The dump hook is armed **before** `graph_compute()` and closed after it. Arming installs
`ggml_backend_sched_set_eval_callback(sched, ld_eval_cb, nullptr)`; the callback is ggml's per-node
scheduler callback (`ggml-backend.cpp §ggml_backend_sched_compute_splits`, the `callback_eval`
branch): it is asked `ask=true` "does the user need this node", and if we say yes for every node then
ggml computes one node per range, synchronizes the split, and calls us back with `ask=false` — which
is the moment at which that node's output is certainly still live. Closing checks
`visited == ggml_graph_n_nodes(gf)` and restores llama.cpp's own `cparams.cb_eval`, to which ours
delegates rather than replacing.

**Why it has to be that way.** ggml's allocator recycles regions of the compute buffer as soon as a
tensor's consumers have run — the scheduler says so itself, in the comment above the split
synchronization: "the allocator may have reused buffer regions across splits". A walk over the graph
after the compute returns therefore reads bytes that later nodes overwrote. The first version of this
instrument did exactly that, and the numbers it produced looked *excellent*: 14 267 files, digests
recorded, run-to-run bit-identical, prompt-sensitive. They were also mostly not the tensor's values:

| symptom in the post-compute walk | what a live read gives |
|---|---|
| 3590 written tensors in one prefill graph collapsed to **128 distinct digests**; largest alias group 353 | 39 blocks × `ffn_moe_out` → **39 distinct digests**; `norm` → 129 distinct digests over 129 rows |
| every block's `ffn_moe_out`, `ffn_out`, `l_out` shared a single digest | each block's `l_out` has its own digest and its own range |
| `ffn_moe_probs` was **byte-identical to `ffn_moe_logits`** in the same block, values ranging −8.13 … 41.60 | `ffn_moe_probs` ∈ (0,1) with each token's 256 entries summing to 1.0000 (sum over the whole `[256, 11]` tensor = 11); `ffn_moe_logits` in −10.5 … −1.6 and never equal to the probs |
| KV / recurrent-state tensors (`cache_k_l*`, `cache_s_l*`, `cache_r_l*`) were sane: 30/30, 150/30, 90/90 distinct | unchanged — they live in persistent buffers, which is what localized the bug to the compute buffer |

**The trap, written down because it generalizes:** a recycled buffer is a *deterministic* function of
the allocation pattern, so both existing controls pass — the two same-prompt runs agree byte for byte,
and changing the prompt still changes the recycled bytes. Aliasing is invisible to determinism and to
prompt-sensitivity alike. What catches it is a **distinctness** requirement: `verify.py self` fails if
any name written in ≥5 blocks of one graph carries one and the same non-zero digest. That check reads
**149 groups** on `data/dumps/2026-10-27-llama/superseded-postcompute-walk/` and **0** on the current
runs, so it is witnessed firing where it must fire rather than being vacuously green.

One consequence worth knowing before reading a rate into anything here: with an eval callback
installed, ggml computes and synchronizes one node at a time instead of a whole split. Values are
unaffected — same kernels, same order, same arithmetic — but this build is not the build to ask about
speed, and no number from it is a timing.

The census is the control that says the fix moved only bytes: the 164 `LLAMA-DUMP CENSUS` lines of the
superseded post-compute run and of the current run for the same tag are byte-identical (`diff` empty).
Names, classes, weight exclusions and duplicate counts come from graph structure, which the fix did not
touch; the two runs differ in nothing but the contents of the `.bin` files.

## The manifest

Top-level keys: `manifest_version`, `lane`, `date`, `instrument`, `gguf`, `model`, `input`, `counts`,
`graphs`, `census`, `dumps`.

- `gguf` — `path`, `sha256`, `pinned_sha256`, `matches_pin`, `discovered_via`. The digest is computed
  by the dumper over the file it actually mapped, found through `/proc/self/maps` + `/proc/self/fd`,
  then compared to `checkpoints/Qwen3.6-35B-A3B.gguf-pin.json`. Every manifest in this lane carries
  `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61` once and `matches_pin: true`.
  `self` re-hashes the 22 GB file on disk rather than trusting that line.
- `instrument` — patch path, `llama_cpp_commit`, build-record path **and its sha256**, the derivation
  script, and the sizes of the derived name sets. The build record itself lives in `$WORK`; its digest
  is what pins which binary produced a dump. One field to read carefully: `build_flags` is empty in the
  three delivered manifests, because the driver extracted it with `grep '^[A-Z_]+='` and the build
  record keeps the flags inside `configure_command=`. It is not "no flags were used" — the record bound
  by `build_record_sha256` holds the whole cmake command line, and §The CPU-only gate quotes it. The
  driver now reads `configure_command=` instead, so a run made from here on fills the field.
- `model` — arch, `n_layer`, `n_layer_all`, `n_expert`, `n_expert_used`, `n_embd`, `linear_layers`
  (the 30 recurrent blocks), `full_attention_layers` (the 10 full blocks: 3, 7, 11 … 39 — every 4th,
  starting at 3) and `is_recr_by_layer`, the same fact as a per-layer 0/1 vector. `self` re-derives all
  of it from `checkpoints/Qwen3.6-35B-A3B.config.json`'s `text_config.layer_types`, so a llama.cpp
  `is_recr()` that disagreed with the pinned checkpoint would be caught rather than quietly labelling
  every census row wrong.
- `input` — the prompt verbatim, the full producing command line, the tag, `graphs_dumped`, and a note
  pointing at `graphs[]` for the token ids.
- `counts` — records, files written, bytes written, census lines, peak RSS in kB.
- `census[]` — one row per `(graph, block)` including `block: -1` (`kind: "model"`). Fields:
  `dumped matched unexpected anonymous state weights_excluded weights_unknown duplicates not_written
  too_big` plus the name sets `seen_expected`, `seen_unexpected`, `parameters`.
- `dumps[]` — one row per graph node considered: `graph block node seq name class op dtype shape bytes
  written sha256 file view_of`. `file` is absolute; `self` checks every written row's file exists, has
  that length and hashes to that digest.

Filenames are `g<graph>_blk_<block:03d>_<sanitized-name>_<seq>.bin` for blocks and
`g<graph>_mdl_-<n>_<sanitized-name>_<seq>.bin` for the model-level bucket, e.g.
`g0_blk_000_attn_norm_1.bin`, `g0_mdl_-01_model.input_embed_1.bin`. Names are sanitized for the
filesystem, so dots and spaces survive badly; the manifest's `name` field is the truth and `file` is
only a pointer to the bytes.

## Class taxonomy

Every node in the compute graph lands in **exactly one** of three write classes, plus parameters which
are never written:

| class | meaning |
|---|---|
| `matched` (`expected`) | the name is in the derived expected set for that block kind |
| `unexpected` | the graph builder named the tensor, and that name is not in the expected set |
| `anonymous` | the graph builder never named it: ggml's `node_%d` / `leaf_%d`, or a view derived from one |
| `state` | *not a fourth bucket* — a write whose view-chain root is a `cache_*` tensor. Drawn from all three classes |
| parameters | `ggml_is_param()`: excluded, never written, name-checked against the derived weight families and listed per block in `census[].parameters` |

### The arithmetic identity

`matched`, `unexpected`, `anonymous` are counts of **classifications**; `dumped` counts those whose
bytes were written and `not_written` counts those skipped (zero bytes, over the size cap, or bytes
turned off for that class). So:

```
dumped + not_written == matched + unexpected + anonymous        (the three classes are disjoint)
missing == expected - |seen_expected|                            (missing is a name set, not a count)
matched >= |seen_expected|, with equality iff no expected name is written twice
state <= dumped                                                  (state is a class of writes)
```

**`matched + missing != expected` and that is not a bug.** `matched` counts writes including
duplicates; `missing` counts distinct names never seen. A linear block has `matched=52` but
`expected - missing = 130 - 80 = 50` distinct names, because `norm` is the RMS_NORM name three times
in a block and `duplicates=10`. This is asserted in `verify.py self` for all 164 census rows, against
both the manifest and the log — the log additionally carries `missing`, so the two sources must agree
row for row and field for field.

`expected` is a **source-derived upper bound**, not a model-derived truth: it is every `cb()` name the
vendored builder can emit for a block of that kind, across every branch. So `missing` is branch
coverage, and it is not a failure. What *is* a failure, and fails the run loudly:
`BLOCK-EMPTY` (a block dumped nothing), `NO-PARAMS` (the parameter guard never fired for a block, so
nothing proves weights were excluded), a parameter whose family is not in the derived table
(`weights_unknown > 0`), a census line count other than `(n_layer + 1) × graphs`, or a GGUF digest
that is not the pin. Both block-level guards exempt `block: -1` — the model-level bucket legitimately
holds no `blk.<il>.*` parameters, which is why its row reads `weights_excluded: 0` and `parameters: []`.

## How ggml names tensors

The model's `build_*` methods do not name tensors; they call a `cb(name, il)` callback and the
builder keeps its own names. Three derivations show up in `name`, and mistaking one for another is the
main way this index gets misread:

1. **`<name>-<il>`** — the callback name with the layer suffix, e.g. `ffn_moe_out-7`. The census strips
   the suffix and looks `ffn_moe_out` up in the expected set for block 7's kind.
2. **`<src> (view)` / `<src> (reshaped)` / `<src> (permuted)` / `<src> (copy of <other>)`** — ggml
   derives a name for a tensor it has to name itself from whatever it was derived from, so
   `cache_s_l1 (reshaped) (view) (view)` is a scale of a view of a reshape of the recurrent-state
   cache. When the source is itself unnamed the prefix is empty and you get a name that is literally
   `" (view)"` — leading space, real name, appears in `seen_unexpected`.
3. **`node_%d` / `leaf_%d`** — ggml's fallback for anything left unnamed (`ggml_graph_add_node`). These
   are the `anonymous` class. They are not junk: `node_151` in a linear block is the
   `GGML_OP_GATED_DELTA_NET` node — the fused GDN kernel output — and `node_334` in a full block is
   `GGML_OP_FLASH_ATTN_EXT`. Attribution of these to a block uses the last block-named tensor in graph
   order, since they carry no `-<il>` suffix themselves.

A name is not a value. Where two classes disagree about what a tensor is, the `op`, `dtype`, `shape`
and `view_of` fields on the record are the evidence; the name is a hint from a third-party builder.

Anonymous nodes have no `-<il>` suffix to attribute them with, so they are assigned to the block of the
last block-named node in graph order. That is a heuristic, and it moves a count between columns and
never a byte; `dumped + not_written == matched + unexpected + anonymous` still holds per block, and the
per-graph totals are printed in `counts` so the attribution can be cross-checked rather than trusted.

## What is NOT dumped, and why

| absent | count | why |
|---|---|---|
| weights / parameters | 19 names per linear block, 16 per full block, listed per block in `census[].parameters` | A weight is reachable only as a *source* of a node, never as a node itself, so the walk finds them by scanning each node's `src[]` for leaves that are not nodes. Each is classified by `ld_param_family` (`blk.<il>.<family>.<suffix>`) and, if the family is in the derived table, counted in `weights_excluded` and never written. The 19 linear families here are `attn_gate, attn_norm, attn_qkv, ffn_down_exps, ffn_down_shexp, ffn_gate_exps, ffn_gate_inp, ffn_gate_inp_shexp, ffn_gate_shexp, ffn_up_exps, ffn_up_shexp, post_attention_norm, ssm_a, ssm_alpha, ssm_beta, ssm_conv1d, ssm_dt, ssm_norm, ssm_out`; the full block's 16 replace the seven `ssm_*`/`attn_qkv`/`attn_gate` with `attn_q, attn_q_norm, attn_k, attn_k_norm, attn_v` |
| `token_embd` and `output` (`lm_head`) | not counted anywhere | also weights, but their names are not `blk.<il>.*`, so `ld_param_family` declines them: they are neither dumped nor tallied. Nothing in a manifest claims their bytes; the only binding to them is the GGUF digest. Their *outputs* are dumped: `model.input_embed` (a `GET_ROWS` on the embedding table) and `result_output` (the logits) |
| the graph **input leaves** | 3 | `inp_tokens`, `inp_pos`, `embd` are graph inputs, not members of `ggml_cgraph::nodes[]` at this revision, so the walk never sees them. Their content is recorded a different way: token ids and `n_pos` come from the `llama_ubatch` the graph was built from, into `manifest.graphs[]` |
| zero-byte tensors | 6 per linear block in the prefill graphs and 10 in the decode graphs; 0 per full block; 1009 records over a 4-graph run | the pre-state reads whose slice has a zero-length dimension at position 0: `cache_r_l<N> (view)`, `cache_s_l<N> (view)` and their copies. Nothing to write; `written=0 reason=zero-bytes`. Full blocks have no recurrent state, so they have none of these |
| size-cap skips | 0 (`too_big=0` everywhere) | the cap is 64 MiB per tensor (`LLAMA_DUMP_MAX_BYTES`, default `64u << 20`) and no activation reached it; the largest tensor written was the fused GDN node at 2 277 376 B |
| every name in `missing` | 80 per linear block, 68 per full block | branch coverage, see below |
| anything outside a graph node | — | the capture is the scheduler's per-node callback over `ggml_cgraph::nodes[]`, so only values that pass through a node are readable. KV-cache contents as *stored* are visible only where a node touches them (`cache_k_l3 (view)`/`cache_v_l3 (view)` as `SET_ROWS`/`VIEW`/`PERMUTE`); `cache_r_l`/`cache_s_l` likewise. There is no cache read-back of the parts no node reads |

`missing` for a linear block is 80 names and they are not mysterious: they are the branches this
checkpoint does not take. All of them are either the **unfused GDN path** (`dnet_add_ar_state`,
`dnet_add_ch_*`, `q_in`, `k_in`, `v_in`, `g_in`, `g_diff`, `g_cs`, `g_last`, `g_last_exp_t`,
`k_cumdecay`, `k_beta_g_exp`, `key_gdiff`, `key_gdiff_t`, `v_prime`, `v_t_new`, `v_attn`, `v_b`,
`k_b`, `b_in`, `decay_mask`, `attn_pre_solve`, `attn_inter`, `output_state`, `v_conv`, `g_last`) or an
**FFN/MoE variant this model does not configure** (`ffn_gelu*`, `ffn_silu*`, `ffn_reglu`,
`ffn_geglu*`, `ffn_swiglu_oai`, `ffn_moe_situ`, `ffn_moe_relu*`, `ffn_moe_silu*`, `ffn_moe_up_*`,
`ffn_moe_gate_up*`, `ffn_moe_weights_softmax`, `ffn_moe_slots`, `ffn_moe_group_topk`,
`ffn_moe_post_exp_norm`, `norm_w`, `post_moe`, `kq`). A full block's 68 are the same FFN/MoE group
plus the paths it does not use: `wqkv`, `wqkv_b`, `wqkv_clamped`, `kq*`, `kqv`, `kqv_out`, `kqv_mla`,
`fattn_mla`, `v_cont`, `Qcur_clamped`, `Kcur_clamped`, `Vcur_clamped`, `norm_w`, `post_moe`, `kq`.
The full list for any block is on the log's `CENSUS-NAMES graph=G block=B missing=...` line, pipe-
separated, and the taken set is `census[].seen_expected`.

The fused path is the interesting consequence: `qwen35moe.cpp` takes
`build_gated_delta_net`'s fused branch, so the per-chunk intermediates of the delta rule never exist
as nodes. What you get instead is `node_151` (`GATED_DELTA_NET`, shape `[4096, 139, 1, 1]` in a
prefill) plus `state_predelta` going in and `new_state` / `attn_output` coming out as views of it. If
you want the unfused intermediates, that is a different build (or the CPU oracle), not this one.

## Controls

`determinism` and `prompt-change` compare **files on disk**, not the manifest's memory of them, keyed
on `(graph, block, name, op, dtype, shape, seq)` — `seq` is the instrument's per-`(graph, block,
name)` occurrence counter, and without it the eight `ffn_moe_weighted-<il> (view)` per block and the
repeated `norm` would collapse into one key and never be compared.

- same prompt, two runs (`run-a`, `run-b`): 14267 written files per run (and 14267 `.bin` files on
  disk, which is what `counts.files_written` claims), `compared=14267 differing=0`. Byte-identical
  across two separate processes of the CPU build at `-t 8`. This is the result that makes the dump
  usable as a reference at all.
- changed prompt (`France` → `Italy`, `run-a` vs `run-c`): `compared=14267 differing=10120`, in every
  block including the last (`blocks with a change: [-1, 0, …, 39]`), with `result_output` differing in
  both the assistant-header graph and the decode graph. Per graph, from
  `llama_dump_probe.py … graphdiff`:

  | graph | written | compared | differing | same |
  |---|---|---|---|---|
  | 0 (init) | 3639 | 3639 | 0 | 3639 |
  | 1 (prompt prefill) | 3590 | 3590 | 3166 | 424 |
  | 2 (assistant header) | 3519 | 3519 | 3482 | 37 |
  | 3 (first decode) | 3519 | 3519 | 3472 | 47 |

  Graph 0 being prompt-independent is the control working, not failing. The 508 unchanged rows inside
  graphs 1–3 are the tensors that cannot depend on the one token that differs — cache reads at
  positions the prompt has not reached, and tensors of the pre-divergence prefix — and per-name
  equality is checkable with `llama_dump_probe.py … same-diff`. Why any individual row coincides is
  not established here; the claim being made is only that a changed prompt moves the logits and the
  last block, which is what this control fails on.
- **the aggregate fingerprint of a run**, for anyone who wants one number to compare:
  `find . -name '*.bin' | sort | xargs sha256sum | sha256sum` over the dump directory. `run-a` and
  `run-b` both give `f78f703e0c556917c2008f9941e895b9b51689fec799aa588f7efe2cd3749a0c`; `run-c` gives
  `97bfe7bce1b43dc3a88444fcd946a39461bdda98ff7f08b81ed9ff6bc1f05b8c`. `sha256sum` prints the file name
  on each line, so the fingerprint covers the names too — a rename changes it.
- aliasing (§Why the capture is a per-node callback): `self` fails if a name written in ≥5 blocks of
  one graph carries a single non-zero digest. 0 groups on the current runs, 149 on the superseded
  post-compute walk. Determinism and prompt-change both pass on aliased data, so this is the only one
  of the three that can see it.

Both controls treat `compared == 0` as a failure. A comparison that compared nothing must not print
PASS (AGENTS.md §a check that can produce no output cannot fail).

## Stage mapping — hypotheses, for the main agent to adjudicate

Suggested correspondences between the llama.cpp names and our stages, offered as guesses a reviewer
should check against `specification/` before anyone leans on them. The dtype column is load-bearing:
llama.cpp runs Q4_K_M weights and accumulates to **f32 activations**; our pinned oracle dumps are
bf16. Any comparison across the two is a comparison between different numeric systems, which is
exactly why no band may be sized from these files.

| our stage (guess) | llama.cpp tensors |
|---|---|
| input embedding | `model.input_embed`, `node_1` (both `GET_ROWS`), graph input `embd` (not dumped) |
| per-block pre-norm | `norm` (RMS_NORM) → `attn_norm` (MUL) |
| GDN qkv projection | `node_117` (MUL_MAT) → `linear_attn_qkv_mixed` → `qkv_mixed_transposed` |
| conv branch | `conv_input` (CONCAT with the state window), `conv_output_raw` (`SSM_CONV`), `conv_output_silu`, `q_conv`/`k_conv`/`v_conv_predelta` |
| recurrent state read/write | `cache_r_l<N>` (conv state) and `cache_s_l<N>` (recurrent state) as `RESHAPE`/`VIEW`/`SCALE`/`GET_ROWS`/`CPY`, `conv_state_last`, `conv_state_update`, `new_state` |
| delta rule itself | `alpha`, `a_softplus`, `beta`, `beta_sigmoid`, `gate`, `state_predelta`, `node_151` (`GATED_DELTA_NET`), `new_state`, `attn_output` |
| GDN output norm + gate | `norm` (seq 2), `node_157` (MUL), `z` (MUL_MAT), `node_160` (UNARY), `node_161` (MUL), `linear_attn_out` |
| attention (full blocks) | `Qcur_full`/`Qcur_reshaped`/`Qcur_normed`/`Qcur`, `Kcur*`, `Vcur*`, `node_334` (`FLASH_ATTN_EXT`), `attn_pregate`, `gate_reshaped`, `gate_sigmoid`, `attn_gated`, `attn_output` |
| router | `ffn_moe_logits` (MUL_MAT on `ffn_gate_inp`), `ffn_moe_probs` (`SOFT_MAX`), `ffn_moe_argsort` (`ARGSORT`, i32), `ffn_moe_topk` (VIEW), `ffn_moe_weights` (`GET_ROWS`), `ffn_moe_weights_sum`, `ffn_moe_weights_sum_clamped` (`CLAMP`), `ffn_moe_weights_norm` (`DIV`) |
| routed experts | `ffn_moe_gate`/`ffn_moe_up`/`ffn_moe_down` (`MUL_MAT_ID`), `ffn_moe_swiglu` (`GLU`), `ffn_moe_weighted`, the eight `ffn_moe_weighted-<il> (view)` slices, `node_192…197` (the six `ADD`s that reduce them), `ffn_moe_out` |
| shared expert | `ffn_gate`, `ffn_up`, `ffn_swiglu`, `ffn_shexp`, `shared_expert_gate`, `shared_expert_gate_sigmoid`, `ffn_shexp_gated` |
| block residual | `attn_residual`, `ffn_out`, `l_out` |
| final norm / logits | `result_norm`, `result_output` |

Two cautions the row above cannot carry: `ffn_moe_weights_norm` is the renormalisation, and prediction
4 in `AGENTS.md` says the HF form casts those weights back to the logits dtype **before** the expert
multiply — llama.cpp's `DIV` here is f32, so this is not the same number and is not evidence about
that ordering. And the eight `ffn_moe_weighted-<il> (view)` slices are the only place the per-expert
outputs are individually visible, so a route-order question is answerable from them and from nothing
else in this dump.

## Reading the values

`tools/llama_dump_probe.py` is a read-only aid over a manifest's `.bin` rows. It prints; it decides
nothing — the gates live in `llama_dump_verify.py`.

| invocation | prints |
|---|---|
| `<manifest> <names> [graph]` | the recorded range and total of each named row: the way to see that `ffn_moe_probs` lies in (0,1) and sums to 1 per token while `ffn_moe_logits` does not |
| `<manifest> digests <names> [graph]` | how many rows of a name share one digest, largest group first — the eyeball form of the aliasing check |
| `<manifest> same-diff <other> <names> [graph]` | which blocks' digests moved between two runs |
| `<manifest> graphdiff <other>` | per-graph written / compared / differing / same counts |
| `<manifest> names <graph> <block>` | every row of one block in node order: node index, class, op, dtype, shape, seq, written flag, verbatim name |
| `<manifest> namelist <graph> <block>` | the same rows as one `class:name[op]` line, for quoting |

## Knobs

| env | instrument default | driver passes | effect |
|---|---|---|---|
| `LLAMA_DUMP_DIR` | unset | the tag's dir | **unset disables the instrument entirely** — the hook returns before touching the graph. Required together with `LLAMA_DUMP_MANIFEST`, else `ld_fatal` |
| `LLAMA_DUMP_MANIFEST` | — | `checkpoints/llama-dump-manifest-<date>-<tag>.json` | where the index is written, once |
| `LLAMA_DUMP_MAX_GRAPHS` | 1 | 4 | hard cap on graphs captured; at the cap the hook returns without a message, and `DONE graphs=N` plus the `graphs` list are how you see the cut |
| `LLAMA_DUMP_MAX_BYTES` | 67108864 | — | per-tensor cap; over it is `reason=size-cap`, counted in `too_big`. The largest activation actually written is 2277376 B — the fused GDN node — so the cap never fired (`too_big=0` in all 164 rows) and `size-cap` never appears in a log |
| `LLAMA_DUMP_UNNAMED` | 0 | 1 | also write the bytes of `anonymous` tensors. Keep it on: in the fused GDN path the node carrying the post-update recurrent state is one of them |
| `LLAMA_DUMP_UNEXPECTED` | 1 | — | also write the bytes of `unexpected` tensors. `state` tensors have no knob of their own: they are written or skipped by whatever class reached them, and counted in `state` either way |
| `LLAMA_DUMP_GGUF_PIN_SHA256` | — | from `checkpoints/Qwen3.6-35B-A3B.gguf-pin.json` | 64 hex chars or the instrument refuses to run; the discovered GGUF digest must equal it |
| `LLAMA_DUMP_GGUF_EXPECTED_PATH` | — | the pinned GGUF | cross-check that the file found in the address space is the one we asked for |
| `LLAMA_DUMP_BUILD_RECORD`, `LLAMA_DUMP_BUILD_FLAGS` | — | from the build record | recorded so the manifest names the binary that produced it |
| `LLAMA_DUMP_CMDLINE`, `LLAMA_DUMP_PROMPT`, `LLAMA_DUMP_TAG` | — | from the driver | recorded verbatim in the manifest |

Two guards, not one convenience: the instrument **creates** its dump directory, but it refuses to run
at all if that directory already exists and is non-empty (`FATAL refusing to write into non-empty dump
dir`, `ld_fatal("append-only dump dir")`) or if the manifest path already exists (`refusing to
overwrite existing manifest`, `write-once manifest`). It also refuses to start with an empty derived
name set, an unhexed pin, or a missing `LLAMA_DUMP_DIR`/`LLAMA_DUMP_MANIFEST`. `tools/llama_dump_run.sh`
repeats the write-once and append-only checks before launching, so a bad tag fails in the shell rather
than inside the model load.

## Regenerating the name sets

`tools/llama_dump_expected_gen.py --write` after any llama.cpp bump, then `regen`, `build`, and re-run.
The counts are printed and asserted in `status`; `weights_unknown` in the census is the witness that
the weight-family table is being exercised (AGENTS.md §a counter that reads exactly 0 is vacuous until
witnessed non-zero somewhere it must fire) — it is 0 in all 164 rows here, and the guard that produces
it is the same code path that produced 19/16 `weights_excluded` per block, so it fired.

If the vendored source changes and the header is stale, the symptom is `unexpected` growing: a name
the builder now emits that the header never heard of. That is the check doing its job, not noise —
re-derive rather than adding the name by hand, because a hand-typed expected set is a set that agrees
with whatever the build happens to produce.

`tools/llama_instrument_apply.sh status` prints HEAD, the porcelain line count, whether the patch is
applied and the build dir. It is the command that answers "did I leave the tree dirty?" without a git
incantation, and it is the last thing to run before you stop.
