# M3 unblocked: the two oracles share a stimulus (2026-10-27)

What this file is: the evidence that the blocker named in
`quantization/evidence/llama-stimulus-alignment-2026-10-27.md` and
`specification/evidence/cross-oracle-three-way-2026-10-27.md` — three sides, three stimuli, therefore
nothing numeric comparable — is gone, and what the aligned comparison actually says. CPU-only
throughout: the oracle binary is the CPU build gated by `tools/llama_instrument_apply.sh` and
`tools/llama_tokens_build.sh` (`ldd`: zero ROCm/HIP libraries, `libllama-common` absent), `-ngl 0`,
no device act, no lock, and **no rate or duration of any kind appears in this file** — host RSS only,
read out of the manifests. `grep -c gpu_lock` is 0 on every tool named here.

No band, floor, tolerance or licensed pin was authored here. The fixture this file describes is
`status=candidate`, `registration=not-registered`: registering pins is the user's reserved act
(DECISIONS.md D-18).

## The door that was left

The earlier finding stands: no **text** reproduces the chain's 32 sampled ids (detokenize → 205 bytes →
re-tokenize → 30 of 32, first divergence index 3, and the byte string is forced by the ids). That file
named two exits; re-dumping the chain would replace pinned fixture material, so this took the other —
an id-input channel. `tools/llama_tokens_dump.cpp` is a *client* of the vendored library, not a patch
to a vendored front end, and `engines/llama.cpp` was never dirty:
`tools/llama_instrument_apply.sh status` at the end of the work reported HEAD
`d327bf1f0780f87b8c01b367b61853b0c291d7c4`, `porcelain : 0 line(s)`, `patch applied : no`. The
instrumented library it links is the one already built at the pinned commit
(`/tmp/llama-instrument-vendored/build-cpu/bin/libllama.so`, carrying `llama_dump_before_graph`).

The input path, named so the claim is falsifiable:

```
tokens file (decimal ids) -> llama_batch.token[i]  ->  llama_decode(ctx, batch)
```

No tokenizer, no chat template, no BOS step — and not only uncalled but unlinkable: `ldd` on
`llama-tokens-dump` shows no `libllama-common`, the library that owns `llama-chat.cpp` and
tokenization in this tree. That gate is witnessed in both directions (gates 5–7 of
`tools/llama_tokens_build.sh`: `libllama-common` must be absent from `llama-tokens-dump` **and**
present in `llama-cli`), so it cannot pass by looking at nothing. `checks ran: 8 / expected 8`.

## Item 1 — the aligned stimulus, verbatim

Driver: `tools/llama_aligned_run.py run d3` (and `run d4`, the same stimulus a second time).

```
chain   n_ids=32 token_ids_sha256=f87335bb69b4e413c4f7de5ed4077ea93acddaa22145ec71c6748539fa419631 encoding=int64-le-per-id recomputes=True
oracle  graphs_captured=1 (cap was 1: one ubatch of 32 tokens is one graph)
oracle  graph 0 n_tokens=32 n_seqs=1 token_ids_sha256=f87335bb69b4e413c4f7de5ed4077ea93acddaa22145ec71c6748539fa419631
align   token_ids_sha256=f87335bb69b4e413c4f7de5ed4077ea93acddaa22145ec71c6748539fa419631  chain_manifest_token_ids_sha256=f87335bb69b4e413c4f7de5ed4077ea93acddaa22145ec71c6748539fa419631  equal=True (elementwise 32/32, length_equal=True)
```

`token_ids_sha256` is the sha256 of the ids the **oracle** recorded consuming — the instrument's own
`graphs[0].token_ids`, read from the `llama_ubatch` inside `llama-context.cpp::process_ubatch()`, not
echoed by the driver — under the encoding the chain manifest itself uses (sha256 of each id as a
little-endian int64; the driver first verifies that this encoding reproduces the chain manifest's own
`token_ids_sha256` field, `recomputes=True`, before trusting an equality computed with it). The check
reddens: feeding only the first 4 ids printed
`equal=False (elementwise 4/32, length_equal=False)` and exited `verdict=REFUSED tag=token-ids`.

`template_applied=no`, both halves:

```
template template_applied=no  (API path: file -> batch.token[] -> llama_decode; the binary does not
        link libllama-common, which owns chat templating and tokenization)
template control-class ids (>= 248044) among the 32 ids consumed: none
template witness: the same detector run over llama-dump-manifest-2026-10-27-run-a.json graphs[1].token_ids
        finds 3 control-class id(s) [248045, 248046, 248045] — the detector is not vacuous
template length witness: the template-wrapped prefill carried 11 ids for a 5-token text; this run
        carries 32 ids and every one of them is a chain id, so nothing was added around it
```

Weights read, and CPU-only, both read off the artifact:

```
gguf    gguf_sha256=ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61 pinned_sha256=ac0e2c11…53e31a61 matches_pin=True
cpuonly n_gpu_layers=0 recorded, no device-shaped flag in the command=True
cpuonly peak_rss_kb=33587880 (host RSS from the manifest; no timing of any kind is recorded here)
```

## The determinism pair (D-26 clause 4, re-established on THIS stimulus)

`tools/llama_aligned_run.py pair d3 d4`, keyed on
`(graph, block, name, op, dtype, shape, seq)` the way `tools/llama_dump_verify.py` keys, so a repeated
name cannot collapse:

```
pair    run d3 rows_written=3639 aggregate_digest=b9f83aa152c1984108d5408aac292f8d3ca8c00bf47298e2d3eca370be72a86c
pair    run d4 rows_written=3639 aggregate_digest=b9f83aa152c1984108d5408aac292f8d3ca8c00bf47298e2d3eca370be72a86c
pair    keys compared=3639 differing=0 only_in_first=0 only_in_second=0 aggregate_equal=True
pair    control: the same comparison against the template-wrapped run-a gives keys_compared=649 differing=229 — the comparator reports a difference when one exists
pair    logits digest equal across the pair=True (n_vocab=248320)
DETERMINISM verdict=DETERMINISTIC pair=d3,d4 tensors_compared=3639 differing=0 aggregate_digest=b9f83aa152c1984108d5408aac292f8d3ca8c00bf47298e2d3eca370be72a86c no timing taken
```

Two anti-vacuity legs are part of that result rather than footnotes to it. The comparator was run against
a *different* stimulus and reported 229 differences out of 649 shared keys, so `differing=0` above is a
property of the oracle and not of the comparison. And the aggregate digest is a function of per-tensor
digests that were themselves re-verified: `460 of 3639` recorded row digests were re-hashed from the
bytes on disk (every written row of the four HF blocks, every model-level row, and every 37th written
row so no block is unrepresented) with 0 mismatches; a single mismatch aborts the run.

## Item 2 — the candidate fixture

`checkpoints/m3-candidate-llama-fixture-2026-10-27.json`, written by
`tools/llama_aligned_run.py candidate d3 d4`. `status=candidate`,
`registration=not-registered — adding or replacing pinned fixtures is reserved to the user (D-18)`.
Fields: `stimulus` (the ids, the sha, the three-part template-free proof), `oracle` (gguf sha + pin +
`matches_pin`, `llama_cpp_commit=d327bf1f…`, front end + its build record, the instrument manifest and
that manifest's own sha256, the producing command), `determinism` (both aggregate digests,
`tensors_compared=3639`, `tensors_differing=0`, and the non-vacuity control's numbers), `logits`,
`coverage`, `crosscheck` (430 per-block rows: block, name, op, dtype, shape, sha256, node, seq, file),
and `not_licensed`.

Logits and greedy argmax:

```
logits  logits_sha256=68315e1a0aae1fea3774104d1f3cdab51719fddd98c51e760c1a08fea44b507f logits_bytes=993280 n_vocab=248320
logits  instrument's own result_output row: graph=0 shape=[248320, 1, 1, 1] bytes=993280 sha256=68315e1a0aae…507f equal_to_front_end_logits=True
logits  argmax=82 argmax_logit=9.12432384 top5=82:9.12432384 7:8.34050655 198:8.29955387 13:7.93430614 62:7.8670187
```

`result_output` is one row because llama.cpp crops to the output position before the last block
(`qwen35moe.cpp`'s `crop_before_nextn(inp_out_ids)` gather at `il == n_layer-1`), so the referent is
"the logits after the 32nd token", and the row the instrument captured and the row the front end read
back through `llama_get_logits` are the same bytes by digest. Greedy argmax is over all 248320 rows
with no penalties, no bad-token mask, no grammar; the rank-1/rank-2 margin is `9.12432 − 8.34051 ≈ 0.78`,
printed so nobody has to guess whether the choice was a near-tie.

Run facts: `records=3819`, `files_written=3639`, `bytes_written=2313656464`, `census_lines=41`
(40 blocks + the model bucket, one graph), `layers_covered=40`, `peak_rss_kb=33587880`.

**The limitation, stated in the manifest and here.** Full-40-block logits/activation truth is available
**only** from the quantized oracle: `models/qwen36-src/` carries 4 of 26 safetensors shards — layers
0, 1, 38, 39 — so HF can calibrate **layer0 and layer1 numerically**, and **layer38/layer39 are
structural-only** because the pinned chain reaches them from layer1 with layers 2..37 absent
(`checkpoints/dump-manifest-chain-2026-10-26.json` `block_set.why_not_contiguous`). There is no bf16
reference for layers 2..37 on this box, and no HF logits record for any block: the chain fixture is
per-block stages and contains no `lm_head` or logits row at all (D-25). HF's four acquired blocks are
therefore the *calibration* surface; the other 36 have only the oracle.

## Item 3 — what the aligned comparison says (measured, not adjudicated)

`engine/cross_oracle.py --mode aligned --aligned-manifest …` — 23 counted preconditions, then the HF ×
llama distance on the two blocks HF can reach, each line labelled
`licence=llama-crosscheck(observation-only)`, `band=NONE`, `adjudicated=no`:

```
RESULT    mode=aligned checks=23 findings=0 aligned=True escalations=14 bands-authored=none
          licence-issued=llama-crosscheck(observation-only) verdict=ALIGNED-OBSERVED no timing taken
layout  cell=layer0.out relL2[token-major re-view]=2.757e-02 relL2[raw re-view]=1.416e+00
        chosen=token-major margin=51.4x
```

The `layout` line is one of the 23 checks, and it is there because HF wrote torch `[tokens, hidden]`
while ggml shapes are `[hidden, tokens]` with `ne[0]` fastest. Both re-views are computed for one cell
and the chosen one must win by >5x — a wrong re-view scrambles element order and lands near √2 relative
error, 51× worse here. Without it, "we transposed" is a sentence, and a sentence cannot fail.

| block.stage | max_abs_delta | relL2 | hf_rms | llama_rms |
|---|---|---|---|---|
| layer0.x (embedding rows) | 2.608e-04 | 5.367e-03 | 0.0126 | 0.0126 |
| layer0.h_in | 2.630e-02 | 5.609e-03 | 1.0294 | 1.0295 |
| layer0.mixer_out | 4.852e-03 | 1.724e-02 | 0.0191 | 0.0191 |
| layer0.x1 | 4.638e-03 | 1.471e-02 | 0.0228 | 0.0228 |
| layer0.h_mlp | 1.445e-01 | 1.832e-02 | 0.7064 | 0.7066 |
| layer0.moe_out | 5.528e-03 | 7.350e-02 | 0.0077 | 0.0077 |
| layer0.out | 5.637e-03 | 2.759e-02 | 0.0236 | 0.0236 |
| layer1.x (== layer0.out) | 5.637e-03 | 2.759e-02 | 0.0236 | 0.0236 |
| layer1.h_in | 2.919e-01 | 2.800e-02 | 1.0424 | 1.0425 |
| layer1.mixer_out | 9.067e-03 | 3.547e-02 | 0.0097 | 0.0097 |
| layer1.x1 | 1.113e-02 | 2.479e-02 | 0.0272 | 0.0272 |
| layer1.h_mlp | 2.145e-01 | 3.327e-02 | 0.7383 | 0.7384 |
| layer1.moe_out | 1.039e-02 | 8.152e-02 | 0.0138 | 0.0138 |
| layer1.out | 1.596e-02 | 4.043e-02 | 0.0323 | 0.0322 |

Three things that follow from those numbers, none of them a band:

1. **The mapping is corroborated by the data, not only by the names.** `layer1.x` and `layer0.out`
   agree to the last printed digit because they are the same tensor reached two ways — the chain feeds
   layer1 from layer0's output, and the mode maps `x` of layer *N* to `l_out` of block *N−1*. Had the
   transpose or the block index been wrong, that identity would not have held.
2. **The disagreement is a width difference, and it is already visible before any block runs.** The
   first row compares the embedding rows HF read from the bf16 shard against llama.cpp's `GET_ROWS` on
   the Q8_0 table: relL2 5.4e-3 with `max_abs_delta` 2.6e-4 on a tensor of rms 0.0126. Everything
   downstream inherits that. This is refused-adjudication #1 (activation and weight width) becoming
   measurable, not settled — D-26 clause 3 keeps the tighter HF limit and forbids adopting the
   quantized side, so all 14 rows are escalations, not accepts.
3. **relL2 grows monotonically along the stages that matter** (moe_out is the largest at ~7–8e-2 on a
   small-rms tensor), which is the shape a logits band would have to price after 38 more blocks. That is
   the honest argument for why the logits band cannot be guessed from a per-block number, and it is in
   `engine/HANDOFF.md` §PASS 15 §6 as the user's choice, not ours.

The prior structural result did not move: `engine/cross_oracle.py` with no flags still prints
`RESULT checks=78 findings=0 aligned=False no bands authored=no licence-issued=none
verdict=STRUCTURAL-OK no timing taken`. What changed in that default path is one wording line —
`align unattainable-through-frontends=yes` became `unattainable-through-text=yes` plus
`attainable-through-raw-ids=yes`, because the text proof is untouched and the id door is now built.
`--mode align --llama-manifest <aligned manifest>` prints `align ALIGNED graphs=0` and exits 0.

## What is still not done

- No engine number. `engine/block_forward.py` is a one-block, one-token lane, so every engine-vs-oracle
  cell is 0 and G1 of `engine/HANDOFF.md` §PASS 15 §5 is a specification, not a run.
- The candidate's bytes are scratch (`/tmp/m3-align/dumps/<tag>/`, digests in the instrument manifest),
  because `data/` was closed to this lane. Moving them into `data/dumps/…` is part of registration and
  is the user's act.
- No band, no licensed pin, no must-catch set, and no adjudicated HF-vs-llama disagreement anywhere in
  this file: 14 divergences printed, 0 adjudicated.

Reproduce:

```
WORK=/tmp/llama-instrument-vendored tools/llama_instrument_apply.sh status      # pristine, pinned
tools/llama_tokens_build.sh                                                   # 8/8, no libllama-common
tools/llama_aligned_run.py run d3 && tools/llama_aligned_run.py run d4
tools/llama_aligned_run.py pair d3 d4
tools/llama_aligned_run.py candidate d3 d4
engine/cross_oracle.py --mode aligned --aligned-manifest /tmp/m3-align/manifests/llama-aligned-d3.json
```

CPU-only work; no device number appears in this file. **no timing taken.**
