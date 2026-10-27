# Stimulus alignment: can llama.cpp be fed the chain's pinned token ids? (2026-10-27)

Question asked by `engine/cross_oracle.py`: the HF chained dumps and the llama.cpp dumps are two
different forwards, so before any per-stage number is compared, does a single stimulus exist — i.e.
can the vendored instrumented llama.cpp be fed **exactly** the 32 `token_ids` the HF chain used,
with the chat template out of the way?

Answer, measured: **no**. Exact raw-token prefill of those ids is unattainable through the vendored
front ends, so `engine/cross_oracle.py` runs structural-only and every numeric line prints
`UNCOMPARABLE-BY-STIMULUS`. No band, floor or tolerance was authored here or by that tool
(`RESULT … licence-issued=none`).

CPU-only throughout: no iGPU, no device lock, no rate of any kind. **no timing taken.**

## The two stimuli, as the manifests state them

| side | stimulus | evidence |
|---|---|---|
| HF chain | 32 ids, `sha256 f87335bb69b4e413c4f7de5ed4077ea93acddaa22145ec71c6748539fa419631` | `checkpoints/dump-manifest-chain-2026-10-26.json` `input.token_ids`, `input.token_ids_sha256` |
| llama.cpp run-a | prefill graph = 11 ids `248045 846 198 760 6511 314 9338 369 248046 198 248045`; next graph = 4 ids `74455 198 248068 198`; decode graph = 1 id `8160`; warm-up = `0 0` | `checkpoints/llama-dump-manifest-2026-10-27-run-a.json` `graphs[]`, `input.prompt` = `"The capital of France is"` |

The chain's 32 ids, as numbers (never as template tokens):

```
75767 48645 25290 78392 55186 7464 2393 70962 18630 58883 68803 22545 13456 67839 23233 60377
30202 19060 79456 93321 35648 86608 64588 87257 60667 40221 9956 21117 27930 80746 5867 87935
```

Two facts about that list matter:

* **max id 93321 < 248044**, so the chain stimulus contains no control-class id at all — it is not a
  rendered conversation.
* the manifest says how it was made: `input.kind = "embedding_slice"`, `input.method = "torch.randint
  over the first 100k vocabulary rows …"`. The ids were **sampled**, not produced by a tokenizer.

llama's `gguf_sha256 = ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`,
`matches_pin = true` against `checkpoints/Qwen3.6-35B-A3B.gguf-pin.json`, instrument =
`tools/llama-instrument.patch` on `d327bf1f0780f87b8c01b367b61853b0c291d7c4` — so the two sides agree
on the weights; they disagree on everything the weights are fed.

## The instrument has no token-id input

`tools/LLAMA_DUMP_NOTES.md` §The CLI surface lists the whole surface of the instrumented binary:
`-m -p -n -ngl -c -b -ub -t --temp -s --advance-token` plus the `--op-dump-*` flags, and the env knobs
`LLAMA_DUMP_DIR`, `LLAMA_DUMP_EAGER`, `LLAMA_DUMP_TOKENS`, `LLAMA_DUMP_DEBUG`. `--op-dump-tokens`
(`tools/llama-instrument.patch`, `dump_want_tokens`) **filters which token positions get written**; it
does not choose the input. Nothing in the patch or the CLI accepts a list of ids. Text is the only
input channel, so the question reduces to: is there a text whose tokenization is those 32 ids?

## The round trip, measured

Built the CPU tokenizer from a **pristine** copy of the vendored tree (no instrument patch,
`-DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF`, build dir `/tmp/llama-align-cpu/build`), then:

1. Detokenized the 32 ids through the byte-level inverse map read out of the GGUF's own vocab
   (`tokenizer.tokens`, via `tools/chat_render.py::read_kv` — `tools/gguf_reader.py` truncates the
   tokens array to `MAX_ARRAY`, so read the full array before using it). Result: **205 bytes, valid
   UTF-8, no control-class token** → `/tmp/wprobe/chain_stimulus.bin`.
2. `llama-tokenize -m <gguf> -f /tmp/wprobe/chain_stimulus.bin --ids` (that is the front end's own
   usage line, `engines/llama.cpp/tools/tokenize/tokenize.cpp:25`; it calls `common_tokenize(vocab,
   prompt, add_bos, parse_special)` and applies no chat template) → 32 ids again, **30 of 32 equal**.
   Divergences:

| index | chain id | re-tokenized id |
|---|---|---|
| 3 | 78392 | 750 |
| 4 | 55186 | 74340 |

3. Control that the channel is template-free: tokenizing the run-a prompt text `"The capital of
   France is"` returns **`760 6511 314 9338 369`** — exactly the prefill graph's inner ids 3..7. The
   wrapper ids (`248045`, `846`, `198`, `248046`, `198`, `248045`) came from the chat template the CLI
   applied, not from the tokenizer.

Why the two mismatches are not a tooling artifact but a proof of impossibility: this is a byte-level
BPE, tokenization preserves bytes, and the token→piece map is a bijection, so any text with
`tokenize(text) == ids` must have `bytes(text) == concat(piece(ids[i]))` — that is, the 205-byte
candidate above is the **only** candidate. It fails at index 3 because the merge boundary between the
pieces of ids 2 and 3 is not the boundary BPE would choose. A sampled id sequence has no reason to be
a fixed point of its own tokenizer, and this one is not.

## Consequences, recorded where they bind

* `engine/cross_oracle.py` treats alignment as a precondition. `align` mode prints this whole
  finding and returns non-zero; `numeric` mode refuses and prints `UNCOMPARABLE-BY-STIMULUS` per pair
  per block with its denominator (0 elements compared of 84 possible stage-pair comparisons).
* Even if alignment were achieved, the honest HF-vs-llama scope is layer0 and layer1 only: the chain
  is `layer0 → layer1 → layer38 → layer39` with layers 2..37 absent
  (`block_set.why_not_contiguous`), so layers 38/39 mix a chain gap with any implementation
  difference. `engine/cross_oracle.py` prints that as a refused adjudication, not a verdict.
* Nothing here needs a new llama run: run-a (`checkpoints/llama-dump-manifest-2026-10-27-run-a.json`)
  was reused, `engines/llama.cpp` stayed pristine (`git status --porcelain` empty, HEAD
  `d327bf1f0780f87b8c01b367b61853b0c291d7c4`), and the tokenizer build was a scratch copy under
  `/tmp`.
* Making the stimuli share a token sequence requires one of: an id-input knob in the vendored front
  end (a new patch), or re-dumping the chain on a text-tokenizable stimulus. The second replaces
  pinned fixture material, which is the user's call, not an agent's.

Reproduce with:

```
# 1. detokenize the pinned ids (byte-level inverse map from the GGUF vocab)
/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python /tmp/wprobe/make_stimulus.py
# 2. rebuild the CPU tokenizer if /tmp/llama-align-cpu is gone
bash /tmp/wprobe/build_tok.sh
# 3. re-tokenize the 205 bytes and compare position by position
LD_LIBRARY_PATH=/tmp/llama-align-cpu/build/lib:/tmp/llama-align-cpu/build/src \
  /tmp/llama-align-cpu/build/bin/llama-tokenize \
    -m models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -f /tmp/wprobe/chain_stimulus.bin --ids
# 4. the template-free control
printf 'The capital of France is' > /tmp/wprobe/plain2.txt
LD_LIBRARY_PATH=/tmp/llama-align-cpu/build/lib:/tmp/llama-align-cpu/build/src \
  /tmp/llama-align-cpu/build/bin/llama-tokenize \
    -m models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -f /tmp/wprobe/plain2.txt --ids
```

Step 4 prints `[760, 6511, 314, 9338, 369]`, which is `graphs[1].token_ids[3:8]` of run-a verbatim;
the wrapper ids `[248045, 846, 198] + [248046, 198, 248045]` are what the CLI's chat template added
around them.

The two `/tmp/wprobe` scripts are scratch, not repo files; the numbers above are theirs to reproduce,
and the claims that bind the repo are the manifest fields cited in this file.

---

CPU-only work; no device number appears in this file. **no timing taken.**
