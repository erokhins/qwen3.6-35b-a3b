# llama.cpp smoke test on the pinned artifact — 2026-10-26

Verdict: **PASS** — the pinned HIP build loads the shipping GGUF and generates from it, with the ROCm
backend exercised at runtime. This is the runtime half of the question that
`quantization/evidence/llama-cpp-build-and-flags-2026-10-26.md` answered only structurally (141/141
code objects `gfx1151`, `.hip_fatbin` 56,112,304 B, 12 HIP dependency lines): a fatbin census proves
device code is present, not that the loader reaches it.

## What was run

`engines/llama.cpp/build-hip/bin/llama-cli` @ `d327bf1f0780f87b8c01b367b61853b0c291d7c4` (tag `b11869`),
model `models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (sha256 `ac0e2c11…53e31a61`, pinned in
`checkpoints/Qwen3.6-35B-A3B.gguf-pin.json`), with:

```
-p "The capital of France is" -n 24 -ngl 1 -s 42 --temp 0 -st --no-display-prompt --no-warmup -v
```

Run by the main agent, in the foreground, holding `tools/gpu_lock.sh` (tag
`35b: main llama.cpp GGUF smoke test 3 (halogen UP, no timing)`), released afterwards — `gpu_lock`
status reads `free`. **Zero subagents active.** `halogen` was resident for the whole run (22.62 GiB),
which is admissible for a correctness run and disqualifying for any rate.

## The evidence

| claim | witness |
|---|---|
| the file loads | exit 0; banner `build: b1-d327bf1`, `ftype: Q4_K - Medium`, `modalities: text` |
| the ROCm backend is live, not just present | `system_info: … \| ROCm : NO_VMM = 1 \| FA_QUANTS = q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16 \| …` |
| device offload actually happens | `load_tensors: offloaded 1/41 layers to GPU` (twice, pre- and post-`mmap`) with `-ngl 1` |
| it generates | `predicted_n: 24`, `finish_reason: "length"`, `stop_type: "limit"`, completion `France` for prompt `The capital of France is` |
| greedy and reproducible | `temperature 0.0`, `seed 42`, `samplers: [penalties, dry, top_n_sigma, top_k, typ_p, top_p, min_p, xtc, temperature]` |

`41` in `1/41` is 40 transformer blocks plus the embedding stage — the same layer count the config and
the tensor census give, so the loader and our own census agree about the model's shape from opposite
ends.

## Two findings this produced, neither of them about speed

**1. `tokenizer.ggml.eos_token_id u32 = 248046`.** The GGUF says 248046 and `text_config` says 248044;
this is the disagreement registered as `specification/ASSUMPTIONS.md` **A-04**, and it is now witnessed
from the artifact as llama.cpp reads it, not just from our own header parse. `generation_config.json`
is still not on disk, so A-04 stays open — but our decode loop must take `STOP_IDS` as a named input
rather than defaulting to either value.

**2. llama.cpp infers a reasoning format for this architecture.** The verbose load shows the template
it chose — `chat_format: peg-native`, `reasoning_format: deepseek`, generation prompt ending in the
think-boundary token, `preserved_tokens [248058, 248059, 248068, 248069]`, `ignore_eos: false`. Those
ids are named numerically on purpose (AGENTS.md: never print chat-template special tokens literally).
This is llama.cpp's *guess* about a checkpoint whose own config is ambiguous, and our engine inherits
none of it by default: whether the think-boundary is emitted, preserved or suppressed is a decision the
engine makes deliberately, and it interacts with A-20 (drafted stop ids) in the MTP path.

## Deliberately absent: every rate

`llama-cli` prints its own timings, and the raw capture contains them. **They are not transcribed here
and must not be quoted anywhere.** The run was co-resident with halogen (22.62 GiB charged to the same
device-visible pool) *and* ran with 40 of 41 layers on CPU, so it is both contaminated and
architecturally unrepresentative; `AGENTS.md` §"GPU execution is serialized" treats a contaminated
number as worse than no number. The raw log lives in gitignored `tmp/evidence/` precisely so it cannot
leak into a commit or a timing doc. The tokens/s ruler is main-agent work in a halogen-down window.

## Traps this lane hit, so the next one doesn't

- **`llama-cli -p … ` alone enters interactive mode.** It printed the prompt, read empty stdin, emitted
  an **empty completion and exited 0**. A smoke test whose pass criterion is the exit code passes on a
  run that generated nothing. The completion text, not the exit code, is the evidence.
- **`--no-cnv` does not exist in this build** (`error: invalid argument`, exit 1). The flag here is
  `-st, --single-turn`; the binary set in `build-hip/bin/` is `llama-cli` and `llama-quantize` only,
  though `libllama-server-impl.so` is built, so the rate ruler can go through the server path without
  a rebuild.
- **No `ggml`/`HIP`/`offload` lines appear without `-v`.** A silent smoke test cannot see the one line
  (`offloaded 1/41 layers to GPU`) that distinguishes a GPU build from a CPU fallback — which is the
  failure mode the whole build audit was chasing.
