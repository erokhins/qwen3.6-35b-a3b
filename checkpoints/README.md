# checkpoints/ — the metadata that our sizing claims are derived from

Small JSON files (**not weights**) committed deliberately: every byte figure in `README.md`,
`quantization/` and `tools/quant_scenarios.py` is derived from these, and `pitfalls/60` says a fact
needed for a task must live in this repo rather than be cited from a website that can move.

| file | what it is | where it came from |
|---|---|---|
| `Qwen3.6-35B-A3B.config.json` | the model `config.json` (text_config + vision_config) | `huggingface.co/Qwen/Qwen3.6-35B-A3B` `/resolve/main/config.json`, fetched 2026-10-26 |
| `Qwen3.6-35B-A3B.safetensors.index.json` | shard index: 1,045 tensor names and `metadata.total_size = 71,903,645,408 B` | same repo, `model.safetensors.index.json`, fetched 2026-10-26 |
| `full-checkpoint-manifest-2026-10-27.json` | per-shard sha256 + byte length + the Hub's LFS etag for all 26 shards of `Qwen/Qwen3.6-35B-A3B`, with `etag_equals_sha256` per shard and the index's tensor→shard map re-derived | `tools/checkpoint_integrity_d29.py build`, fetched Hub tree in `specification/evidence/full-checkpoint-hub-tree-2026-10-27.json` |
| `m3-candidate-hf-fixture-2026-10-27.json` | the HF (bf16, unquantized) M3 fixture: 282 records over all 40 blocks + `normed` + the logits row; `licence=d29-hf-bf16-authoritative(...)` after D-29 | `tools/hf_full_forward_d29.py candidate` |
| `m3-candidate-llama-fixture-2026-10-27.json` | the llama.cpp (quantized) M3 fixture: 430 cross-check rows + logits/argmax; `licence=llama-crosscheck(observation-only)` | `tools/llama_aligned_run.py candidate` |
| `Qwen3.8-Flash-Next.config.json` | the end-goal model's config, kept so the transfer argument in `README.md` is checkable | `huggingface.co/Qwen/Qwen3.8-Flash-Next`, fetched 2026-10-26 |

What they are good for, and what they are not:

- **Good for**: shapes, counts, dtype, `num_experts`, `full_attention_interval`, `head_dim`,
  `vocab_size`, tie status, and the `total_size` used to validate a derived census to within ~2 %.
  `python3 tools/quant_scenarios.py --census` reads exactly these two files.
- **Not good for**: anything about *values*. Quantization bands, per-tensor sensitivity and
  activation ranges cannot be derived from a config; they come from `speccheck/` dumps and from the
  evidence under `quantization/evidence/`.
- The index gives tensor **names and shards but no shapes**. Shapes in the census come from
  `config.json` geometry; that is why `--census` prints the delta against `total_size` (1.86 %, which
  is the vision tower plus small tensors) instead of asserting a match. Re-validate after any
  checkpoint change: a silent shape change is exactly what a 1-ULP-shaped bug looks like downstream.

The weights themselves stay out of git and live under `models/` (gitignored) once
`DECISIONS.md` D-03 is answered.
