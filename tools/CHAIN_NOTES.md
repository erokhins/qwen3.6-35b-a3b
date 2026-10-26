# CHAIN_NOTES — `tools/dump_chain.py`, P1 of D-25

Status: **P1 done, all gates green.** CPU only — torch-CPU, no device allocation, no HIP, no
`gpu_lock`, and no device clock is read. The only timings here are host CPU wall and host RSS;
**no device rate, no tok/s**. `data/dumps/2026-10-26/`, `data/dumps/2026-10-26-gdn/` and
`checkpoints/dump-manifest-2026-10-26.json` are untouched: the tool refuses to write any path
whose first two components are `data/dumps` but are not the chain directory, asserts the pins'
mtimes are unchanged across every mode, and — as an independent witness — `--mode guards`
re-hash of the existing records against the existing manifest passes inside the same run.

```
existing_records_vs_manifest[existing]        PASS      104/104 records verified, 0 bad
existing input digests, unchanged by this run: 162,519,680 B over 178 records (+256 B sidecars)
```

## What is actually chained, and why it is not blocks 0..3

`--chain-layers` defaults to `0,1,38,39` because **that is all that is on disk**. A shard census
run in the same process as the dumps, derived from `model.safetensors.index.md`'s own
shard→tensor map rather than typed in, says:

```
shards 4/26, tensors 424/1045, layers complete [0, 1, 38, 39], missing_layers count=36
missing_layers = [2..37]
```

So this P1 is a **structural proof over the four complete decoder layers** — layer0, layer1,
layer38, layer39 — chained end to end, not the P2 deliverable (all 40 blocks, true adjacent
hops). The gap is structural, not narrative:

* `missing_layers` in the manifest is **derived** from the index at run time (`census()`, with
  `source: "derived: index.md shard->tensor map"` and the population in `counts`);
* **every** record carries `contiguous: false` while any layer is missing, plus a per-hop
  `chain_gap: {from_index, to_index, layers_between_missing}` — `[]` for the two real adjacent
  hops (`layer0→layer1`, `layer38→layer39`), the full list `[2..37]` for `layer1→layer38`, whose
  activation is **not** a true residual stream and is labelled `chain_head: "SUBSTITUTE CHAIN"`;
* `block_set.contiguous: false`, and `--mode guards` refuses outright if the census is complete
  yet any record is non-contiguous, or if the census is incomplete yet `contiguous` is true.

The two checks that police this each have a mutated-copy control that must fire
(`pitfalls/40-gates-and-admissions.md §5`): `CONTROL gap_declared_for_gap catches a stripped
record` and `CONTROL no_contiguity_claim catches a lying record`. Both PASS.

## Seeding and the two substitutions

Seeding is real: `layer0.x` is rows `token_ids[seed=20261026]` of
`model.language_model.embed_tokens.weight`, read straight out of
`model-00001-of-0026.safetensors`. `inputs_not_collapsed` does **not** fire on it
(`embedding_rows_come_from_the_shard` PASS, rows sha `10a98057dbbd27a4…`), which is the
asymmetry D-25 §"The counter's own weakness" asked for. token ids and their sha256 are pinned, so
the seed is reproducible and the substitution is auditable rather than implied.

Two substitutions, both witnessed twice as the supervisor required:

**1. The block is loaded by name from the checkpoint** (`load_layer_params` →
`layer_params(layer_idx, cfg)`) **rather than by living inside a materialised `Qwen3_5MoeForCausalLM`.**
The model has 1,045 tensors and 66 GiB of weights; the host has 94 GiB and `from_pretrained` does
not accept `weights_only=False` for a `from_config` skeleton (it raises
`NotImplementedError: ... weights_only=False is only supported for weights that are already
materialized`). Witnessed twice:
  * **per parameter**, live tensor digest vs an independent re-read of the same tensor from the
    safetensors shard — 18/18 parameters for the three `linear_attention` blocks, 15/15 for the
    `full_attention` block, 69 rows total, every one `live==shard`
    (`substitution_complete[layer0]` … `[layer39]`, all PASS);
  * **as a whole chain**, `--mode peer`: all four blocks built the ordinary installed-transformers
    way (real `Qwen3_5MoeDecoderLayer` objects under `from_config`, parameters materialised),
    `load_state_dict(strict=True)` against exactly the checkpoint's own key set, and the same
    chained forward run beside the meta-substituted one. **28/28 stage digests agree bit for
    bit** (`peer[layer0/x]` … `peer[layer39/out]`, 28 PASS). `load_state_dict`'s own missing/
    unexpected report must be clean for the row to count, and `peer_forward_is_cheap` guards
    against a second oracle that is really the engine under test (`speccheck/` is the only
    independent implementation allowed here).

**2. Substituted parameters live on the meta device** until the shard gives them bytes
(`torch.set_grad_enabled(False)` + `torch.device("meta")` at construction), because 2 GiB of
`embed_tokens` plus every block cannot be resident at once. That is a real risk — a parameter
that never got its bytes would look like a dtype-correct zero and produce a plausible forward.
The witness against it is `--mode control --control layer38:mlp.experts.down_proj`, which leaves
that one parameter on meta (all its reads raise), rebuilds, and requires the output to move:

```
broken_substitution_is_local[layer0]      PASS  upstream output unchanged
broken_substitution_is_local[layer1]      PASS  upstream output unchanged
broken_substitution_moves_output[layer38] PASS  output CHANGED, the substitution is load-bearing
broken_substitution_propagates[layer39]   PASS  downstream output changed, as it must
```

Upstream cannot move if the chain is real, and downstream cannot move if the substituted weight
were decorative; both halves are required, so a build that silently zeroed a whole tensor would
not pass `--mode chain`'s `substitution_complete` **or** survive `--mode control`.

## The seven rows per block

`x, h_in, mixer_out, x1, h_mlp, moe_out, out` — `specification/03 §03.2`, keyed
`chain.<block>.<stage>`, and every value is taken by a **forward hook on the reference module**,
not computed by the dumper: `pre` on `linear_attn`/`self_attn` for `h_in`, its `post` for
`mixer_out`, `pre` on `mlp` for `h_mlp`, `post` on `mlp` for `moe_out`; `x`, `x1`, `out` are the
block's entry, the first residual add, and the second, each a fresh `.clone().contiguous()`.
`h = rms_norm(x, input_layernorm)` is asserted bit-exact against the module's own output, so the
engine cannot get away with an unnormalised residual (`residual_identity[...]`, 4 PASS,
`ndiff=0/65536`).

## Running it

```sh
cd /home/erokhins/erokhins_projects/qwen3.6-35b-a3b
PY=/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python   # torch 2.14.1+cpu

"$PY" tools/dump_chain.py --mode chain      --tokens 32 --seed 20261026 --threads 8
"$PY" tools/dump_chain.py --mode guards
"$PY" tools/dump_chain.py --mode perturb    --perturb layer38:mlp.experts.down_proj:row=7
"$PY" tools/dump_chain.py --mode peer
"$PY" tools/dump_chain.py --mode control    --control layer38:mlp.experts.down_proj
"$PY" tools/dump_chain.py --mode manifest   # refuses until the four evidence files exist
"$PY" tools/dump_chain.py --mode verify
```

Append-only: `--mode chain` refuses if the chain directory exists, `--mode manifest` refuses if
the manifest exists, and `--mode manifest` refuses if any evidence file is missing — so the
manifest cannot be written from an unpolicied chain. `--mode verify` is read-only and is the mode
a reviewer runs.

## Measured cost, and the honest full-40 estimate

Per block, from the pinned generation (`evidence/chain.json`, which also carries 23 timestamped
phase marks; five independent generations produced byte-identical stage digests, so these are
stable and the forward is deterministic on CPU):

| block | type | load s | forward s | oracle s | write s | RSS after |
|---|---|---|---|---|---|---|
| layer0 | linear_attention | 2.96 | 0.082 | 0.088 | 0.000 | 1.96 GiB |
| layer1 | linear_attention | 4.11 | 0.053 | 0.060 | 0.000 | 1.96 GiB |
| layer38 | linear_attention | 2.97 | 0.041 | 0.042 | 0.000 | 1.97 GiB |
| layer39 | full_attention | 3.75 | 0.033 | 0.037 | 0.000 | 1.95 GiB |

Load is the only variable term and it is page-cache dependent: the same four blocks cost
2.96–4.11 s with the shards cached and 5.51–6.82 s with them cold, which is why the estimate
below is given as a range.

Peak host RSS: **3.95 GiB** for `--mode chain` (one block + the 2 GiB embedding table; RSS drops
back to 0.39 GiB between blocks, so the "one block at a time" claim is measured, not asserted),
**8.57 GiB** for `--mode peer` (all four blocks materialised at once), 3.95 GiB for `--mode
perturb`/`--mode control`, ≈0.21 GiB for the file-only modes.

**Full 40 blocks, linear in the measured numbers:** load 3.45 s/block warm (6.3 s cold) ⇒
**138–252 s**; forward 0.052 s/block ⇒ 2.1 s; oracle 0.057 s/block ⇒ 2.3 s; writes ≈105 MiB
total; plus 0.27 s to slice the embedding table and ≈2 s of interpreter/import. So `--mode chain`
over all 40 blocks is **≈2.5 minutes warm, ≈4.5 minutes cold, and 97 % of it is weight
loading** — the CPU forward is free (2.1 s for the whole model), so anyone tempted to optimise it
should instead make the loader read each tensor once. Two caveats: the 22 unacquired shards
(~56 GiB) are not on this box, so the cold figure is the one that will actually be paid on the
first full run; and per-block load is safetensors materialisation as much as disk.

`--mode peer` does **not** extrapolate as written: 2.1 GiB per materialised block ⇒ ≈85 GiB for
40, next to nothing left of the 94 GiB host. P2 must chunk the peer view (build a few blocks at a
time, run, free) — flagged, not implemented, because it is P2's problem and the current peer mode
is exactly the witness P1 needs.

## What each guard checks, and what it would catch

| check | catches | its own control |
|---|---|---|
| `inputs_not_collapsed[chain]` | the fixture being one tensor dressed as four | `CONTROL all-chain-inputs-collapsed` (in-process mutated copy) |
| `inputs_not_collapsed[existing per-block manifest]` | — | this **is** the real RED: the four existing `*.input.bin` are one tensor, `f82d402d574a785b…`, printed and cross-checked against the existing manifest, never trusted from prose |
| `chained_input_equals_previous_output[b]` | chaining asserted rather than done: re-hashes `blockN-1.out.bin` from disk and compares to `blockN.x.bin` and to `input_from.sha256`, bit-exact | three independent hash sources per row |
| `independent_oracle[b/moe_out/B/moe_ref]` | an MoE that only agrees with itself | `speccheck/moe_ref.moe_forward_ref` re-derives `moe_out` per row; `ndiff=0/65536`, `maxabs=0.0`, bit-exact 4/4 |
| `independent_oracle[b/mixer_out/B/gdn_ref]` | a GDN that only agrees with itself | `speccheck/gdn_ref.gdn_block` re-derives `mixer_out` for the three linear-attention blocks, `bit_exact=True`; **layer39 is `full_attention` and has no independent mixer oracle in `speccheck/`** — its witness is the MoE row, and the row prints `independent_oracle=none` rather than pretending |
| `independent_oracle_covers_every_block` | 3 of 4 blocks secretly unpoliced | denominator is `block_set.blocks`, not `rows` |
| `residual_identity[b]` | `h` not being `rms_norm(x)` bit-exactly | `ndiff=0/65536` vs the module's own output |
| `substitution_complete[b]` | a parameter that never got its bytes | per-parameter live-vs-shard digest, population = `live ∩ expected` |
| `peer[b/stage]` | a meta-substituted forward that diverges from the installed-transformers way | 28 rows, `strict=True` load report must be clean |
| `gap_declared_for_gap`, `no_contiguity_claim_across_gap` | reading the substitute chain as real activations | mutated-copy controls for both |
| `verify` tamper arms | a hand-edited manifest | a manifest with a forged sha256, a wrong `bytes`, a `contiguous: true` across the gap, or a stripped `input_from` is **rejected with rc=1** (all four arms tested; the fourth is why `--mode verify` re-asserts chaining, not just hashes) |

## For P2

* **The gap is the only blocker.** `missing_layers` is 36 of 40; `--mode chain` refuses to claim
  contiguity until the census says otherwise, and its guards are written so that completing the
  download turns `contiguous` true by derivation, not by editing a flag.
* **Chunk the peer view** (≈2.1 GiB/materialised block) and re-check `--mode peer`'s RSS claim
  then, not before.
* `--mode perturb` currently perturbs one row of one expert's `down_proj`
  (`layer38:mlp.experts.down_proj:row=7`, ×1.5): the change map is
  `layer38.mlp.experts.down_proj row 7 x 1.5` → `layer0.x/out` unchanged
  (`10a98057dbbd…`, `1b5e3248d29b…`), `layer1.x/out` unchanged (`1b5e3248d29b…`,
  `35038f951946…`), `layer38.x` unchanged (`35038f951946…`) while `layer38.out` moves
  (`15366c63d165… → 6ce4e8380ba9…`), `layer39.x` **equals that new value** and `layer39.out`
  moves (`0e0452623e58… → 1228e08007d8…`). For P2 prefer a perturbation that must move *every*
  expert (e.g. the shared expert, or the router) alongside a single-expert one: the single-expert
  arm proves locality but can be satisfied by a dead expert, and only `nonzero_output` plus the
  `independent_oracle` row rule that out.
* Band heads-up for M2: `h_mlp - x1` is the largest cancellation in this chain
  (`|out - x| / max|x| = 0.8993` at layer0), in the same regime as the 0.8876 chaos floor in
  `pitfalls/30-numerics.md`; bands derived from a floor probe will be wide there for a real
  reason, and a gate that fails at 1-ULP is the wrong instrument for that row.
