# GGUF conformance audit of the shipping artifact (2026-10-27)

D-27 ships `models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` as the engine's weight source, and
requires a **conformance** audit in its place — there is no conversion step here, so there is nothing
for a conversion audit to check. This file records the first run of `tools/gguf_conformance_audit.py`.

Tool: `tools/gguf_conformance_audit.py` (this repo). Machine-readable output:
`checkpoints/gguf-conformance-audit-2026-10-27.json` (write-once; the tool refuses to overwrite it).
Full log: `engine/work/gguf-conformance-audit-2026-10-27.log`. Two earlier runs of the same command
were superseded while the audit itself was being corrected (one died at the engine leg, one still
folded the document's GiB column into `findings`); their JSON was deleted before this manifest was
written, so the file on disk is the run this page describes.

CPU-only. The only file touched read-only besides the GGUF is `checkpoints/`. No device lock is taken
because nothing here touches the iGPU, and no rate appears anywhere: the run ends with
`no timing taken`.

## Verdict

```
RESULT verdict=CONFORMS sha256=ac0e2c1189e0 records=733/733 families=16 families_tensor_total=733
types=5 width_map_rows=12 divergences=3 findings=0 doc_column_deltas=2 controls=2 fired=2 legs=4/4
draft_tensors_expected=19 no timing taken
```

Run as: `/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python tools/gguf_conformance_audit.py
--out-json checkpoints/gguf-conformance-audit-2026-10-27.json`.

## Identity, parse, byte identity

* `sha256 = ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61` — matches
  `checkpoints/Qwen3.6-35B-A3B.gguf-pin.json`; the audit **refuses** to continue on any other digest,
  and `--skip-sha` does not exist.
* `version=3 arch=qwen35moe n_kv=54 records_parsed=733 records_declared=733
  expected_from_index=733 header_end=10990035 data_start=10990048 alignment=32`. The denominator
  `expected_from_index` comes from `engine/artifact.py::expected_records()` — the committed
  safetensors index crossed with config-derived shapes — not from the header agreeing with itself.
* Byte identity: `sum(row_bytes) = 22123538944 + data_start = 10990048 == file size 22134528992`,
  with row bytes from `tools/ggml_block_geom.py` (struct-derived block geometry, not a hand-typed
  bpw table).
* Independent byte authority: `quantization/evidence/audit-hash-2026-10-26.json` records
  `weight_bytes / bytes_on_disk / n_tensors = 22123538944 / 22134528992 / 733`. Derived values match
  it exactly, `delta=+0`. This is an exact equality with no tolerance.
* Three header walks over the same bytes — this audit's own, `engine/loader.py::parse_records`, and
  `tools/gguf_width_table.py::parse_gguf` — agree on all 733 type fields (`differing=0`).
  `engine/artifact.py`'s walk is not a fourth walk here: `Artifact()` cannot be constructed on this
  file at all, because `_c_draft_block` refuses an artifact with no draft block and this file carries
  zero `mtp.*` tensors (§What else the file carries, item 3 of the width map;
  `tools/mtp_shard_cost.py` prices the fix at 2 shards ≈ 5.2 GiB). That is a property of the
  artifact, so the engine leg parses through `loader.parse_records` + `loader.ruler_record`.

## Family map, derived from the file's own header

`stored types (derived count)` is derived: types come from the record, counts from the population,
bytes from struct geometry. Nothing in the tool's family column was typed by hand.

| family | stored types (derived count) | tensors | stored bytes | GiB |
|---|---|---|---|---|
| GDN conv1d | F32 x30 | 30 | 3932160 | 0.0037 |
| GDN other (a/b/gate/beta/A_log/dt_bias) | F32 x150 | 150 | 15751680 | 0.0147 |
| GDN out_proj | Q8_0 x30 | 30 | 267386880 | 0.2490 |
| full-attn k | Q8_0 x10 | 10 | 11141120 | 0.0104 |
| full-attn output | Q8_0 x10 | 10 | 89128960 | 0.0830 |
| full-attn q | Q8_0 x10 | 10 | 178257920 | 0.1660 |
| full-attn v | Q8_0 x10 | 10 | 11141120 | 0.0104 |
| linear-attn in_proj (attn_qkv) | Q8_0 x30 | 30 | 534773760 | 0.4980 |
| linear-attn z-gate (attn_gate) | Q8_0 x30 | 30 | 267386880 | 0.2490 |
| norms | F32 x101 | 101 | 684032 | 0.0006 |
| output (lm_head) | Q6_K x1 | 1 | 417177600 | 0.3885 |
| routed experts down | Q5_K x37, Q6_K x3 | 40 | 7488929792 | 6.9746 |
| routed experts gate/up | Q4_K x80 | 80 | 12079595520 | 11.2500 |
| router (ffn_gate_inp) | F32 x80 | 80 | 84213760 | 0.0784 |
| shared expert | Q8_0 x120 | 120 | 133693440 | 0.1245 |
| token_embd (embeddings) | Q8_0 x1 | 1 | 540344320 | 0.5032 |
| **TOTAL** | | **733** | **22123538944** | **20.6042** |

## Against the published width map

The map is `quantization/evidence/q4km-published-width-map-2026-10-26.md`; the tool parses that
document's own table at run time (row labels, `×N` counts, GiB column, and the bpw note in the
caveats), so the comparison is document-vs-file, not document-vs-this-file's-memory-of-the-document.
All 12 rows map onto file families and all 12 agree on type **and** count:

```
width map     rows_parsed=12 rows_compared=12 doc_total_gib=20.676 derived_total_gib=20.6042
              bpw_note_types=4
```

Two rows' printed GiB do not reproduce from the document's own bpw note, and are printed as
`doc_column_deltas=2` rather than asserted:

* `full attention q/k/v/output` — document 0.269 GiB, reproduction 0.270 GiB (the row is four
  families rounded once each against this tool rounding once per family);
* `GDN control (ssm_a/alpha/beta/dt, conv1d, ssm_norm)` — document 0.019 GiB, reproduction 0.018
  GiB (the note prices the quantized types only; F32 is priced here by the struct table).

Why that is reported instead of asserted: the document's GiB column was typed from the hand-written
bpw table that `quantization/evidence/audit-block-geometry-2026-10-26.md` retired — Q5_K 5.5625
against the struct's 5.5, a 74 MiB overcharge over the 37 `down_proj` tensors, which is exactly the
0.0718 GiB gap between the document's 20.676 total and the file's 20.6042. Asserting the document's
arithmetic would mean asserting a table this repo already overruled. The width map's authority is its
type-and-count map, and that matched exactly; the byte authority is `audit-hash-2026-10-26.json`, and
that matched exactly too. **This is a reporting decision by the agent, not a loosened gate**: no
assertion was removed, the deltas are printed on every run, and a *type* disagreement still reddens
the audit (`width-map-types`, exercised by the control below).

## The two known divergences (engine declared vs file stored)

The declared width is read out of the engine's own tables — `engine/layout.py::EXACT_ROLES` plus
`engine/loader.py::EXACT_KINDS` for never-quantized roles, `quant_scenarios.POLICY_TYPE` (policy A)
for everything else — never typed in this file. 692 of 733 tensors agree exactly.

```
engine   records=733 classified=733 equal=692 wider=40 narrower=1
         loader-admit-notes={"at-or-above-floor": 371, "exact": 361, "waived": 1}
divergence  wider     role=blk.N.ffn_down_exps.weight  engine declares Q4_K  file stores Q5_K  x37  EXPECTED(D-27)
divergence  wider     role=blk.N.ffn_down_exps.weight  engine declares Q4_K  file stores Q6_K  x3   EXPECTED(D-27)
divergence  narrower  role=output.weight               engine declares Q8_0  file stores Q6_K  x1   EXPECTED(D-27)
```

`judge_divergences` makes a *third* divergence a finding, and makes one of these three disappearing a
finding too: if the engine's declared width moved to match the file, the check would notice that the
divergence went silent, which is the stale-constant failure this audit exists to catch. `loader.admit`
is called on every record and refuses nothing: the wider experts clear its floor, and `output.weight`
is admitted through the waiver `loader.WIDTH_WAIVERS` already carries (so the waiver is *exercised*,
not assumed).

## Controls — both seen to go red

```
control mistyped-header-record: blk.0.attn_qkv.weight Q8_0->Q4_K: 1 width-map findings, 2 engine
        findings (['narrower-undeclared', 'width-map-types']) RED
control mistyped-header-record: blk.0.ffn_gate_inp.weight F32->Q8_0: 2 width-map findings, 2 engine
        findings (['narrower-undeclared']) RED
control zero-record-parse: truncated-header: [parse-incomplete] parsed 0 of 733 declared tensor-info
        records before the walk failed (EOFError: need 2 more bytes (fetched 4096)) on a 4096-byte
        read. A short parse of a width policy reads like a complete one, so this refuses.
control zero-record-parse: empty-population: [empty-population] 0 records to build a family map from:
        an empty census of a width policy prints an empty table and a zero total, which is a ...
CONTROL mistyped-header-record fired=2 of 2 RED      CONTROL zero-record-parse fired=2 of 2 RED
```

The counts differ between the two mis-typed cases because one record's type feeds several width-map
rows: `ffn_gate_inp` is named by the router row and the shared-expert row it belongs to, so a single
wrong type there moves two rows. Both cases redden *both* sides, which is what the control is for.

The mis-typed cases are run on both sides at once: the header rows fed to the family map *and* the
records fed to the engine leg, so the control proves the width map and the engine leg each catch it,
by name. The zero-record control prints parsed-vs-expected counts (`0 of 733`) inside the refusal, so
the "print the denominator" requirement is witnessed non-vacuous rather than asserted.

## Draft block

The file carries 0 tensors naming `mtp`; `engine/artifact.py::expected_records()` derives 19 draft
tensors from the committed index. Recorded here because it is the reason `Artifact()` is not usable on
this artifact, and because D-05's MTP weights are not in the shipping file
(`tools/mtp_shard_cost.py`).

## What a later run must re-check

Re-running is cheap to state and not cheap to do: the sha pass is one full read of 22,134,528,992 B.
If the width map document is edited, this audit changes with it by construction — that is the point of
parsing the document instead of copying it. If a tensor's stored type changes, the sha check fires
first; if only the engine's declared width changes, `divergences=` and `judge_divergences` fire.

---

CPU-only work; no device number appears in this file. **no timing taken.**
