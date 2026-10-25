#!/usr/bin/env python3
"""tools/dump_reference_gdn.py — pin the GDN block's reference tensors for the two pinned
`linear_attention` blocks (layer0, layer1), in BOTH dispatch shapes: the chunked prefill of
§04b.1 and the cached recurrent decode step.

CPU only: torch-CPU, no device allocation anywhere, and no clock is read — this file prints no
ms, no GB/s, no tok/s (`AGENTS.md` §GPU execution). Nothing here needs the iGPU: the oracle
path runs in host RAM against the four acquired bf16 shards.

NEW PATHS ONLY, and that is a hard rule, not a convention. This file writes
`data/dumps/2026-10-26-gdn/` and `checkpoints/dump-manifest-gdn-2026-10-26.json` and nothing
else; `data/dumps/2026-10-26/` and `checkpoints/dump-manifest-2026-10-26.json` are live
infrastructure that the MoE lanes' gates read, and `refuse_live_paths()` refuses any `--out`
or `--manifest` that is, or contains, or is contained by one of them. The MoE manifest is
*read* (to witness that the two dumps describe the same forward pass) and never written.

What is pinned, per block (`tools/dump_reference.py`'s byte contract, shared by import so
there is one reader for both dumps):

    weights.{role}            9 bf16   the block's own tensors, as stored in the shard
    prefill.input             [T,H]    bf16   the SAME seeded fixture the MoE dump pins
    prefill.<stage>           19 stages, the §04.2-§04.8 pipeline in order
    decode{s}.input           [1,H]    bf16   the next row of the same generator draw
    decode{s}.<stage>         19 stages, per cached step, s = 0 .. k_conv-2

Why 19 stages and both modes: `speccheck/gdn_ref.py` §The stages is the attribution vocabulary
the gates use, and §04b.1's dispatch table says a prefill and a decode step are different
programs — a dump that pins only one of them cannot attribute a decode-path bug.

The witness. Like the MoE dumper, this refuses to write a byte the installed reference does not
agree with: for every block it runs the reference's own `Qwen3_5MoeGatedDeltaNet` over the same
input and the same weight tensors (`gdn_ref.build_hf_gdn_block` witnesses that the nine
parameters ARE the oracle's storage) and compares all 19 stages in prefill and at every decode
step bit for bit. Agreement is recorded in the manifest per block, so a later reader sees what
was checked instead of being asked to trust it.

Reproduce:
    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python \\
        tools/dump_reference_gdn.py --blocks layer0,layer1 --tokens 32 --seed 20261026 \\
        --threads 8 --out data/dumps/2026-10-26-gdn \\
        --manifest checkpoints/dump-manifest-gdn-2026-10-26.json
Verify (pinned mode — re-hash every file, reconcile the byte totals, assert the counts; it
REFUSES to run at all without the manifest):
    ... tools/dump_reference_gdn.py --verify --out data/dumps/2026-10-26-gdn \\
        --manifest checkpoints/dump-manifest-gdn-2026-10-26.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "speccheck"))
sys.path.insert(0, str(REPO / "tools"))

import gdn_ref as G  # noqa: E402  (the oracle, the HF witness, the stage list)
import moe_ref as R  # noqa: E402  (read-only: set_threads, the shared fixture generator)
import dump_reference as D  # noqa: E402  (reuse, do not fork: ONE byte contract for both
#                                           dumps — sha256, raw little-endian payload, the
#                                           dtype-by-filesize check; pitfalls/40)

DEFAULT_OUT = "data/dumps/2026-10-26-gdn"
DEFAULT_MANIFEST = "checkpoints/dump-manifest-gdn-2026-10-26.json"

# The live infrastructure of other lanes. Never written, never used as an output directory,
# never used as a parent of one.
LIVE_PATHS = ("checkpoints/dump-manifest-2026-10-26.json", "data/dumps/2026-10-26")
MOE_MANIFEST = REPO / "checkpoints" / "dump-manifest-2026-10-26.json"

# Which line of the spec produces each stage, recorded per dump row so a reader can attribute a
# mismatch to a section instead of to "the block". Same vocabulary as `gdn_ref.STAGES`.
STAGE_SOURCE = {
    "mixed_qkv": "§04.2 in_proj_qkv(x)",
    "z": "§04.2 in_proj_z(x), head-shaped",
    "b_raw": "§04.2 in_proj_b(x)",
    "a_raw": "§04.2 in_proj_a(x)",
    "conv_pre": "§04.3 causal conv, pre-activation",
    "conv_out": "§04.3 conv then SiLU, all conv_dim channels",
    "conv_window": "§04.3 consequence 3 / A-09: a window of k_conv INPUT columns",
    "q_pre": "§04.2 split of conv_out, h_k heads, pre-expansion",
    "k_pre": "§04.2 split of conv_out, h_k heads, pre-expansion",
    "v_pre": "§04.2 split of conv_out, h_v heads",
    "beta": "§04.4/§04.5 sigmoid(in_proj_b x), activation dtype",
    "g_decay": "§04.5 -exp(A_log) * softplus(a + dt_bias), F32",
    "q_hat": "§04.4 l2norm(q) then 1/sqrt(d_k), scan dtype",
    "k_hat": "§04.4 l2norm(k), scan dtype",
    "scan_out": "§04.6/§04b.5 the scan's F32 core_attn_out, pre-cast",
    "core_attn": "§04.6/§04b.5 the scan output in the activation dtype",
    "state": "§04.10 [h_v, d_k, d_v] F32 recurrent state, after this pass",
    "normed": "§04.7 Qwen3_5MoeRMSNormGated(core_attn, z)",
    "y": "§04.8 out_proj, no bias",
}


def refuse_live_paths(*paths: Path) -> None:
    """Refuse any output path that is, contains, or is contained by another lane's live dump.

    A prefix match would be wrong here: `data/dumps/2026-10-26-gdn` shares a string prefix with
    the live `data/dumps/2026-10-26` and is a sibling, not a child, so the test is on resolved
    path components (`pitfalls/50` §a check whose predicate is a string prefix)."""
    live = [(REPO / p).resolve() for p in LIVE_PATHS]
    for raw in paths:
        p = Path(raw).resolve()
        for lp in live:
            if p == lp:
                raise AssertionError(
                    f"{p}: this is live infrastructure that other lanes' gates read. "
                    f"dump_reference_gdn.py writes only {DEFAULT_OUT} and "
                    f"{DEFAULT_MANIFEST}; it will not append to or rewrite a MoE path.")
            if lp in p.parents or p in lp.parents:
                raise AssertionError(
                    f"{p}: contains or is contained by the live path {lp}; writing here could "
                    "orphan or shadow another lane's dump. Refused.")


def dtype_size(dtype: str) -> int:
    return D.dtype_size(dtype)


def element_count(shape) -> int:
    return D.element_count(shape)


def stage_records(block: G.Block, stages: dict, mode: str, step, out_dir: Path,
                  command: str, x_row: torch.Tensor, g: G.GdnGeometry, seed: int) -> list:
    """Write the pinned input row plus every stage of one pass, in `G.STAGES` order."""
    rows = []
    what = mode if step is None else f"{mode}{step}"
    n_rows = x_row.numel() // g.hidden_size
    rec = D.write_tensor(x_row, out_dir / f"{block.name}.{what}.input.bin")
    rec.update({"name": f"{block.name}.{what}.input", "tensor_key": "input",
                "block": block.name, "mode": what, "step": step, "arm": "ref",
                "source_shard": "seeded fixture, not a shard",
                "source_tensor": f"moe_ref.make_input(seed {seed}), {n_rows} row(s) of the "
                                 f"tokens+steps draw",
                "computed_by": "gdn_ref.gdn_input — the same generator the MoE dump uses",
                "command": command})
    rows.append(rec)
    for stage in G.STAGES:
        t = stages[stage]
        if not torch.is_tensor(t):
            raise AssertionError(f"{block.name}/{what}/{stage}: the oracle handed a "
                                 f"{type(t).__name__}, not a tensor; a non-numeric stage has no "
                                 "place in this dump")
        path = out_dir / f"{block.name}.{what}.{stage}.bin"
        rec = D.write_tensor(t, path)
        rec.update({"name": f"{block.name}.{what}.{stage}", "tensor_key": stage,
                    "block": block.name, "mode": what, "step": step, "arm": "ref",
                    "computed_by": STAGE_SOURCE[stage],
                    "produced_by": "speccheck/gdn_ref.py gdn_block/gdn_sequence under NO_FAULTS",
                    "command": command})
        if stage == "scan_out":
            rec["orientation"] = "[B, h_v, T, d_v] — the scan's own layout; `core_attn` is the " \
                                 "same values transposed to [B, T, h_v, d_v] and cast to the " \
                                 "activation dtype (§04b.5's exit cast)"
        rows.append(rec)
    return rows


def witness_against_reference(block: G.Block, w: G.GdnWeights, g: G.GdnGeometry,
                              x_pf: torch.Tensor, x_steps: torch.Tensor,
                              prefill: dict, steps: list) -> dict:
    """Refuse to pin anything the installed reference does not reproduce, bit for bit.

    The comparison is the oracle's own `_bit_diff` — shape- and dtype-checked, widened with an
    explicit `.to(f64)` — so the dumper and `--compare-hf` cannot drift into two different
    notions of agreement (`pitfalls/40` §one comparison, two readers)."""
    hf, ran = G.hf_reference_stages(block, w, g, x_pf, x_steps)
    checks, worst, worst_name = 0, 0.0, ""
    modes = [("prefill", prefill, hf["prefill"])]
    modes += [(f"decode{i}", ours, theirs) for i, (ours, theirs)
              in enumerate(zip(steps, hf["decode"]))]
    if len(steps) != len(hf["decode"]):
        raise AssertionError(f"{block.name}: the reference produced {len(hf['decode'])} decode "
                             f"steps, the oracle {len(steps)}; nothing is comparable")
    for what, ours, theirs in modes:
        for stage in G.STAGES:
            try:
                d = G._bit_diff(ours[stage], theirs[stage])
            except AssertionError as exc:
                raise AssertionError(f"{block.name}/{what}/{stage}: the dumper refuses to pin a "
                                     f"tensor the reference does not agree with — {exc}")
            checks += 1
            if not d["bit_identical"]:
                raise AssertionError(
                    f"{block.name}/{what}/{stage}: the independent implementation differs from "
                    f"the installed reference (ndiff={d['ndiff']}/{d['n']}, "
                    f"maxabs={d['max_abs']:.3e}). Not writing dumps — run "
                    "`speccheck/gdn_ref.py --compare-hf` for the table.")
            if d["max_abs"] > worst:
                worst, worst_name = d["max_abs"], f"{block.name}/{what}/{stage}"
    return {"stage_checks_bit_exact": checks, "expected_checks": len(modes) * len(G.STAGES),
            "max_abs_seen": worst, "where": worst_name,
            "reference": {"transformers": ran["transformers"],
                          "scan_chunked": ran["scan_impl"]["chunked"],
                          "scan_recurrent": ran["scan_impl"]["recurrent"],
                          "conv": ran["conv_impl"]["fn"], "conv_update":
                          ran["conv_impl"]["update"], "dispatch": ran["scans_seen"]}}


def dump_block(block: G.Block, g: G.GdnGeometry, out_dir: Path, tokens: int, steps: int,
               seed: int, command: str) -> tuple:
    if block.kind != "layer" or block.index not in g.linear_layers:
        raise AssertionError(f"{block.name}: not a linear_attention decoder layer; this dumper "
                             "serves the GDN block only (file 04's header)")
    w = G.load_gdn_block_weights(block, g)
    x_all = G.gdn_input(tokens + steps, g, seed).unsqueeze(0)
    x_pf, x_st = x_all[:, :tokens], x_all[:, tokens:]
    if x_st.shape[1] != steps:
        raise AssertionError("the decode rows are not the requested step count")
    prefill = G.gdn_block(x_pf, w, g, "chunked", G.NO_FAULTS, mode="prefill")
    seq = G.gdn_sequence(x_pf, x_st, w, g, "chunked", "recurrent", faults=G.NO_FAULTS)
    if len(seq["all_steps"]) != steps:
        raise AssertionError("gdn_sequence did not produce one stage dict per decode row")
    agree = witness_against_reference(block, w, g, x_pf, x_st, prefill, seq["all_steps"])

    rows = []
    for role, suffix in G.SUFFIXES.items():
        name = block.prefix + suffix
        t = w.tensors[role]
        rec = D.write_tensor(t, out_dir / f"{block.name}.weights.{role}.bin")
        rec.update({"name": f"{block.name}.weights.{role}", "tensor_key": suffix,
                    "block": block.name, "mode": "stored", "step": None, "arm": "stored",
                    "source_shard": w.source_shards[role], "source_tensor": name,
                    "gguf_name": G.GGUF_NAMES[role],
                    "computed_by": "read from the acquired shard, unchanged",
                    "command": command})
        rows.append(rec)

    rows += stage_records(block, prefill, "prefill", None, out_dir, command, x_pf, g, seed)
    for i, st in enumerate(seq["all_steps"]):
        rows += stage_records(block, st, "decode", i, out_dir, command, x_st[:, i:i + 1], g,
                              seed)
    del w
    return rows, agree


def moe_input_crosscheck(prefill_input_sha: str, tokens: int) -> dict:
    """Witness that this dump's prefill input IS the MoE dump's pinned input, byte for byte.

    The MoE manifest is opened read-only. 'The same forward pass' is a claim about the fixture
    generator, not about a real forward: both lanes feed an A-01 RMS-1 stand-in for a
    layernorm output, and this says so in the manifest rather than letting the phrase float."""
    if not MOE_MANIFEST.exists():
        raise AssertionError(f"{MOE_MANIFEST} is missing; the cross-reference that makes "
                             "'the two dumps describe the same input' a checked claim cannot "
                             "run. Refusing to write an unreconciled dump.")
    m = json.loads(MOE_MANIFEST.read_text())
    rec = next((r for r in m["dumps"] if r.get("name") == "layer0.input"), None)
    if rec is None:
        raise AssertionError("the MoE manifest has no layer0.input record to compare against")
    same = rec["sha256"] == prefill_input_sha
    if not same:
        raise AssertionError(
            f"the GDN prefill input does not equal the MoE dump's pinned input "
            f"({prefill_input_sha[:16]}… vs {rec['sha256'][:16]}…). The two dumps would not "
            "describe one forward pass; check the seed, the token count and the threads.")
    return {"moe_manifest": str(MOE_MANIFEST.relative_to(REPO)),
            "moe_layer0_input_sha256": rec["sha256"],
            "gdn_prefill_input_sha256": prefill_input_sha,
            "n_tokens": tokens, "equal": bool(same),
            "moe_producing_command": m.get("producing_command"),
            "reading": "byte-identical because both are speccheck/moe_ref.py make_input over "
                       "the same seed and row count; the GDN dump draws tokens+k_conv-1 rows "
                       "and slices the first `tokens` of the same generator stream",
            "caveat": "[ASSUMPTION A-01] neither is a real activation: the MoE dump models "
                      "post_attention_layernorm output and this one models input_layernorm "
                      "output (file 03 §03.3), and both are RMS-1 without the (1+w) "
                      "modulation. Same fixture, not the same layer of a real pass."}


def build_manifest(blocks, rows, agreement, cross, g, args, command, threads) -> dict:
    expected = len(blocks) * (len(G.SUFFIXES) + (1 + len(G.STAGES))
                              + args.steps * (1 + len(G.STAGES)))
    declared = sum(r["bytes"] for r in rows)
    for r in rows:
        if r["bytes"] != r["elements"] * dtype_size(r["dtype"]):
            raise AssertionError(f"{r['file']}: {r['bytes']} B cannot hold {r['elements']} x "
                                 f"{r['dtype']} (pitfalls/30 §read a dump's dtype off its size)")
    return {
        "manifest_version": 1,
        "lane": "gdn",
        "date": "2026-10-26",
        "producing_command": command,
        "interpreter": {"executable": sys.executable, "version": sys.version.split()[0],
                        "torch": torch.__version__, "threads": threads},
        "geometry": {"hidden_size": g.hidden_size, "linear_num_key_heads": g.h_k,
                     "linear_num_value_heads": g.h_v, "linear_key_head_dim": g.d_k,
                     "linear_value_head_dim": g.d_v, "linear_conv_kernel_dim": g.k_conv,
                     "key_dim": g.key_dim, "value_dim": g.value_dim, "conv_dim": g.conv_dim,
                     "expansion": g.expansion, "num_hidden_layers": g.num_hidden_layers,
                     "linear_layers": list(g.linear_layers),
                     "blocks_dumped": [b.name for b in blocks],
                     "hidden_act": g.hidden_act, "mamba_ssm_dtype": g.mamba_ssm_dtype,
                     "rms_norm_eps": g.rms_norm_eps, "chunk_width": g.chunk_width,
                     "l2norm_eps": g.l2norm_eps, "config": g.config_path},
        "dispatch": {"prefill": "chunked UT solve (§04b.1, §04b.3-§04b.5)",
                     "decode": "recurrent, cached conv window + recurrent state (§04b.1)",
                     "chunk_width_used": g.chunk_width, "solve_branch": "triangular (A-16: the "
                     "substitution branch is a different number and is not what was pinned)",
                     "scan_dtype": g.mamba_ssm_dtype,
                     "reference_impl": "the installed torch fallbacks; which function objects "
                                       "ran is recorded per block under agreement_at_write_time"},
        "input": {"n_tokens": args.tokens, "n_decode_steps": args.steps, "seed": args.seed,
                  "dtype": "bfloat16",
                  "method": "speccheck/moe_ref.py §Input generation via gdn_ref.gdn_input: fp32 "
                            "torch.randn from a private torch.Generator('cpu')"
                            ".manual_seed(seed), each row rescaled to RMS exactly 1.0, cast to "
                            "bf16. [ASSUMPTION A-01] RMS-1 models a layernorm output; it does "
                            "NOT carry the (1+weight) modulation of a real activation"},
        "same_input_as_the_moe_dump": cross,
        "reading_rules": {
            "payload": "raw little-endian element bytes, no header; bytes == elements x "
                       "itemsize, and `--verify` refuses a record whose size cannot hold its "
                       "own declared shape (pitfalls/30 §read a dump's dtype off its file size)",
            "bf16": "two bytes per element, bit image — read with a uint16 view then bitcast, "
                    "never cast the uint16 to a float (pitfalls/30 §Tensor.to casts, .view "
                    "bitcasts)",
            "orientations": "conv_pre/conv_out/conv_window are [B, conv_dim, T] (the conv's own "
                            "orientation, §04.3); q_pre/k_pre/v_pre are [B,T,h,d]; q_hat/k_hat/"
                            "scan_out are [B,h_v,T,d] in the scan dtype; core_attn/normed are "
                            "[B,T,h_v,d_v] in the activation dtype; state is [B,h_v,d_k,d_v] F32",
            "scan_out_vs_core_attn": "the same values, transposed and cast — pinning only "
                                     "core_attn would hide an exit-cast bug (pitfalls/30 §a hook "
                                     "that reads output can pin the answer)",
            "decode_rows": "decode{s} is step s of the cached sequence, fed row s of the same "
                           "generator draw AFTER the pinned prefill produced the state and "
                           "window in prefill.state / prefill.conv_window; the decode rows are "
                           "not independent of the prefill rows",
        },
        "dumps": rows,
        "agreement_at_write_time": agreement,
        "counts": {"dumps": len(rows), "expected": expected, "blocks": len(blocks),
                   "stages_per_pass": len(G.STAGES),
                   "passes_per_block": 1 + args.steps, "weights_per_block": len(G.SUFFIXES)},
        "bytes": {"declared_records": declared,
                  "formula": "sum of every per-record `bytes`; --verify re-sums the directory "
                             "and requires records == declared == files on disk"},
    }


def verify(manifest_path: Path, out_dir: Path) -> int:
    # §Pinned mode. No manifest, no run: the manifest is the only thing that says what these
    # bytes are supposed to be, and a verifier that can run without one can print PASS over an
    # empty or swapped directory (`pitfalls/40` §a check that produces no output cannot fail).
    if not manifest_path.exists():
        print(f"PINNED MODE REFUSED: {manifest_path} does not exist. This verifier will not "
              "run against an unmanifested dump directory: without the manifest there is no "
              "dtype, shape or digest to check the bytes against, and 'no files bad' would be "
              "vacuous. Restore the manifest or produce a fresh dump with the write path.")
        return 2
    try:
        m = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"PINNED MODE REFUSED: {manifest_path} is not parseable JSON ({exc})")
        return 2
    if not isinstance(m.get("dumps"), list) or not m["dumps"]:
        print(f"PINNED MODE REFUSED: {manifest_path} carries no dump records")
        return 2
    if m.get("lane") != "gdn":
        print(f"PINNED MODE REFUSED: {manifest_path} is a {m.get('lane')!r} manifest; this "
              "verifier reads the GDN one only")
        return 2

    bad, checked = [], 0
    listed = set()
    for rec in m["dumps"]:
        p = Path(rec["file"])
        if not p.is_absolute():
            p = REPO / p
        listed.add(p.name)
        if not p.exists():
            bad.append(f"MISSING {rec['file']}")
            continue
        got = D.sha256_file(p)
        size = p.stat().st_size
        if got != rec["sha256"]:
            bad.append(f"HASH {rec['file']}")
        elif size != rec["bytes"]:
            bad.append(f"SIZE {rec['file']} {size} != {rec['bytes']}")
        elif size != rec["elements"] * dtype_size(rec["dtype"]):
            bad.append(f"BYTES-vs-SHAPE {rec['file']}: {size} B cannot hold "
                       f"{rec['elements']} x {rec['dtype']} — the dtype-by-filesize check "
                       "(pitfalls/30)")
        else:
            checked += 1
    on_disk = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    extra = [n for n in on_disk if n not in listed]
    missing = [n for n in listed if n not in on_disk]
    for n in extra:
        bad.append(f"UNLISTED {n} — a file in the dump directory that the manifest does not "
                   "name is a named-and-skipped file, never a silently ignored one")
    for n in missing:
        bad.append(f"NOT-ON-DISK {n}")
    dir_total = sum(p.stat().st_size for p in out_dir.iterdir() if p.is_file())
    record_total = sum(r["bytes"] for r in m["dumps"])
    declared = m.get("bytes", {}).get("declared_records")
    n_expected = m.get("counts", {}).get("expected")
    print(f"# pinned verify: {len(m['dumps'])} manifest records, {checked} verified, "
          f"{len(bad)} bad, expected count {n_expected}")
    print(f"# byte reconciliation: sum(records.bytes)={record_total:,}  "
          f"manifest.declared={declared}  directory_total={dir_total:,}  "
          f"files_on_disk={len(on_disk)} listed={len(listed)}")
    if record_total != dir_total:
        bad.append(f"BYTE RECONCILIATION: records sum to {record_total:,} but the directory "
                   f"holds {dir_total:,}")
    if declared != record_total:
        bad.append(f"BYTE RECONCILIATION: the manifest declares {declared} but its records sum "
                   f"to {record_total:,}")
    if len(on_disk) != len(m["dumps"]):
        bad.append(f"FILE COUNT: {len(on_disk)} files on disk vs {len(m['dumps'])} records")
    if n_expected != len(m["dumps"]):
        bad.append(f"COUNT: {len(m['dumps'])} records vs derived expectation {n_expected}")
    for b in bad:
        print("  " + b)
    if bad:
        print("VERIFY FAIL")
        return 1
    if checked != len(m["dumps"]) or checked != n_expected:
        print("VERIFY FAIL: a verified count that does not equal its own denominator cannot "
              "pass (pitfalls/40)")
        return 1
    print("VERIFY PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--blocks", default="layer0,layer1")
    ap.add_argument("--tokens", type=int, default=32,
                    help="prefill length; 32 + seed 20261026 is what the MoE dump pins")
    ap.add_argument("--steps", type=int, default=0,
                    help="cached decode steps; 0 = k_conv-1, derived from the geometry")
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--verify", action="store_true", help="pinned mode; refuses without a "
                                                          "manifest")
    ap.add_argument("--overwrite", action="store_true",
                    help="permit writing into a non-empty dump directory")
    args = ap.parse_args()

    out_dir = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    manifest_path = (REPO / args.manifest) if not Path(args.manifest).is_absolute() \
        else Path(args.manifest)
    refuse_live_paths(out_dir, manifest_path)
    if manifest_path.resolve().parent == out_dir.resolve():
        raise AssertionError(
            f"{manifest_path} sits inside the dump directory {out_dir}. The verifier's contract "
            "is that the directory's file set equals the manifest's record set exactly, so a "
            "manifest kept there could never verify — it would be an unlisted file. Keep the "
            "manifest outside the dump (the pinned one lives in checkpoints/).")
    if args.verify:
        return verify(manifest_path, out_dir)

    command = " ".join([sys.executable] + [str(Path(__file__).resolve())] + sys.argv[1:])
    threads = R.set_threads(args.threads)
    g = G.load_gdn_geometry()
    blocks = [G.Block.parse(s) for s in args.blocks.split(",") if s.strip()]
    if not blocks:
        raise AssertionError("no blocks selected; an empty dump set cannot pass")
    steps = args.steps if args.steps else max(2, g.k_conv - 1)
    args.steps = steps
    if steps < 1:
        raise AssertionError("at least one cached decode step is required")

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise AssertionError(f"{out_dir} is not empty; refuse to write over it without "
                             "--overwrite (a pinned dump is replaced whole or not at all)")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, agreement = [], {}
    for block in blocks:
        r, ag = dump_block(block, g, out_dir, args.tokens, steps, args.seed, command)
        rows += r
        agreement[block.name] = ag
        print(f"# {block.name}: {len(r)} tensors pinned "
              f"({len(G.SUFFIXES)} weights + 1 prefill input + {len(G.STAGES)} prefill stages "
              f"+ {steps} x (1 input + {len(G.STAGES)} decode stages) = "
              f"{sum(x['bytes'] for x in r):,} B), reference agreed on "
              f"{ag['stage_checks_bit_exact']}/{ag['expected_checks']} stage checks")

    prefill_inputs = [r for r in rows
                      if r["tensor_key"] == "input" and r["mode"] == "prefill"]
    if len(prefill_inputs) != len(blocks):
        raise AssertionError("every block must pin exactly one prefill input row")
    if len({r["sha256"] for r in prefill_inputs}) != 1:
        raise AssertionError("the blocks' prefill inputs differ; they are meant to be one "
                             "fixture draw shared across blocks")
    cross = moe_input_crosscheck(prefill_inputs[0]["sha256"], args.tokens)

    manifest = build_manifest(blocks, rows, agreement, cross, g, args, command, threads)
    if len(rows) != manifest["counts"]["expected"]:
        print(f"MANIFEST FAIL: wrote {len(rows)} records, expected "
              f"{manifest['counts']['expected']}")
        return 1
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n")
    print(f"# manifest: {manifest_path} ({manifest_path.stat().st_size:,} B), "
          f"{len(rows)} records, {manifest['bytes']['declared_records']:,} B declared")
    print(f"# same input as the MoE dump: sha256 {cross['gdn_prefill_input_sha256'][:16]}… "
          f"equal={cross['equal']} (A-01 caveat recorded in the manifest)")
    rc = verify(manifest_path, out_dir)
    print("no timing taken")
    return rc


if __name__ == "__main__":
    sys.exit(main())
