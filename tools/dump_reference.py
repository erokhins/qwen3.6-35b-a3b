#!/usr/bin/env python3
"""tools/dump_reference.py — pin the MoE block's bf16 reference tensors for
layers 0, 1, 39 and the MTP draft block.

CPU only: torch-CPU, no device allocation anywhere, and no clock is read — this file
prints no ms, no GB/s, no tok/s (AGENTS.md §GPU execution). Nothing here needs the iGPU:
the oracle path runs in host RAM against the four acquired bf16 shards.

What is pinned, per block (stage names are `speccheck/moe_ref.py` §The block, in words):

    input                      [T,H]   bf16   seeded, method recorded in the manifest
    mlp.gate.weight            [E,H]   bf16   the router weights, as stored
    router_logits              [T,E]   bf16   model dtype, as the reference computes them
    topk_ids                   [T,K]   int64
    topk_weights               [T,K]   bf16   renormalised, cast to the logits dtype
                                               BEFORE the multiply — the authoritative one
    topk_weights_f32           [T,K]   f32    the same values before that cast
    expert_raw                 [T,K,H] bf16   each selected expert's output BEFORE weighting
    routed_sum                 [T,H]   bf16   the weighted sum in the reference form
    shared_raw                 [T,H]   bf16   shared MLP output, pre-sigmoid-gate
    shared_gate_prob           [T,1]   bf16   sigmoid(shared_expert_gate(x))
    shared_out                 [T,H]   bf16   the gated shared expert that is actually added
    combined_out               [T,H]   bf16   routed_sum + shared_out
    *_spec                     ...     f32    our independent form (per-slot GEMM, f32
                                               accumulation, no bf16 rounding of the weights)
    *_exact                    ...     f32    the same math with no bf16 rounding anywhere
                                               (computed in f64, stored as f32)

Who produces the numbers: `speccheck/moe_ref.py`, the independent implementation, which is
this project's oracle (AGENTS.md §Standing rules). The installed `transformers` reference
is still the witness — the dumper runs the reference's own classes over the same input and
refuses to write a byte unless the two agree exactly on `topk_ids` and `routed_sum` and
within the structural bug detector elsewhere. Agreement is recorded in the manifest, so a
later reader sees what was checked rather than being asked to trust it.

Reproduce:
    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python \\
        tools/dump_reference.py --blocks layer0,layer1,layer39,mtp0 --tokens 32 \\
        --seed 20261026 --threads 8 \\
        --out data/dumps/2026-10-26 --manifest checkpoints/dump-manifest-2026-10-26.json
Verify (re-hash every file, compare to the manifest, assert the counts):
    ... tools/dump_reference.py --verify --out data/dumps/2026-10-26 \\
        --manifest checkpoints/dump-manifest-2026-10-26.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "speccheck"))

import moe_ref as R  # noqa: E402
import cross_check_hf as C  # noqa: E402  (the reference side: reuse, do not fork)

# Stage -> (arm, dtype we write, dtype the arm computes in). Kept as data so the manifest
# writer, the verifier and the count assertion all read one table (pitfalls/40: one address
# expression shared by writer and reader).
STAGE_PLAN = [
    # (key suffix, stage, arm, dtype written)
    ("input", "input", "ref", torch.bfloat16),
    ("router_logits", "router_logits", "ref", torch.bfloat16),
    ("topk_ids", "topk_ids", "ref", torch.int64),
    ("topk_weights", "topk_weights", "ref", torch.bfloat16),
    ("topk_weights_f32", "topk_weights", "spec", torch.float32),
    ("expert_raw", "expert_raw", "ref", torch.bfloat16),
    ("routed_sum", "routed_sum", "ref", torch.bfloat16),
    ("shared_raw", "shared_raw", "ref", torch.bfloat16),
    ("shared_gate_prob", "shared_gate_prob", "ref", torch.bfloat16),
    ("shared_out", "shared_out", "ref", torch.bfloat16),
    ("combined_out", "combined_out", "ref", torch.bfloat16),
    ("expert_raw_spec", "expert_raw", "spec", torch.float32),
    ("routed_sum_spec", "routed_sum", "spec", torch.float32),
    ("combined_out_spec", "combined_out", "spec", torch.float32),
    ("router_logits_exact", "router_logits", "exact", torch.float32),
    ("topk_ids_exact", "topk_ids", "exact", torch.int64),
    ("topk_weights_exact", "topk_weights", "exact", torch.float32),
    ("expert_raw_exact", "expert_raw", "exact", torch.float32),
    ("routed_sum_exact", "routed_sum", "exact", torch.float32),
    ("shared_out_exact", "shared_out", "exact", torch.float32),
    ("combined_out_exact", "combined_out", "exact", torch.float32),
]

# The seven weights a block needs. The routed expert stack (1.5 GiB per block) is
# deliberately NOT copied into data/dumps: the shard is the authority and a gate can read
# exactly the experts it needs from the offsets recorded in `referenced_not_dumped`. A
# carried file is named and skipped, never silently ignored (pitfalls/50).
HELD_OUT = {
    "gate_up": "routed expert stack is 1.5 GiB per block and the shard is the authority; "
               "the manifest carries shape/dtype/shard/byte-offsets so a gate can read the "
               "selected experts directly",
    "down": "see gate_up",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tensor(t: torch.Tensor, path: Path, base: Path = REPO) -> dict:
    """Write one tensor as raw little-endian bytes plus a sidecar header.

    `torch.save` is deliberately not used: a gate that reads a dump must be able to get the
    dtype from the file size (pitfalls/30 §Read a dump's dtype off its file size), so the
    payload is element-type-free bytes whose length is elements × itemsize, and the shape
    and dtype live in the manifest next to the sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = t.detach().contiguous()
    if raw.dtype is torch.bfloat16:
        payload = raw.view(torch.uint16).numpy().tobytes()      # bitcast: correct here
    elif raw.dtype is torch.float16:
        payload = raw.view(torch.uint16).numpy().tobytes()
    else:
        payload = raw.numpy().tobytes()                          # widening casts already done
    with open(path, "wb") as fh:
        fh.write(payload)
    if len(payload) != raw.numel() * raw.element_size():
        raise AssertionError(f"{path.name}: wrote {len(payload)} B for "
                             f"{raw.numel()} x {raw.element_size()} B")
    try:
        rel = str(path.relative_to(base))
    except ValueError:      # a scratch --out outside the repo: keep the absolute path
        rel = str(path)
    return {"file": rel, "bytes": len(payload),
            "shape": list(raw.shape), "dtype": str(raw.dtype),
            "elements": int(raw.numel()), "sha256": hashlib.sha256(payload).hexdigest()}


def element_count(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def dump_block(block: R.Block, g: R.Geometry, out_dir: Path, tokens: int, seed: int,
               command: str, cfg_text) -> tuple[list, list, dict]:
    """Produce every pinned tensor for one block, witnessed against the reference."""
    rows, held = [], []
    x = R.make_input(tokens, g, seed)
    w = R.load_block_weights(block, g)

    # --- the witness: the installed reference's own classes over the same input and the
    # same tensors. If this disagrees, the dumps are not a reference and we must not write.
    hf, _M = C.build_hf_block(block, w, cfg_text)
    h = C.hf_block_outputs(hf, x)
    h["expert_raw"] = C.recompute_expert_raw(x, h["topk_ids"], w, g)

    arms = {arm: R.moe_block(x, w, g, arm) for arm in ("ref", "spec", "exact")}
    ref, spec, exact = arms["ref"], arms["spec"], arms["exact"]

    agree = {}
    ids_exact = bool((ref["topk_ids"] == h["topk_ids"]).all().item())
    ids_set = int(sum(1 for t in range(tokens)
                      if sorted(ref["topk_ids"][t].tolist())
                      == sorted(h["topk_ids"][t].tolist())))
    routed_exact = bool((ref["routed_sum"] == h["routed_sum"]).all().item())
    agree["topk_ids_bitexact_vs_reference"] = ids_exact
    agree["topk_ids_rows_set_equal"] = f"{ids_set}/{tokens}"
    agree["routed_sum_bitexact_vs_reference"] = routed_exact
    for stage in ("router_logits", "expert_raw", "routed_sum", "shared_out", "combined_out"):
        a = (C.canon_raw(ref[stage], ref["topk_ids"]) if stage == "expert_raw"
             else C.canon(ref[stage]))
        b = (C.canon_raw(h[stage], h["topk_ids"]) if stage == "expert_raw"
             else C.canon(h[stage]))
        m = R.metrics(a, b, name=f"{block.name}/{stage}")
        agree[stage] = {"max_abs": m["max_abs"], "one_minus_cos": m["one_minus_cos"],
                        "ndiff": m["ndiff"], "n": m["n"]}
        if m["one_minus_cos"] > C.BUG_DETECTOR_ONE_MINUS_COS:
            raise AssertionError(f"{block.name}/{stage}: ref-vs-reference 1-cos="
                                 f"{m['one_minus_cos']:.3e} above the bug detector; the "
                                 "dumper refuses to pin numbers it cannot explain")
    if not ids_exact or not routed_exact or ids_set != tokens:
        raise AssertionError(f"{block.name}: the independent implementation does not match "
                             "the installed reference (see the agreement block). Not "
                             "writing dumps — run speccheck/cross_check_hf.py for the table")

    # --- internal consistency, and its witness: applying the routing weights to the pinned
    # `expert_raw` in the reference order must reproduce `routed_sum` bit for bit, and the
    # post-weight tensor must NOT be equal to `expert_raw` (otherwise "before weighting"
    # would be an unverified claim — the pitfalls/30 hook trap).
    recomposed = torch.zeros(tokens, g.hidden_size, dtype=torch.bfloat16)
    order = torch.argsort(ref["topk_ids"], dim=-1, stable=True)
    for t in range(tokens):
        for j in range(g.top_k):
            k = int(order[t, j])
            recomposed[t] += (ref["expert_raw"][t, k] * ref["topk_weights"][t, k])
    if not torch.equal(recomposed, ref["routed_sum"]):
        raise AssertionError(f"{block.name}: expert_raw x topk_weights does not reproduce "
                             "routed_sum; the pinned intermediate is the wrong tensor number")
    post_weight = ref["expert_raw"] * ref["topk_weights"].unsqueeze(-1)
    n_diff = int((post_weight != ref["expert_raw"]).sum().item())
    if n_diff == 0:
        raise AssertionError(f"{block.name}: post-weight tensor equals expert_raw — every "
                             "routing weight is 1.0 and the 'before weighting' claim is "
                             "vacuous on this input")
    agree["raw_vs_post_weight_elements_differ"] = n_diff
    agree["recompose_routed_sum_bitexact"] = True

    # --- write the tensors
    for suffix, stage, arm, dtype in STAGE_PLAN:
        t = arms[arm][stage]
        if stage == "topk_weights" and suffix == "topk_weights_f32":
            t = spec["topk_weights"]
        if t.dtype is torch.int64 and dtype is not torch.int64:
            raise AssertionError(f"{suffix}: int64 stage asked to write {dtype}")
        t = t.detach()
        if t.dtype is not dtype:
            if dtype is torch.float32 and t.dtype in (torch.float64, torch.float32):
                t = t.to(torch.float32)
            else:
                raise AssertionError(f"{block.name}/{suffix}: arm {arm} gives {t.dtype}, "
                                     f"plan says write {dtype}")
        name = f"{block.name}.{suffix}"
        rec = write_tensor(t, out_dir / f"{name}.bin")
        rec.update({"name": name, "tensor_key": stage, "block": block.name, "arm": arm,
                    "source_shard": source_for(block, stage, w),
                    "source_tensor": source_tensor_for(block, stage, w),
                    "command": command})
        rows.append(rec)

    # the router weights are a dump (1 MiB); the routed stack is named and held out
    rec = write_tensor(w.tensors["router"], out_dir / f"{block.name}.mlp.gate.weight.bin")
    rec.update({"name": f"{block.name}.mlp.gate.weight", "tensor_key": "mlp.gate.weight",
                "block": block.name, "arm": "stored",
                "source_shard": w.source_shards["router"],
                "source_tensor": block.prefix + R.SUFFIXES["router"],
                "command": command})
    rows.append(rec)

    for role in ("shared_gate", "shared_up", "shared_down", "shared_scalar_gate"):
        rec = write_tensor(w.tensors[role], out_dir / f"{block.name}.{R.SUFFIXES[role]}.bin")
        rec.update({"name": f"{block.name}.{R.SUFFIXES[role]}", "tensor_key": R.SUFFIXES[role],
                    "block": block.name, "arm": "stored",
                    "source_shard": w.source_shards[role],
                    "source_tensor": block.prefix + R.SUFFIXES[role],
                    "command": command})
        rows.append(rec)

    index = R.load_index()
    for role, reason in HELD_OUT.items():
        tensor_name = block.prefix + R.SUFFIXES[role]
        shard = R.shard_of(tensor_name, index)
        hdr, _ = R.read_header(R.SHARD_DIR / shard)
        if tensor_name not in hdr:
            raise AssertionError(f"{tensor_name} is not in {shard}'s header; the shard index "
                                 "and the acquired file disagree")
        meta = hdr[tensor_name]
        held.append({
            "block": block.name, "tensor_key": R.SUFFIXES[role], "reason": reason,
            "source_shard": shard, "source_tensor": tensor_name,
            "shape": meta["shape"], "dtype": meta["dtype"],
            "elements": element_count(meta["shape"]),
            "bytes": meta["data_offsets"][1] - meta["data_offsets"][0],
            "data_offsets": meta["data_offsets"],
            "header_bytes": None,   # filled by the caller once known
        })
    del hf, w, arms
    return rows, held, agree


def source_for(block: R.Block, stage: str, w: R.BlockWeights):
    """Which shard the stage's weights came from — recorded per dump as the brief asks."""
    table = {"router_logits": "router", "topk_ids": "router", "topk_weights": "router",
             "expert_raw": "gate_up", "routed_sum": "gate_up",
             "shared_raw": "shared_gate", "shared_gate_prob": "shared_scalar_gate",
             "shared_out": "shared_gate", "combined_out": "gate_up", "input": None}
    role = table.get(stage)
    return w.source_shards[role] if role else "seeded fixture, not a shard"


def source_tensor_for(block: R.Block, stage: str, w: R.BlockWeights):
    table = {"router_logits": "router", "topk_ids": "router", "topk_weights": "router",
             "expert_raw": "gate_up", "routed_sum": "gate_up",
             "shared_raw": "shared_gate", "shared_gate_prob": "shared_scalar_gate",
             "shared_out": "shared_gate", "combined_out": "gate_up", "input": None}
    role = table.get(stage)
    return block.prefix + R.SUFFIXES[role] if role else "make_input(seed, T)"


def verify(manifest_path: Path, out_dir: Path) -> int:
    m = json.loads(manifest_path.read_text())
    bad, checked = [], 0
    for rec in m["dumps"]:
        p = Path(rec["file"])
        if not p.is_absolute():
            p = REPO / p
        if not p.exists():
            bad.append(f"MISSING {rec['file']}")
            continue
        got = sha256_file(p)
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
    n_expected = m.get("counts", {}).get("expected")
    print(f"# verify: {len(m['dumps'])} manifest records, {checked} verified, "
          f"{len(bad)} bad, expected count {n_expected}")
    for b in bad:
        print("  " + b)
    if bad:
        print("VERIFY FAIL")
        return 1
    if not m["dumps"] or checked != len(m["dumps"]) or checked != n_expected:
        print("VERIFY FAIL: a verified count that does not equal its own denominator "
              "cannot pass (pitfalls/40)")
        return 1
    print("VERIFY PASS")
    return 0


def dtype_size(dtype: str) -> int:
    return {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4,
            "torch.float64": 8, "torch.int64": 8}[dtype]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="layer0,layer1,layer39,mtp0")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="data/dumps/2026-10-26")
    ap.add_argument("--manifest", default="checkpoints/dump-manifest-2026-10-26.json")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    out_dir = REPO / args.out
    manifest_path = REPO / args.manifest
    if args.verify:
        return verify(manifest_path, out_dir)

    command = " ".join([sys.executable] + [str(Path(__file__).resolve())] + sys.argv[1:])
    threads = R.set_threads(args.threads)
    g = R.load_geometry()
    blocks = [R.Block.parse(s) for s in args.blocks.split(",") if s.strip()]
    if not blocks:
        raise AssertionError("no blocks selected; an empty dump set cannot pass")
    # The MTP block is not a footnote: it carries its own 256-expert stack, so it is dumped
    # on the same footing as a decoder layer and its expert stack is read from the shards
    # (the published quant carries no draft block at all — see the manifest's
    # referenced_not_dumped).
    if not any(b.kind == "mtp" for b in blocks):
        print("# NOTE: no MTP block in this run; checkpoints/ dump manifest will not carry "
              "draft-block rows")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows, held, agreement = [], [], {}
    for block in blocks:
        r, hd, ag = dump_block(block, g, out_dir, args.tokens, args.seed, command,
                               _text_config())
        rows += r
        held += hd
        agreement[block.name] = ag
        print(f"# {block.name}: {len(r)} tensors pinned, "
              f"{sum(x['bytes'] for x in r):,} B written")

    # header byte offsets for the held-out stacks
    for rec in held:
        hdr, hdr_bytes = R.read_header(R.SHARD_DIR / rec["source_shard"])
        rec["header_bytes"] = hdr_bytes

    expected = len(blocks) * (len(STAGE_PLAN) + 1 + 4)
    manifest = {
        "manifest_version": 1,
        "date": "2026-10-26",
        "producing_command": command,
        "interpreter": {"executable": sys.executable, "version": sys.version.split()[0],
                        "torch": torch.__version__, "threads": threads},
        "geometry": {"num_experts": g.num_experts, "top_k": g.top_k,
                     "moe_intermediate_size": g.moe_intermediate_size,
                     "shared_expert_intermediate_size": g.shared_expert_intermediate_size,
                     "hidden_size": g.hidden_size, "num_hidden_layers": g.num_hidden_layers,
                     "mtp_num_hidden_layers": g.mtp_num_hidden_layers,
                     "blocks_in_text_path": g.n_blocks,
                     "blocks_dumped": [b.name for b in blocks],
                     "expert_bytes": g.expert_bytes,
                     "config": g.config_path},
        "input": {"n_tokens": args.tokens, "seed": args.seed,
                  "dtype": "bfloat16",
                  "method": "speccheck/moe_ref.py §Input generation: fp32 torch.randn from "
                            "a private torch.Generator('cpu').manual_seed(seed), each row "
                            "rescaled to RMS exactly 1.0, cast to bf16. [ASSUMPTION A-01] "
                            "RMS-1 models post_attention_layernorm output; it does NOT "
                            "carry the (1+weight) modulation of a real activation"},
        "reading_rules": {
            "payload": "raw little-endian element bytes, no header; bytes == elements x "
                       "itemsize, and `verify` refuses a record whose size cannot hold its "
                       "own declared shape (pitfalls/30 §Read a dump's dtype off its file size)",
            "bf16": "two bytes per element, bit image — read with a uint16 view then bitcast, "
                    "never cast the uint16 to a float (pitfalls/30 §Tensor.to casts, .view bitcasts)",
            "slot axis": "expert_raw/routed_sum are indexed by top-k SLOT. The *_exact arm's "
                         "slots follow the fp64 router's own rank order, which can permute a "
                         "tie relative to the bf16 router; compare those along K sorted by "
                         "topk_ids_exact (speccheck/cross_check_hf.py §canon_raw)",
            "authoritative": "topk_weights is the cast-BEFORE tensor (the reference's form, "
                             "README prediction 4 verified in source); topk_weights_f32 is the "
                             "pre-cast value, kept so the cast-after form can be measured",
        },
        "dumps": rows,
        "referenced_not_dumped": held,
        "agreement_at_write_time": agreement,
        "counts": {"dumps": len(rows), "expected": expected,
                   "blocks": len(blocks), "referenced_not_dumped": len(held)},
    }
    if len(rows) != expected:
        print(f"MANIFEST FAIL: wrote {len(rows)} records, expected {expected} "
              f"({len(blocks)} blocks x {len(STAGE_PLAN) + 5} stages)")
        return 1
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n")
    print(f"# manifest: {manifest_path} ({manifest_path.stat().st_size:,} B), "
          f"{len(rows)} dumps + {len(held)} named-and-held-out stacks")
    return verify(manifest_path, out_dir)


def _text_config():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    return Qwen3_5MoeTextConfig(**json.loads(R.CONFIG_PATH.read_text())["text_config"])


if __name__ == "__main__":
    sys.exit(main())
