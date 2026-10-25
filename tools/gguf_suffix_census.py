#!/usr/bin/env python3
"""Suffix census of the pinned GGUF: what is actually in the shipping artifact, by tensor family.

Per-family cut of the shipping artifact. The authoritative TOTALS are tools/bf16_census.py and
quantization/evidence/bf16-census-adjudication-2026-10-26.md; this is not a second ruler over the grand
total, it is the only thing that prints the breakdown by tensor suffix -- and the breakdown is what
caused the surprise that started the census dispute (see below), so the cut has to be reproducible.

Why it exists: the GGUF header's own sum(elts)x2 = 64.5604 GiB matched neither tools/quant_scenarios.py
(65.7198) nor the first census pass (66.1337). The resolution, in
quantization/evidence/bf16-census-adjudication-2026-10-26.md, was that 64.5604 is text-minus-draft (the
GGUF carries zero mtp.* tensors) and quant_scenarios.py had four wrong terms. Neither side was right
about the thing it was defending.

A total that disagrees with two config-driven
censuses by more than 1 GiB has only two explanations: the censuses model tensors that the artifact
does not carry, or the artifact legitimately omits tensors the HF text model has. Which one, and
where, is a name question -- so this enumerates every one of the 733 records by its stripped suffix
instead of by a hand-written substring classifier. (An earlier hand-written bucketing of the same
records mislabelled the GDN family entirely: this model spells it `attn_qkv` / `attn_gate` /
`ssm_out`, not `in_proj` / `linear_attn`, so name-substring buckets are the thing not to trust.)
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from gguf_reader import Gguf  # noqa: E402

GGUF = "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
g = Gguf(GGUF)
GiB = 2.0 ** 30

groups = {}
for t in g.tensors:
    nm = t.name
    if nm.startswith("blk."):
        stripped = re.sub(r"^blk\.\d+\.", "blk.N.", nm)
    else:
        stripped = nm
    d = groups.setdefault(stripped, {"n": 0, "elts": 0, "types": set(), "dims": set()})
    elts = 1
    for x in t.dims:
        elts *= int(x)
    d["n"] += 1
    d["elts"] += elts
    d["types"].add(t.type_name if hasattr(t, "type_name") else str(t.type_id))
    d["dims"].add(tuple(t.dims))

total = sum(d["elts"] for d in groups.values())
print("distinct stripped suffixes: %d   tensors: %d   sum(elts): %d = %.4f GiB bf16"
      % (len(groups), len(g.tensors), total, total * 2.0 / GiB))
print()
print("%-46s %6s %16s %10s  %s" % ("suffix", "count", "elts", "GiB", "dims / types"))
for key in sorted(groups, key=lambda k: -groups[k]["elts"]):
    d = groups[key]
    dims = ", ".join("x".join(str(x) for x in dd) for dd in sorted(d["dims"])[:3])
    tys = ",".join(sorted(d["types"]))
    print("%-46s %6d %16d %10.4f  %s [%s]" % (key, d["n"], d["elts"], d["elts"] * 2.0 / GiB,
                                              dims, tys))

# the specific question that settles the dispute: does the shipping artifact carry the GDN input
# projections at all, and the out_proj W7 says quant_scenarios.py omits?
print()
print("--- targeted: the tensors the dispute turns on ---")
for pat in ("in_proj", "ssm_out", "conv1d", "A_log", "dt_bias", "ssm_alpha", "ssm_beta",
            "gate_up", "ffn_gate", "ffn_up", "ffn_down", "shared_expert", "post_attention_norm"):
    hits = [k for k in groups if pat in k]
    if not hits:
        print("  %-20s NONE -- no tensor name contains this" % pat)
    else:
        for k in sorted(hits):
            print("  %-20s %-34s x%-4d %14d elts" % (pat, k, groups[k]["n"], groups[k]["elts"]))
