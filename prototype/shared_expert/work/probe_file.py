#!/usr/bin/env python3
"""probe_file.py — scratch probe: what does the pinned GGUF hold for the shared expert?

Read-only, structure only (no data window is opened here). Run with plain python3.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
import gguf_reader as gr          # noqa: E402
import gguf_width_table as GWT    # noqa: E402

GGUF = os.path.join(REPO, "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

g = gr.Gguf(GGUF)
print("file=%s n_tensors=%d data_start=%d" % (os.path.basename(g.path), g.n_tensors, g.data_start))
print("derived block geom Q8_0=%s F32=%s" % (gr.BLOCK["Q8_0"], gr.BLOCK["F32"]))
print("width-table   Q8_0=%s F32=%s" % (GWT.BLK["Q8_0"][:2], GWT.BLK["F32"][:2]))
hits = [t for t in g.tensors if "shexp" in t.name]
print("shexp tensors=%d" % len(hits))
for t in hits:
    if t.name.startswith(("blk.0.", "blk.39.", "blk.40.", "blk.41.")):
        print("  %-40s dims=%s shape_hf=%s type=%s n_elts=%d bytes=%d"
              % (t.name, t.dims, t.shape_hf, t.type_name, t.n_elts, t.row))
byblk = {}
for t in hits:
    b = t.name.split(".")[1]
    byblk[b] = byblk.get(b, 0) + 1
print("shexp tensors per block index:", " ".join("%s:%d" % (k, byblk[k]) for k in sorted(byblk, key=int)))
print("count blocks with shexp:", len(byblk))
# the pinned dumps: what shapes do the shared tensors declare?
man = json.load(open(os.path.join(REPO, "checkpoints/dump-manifest-2026-10-26.json")))
print("manifest geometry=%s" % json.dumps(man["geometry"]))
print("manifest input=%s" % json.dumps(man["input"])[:400])
print("manifest counts=%s" % json.dumps(man["counts"])[:400])
print("reading_rules keys=%s" % json.dumps(man["reading_rules"])[:900])
recs = man["dumps"]
print("manifest dumps=%d" % len(recs))
for r in recs:
    n = r.get("name", "")
    if "shared" in n and (n.startswith("layer0.") or n.startswith("mtp0.")):
        print("  dump %-46s shape=%s dtype=%s bytes=%d" % (n, r["shape"], r["dtype"], r["bytes"]))
print("--- gguf names containing mtp:", [t.name for t in g.tensors if "mtp" in t.name][:5])
print("--- gguf max blk index:", max(int(t.name.split(".")[1]) for t in g.tensors if t.name.startswith("blk.")))
