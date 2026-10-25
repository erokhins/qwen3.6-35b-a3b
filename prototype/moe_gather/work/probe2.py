"""Decisive probe: do the locally acquired safetensors reproduce every DUMPED weight tensor,
and does speccheck's own ref arm reproduce the pinned expert_raw?"""
import json, os, sys, hashlib
REPO='/home/erokhins/erokhins_projects/qwen3.6-35b-a3b'
sys.path.insert(0, REPO+'/speccheck')
import numpy as np, torch
import moe_ref as R

m=json.load(open(REPO+'/checkpoints/dump-manifest-2026-10-26.json'))
g=R.load_geometry()
weights=[r for r in m['dumps'] if '.weight' in r['name']]
print("dumped weight tensors:", len(weights))
bad=0
for r in weights:
    block=r['block']; suffix=r['name'][len(block)+1:]
    role={v:k for k,v in R.SUFFIXES.items()}.get(suffix)
    if role is None:
        print("  ? unknown suffix", r['name']); bad+=1; continue
    b=R.Block.parse(block)
    t=R.load_block_weights(b, g, roles=(role,)).tensors[role]
    raw=t.contiguous().view(torch.uint16).numpy().tobytes()
    got=hashlib.sha256(raw).hexdigest()
    ok = got==r['sha256']
    bad += 0 if ok else 1
    print("  %-46s role=%-17s dump=%s local=%s %s" % (r['name'], role, r['sha256'][:12], got[:12],
          "MATCH" if ok else "DIFFER"))
print("mismatched weight tensors:", bad, "of", len(weights))
