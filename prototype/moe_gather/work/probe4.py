"""Confirm the 8-byte header-offset bug in the router lane's P3 probe, and adjudicate P3."""
import json, os, sys, struct, time
REPO='/home/erokhins/erokhins_projects/qwen3.6-35b-a3b'
sys.path.insert(0, REPO+'/speccheck')
import numpy as np, torch
import moe_ref as R

g=R.load_geometry(); E,twoI,H,I = 256,1024,2048,512
m=json.load(open(REPO+'/checkpoints/dump-manifest-2026-10-26.json'))
recs={r['name']:r for r in m['dumps']}
def arr(name):
    r=recs[name]; raw=open(os.path.join(REPO,r['file']),'rb').read()
    if r['dtype']=='torch.bfloat16':
        u=np.frombuffer(raw,dtype=np.uint16).reshape(r['shape']); return (np.asarray(u,dtype=np.uint32)<<16).view(np.float32), u
    if r['dtype']=='torch.float32': return np.frombuffer(raw,dtype=np.float32).reshape(r['shape']), None
    if r['dtype']=='torch.int64':   return np.frombuffer(raw,dtype=np.int64).reshape(r['shape']), None
    raise AssertionError(r['dtype'])

blk='layer0'; bo=R.Block.parse(blk); idx=R.load_index()
gu_n=bo.prefix+R.SUFFIXES['gate_up']; dn_n=bo.prefix+R.SUFFIXES['down']
gu_s=os.path.join(str(R.SHARD_DIR), R.shard_of(gu_n, idx)); dn_s=os.path.join(str(R.SHARD_DIR), R.shard_of(dn_n, idx))
gu_h,gu_hn=R.read_header(gu_s); dn_h,dn_hn=R.read_header(dn_s)
print("gu header json bytes=%d  first data_offset=%d" % (gu_hn, gu_h[gu_n]['data_offsets'][0]))
with open(gu_s,'rb') as fh:
    fh.seek(gu_hn+gu_h[gu_n]['data_offsets'][0]); a=fh.read(8)
    fh.seek(8+gu_hn+gu_h[gu_n]['data_offsets'][0]); b=fh.read(8)
print("bytes at (header_json_len + off) :", a.hex())
print("bytes at (8 + header_json_len + off):", b.hex())
# what does safe_open say the first row of expert 0 is?
from safetensors import safe_open
with safe_open(gu_s, framework='pt') as f:
    t0=f.get_tensor(gu_n)[0,0,:4]
print("safe_open expert0 row0[:4] =", [hex(v) for v in t0.view(torch.uint16).tolist()],
      " bf16 vals=", t0.tolist())
