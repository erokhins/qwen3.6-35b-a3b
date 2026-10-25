"""Sizing probes: distinct experts per token per block; dequant time for one expert of each width."""
import json, os, sys, time
REPO='/home/erokhins/erokhins_projects/qwen3.6-35b-a3b'
sys.path.insert(0, REPO+'/speccheck'); sys.path.insert(0, REPO+'/tools')
import numpy as np, torch
import moe_ref as R, gguf_reader as gr, audit_published_quant as apq

m=json.load(open(REPO+'/checkpoints/dump-manifest-2026-10-26.json'))
recs={r['name']:r for r in m['dumps']}
def arr(name):
    r=recs[name]; raw=open(os.path.join(REPO,r['file']),'rb').read()
    if r['dtype']=='torch.bfloat16':
        u=np.frombuffer(raw,dtype=np.uint16).reshape(r['shape']); return (np.asarray(u,dtype=np.uint32)<<16).view(np.float32), u
    if r['dtype']=='torch.float32': return np.frombuffer(raw,dtype=np.float32).reshape(r['shape']), None
    if r['dtype']=='torch.int64':   return np.frombuffer(raw,dtype=np.int64).reshape(r['shape']), None
    raise AssertionError(r['dtype'])
for blk in ['layer0','layer1','layer39','mtp0']:
    ids=arr(blk+'.topk_ids')[0]
    tot=set(); per=[len(set(ids[t].tolist())) for t in range(ids.shape[0])]
    for t in range(4): tot |= set(ids[t].tolist())
    print("%s ids shape %s  distinct-per-token=%s  union(rows0..3)=%d  union(all)=%d" %
          (blk, ids.shape, per[:8], len(tot), len(set(ids.reshape(-1).tolist()))))
    w=arr(blk+'.topk_weights')[0]
    print("   weights row0 bits:", ["%04x"%int((np.float32(v).view(np.uint32)>>16)) for v in w[0]])
    r=recs[blk+'.topk_weights']; print("   manifest dtype", r['dtype'])
# dequant timing for one expert of each width
gr.grant_selftest_token(apq.SELFTEST_TOKEN)
gg=gr.Gguf(apq.GGUF_PATH)
tmap={t.name:t for t in gg.tensors}
for name,tname in [('blk.0.ffn_gate_exps.weight','Q4_K'),('blk.0.ffn_down_exps.weight','Q5_K'),
                   ('blk.39.ffn_down_exps.weight','Q6_K')]:
    t=tmap[name]; bb=gr.BLOCK[tname][1]; nelts=512*2048 if 'gate' in name or 'up' in name else 2048*512
    nb=nelts//256
    t0=time.time(); buf=gg.read(apq.SELFTEST_TOKEN, gg.data_start+t.offset, nb*bb, "probe5 timing"); t1=time.time()
    vals=gr.dequant(buf, tname); t2=time.time()
    a=np.asarray(vals,dtype=np.float32)
    print("%-28s %s blocks=%d bytes=%d read=%.3fs dequant=%.2fs absmax=%.3f nonzero=%d" %
          (name,tname,nb,len(buf),t1-t0,t2-t1,float(np.abs(a).max()),int((a!=0).sum())))
