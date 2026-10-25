"""Does speccheck's own ref arm reproduce the pinned expert_raw / routed_sum for layer0?"""
import json, os, sys, time
REPO='/home/erokhins/erokhins_projects/qwen3.6-35b-a3b'
sys.path.insert(0, REPO+'/speccheck')
import numpy as np, torch
import moe_ref as R

m=json.load(open(REPO+'/checkpoints/dump-manifest-2026-10-26.json'))
recs={r['name']:r for r in m['dumps']}
g=R.load_geometry()
def arr(name):
    r=recs[name]; raw=open(os.path.join(REPO,r['file']),'rb').read()
    assert len(raw)==r['bytes']
    if r['dtype']=='torch.bfloat16':
        u=np.frombuffer(raw,dtype=np.uint16).reshape(r['shape'])
        return (np.asarray(u,dtype=np.uint32)<<16).view(np.float32), u
    if r['dtype']=='torch.float32': return np.frombuffer(raw,dtype=np.float32).reshape(r['shape']), None
    if r['dtype']=='torch.int64':   return np.frombuffer(raw,dtype=np.int64).reshape(r['shape']), None
    raise AssertionError(r['dtype'])

blk='layer0'
x_pin,_=arr(blk+'.input')
er_pin,er_pin_u16=arr(blk+'.expert_raw')
rt_pin,rt_pin_u16=arr(blk+'.routed_sum')
ids_pin,_=arr(blk+'.topk_ids')
w_pin,_=arr(blk+'.topk_weights')
print("pinned ids row0:", ids_pin[0].tolist(), "w row0:", [float(v) for v in w_pin[0]])

b=R.Block.parse(blk)
t0=time.time()
w=R.load_block_weights(b,g)
print("loaded weights in %.1fs" % (time.time()-t0), {k:tuple(v.shape) for k,v in w.tensors.items()})
x=torch.from_numpy(x_pin.copy()).to(torch.bfloat16)
print("x bit-identity vs pinned input:", bool((x.view(torch.uint16).numpy()==np.frombuffer(open(os.path.join(REPO,recs[blk+'.input']['file']),'rb').read(),dtype=np.uint16).reshape(x_pin.shape)).all()))
torch.set_num_threads(8)
t0=time.time()
out=R.moe_block(x,w,g,'ref')
print("moe_block ref in %.1fs" % (time.time()-t0))
er=out['expert_raw'].view(torch.uint16).numpy()
rt=out['routed_sum'].view(torch.uint16).numpy()
print("expert_raw bit-equal:", bool((er==er_pin_u16).all()), " ndiff:", int((er!=er_pin_u16).sum()), "/", er.size)
print("routed_sum bit-equal:", bool((rt==rt_pin_u16).all()), " ndiff:", int((rt!=rt_pin_u16).sum()), "/", rt.size)
# per-row cos of row0 slot0
a=out['expert_raw'][0,0].double().numpy(); bref=er_pin[0,0]
print("row0 slot0: cos=%.6f maxabs=%.3e refmax=%.3e" % (
  float(a@bref/ (np.linalg.norm(a)*np.linalg.norm(bref)+1e-30)), float(np.abs(a-bref).max()), float(np.abs(bref).max())))
print("ids equal:", bool((out['topk_ids'].numpy()==ids_pin).all()))
