"""Which combine form reproduces the pinned routed_sum, bit for bit?"""
import json, os, sys
REPO='/home/erokhins/erokhins_projects/qwen3.6-35b-a3b'
import numpy as np
m=json.load(open(REPO+'/checkpoints/dump-manifest-2026-10-26.json'))
recs={r['name']:r for r in m['dumps']}
def u16(name):
    r=recs[name]; raw=open(os.path.join(REPO,r['file']),'rb').read()
    assert r['dtype']=='torch.bfloat16', r['dtype']
    return np.frombuffer(raw,dtype=np.uint16).reshape(r['shape'])
def f32of(u): return (np.asarray(u,dtype=np.uint32)<<16).view(np.float32)
def rb(f): return np.frombuffer(np.asarray(f,dtype=np.float32).tobytes(),dtype=np.uint32).astype(np.uint64)
def to_bf16_bits(f):
    b=(np.asarray(f,dtype=np.float32).view(np.uint32).astype(np.uint64))
    lb=(b & 0xFFFF).astype(np.int64)
    r=(b>>16).astype(np.uint32)
    r=(r + ((lb>0x8000)|(((lb==0x8000))&(r&1))).astype(np.uint32)).astype(np.uint16)
    return r
def bf16_of_bits(u): return (u.astype(np.uint32)<<16).view(np.float32)

for blk in ['layer0','layer1','layer39','mtp0']:
    raw_u=u16(blk+'.expert_raw'); w_u=u16(blk+'.topk_weights'); ids=u16  # noqa
    rt_u=u16(blk+'.routed_sum'); idsa=np.frombuffer(open(os.path.join(REPO,recs[blk+'.topk_ids']['file']),'rb').read(),dtype=np.int64).reshape(32,8)
    T,Hh=rt_u.shape; K=raw_u.shape[1]
    rawf=f32of(raw_u); wf=f32of(w_u)
    # form A: product rounded to bf16, bf16 accumulator, ascending expert id
    A=np.zeros((T,Hh),dtype=np.uint16)
    for t in range(T):
        order=np.argsort(idsa[t],kind='stable')
        acc=np.zeros(Hh,dtype=np.uint16)
        for j in order:
            prod=to_bf16_bits(bf16_of_bits(raw_u[t,j])*wf[t,j])
            acc=to_bf16_bits(bf16_of_bits(acc)+bf16_of_bits(prod))
        A[t]=acc
    # form B: f32 products, f32 accumulator, one rounding at the end, ascending id
    B=np.zeros((T,Hh),dtype=np.uint16)
    for t in range(T):
        order=np.argsort(idsa[t],kind='stable')
        acc=np.zeros(Hh,dtype=np.float32)
        for j in order: acc=acc+bf16_of_bits(raw_u[t,j])*wf[t,j]
        B[t]=to_bf16_bits(acc)
    # form C: like B but rank order (no ascending re-sort)
    C=np.zeros((T,Hh),dtype=np.uint16)
    for t in range(T):
        acc=np.zeros(Hh,dtype=np.float32)
        for j in range(K): acc=acc+bf16_of_bits(raw_u[t,j])*wf[t,j]
        C[t]=to_bf16_bits(acc)
    # form D: cast-after: f32 weight (from the f32 dump) times raw, f32 acc, one rounding
    wf32=np.frombuffer(open(os.path.join(REPO,recs[blk+'.topk_weights_f32']['file']),'rb').read(),dtype=np.float32).reshape(32,8)
    D=np.zeros((T,Hh),dtype=np.uint16)
    for t in range(T):
        acc=np.zeros(Hh,dtype=np.float32)
        for j in np.argsort(idsa[t],kind='stable'): acc=acc+bf16_of_bits(raw_u[t,j])*wf32[t,j]
        D[t]=to_bf16_bits(acc)
    def cmp(name,X):
        nd=int((X!=rt_u).sum()); rows=int((X!=rt_u).any(axis=1).sum())
        print("   %-22s ndiff=%6d/%d rows=%d" % (name,nd,X.size,rows))
    print(blk)
    cmp('A bf16 prod+bf16 acc',A); cmp('B f32 acc asc',B); cmp('C f32 acc rank',C); cmp('D cast-after',D)
