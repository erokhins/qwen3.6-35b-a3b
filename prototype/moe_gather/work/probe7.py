"""Find a 4-value multiset whose bf16 left-to-right sum depends on the order (for the selftest)."""
import numpy as np, itertools
def to_bf16_bits(f):
    b=np.asarray(f,dtype=np.float32).view(np.uint32).astype(np.uint64)
    lb=(b & 0xFFFF).astype(np.int64); r=(b>>16).astype(np.uint32)
    r=(r+((lb>0x8000)|(lb==0x8000)&(r&1)).astype(np.uint32)).astype(np.uint16); return r
def bf(v): return (v.astype(np.uint32)<<16).view(np.float32)
def seq(vals):
    acc=np.uint16(0)
    for v in vals:
        prod=to_bf16_bits(np.float32(v))       # weight 1.0 -> product rounding is a no-op for bf16 in
        acc=to_bf16_bits(bf(acc)+bf(prod))
    return bf(acc)
cands=[1.0,2.0,0.5,0.25,2**-9,2**-8,3*2**-9,1.5,1.25,4.0,6.0,2**-7,5*2**-9,2**-6,1.0078125,3.0]
found=[]
for combo in itertools.combinations_with_replacement(cands,4):
    s=set(itertools.permutations(combo))
    vals={float(seq(np.float32(list(p)))) for p in s}
    if len(vals)>1: found.append((combo,sorted(vals)))
print("orders-sensitive multisets:", len(found))
for c,v in found[:8]: print("   ", c, v)
