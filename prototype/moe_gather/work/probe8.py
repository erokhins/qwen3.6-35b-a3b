import numpy as np, itertools
def to_bf16_bits(f):
    b=np.asarray(f,dtype=np.float32).view(np.uint32).astype(np.uint64)
    lb=(b & 0xFFFF).astype(np.int64); r=(b>>16).astype(np.uint32)
    r=(r+((lb>0x8000)|(lb==0x8000)&(r&1)).astype(np.uint32)).astype(np.uint16); return r
def bf(v): return (v.astype(np.uint32)<<16).view(np.float32)
def seq(vals):
    acc=np.uint16(0)
    for v in vals:
        prod=to_bf16_bits(np.float32(v)); acc=to_bf16_bits(bf(acc)+bf(prod))
    return float(bf(acc))
for combo in [(1.0,1.0,1.0,0.009765625),(1.0,1.0,0.5,0.009765625)]:
    print(combo)
    for p in sorted(set(itertools.permutations(combo))):
        print("   ", p, seq(list(p)), "bits=%08x"%np.float32(seq(list(p))).view(np.uint32))
