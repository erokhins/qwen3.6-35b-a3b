p = "oracle_arm.py"
s = open(p).read()
old = '''    def raw_of(grows, urows):
        g = bf(xt @ torch.from_numpy(wg[grows]).T)      # HF row-major [I,H]: the dot is x @ W^T
        u = bf(xt @ torch.from_numpy(wu[urows]).T)'''
new = '''    fused = fused_of(wg, wu)          # the [2I,H] slab the kernel actually reads; a candidate row map
                                      # is an index into THIS, not into the two tensors

    def raw_of(grows, urows):
        g = bf(xt @ torch.from_numpy(fused[grows]).T)   # row-major rows: the dot is x @ W^T
        u = bf(xt @ torch.from_numpy(fused[urows]).T)'''
assert old in s
open(p, "w").write(s.replace(old, new))
print("ok")
