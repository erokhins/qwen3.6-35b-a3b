p = "oracle_arm.py"
s = open(p).read()
s = s.replace("        g = bf(xt @ torch.from_numpy(wg[grows]))\n        u = bf(xt @ torch.from_numpy(wu[urows]))",
              "        g = bf(xt @ torch.from_numpy(wg[grows]).T)      # HF row-major [I,H]: the dot is x @ W^T\n"
              "        u = bf(xt @ torch.from_numpy(wu[urows]).T)")
open(p, "w").write(s)
print("ok")
