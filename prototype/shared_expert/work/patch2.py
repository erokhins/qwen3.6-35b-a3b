p = "oracle_arm.py"
s = open(p).read()
old = '''    norms = np.linalg.norm(fused, axis=1)
    for lay in LAYOUTS:
        hits = 0
        for src, which in ((gate_vals, 0), (up_vals, 1)):
            for j in range(n):
                want = layout_rows(lay, j)[which]
                if int(np.argmin(np.abs(norms - np.linalg.norm(src[j])))) == want:
                    hits += 1
'''
new = '''    for lay in LAYOUTS:
        hits = 0
        for src, which in ((gate_vals, 0), (up_vals, 1)):
            # nearest by full row distance. A norm test would not do: dozens of rows of a weight matrix
            # share a norm, which is exactly how a proxy looks like a failure on the honest candidate.
            d = np.sqrt(((fused[None, :, :] - src[:n][:, None, :]) ** 2).sum(axis=2))
            near = np.argmin(d, axis=1)
            for j in range(n):
                if near[j] == layout_rows(lay, j)[which]:
                    hits += 1
'''
assert old in s
open(p, "w").write(s.replace(old, new))
print("patched")
