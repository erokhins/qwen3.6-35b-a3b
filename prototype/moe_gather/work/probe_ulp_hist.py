"""Probe (throwaway): what does the host-vs-device divergence look like on the ordered bit image?

The question this answers before any policy is wired: are the 6,631 cross-arm differing words isolated
single steps (a transcendental route that rounds differently on a few elements) or a systematic shift
(every element moved, which is a different function, not rounding)? Counts only; no timing. CPU-only,
reads the words already on disk from the committed device run.
"""
import sys
from array import array
from collections import Counter

sys.path.insert(0, "/home/erokhins/erokhins_projects/qwen3.6-35b-a3b/tools")
import ulp_policy as UP

I, H = 512, 2048
FIELDS = [("gate", 0, I), ("up", I, I), ("s", 2 * I, I), ("raw", 3 * I, H)]


def words(p):
    a = array("I")
    a.frombytes(open(p, "rb").read())
    return a


def cases():
    out = []
    for line in open("work/cases.tsv"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        out.append(dict(idx=int(f[0]), kind=int(f[1]), mode=f[3], dst=int(f[12]),
                        n=int(f[14]), field_exact=f[16]))
    return out


host, dev, exp = words("work/host_out.f32"), words("work/out.f32"), words("work/expect.f32")
r = UP.Ruler("f32")
tot = Counter()
gu = [c for c in cases() if c["kind"] == 0 and c["field_exact"] == "gate_up"]
print("gate_up expert cases=%d" % len(gu))
for c in gu:
    for name, off, n in FIELDS:
        cen = r.census((i, host[c["dst"] + off + i], dev[c["dst"] + off + i]) for i in range(n))
        if cen.diverging:
            print("case=%3d field=%-4s judged=%5d differing=%5d hist=%s max=%s where=%s"
                  % (c["idx"], name, n, cen.diverging, cen.hist(), cen.max_steps, cen.where()))
        tot[name] += cen.diverging
        tot[name + "_judged"] += n
print("TOTAL host-vs-device by field:", dict(tot))
print()
for name, off, n in FIELDS:
    ho = de = hv = 0
    for c in gu:
        ho += sum(1 for i in range(n) if host[c["dst"] + off + i] != exp[c["dst"] + off + i])
        de += sum(1 for i in range(n) if dev[c["dst"] + off + i] != exp[c["dst"] + off + i])
        hv += sum(1 for i in range(n) if host[c["dst"] + off + i] != dev[c["dst"] + off + i])
    print("vs-oracle field=%-4s host=%6d device=%6d host_vs_device=%6d" % (name, ho, de, hv))
# and everything OUTSIDE the declared cases must be untouched between the arms
others = [c for c in cases() if c["field_exact"] != "gate_up"]
nd = 0
for c in others:
    nd += sum(1 for i in range(c["n"]) if host[c["dst"] + i] != dev[c["dst"] + i])
print("differing words outside the gate_up cases: %d of %d cases" % (nd, len(others)))
