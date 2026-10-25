"""Probe (throwaway): exercise gate.cross_arm() offline on the COMMITTED device-run words.

Purpose: the cross-arm rule must be validated before a device run is spent on it. This reads the words
already on disk (work/host_out.f32, work/out.f32, work/expect.f32, work/cases.tsv) from the committed
red run and asserts the four things D-16 option A promises: gate/up bit-identical between arms, nothing
outside the 18 declared cases moved, the divergence attributed, and the divergence isolated. Counts
only; no timing; no device.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(LANE))
sys.path.insert(0, LANE)
os.environ["MG_I"] = "512"
os.environ["MG_H"] = "2048"
import gate as G                                              # noqa: E402

work = os.path.join(LANE, "work")
cases = G.read_cases(os.path.join(work, "cases.tsv"))
nf = sum(c["n_floats"] for c in cases)
hostw = G.read_words(os.path.join(work, "host_out.f32"), nf)
devw = G.read_words(os.path.join(work, "out.f32"), nf)
expw = G.read_words(os.path.join(work, "expect.f32"), nf)
v = G.V()
G.cross_arm(v, cases, hostw, devw, expw)
print("PROBE checks=%d findings=%d bad=%s" % (v.n, len(v.bad), v.bad or "none"))
# and the number the decision was taken on, re-derived independently of the census
nd = G.n_diff(hostw, devw)
print("PROBE whole-buffer differing=%d of %d (the record says 6631 of 306176)" % (nd, nf))
