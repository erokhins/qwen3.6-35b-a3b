"""Does each ALT case's DERIVED expectation actually differ from its honest counterpart's?

If a mode's alternative is bit-identical to the honest expectation, the mode cannot move that record
and the case is immune by construction, whatever the plan called it. This reads the shipped
expect.f32 + cases.tsv and says, per ALT case, whether the two derived spans coincide — that is a
property of the arithmetic, not of the kernel, and it is measurable before anyone argues about it.
"""
import struct
import sys

NAN_MASK = 0x7F800000


def words(p):
    with open(p, "rb") as fh:
        raw = fh.read()
    return struct.unpack("<%dI" % (len(raw) // 4), raw)


def load_cases(path):
    out = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        out.append(dict(case_index=int(f[0]), kind=int(f[1]), mode_name=f[3], block=int(f[4]),
                        row=int(f[6]), slot=int(f[7]), expect_name=f[9], dst_off=int(f[12]),
                        exp_off=int(f[13]), n_floats=int(f[14]), field_exact=f[16]))
    return out


exp = words(sys.argv[1] if len(sys.argv) > 1 else "work/expect.f32")
cases = load_cases("work/cases.tsv")
honest = {}
for c in cases:
    if c["mode_name"] == "MG_MODE_CORRECT":
        honest[(c["kind"], c["block"], c["row"], c["slot"])] = c
same = diff = 0
print("ALT/IMMUNE cases: does the derived expectation differ from the honest one?")
for c in cases:
    if c["expect_name"] not in ("MG_EXP_ALT", "MG_EXP_IMMUNE"):
        continue
    h = honest.get((c["kind"], c["block"], c["row"], c["slot"]))
    if h is None or h["case_index"] == c["case_index"]:
        continue
    n = c["n_floats"]
    nd = sum(1 for i in range(n)
             if exp[c["exp_off"] + i] != exp[h["exp_off"] + i])
    tag = "COINCIDE" if nd == 0 else "differs "
    if nd == 0:
        same += 1
    else:
        diff += 1
    if nd == 0 or c["kind"] == 1:
        print("  %s case=%-3d %-34s kind=%d block=%d row=%-5d class=%-14s expectation differs in "
              "%d/%d words" % (tag, c["case_index"], c["mode_name"], c["kind"], c["block"],
                               c["row"], c["expect_name"], nd, n))
print("COUNTS coincide=%d differs=%d (a case whose derived alternative coincides is immune by "
      "construction: asserting it must differ asserts the impossible)" % (same, diff))
