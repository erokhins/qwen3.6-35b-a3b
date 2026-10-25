"""Attribute the host-arm vs device-arm word differences, per case and per field.

The device gate's single finding is that the two arms' output buffers are not word-for-word identical.
"This lane" already declares, per case, which fields are bit-comparable: field_exact=all means every
word is an integer-exact fixture, field_exact=gate_up means gate and up are fp32 dots over published
bytes while s and raw cross expf (fp32 expf in each compiler, which the lane counts and never bands).
This script says whether the differences live entirely in the fields the lane refuses to band, or
somewhere that was supposed to be identical. Counts only; no timing.
"""
import struct
from collections import Counter


def words(p):
    with open(p, "rb") as fh:
        raw = fh.read()
    return struct.unpack("<%dI" % (len(raw) // 4), raw)


def cases():
    out = []
    for line in open("work/cases.tsv"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        out.append(dict(idx=int(f[0]), kind=int(f[1]), mode_name=f[3], block=int(f[4]),
                        expect_name=f[9], dst=int(f[12]), n=int(f[14]), field_exact=f[16]))
    return out


I, H = 512, 2048
FIELDS = [("gate", 0, I), ("up", I, I), ("s", 2 * I, I), ("raw", 3 * I, H)]
host = words("work/host_out.f32")
dev = words("work/out.f32")
exp = words("work/expect.f32")
assert len(host) == len(dev) == len(exp)
by_field = Counter()
by_class = Counter()
by_case = Counter()
worse_than_oracle = 0
n_cases_diff = 0
for c in cases():
    if c["kind"] != 0:
        flds = [("routed", 0, H)]
    else:
        flds = FIELDS
    for name, off, n in flds:
        d = sum(1 for i in range(n)
                if host[c["dst"] + off + i] != dev[c["dst"] + off + i])
        if d:
            by_field[name] += d
            by_class["%s/%s" % (c["mode_name"], c["field_exact"])] += d
            by_case[c["idx"]] += d
        n_cases_diff += 1 if d else 0
print("DIFF host-vs-device total=%d by field: %s" % (sum(by_field.values()), dict(by_field)))
print("DIFF by mode/field_exact: %s" % dict(by_class))
print("DIFF cases affected=%d of %d cases" % (len(by_case), len(cases())))
# And per field: how far is EACH arm from the oracle? If both arms differ from the oracle in the same
# field and from each other, the field is a route two compilers round differently, which is what the
# lane counts. If one arm matches the oracle exactly and the other does not, that is a device finding.
print("per field, over the cases whose declaration says the field crosses the transcendental:")
for name, off, n in FIELDS:
    ho = de = hv = dv = 0
    for c in cases():
        if c["kind"] != 0 or c["field_exact"] != "gate_up":
            continue
        # each case's expectation span starts at its own exp_off; cases.tsv carries dst only, so use
        # the invariant that exp and out are laid out with the same offsets in this plan
        eo = c["dst"]
        ho += sum(1 for i in range(n) if host[c["dst"] + off + i] != exp[eo + off + i])
        de += sum(1 for i in range(n) if dev[c["dst"] + off + i] != exp[eo + off + i])
        hv += sum(1 for i in range(n) if host[c["dst"] + off + i] != dev[c["dst"] + off + i])
    print("  field %-5s host_vs_oracle=%d device_vs_oracle=%d host_vs_device=%d" % (name, ho, de, hv))
print("TOTALS differing_words_host_vs_device=%d of %d"
      % (sum(1 for i in range(len(host)) if host[i] != dev[i]), len(host)))
