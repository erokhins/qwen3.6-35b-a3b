"""Diagnostic: for every EXPERT case, what is actually in its output span?

The driver pre-fills `out` with mg_nan() (host_probe.cpp, the loop after calloc), so the three
states are distinguishable from the words alone:
  never written        -> every word == MG_NAN_BITS (0x7FC00000)
  written, decoded zero -> every word == 0x00000000 (or -0.0 == 0x80000000)
  written, wrong       -> some other pattern
This prints that census per case, plus the per-field census, so "all zero" is a measurement and not
a prose claim (pitfalls/40 §"a counter that reads exactly 0 is vacuous until witnessed non-zero").
"""
import struct
import sys
from collections import Counter

NAN = 0x7FC00000
I, H = 512, 2048
FIELDS = [("gate", 0, I), ("up", I, I), ("s", 2 * I, I), ("raw", 3 * I, H)]


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
        out.append(dict(case_index=int(f[0]), kind=int(f[1]), mode=int(f[2]), mode_name=f[3],
                        block=int(f[4]), row=int(f[6]), slot=int(f[7]), expect_name=f[9],
                        dst_off=int(f[12]), exp_off=int(f[13]), n_floats=int(f[14])))
    return out


def census(vals):
    c = Counter()
    for w in vals:
        if w == 0:
            c["zero"] += 1
        elif w == 0x80000000:
            c["negzero"] += 1
        elif w == NAN:
            c["nan"] += 1
        elif (w & 0x7F800000) == 0x7F800000 and (w & 0x7FFFFF):
            c["other_nan"] += 1
        else:
            c["finite_other"] += 1
    return c


def main():
    out = words("work/host_out.f32")
    exp = words("work/expect.f32")
    print("WORDS out=%d expect=%d" % (len(out), len(exp)))
    print("OUT-CENSUS %s" % dict(census(out)))
    print("EXP-CENSUS %s" % dict(census(exp)))
    cases = load_cases("work/cases.tsv")
    n_exp = n_com = 0
    allzero_exp = 0
    allnan_exp = 0
    other_exp = 0
    for c in cases:
        if c["kind"] != 0:
            n_com += 1
            continue
        n_exp += 1
        s, n = c["dst_off"], c["n_floats"]
        cc = census(out[s:s + n])
        if cc.get("zero", 0) + cc.get("negzero", 0) == n:
            allzero_exp += 1
            shape = "ALL-ZERO"
        elif cc.get("nan", 0) == n:
            allnan_exp += 1
            shape = "ALL-NAN"
        else:
            other_exp += 1
            shape = "MIXED"
        if shape != "ALL-ZERO":
            print("CASE %d %s block=%s row=%s slot=%s %s span=%d %s"
                  % (c["case_index"], c["mode_name"], c["block"], c["row"], c["slot"], shape, n,
                     dict(cc)))
            for fname, fo, fn in FIELDS:
                sub = out[s + fo:s + fo + fn]
                print("   field %-5s %s first4=%s"
                      % (fname, dict(census(sub)), [hex(w) for w in sub[:4]]))
    print("EXPERT cases=%d all_zero=%d all_nan=%d mixed=%d" % (n_exp, allzero_exp, allnan_exp,
                                                              other_exp))
    print("COMBINE cases=%d" % n_com)
    # how many out words anywhere are NaN (unwritten) vs written
    print("OUT nan-words=%d zero-words=%d total=%d"
          % (out.count(NAN), out.count(0), len(out)))


main()
