#!/usr/bin/env python3
"""Per-field diff of one case against the honest case of the same block and row.

The gate reported `case 6 gate=0 up=0 act=0` for layout_pair_swap, which would mean the kernel's
pair-swap produces the SAME act as the honest case -- the opposite of the declared blindness (the C
comment says 505 of 512 act words differ). Either the comparison is aimed at the wrong spans or the
swap is not happening. This prints the field-by-field word differences so the two are told apart.

    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python work/dbg_case_fields.py CASE
No device, no lock, no timing.
"""
import os
import sys
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
FIELD_OFF = {"gate": 0, "up": 512, "act": 1024, "raw": 1536, "out": 3584, "prob": 5632}
FIELD_LEN = {"gate": 512, "up": 512, "act": 512, "raw": 2048, "out": 2048, "prob": 1}
CASE_FLOATS = 5633


def read_words(path):
    a = array("I")
    with open(path, "rb") as fh:
        a.frombytes(fh.read())
    return a


def cases(path):
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        rows.append(dict(index=int(f[0]), label=f[1], mode=f[2], cast=f[3], sgate=f[4],
                         block=f[5], row=int(f[6]), dst_off=int(f[7]), exp_off=int(f[8]),
                         n_floats=int(f[9])))
    return rows


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "6"
    buf = sys.argv[2] if len(sys.argv) > 2 else "host_out.f32"
    cs = cases(os.path.join(HERE, "cases.tsv"))
    words = read_words(os.path.join(HERE, buf))
    exp = read_words(os.path.join(HERE, "inputs.bin"))
    by = {c["index"]: c for c in cs}
    tgt = by[int(which)]
    hon = [c for c in cs if c["label"] == "honest" and c["block"] == tgt["block"]
           and c["row"] == tgt["row"]][0]
    print("TARGET case=%d label=%s mode=%s cast=%s block=%s row=%d dst=%d | HONEST case=%d dst=%d"
          % (tgt["index"], tgt["label"], tgt["mode"], tgt["cast"], tgt["block"], tgt["row"],
             tgt["dst_off"], hon["index"], hon["dst_off"]))
    print("buffer_words=%d case_span=%d" % (len(words), CASE_FLOATS))
    for fname, off in FIELD_OFF.items():
        a = tgt["dst_off"] + off
        b = hon["dst_off"] + off
        n = FIELD_LEN[fname]
        d_out = sum(1 for i in range(n) if words[a + i] != words[b + i])
        d_exp = sum(1 for i in range(n) if exp[tgt["exp_off"] + off + i] != exp[hon["exp_off"] + off + i])
        d_arm_vs_exp = sum(1 for i in range(n) if words[a + i] != exp[tgt["exp_off"] + off + i])
        print("FIELD %-5s n=%-5d arm_vs_honest_arm=%-5d oracle_vs_honest_oracle=%-5d arm_vs_own_expect=%d"
              % (fname, n, d_out, d_exp, d_arm_vs_exp))
    print("first 4 act words arm=%s honest=%s" %
          ([hex(words[tgt["dst_off"] + 1024 + i]) for i in range(4)],
           [hex(words[hon["dst_off"] + 1024 + i]) for i in range(4)]))
    print("no timing taken")
    return 0


if __name__ == "__main__":
    sys.exit(main())
