#!/usr/bin/env python3
"""Where exactly do the device and host arms disagree, and is it a rounding tie or a wrong index?

The device gate reports `case N field raw: 1 of 2048 words differ` for many cases and
`host and device agree bit-for-bit ... got=65`. A claim of divergence is only useful with a location:
this prints which output column differs in which case, in which direction, and whether the two bf16
values are ADJACENT (one step apart, which is what a rounding difference looks like) or far apart
(which is what a wrong row/column index looks like).

    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python work/dbg_device_word_diff.py
Reads work/host_out.f32 and work/dev_out.f32 produced by the device gate. No device, no lock, no timing.
"""
import os
import sys
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
FIELD_OFF = {"gate": 0, "up": 512, "act": 1024, "raw": 1536, "out": 3584, "prob": 5632}
FIELD_LEN = {"gate": 512, "up": 512, "act": 512, "raw": 2048, "out": 2048, "prob": 1}


def rw(p):
    a = array("I")
    with open(os.path.join(HERE, p), "rb") as fh:
        a.frombytes(fh.read())
    return a


def main():
    host, dev = rw("host_out.f32"), rw("dev_out.f32")
    cases = []
    for line in open(os.path.join(HERE, "cases.tsv")):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        cases.append((int(f[0]), f[1], int(f[7])))
    hist = {}
    for idx, label, dst in cases:
        for fname in ("raw", "out"):
            off = dst + FIELD_OFF[fname]
            for i in range(FIELD_LEN[fname]):
                a, b = host[off + i], dev[off + i]
                if a == b:
                    continue
                ah, bh = a >> 16, b >> 16
                step = (bh | 0) - (ah | 0)
                fv = lambda h: array("f", [h << 16]).tolist()[0]
                hist.setdefault((fname, step > 0), 0)
                hist[(fname, step > 0)] += 1
                print("DIFF case=%-3d label=%-14s field=%-4s idx=%-5d host_bf16=0x%04X dev_bf16=0x%04X "
                      "bitpattern_delta=%+d host_val=%.9g dev_val=%.9g"
                      % (idx, label, fname, i, ah, bh, step, fv(ah), fv(bh)))
    print("SUMMARY by field and sign: %s" % (", ".join("%s/%s=%d" % (f, ("dev_higher" if s else "dev_lower"), n)
                                                     for (f, s), n in sorted(hist.items()))))
    print("no timing taken")
    return 0


if __name__ == "__main__":
    sys.exit(main())
