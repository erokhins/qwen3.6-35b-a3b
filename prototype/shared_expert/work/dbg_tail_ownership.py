#!/usr/bin/env python3
"""Who owns the last column of `raw` and of `out` — sentinel test plus an arithmetic prediction.

The defect being diagnosed: every device-vs-host (and device-vs-oracle) divergent word sits at
index 2047, the last column of `raw` and of `out`, and the count of them moves between identical
device runs (73 -> 56 -> 48). A moving count is not a rounding rule. This script does not speculate;
it asks three questions of the buffers the device gate already wrote:

  Q1 SENTINEL. `shared_expert.hip` pre-fills the device output with the poison NaN (SE_NAN_BITS)
     before launching, so an unwritten slot still reads 0x7FC00000. Does index 2047 hold the
     sentinel (never written), or a value (written)?
  Q2 IDENTITY. Is the device value at `raw` 2047 the value the DOWN stage computed (i.e. the host
     value, the only legitimate writer of that word), or something else?
  PREDICTION. If `se_k_out` runs its out-channel body for the stage-3 SENTINEL item (the item whose
     `base` is -1, which belongs to the prob stage), then its index is h = -1 + threadIdx.x and it
       reads  out[dst + SE_OFF_RAW - 1]  == act[511]            (one word before `raw`)
       writes out[dst + SE_OFF_OUT - 1]  == raw[2047]           (one word before `out`)
     so the word at raw 2047 must equal bf16_rt(prob * act[511]) EXACTLY. That is a bit-exact
     prediction of the mechanism, not a vibe. Cases whose sgate form is per-channel take a different
     branch (they read x[-1] and sg[-1], outside the tensors) and are reported as not-predictable.
  Q3 RACE SIGNATURE. `out` 2047 is written by the legitimate last block, but it READS raw[2047] --
     the word block 0 is clobbering in the same launch. So out[2047] must match either the true raw
     (read before the clobber) or the clobbered raw (read after it). Which one it matches, per case,
     is the race, and the number of "after" cases is what moves from run to run.

    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python work/dbg_tail_ownership.py
Reads work/host_out.f32, work/dev_out.f32, work/cases.tsv. No device, no lock, no timing.
"""
import os
import struct
import sys
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
OFF_RAW, OFF_OUT, OFF_ACT, OFF_PROB = 1536, 3584, 1024, 5632
LEN_RAW, LEN_OUT, LEN_ACT = 2048, 2048, 512
SENTINEL = 0x7FC00000


def rw(name):
    a = array("I")
    with open(os.path.join(HERE, name), "rb") as fh:
        a.frombytes(fh.read())
    return a


def bits_of_f32(x):
    return struct.unpack("I", struct.pack("f", x))[0]


def f32_of_bits(b):
    return struct.unpack("f", struct.pack("I", b & 0xFFFFFFFF))[0]


def bf16_rt_bits(b):
    """f32 bit image -> bf16 bit image, round-to-nearest-even: the kernel's own rule."""
    if (b & 0x7F800000) == 0x7F800000 and (b & 0x007FFFFF):
        return (b >> 16) | 0x0040
    return (b + 0x7FFF + ((b >> 16) & 1)) >> 16


def bf16_rt_f32(x):
    """The product of two bf16-representable f32s is exact in a double (<=16 significand bits),
    so packing the double as f32 rounds once, which is what the f32 multiply does. Returned as the
    FULL f32 bit image the arm stores (bf16 image << 16), because that is what the buffers hold."""
    return bf16_rt_bits(bits_of_f32(x)) << 16


def main():
    host, dev = rw("host_out.f32"), rw("dev_out.f32")
    cases = []
    for line in open(os.path.join(HERE, "cases.tsv")):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        cases.append({"idx": int(f[0]), "label": f[1], "sgate": f[4], "dst": int(f[7])})
    print("CASES parsed=%d (cases.tsv lines, must equal the printed per-case lines below)" % len(cases))

    tally = {}
    def bump(key):
        tally[key] = tally.get(key, 0) + 1

    n_lines = 0
    for c in cases:
        dst = c["dst"]
        rt_raw, rt_out = dst + OFF_RAW + LEN_RAW - 1, dst + OFF_OUT + LEN_OUT - 1
        act_last = dev[dst + OFF_ACT + LEN_ACT - 1]          # act[511] == raw[-1]
        prob = dev[dst + OFF_PROB]
        dv, hv = dev[rt_raw], host[rt_raw]
        scalar = c["sgate"] == "scalar"
        pred = bf16_rt_f32(f32_of_bits(prob) * f32_of_bits(act_last)) if scalar else None

        if dv == SENTINEL:
            verdict = "sentinel-never-written"
        elif dv == hv:
            verdict = "written-by-down-stage(matches host)"
        elif scalar and dv == pred:
            verdict = "written-by-out-stage-h-minus-1(matches prediction EXACTLY)"
        else:
            verdict = "written-but-unexplained"
        bump("raw2047:" + verdict)

        # --- out[2047]: did its read of raw[2047] land before or after the clobber?
        if scalar:
            clean = bf16_rt_f32(f32_of_bits(prob) * f32_of_bits(hv))
            raced = bf16_rt_f32(f32_of_bits(prob) * f32_of_bits(dv))
            ov = dev[rt_out]
            if ov == host[rt_out]:
                o = "out2047=read-clean-raw"
            elif ov == raced and raced != clean:
                o = "out2047=read-CLOBBERED-raw(race loser)"
            elif ov == clean:
                o = "out2047=values-coincide"
            else:
                o = "out2047=unexplained"
            bump(o)
        else:
            bump("out2047:perchan-not-predictable")

        print("TAIL case=%-3d label=%-14s sgate=%-6s dev_raw2047=0x%08X host_raw2047=0x%08X "
              "pred(h=-1)=0x%s sentinel=0x%08X prob=0x%08X act511=0x%08X verdict=%s"
              % (c["idx"], c["label"], c["sgate"], dv, hv,
                 ("%08X" % (pred & 0xFFFFFFFF)) if pred is not None else "   n/a  ",
                 SENTINEL, prob, act_last, verdict))
        n_lines += 1

    print("PARSE witness: printed=%d parsed=%d %s" % (n_lines, len(cases),
                                                      "OK" if n_lines == len(cases) else "MISMATCH"))
    if n_lines != len(cases):
        print("FAIL: per-case lines and case records disagree; a tally from this run is worthless")
        return 1
    for k in sorted(tally):
        print("TALLY %-58s %d" % (k, tally[k]))
    print("no timing taken")
    return 0


if __name__ == "__main__":
    sys.exit(main())
