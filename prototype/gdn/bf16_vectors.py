#!/usr/bin/env python3
"""bf16_vectors.py — derive the self-test's bf16 expectation table from torch, not from arithmetic.

    <venv>/bin/python prototype/gdn/bf16_vectors.py [--sample N]

Why this file exists: the header `gdn_kernels.h` states that `gdn_bf16_of_f32` implements
round-to-nearest-even, "the rule torch's `.to(bf16)` uses". A hand-computed expectation table for
that rule is how the previous lane's self-test shipped four wrong rows (65504, 1e-8, a NaN
round-trip and a sign-flipped solve residual) and never noticed, because nothing ran it. Here the
expected bit images come from torch itself, so the self-test compares the C rule against the rule
the oracle's own casts use, which is the claim actually being made.

Also prints:
  * a random sweep of f32 bit patterns, C rule vs torch, which is the width claim being tested
    rather than asserted (`pitfalls/30` §byte-identity width claims);
  * the round-trip identity over all 65,536 bf16 images, split into finite and non-finite, because
    NaN quieting is NOT identity and the self-test must not pretend it is.

CPU only; no device, no lock, no timer, no ms on any printed line.
"""
import argparse
import struct
import sys

import torch


def c_rule(f: float) -> int:
    """`gdn_bf16_of_f32`, transcribed in python, kept line-for-line with the C."""
    u = struct.unpack("<I", struct.pack("<f", f))[0]
    if ((u >> 23) & 0xFF) == 0xFF and (u & 0x7FFFFF) != 0:
        return (u >> 16) | 0x40
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
    return (u >> 16) & 0xFFFF


def torch_rule(f: float) -> int:
    return torch.tensor(f).to(torch.bfloat16).view(torch.uint16).item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=20261026)
    a = ap.parse_args()
    vals = [0.0, 1.0, -1.0, 0.5, 2.0, 65504.0, 1.0 / 3.0, -1.0 / 3.0, 1e-8, -2.5,
            3.4028235e38, 1e-45, 0.00390625, 127.5, -0.0]
    bad = 0
    for v in vals:
        t, m = torch_rule(v), c_rule(v)
        same = t == m
        bad += 0 if same else 1
        print("BF16VEC f=%-16r want=0x%04x c=0x%04x agree=%d" % (v, t, m, 1 if same else 0))
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    raw = torch.randint(0, 1 << 32, (a.sample,), generator=g, dtype=torch.int64).to(torch.int32)
    f32 = raw.view(torch.float32)
    # Vectorised C rule: same three expressions as c_rule(), applied to the whole sample. Doing it
    # per element through torch.tensor() would be slower than the sweep is worth, and the rule is
    # pure bit arithmetic, so the vector form and the scalar form are the same function.
    u = raw.to(torch.int64) & 0xFFFFFFFF
    exp_all = ((u >> 23) & 0xFF) == 0xFF
    frac = (u & 0x7FFFFF) != 0
    nan_like = exp_all & frac
    rnd = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    cvec = torch.where(nan_like, (u >> 16) | 0x40, rnd) & 0xFFFF
    tvec = f32.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF
    fin = torch.isfinite(f32)
    mism = int(((cvec != tvec) & fin).sum().item())
    for i in torch.nonzero((cvec != tvec) & fin)[:3].flatten().tolist():
        print("BF16MISMATCH f=%r torch=0x%04x c=0x%04x"
              % (f32[i].item(), int(tvec[i].item()), int(cvec[i].item())))
    nonid, nonid_fin = [], 0
    for i in range(1 << 16):
        f = struct.unpack("<f", struct.pack("<I", i << 16))[0]
        if c_rule(f) != i:
            nonid.append(i)
            if (i >> 7) & 0xFF != 0xFF:
                nonid_fin += 1
    print("BF16SWEEP samples=%d mismatches=%d table=%d table_bad=%d"
          % (a.sample, mism, len(vals), bad))
    print("BF16ROUNDTRIP patterns=%d non_identity=%d of_which_finite=%d "
          "note=non-finite non-identity is NaN quieting, not a defect"
          % (1 << 16, len(nonid), nonid_fin))
    if mism or bad or nonid_fin:
        print("RESULT FAIL bf16 rule disagrees with torch")
        return 1
    print("RESULT PASS bf16 rule agrees with torch over the table and the sample")
    return 0


if __name__ == "__main__":
    sys.exit(main())
