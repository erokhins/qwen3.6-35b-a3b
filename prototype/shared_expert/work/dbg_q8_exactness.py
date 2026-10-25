#!/usr/bin/env python3
"""CPU-only: what does a CONFORMANT Q8_0 pack of these weights look like, before I change anything?

Three numbers the fix has to be measured against, none of them typed:
  * clamp rate at amax/127 vs the lane's current amax/512 (expected order 1 element per 32-block,
    because the block max is what maps to +-127; ties and f16 scale rounding can add a few);
  * whether f16(scale) * int8(q) is EXACT in f32 -- measured by comparing the f32 product against the
    same product in f64, element by element, over all three families;
  * the significand-bit width the decoded values actually need (the header's "<=11+7 bits" claim).

    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python work/dbg_q8_exactness.py
No device, no lock, no timing.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(LANE))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, LANE)
import oracle_arm as oa          # noqa: E402

DUMP = os.path.join(REPO, "data/dumps/2026-10-26")
H, I, Q8 = oa.H, oa.I, oa.Q8_ELTS


def quant(w, div):
    rows, cols = w.shape
    blk = w.reshape(rows, cols // Q8, Q8)
    amax = np.abs(blk).max(axis=2)
    d32 = (amax / np.float32(div)).astype(np.float32)
    dh = d32.astype(np.float16)
    d = dh.astype(np.float32)
    inv = np.where(d > 0, np.float32(1.0) / np.where(d > 0, d, np.float32(1.0)), np.float32(0.0))
    prod = blk * inv[:, :, None]
    q = np.where(prod >= 0, np.floor(prod + 0.5), np.ceil(prod - 0.5))
    q = np.clip(q, -127.0, 127.0)
    return dh, q.astype(np.int8).reshape(rows, cols), amax


def bits_needed(vals):
    """Max number of significand bits needed to write |v| exactly as 1.m * 2^e (m plus the implicit 1)."""
    b = np.asarray(vals, dtype=np.float32).view(np.uint32)
    exp = (b >> np.uint32(23)) & np.uint32(0xFF)
    mant = b & np.uint32(0x7FFFFF)
    frac = mant.astype(np.int64)
    frac = np.where(frac == 0, np.int64(0), 24 - np.floor(np.log2(np.maximum(frac, 1))).astype(np.int64))
    normal = (exp != 0) & (exp != 255)
    return int(np.where(normal, frac + 1, 0).max())


def main():
    f0 = oa.pinned_files(DUMP, "layer0")
    fams = (("gate", oa.load_bf16(f0["gate"], (I, H))[0]),
            ("up", oa.load_bf16(f0["up"], (I, H))[0]),
            ("down", oa.load_bf16(f0["down"], (H, I))[0]))
    for nm, w in fams:
        for div in (127, 512):
            dh, q, amax = quant(w, div)
            d = dh.astype(np.float32)
            deq = (d[:, :, None] * q.reshape(w.shape[0], -1, Q8).astype(np.float32)).reshape(w.shape)
            exact64 = (d.astype(np.float64)[:, :, None]
                       * q.reshape(w.shape[0], -1, Q8).astype(np.float64)).reshape(w.shape)
            inexact = int((deq.astype(np.float64) != exact64).sum())
            clamped = int((np.abs(q.astype(np.int32)) == 127).sum())
            is_block_max = int((np.abs(w.reshape(w.shape[0], -1, Q8)) == amax[:, :, None]).sum())
            rel = float(np.linalg.norm(deq - w) / np.linalg.norm(w))
            print("Q8PACK family=%-5s divisor=%-4d clamp_eq127=%d/%d (%.4f%%) elements_equal_block_max=%d "
                  "(%.4f%%, 1/32=%.4f%%) relL2_vs_dump=%.7g f32_vs_f64_inexact=%d max_sig_bits=%d"
                  % (nm, div, clamped, q.size, 100.0 * clamped / q.size, is_block_max,
                     100.0 * is_block_max / q.size, 100.0 / Q8, rel, inexact,
                     bits_needed(deq)))
    print("no timing taken")
    return 0


if __name__ == "__main__":
    sys.exit(main())
