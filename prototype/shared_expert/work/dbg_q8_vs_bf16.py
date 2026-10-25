#!/usr/bin/env python3
"""CPU-only diagnostic: how big is the Q8_0 round-trip on these weights, and does the pinned
`shared_raw` get reproduced by the bf16 weights but not by the Q8 round-trip?

Why this exists: route_discriminates() reports the honest candidate at relL2=0.91 / cos=0.785 against
the pinned eager shared_raw (work/oracle-derive.log). A Q8_0 round-trip on weights should cost well
under 1 %, so a 91 % distance cannot be explained by 8-bit weights. This separates the two possible
causes: (a) the Q8 pack/unpack is wrong, (b) the distance itself is computed against the wrong thing.

Run: /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python work/dbg_q8_vs_bf16.py
No device work, no lock, no timing.
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
H, I = oa.H, oa.I


def main():
    import torch
    torch.set_num_threads(8)
    import torch.nn.functional as F
    f0 = oa.pinned_files(DUMP, "layer0")
    x, _ = oa.load_bf16(f0["input"], (-1, H))
    gw, _ = oa.load_bf16(f0["gate"], (I, H))
    uw, _ = oa.load_bf16(f0["up"], (I, H))
    dw, _ = oa.load_bf16(f0["down"], (H, I))
    sg, _ = oa.load_bf16(f0["sg"], (1, H))
    ref_raw, _ = oa.load_bf16(f0["shared_raw"], (-1, H))
    ref_prob, _ = oa.load_bf16(f0["shared_gate_prob"], (-1, 1))

    for nm, w in (("gate", gw), ("up", uw), ("down", dw)):
        sc, q = oa.q8_quantize(w)
        deq = oa.q8_decode(sc, q)
        print("Q8ERR family=%-5s shape=%s orig_rms=%.6g deq_rms=%.6g relL2=%.6g maxabs=%.6g "
              "zero_scales=%d" % (nm, w.shape, float(np.sqrt((w * w).mean())),
                                  float(np.sqrt((deq * deq).mean())),
                                  float(np.linalg.norm(deq - w) / np.linalg.norm(w)),
                                  float(np.abs(deq - w).max()), int((sc == 0).sum())))

    def t(a):
        return torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16)

    def raw_bf16(xa, ga, ua, da):
        xt = torch.from_numpy(xa.astype(np.float32)).to(torch.bfloat16)
        g = F.linear(xt, torch.from_numpy(ga.astype(np.float32)).to(torch.bfloat16))
        u = F.linear(xt, torch.from_numpy(ua.astype(np.float32)).to(torch.bfloat16))
        act = F.silu(g) * u
        return F.linear(act, torch.from_numpy(da.astype(np.float32)).to(torch.bfloat16)
                        ).to(torch.float32).numpy()

    ref = ref_raw
    def dist(nm, got):
        d = got - ref
        print("RAWDIST %-22s maxabs=%.6g relL2=%.6g cos=%.9f norm_ratio=%.4f"
              % (nm, float(np.abs(d).max()), float(np.linalg.norm(d) / np.linalg.norm(ref)),
                 float((got * ref).sum() / (np.linalg.norm(got) * np.linalg.norm(ref))),
                 float(np.linalg.norm(got) / np.linalg.norm(ref))))

    dist("bf16_weights_exact", raw_bf16(x, gw, uw, dw))
    gsc, gq = oa.q8_quantize(gw); usc, uq = oa.q8_quantize(uw); dsc, dq = oa.q8_quantize(dw)
    dist("q8_weights", raw_bf16(x, oa.q8_decode(gsc, gq), oa.q8_decode(usc, uq),
                                oa.q8_decode(dsc, dq)))
    # and the arms' own route, dot by dot, on the Q8-decoded weights
    Wg, Wu, Wd = oa.q8_decode(gsc, gq), oa.q8_decode(usc, uq), oa.q8_decode(dsc, dq)
    F1 = oa.route_block(x[:1], (Wg, Wu, Wd, Wg, Wu, Wd), sg.reshape(H).astype(np.float32))
    d = F1["raw_full"][0:1] - ref[0:1]
    print("RAWDIST route_block_row0     maxabs=%.6g relL2=%.6g cos=%.9f"
          % (float(np.abs(d).max()), float(np.linalg.norm(d) / np.linalg.norm(ref[0:1])),
             float((F1["raw_full"][0:1] * ref[0:1]).sum()
                   / (np.linalg.norm(F1["raw_full"][0:1]) * np.linalg.norm(ref[0:1])))))
    dp = F1["prob"][0] - ref_prob[0]
    print("PROBDIST route_block_row0    abs=%.6g (ref_prob=%.6g ours=%.6g)"
          % (float(abs(dp)), float(ref_prob[0]), float(F1["prob"][0])))
    print("no timing taken")
    return 0


if __name__ == "__main__":
    sys.exit(main())
