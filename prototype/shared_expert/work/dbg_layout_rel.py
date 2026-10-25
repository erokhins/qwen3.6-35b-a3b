#!/usr/bin/env python3
"""CPU-only diagnostic for the layout discriminator (question (a)).

route_discriminates() in oracle_arm.py currently cannot pass: its wrong-candidate minimum always
contains pair_swap, whose row map is IDENTICAL to gate_first by construction (layout_rows returns
(j, I+j) for both), so worst_wrong == honest for any build. And maxabs alone does not separate the
visible candidates (honest 0.96875 vs up_first 0.961426 in work/oracle-derive.log).

This script does not change any verdict. It prints, for every candidate reading, the three distances
the derive test could use -- maxabs, relative L2, cos -- plus two candidates the current test does not
model at all: the one-row-early control, and a pair_swap that actually swaps which half gets silu
(which is what the pair_swap case in the fixtures does, and the only way a route distance can see it).

Run: /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python work/dbg_layout_rel.py
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
import oracle_arm as oa          # noqa: E402  (needs the sys.path lines above)

DUMP = os.path.join(REPO, "data/dumps/2026-10-26")
H, I, TWO_I = oa.H, oa.I, oa.TWO_I


def main():
    import torch
    f0 = oa.pinned_files(DUMP, "layer0")
    gw, _ = oa.load_bf16(f0["gate"], (I, H))
    uw, _ = oa.load_bf16(f0["up"], (I, H))
    dw, _ = oa.load_bf16(f0["down"], (H, I))
    x0, _ = oa.load_bf16(f0["input"], (-1, H))
    ref_raw, _ = oa.load_bf16(f0["shared_raw"], (-1, H))
    gsc, gq = oa.q8_quantize(gw)
    usc, uq = oa.q8_quantize(uw)
    dsc, dq = oa.q8_quantize(dw)
    Wg, Wu, Wd = oa.q8_decode(gsc, gq), oa.q8_decode(usc, uq), oa.q8_decode(dsc, dq)
    fused = np.concatenate([Wg, Wu], axis=0)
    xt = torch.from_numpy(x0.astype(np.float32))
    ref = torch.from_numpy(ref_raw)

    def bf(a):
        return a.to(torch.bfloat16).to(torch.float32)

    ft = torch.from_numpy(fused)
    dt = torch.from_numpy(Wd.T)

    def raw_of(grows, urows, silu_on_up=False):
        g = bf(xt @ ft[torch.from_numpy(grows)].T)
        u = bf(xt @ ft[torch.from_numpy(urows)].T)
        a, b = (u, g) if silu_on_up else (g, u)
        act = bf(bf(a * torch.sigmoid(a)) * b)
        return bf(act @ dt)

    rows = {lay: np.array([oa.layout_rows(lay, j) for j in range(I)]) for lay in oa.LAYOUTS}
    lim = TWO_I - 1
    cases = []
    for lay in oa.LAYOUTS:
        cases.append((lay, rows[lay][:, 0], rows[lay][:, 1], False))
    cases.append(("row_shift1_control",
                  np.minimum(rows["gate_first"][:, 0] + 1, lim),
                  np.minimum(rows["gate_first"][:, 1] + 1, lim), False))
    cases.append(("col_shift1_control",
                  np.maximum(rows["gate_first"][:, 0] - 1, 0),
                  np.maximum(rows["gate_first"][:, 1] - 1, 0), False))
    cases.append(("pair_swap_silu_swapped", rows["pair_swap"][:, 0], rows["pair_swap"][:, 1], True))
    cases.append(("gate_last_control", rows["up_first"][:, 0], rows["up_first"][:, 1], False))

    refn = float(ref.norm())
    print("REF rows=%d cols=%d norm=%.6g absmax=%.6g (the scale maxabs has to be read against)"
          % (ref.shape[0], ref.shape[1], refn, float(ref.abs().max())))
    out = {}
    for name, gr, ur, swap in cases:
        raw = raw_of(gr, ur, swap)
        d = raw - ref
        maxabs = float(d.abs().max())
        rel2 = float(d.norm() / ref.norm())
        cos = float((raw * ref).sum() / (raw.norm() * ref.norm()))
        out[name] = (maxabs, rel2, cos)
        print("CAND %-24s maxabs=%.6g relL2=%.6g cos=%.9f  (rowmap[0]=(%d,%d) rowmap[-1]=(%d,%d))"
              % (name, maxabs, rel2, cos, gr[0], ur[0], gr[-1], ur[-1]))
    honest = out["gate_first"]
    print("RATIO honest_vs_up_first   maxabs=%.3fx relL2=%.3fx cos_ratio=%.3fx"
          % (out["up_first"][0] / honest[0], out["up_first"][1] / honest[1],
             honest[2] / out["up_first"][2]))
    print("RATIO honest_vs_interleave maxabs=%.3fx relL2=%.3fx cos_ratio=%.3fx"
          % (out["row_interleaved"][0] / honest[0], out["row_interleaved"][1] / honest[1],
             honest[2] / out["row_interleaved"][2]))
    vis_wrong = [v for k, v in out.items() if k not in ("gate_first", "pair_swap")]
    print("VISIBLE_WRONG (pair_swap excluded as blind-by-construction in a row-map-only measure):")
    for k in sorted(out):
        if k in ("gate_first", "pair_swap"):
            continue
        print("  %-24s honest_worse_by maxabs=%.3fx relL2=%.3fx cos=%.3fx  honest_strictly_better=%s/%s/%s"
              % (k, out[k][0] / honest[0], out[k][1] / honest[1], honest[2] / out[k][2],
                 out[k][0] > honest[0], out[k][1] > honest[1], out[k][2] < honest[2]))
    print("BLINDNESS pair_swap row rows_vs_honest maxabs=%.9g relL2=%.9g cos_diff=%.3g "
          "(exactly 0 => a row-map-only distance can never see pair_swap)"
          % (abs(out["pair_swap"][0] - honest[0]), abs(out["pair_swap"][1] - honest[1]),
             abs(out["pair_swap"][2] - honest[2])))
    print("PAIRSWAP_VISIBLE_WITH_SILU_SWAP relL2_ratio_vs_honest=%.3fx (wrong spelling must be worse)"
          % (out["pair_swap_silu_swapped"][1] / honest[1]))
    print("no timing taken")
    return 0


if __name__ == "__main__":
    sys.exit(main())
