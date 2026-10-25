#!/usr/bin/env python3
"""floor_probe_exp_np.py — the other half of the exp floor probe.

`work/floor_probe_exp.c` proved C `expf` equals double-exp-rounded-to-f32 on every bf16 input. This
script closes the loop against the tool the oracle is written in: numpy's float32 `exp`, `reciprocal`
and the bf16 store. If numpy and glibc disagree by 1 ulp anywhere on the grid, the oracle's `prob`
field is not an exactness claim and the gate would be judging a measured width, so the disagreement
count has to be a number and not a hope.

The distances are measured with the repo's one ruler (`tools/ulp_policy.py`), not a local ulp diff,
and the SiLU is derived from the same table: silu(t) = t * sigmoid(t), so the population is the whole
bf16 grid again, which is exactly the set of values the kernel can hand to the transcendental in the
bf16 route (the f32-act route feeds it raw f32 and is therefore NOT covered by this probe — that is
why the f32-act field is declared banded rather than exact).

    work/floor_probe_exp --emit work/c_exp_table.tsv
    <venv python> work/floor_probe_exp_np.py work/c_exp_table.tsv
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools"))
import ulp_policy as UP                                      # noqa: E402  the one ruler (D-17)


def bf16_rt(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even to bf16, kept widened in f32 (the same convention the C side uses)."""
    b = np.asarray(x, dtype=np.float32).view(np.uint32)
    r = ((b >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((b + r) & np.uint32(0xFFFF0000)).view(np.float32)


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    pat, cexp, csig, csig_bf16 = [], [], [], []
    with open(argv[1]) as fh:
        head = fh.readline()
        assert head.startswith("pattern"), "table header missing"
        for line in fh:
            f = line.split("\t")
            if len(f) != 4:
                raise SystemExit(f"malformed row {line!r}")
            pat.append(int(f[0]))  # the C emitter prints the pattern in decimal
            cexp.append(int(f[1], 16))
            csig.append(int(f[2], 16))
            csig_bf16.append(int(f[3], 16))
    n = len(pat)
    if n == 0:
        raise SystemExit("empty table: nothing was compared")
    pat = np.array(pat, dtype=np.uint32)
    t = (pat << np.uint32(16)).view(np.float32)
    np.seterr(over="ignore")   # both sides saturate to +inf for t < -88; the sigmoid is 0 either way
    np_exp = np.exp(-t).astype(np.float32)   # fp32 saturates to inf at |t| > ~88, and so does expf
    np_sig = (np.float32(1.0) / (np.float32(1.0) + np_exp)).astype(np.float32)
    np_sig_bf16 = bf16_rt(np_sig)

    cexp_bits = np.array(cexp, dtype=np.uint32)
    csig_bits = np.array(csig, dtype=np.uint32)
    csig_b = np.array(csig_bf16, dtype=np.uint32)

    d_exp = int((np_exp.view(np.uint32) != cexp_bits).sum())
    d_sig = int((np_sig.view(np.uint32) != csig_bits).sum())
    d_bf16 = int(((np_sig_bf16.view(np.uint32) >> np.uint32(16)) != csig_b).sum())
    # distances in representable steps, from the repo's ruler — not a local ulp diff
    r32 = UP.Ruler("f32")
    # the ruler's census consumes (index, a, b) triples, not pairs
    cen_sig = r32.census([(i, int(a), int(b)) for i, (a, b) in
                          enumerate(zip(csig_bits.tolist(), np_sig.view(np.uint32).tolist()))])
    c_silu = (t * csig_bits.view(np.uint32).view(np.float32)).astype(np.float32)
    n_silu = (t * np_sig).astype(np.float32)
    cen_silu = r32.census([(i, int(a), int(b)) for i, (a, b) in
                           enumerate(zip(c_silu.view(np.uint32).tolist(),
                                         n_silu.view(np.uint32).tolist()))])
    print(cen_sig.line("sigmoid(f32)", extra="probe=c-vs-numpy-over-every-bf16-input")
          .replace("CENSUS", "CENSUSEXP"))
    print(cen_silu.line("silu(f32)", extra="probe=c-vs-numpy-over-every-bf16-input")
          .replace("CENSUS", "CENSUSEXP"))
    print("FLOOREXP_NP max_ulp_f32=%d (the widest f32 divergence the two libms produce on any bf16 "
          "input; the band on an f32-carrying transcendental field is this number, derived here and "
          "not chosen)" % max(cen_sig.max_steps, cen_silu.max_steps))
    print(f"FLOOREXP_NP rows={n} exp_diff_f32={d_exp} sigmoid_diff_f32={d_sig} "
          f"sigmoid_diff_after_bf16_store={d_bf16}")
    print("FLOOREXP_NP verdict=" + ("stored-prob-field-exact: the bf16 store is coarser than every "
                                   "libm-vs-numpy disagreement"
                                   if d_bf16 == 0 else
                                   "NOT exact: the two libms disagree in the stored bf16, so the "
                                   "prob band is a measured width and not an exactness claim"))
    if d_bf16:
        bad = np.where(((np_sig_bf16.view(np.uint32) >> np.uint32(16)) != csig_b))[0][:10]
        for i in bad:
            c_val = ((csig_b[i] << np.uint32(16)).astype(np.uint32)).view(np.float32)
            print(f"  t={float(t[i]):.9g} numpy={float(np_sig_bf16[i]):.9g} c={float(c_val):.9g}")
    print("no timing taken")
    return 0 if d_bf16 == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
