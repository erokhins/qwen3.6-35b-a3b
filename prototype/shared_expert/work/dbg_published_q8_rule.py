#!/usr/bin/env python3
"""CPU-only, file-anchored: which scale rule do the published Q8_0 bytes actually use?

Why this exists. oracle_arm.py's q8_quantize() writes `scale = amax/512` and calls the result Q8_0.
Both in-repo references say the rule is `d = amax/127`: tools/gguf_reader.py §_qref_q8_0 (line ~373,
`d = amax / 127.0`) and tools/moe_stats.py §q8_0_proxy (`d = _f16(amax / 127.0)`), each stated as a
mirror of ggml's quantize_row_q8_0_ref. A /512 packer with the usual [-127,127] clamp clips every
element above 127/512 = 24.8 % of its block max, so it is not a finer Q8_0, it is a damaged one.

This does not argue from memory: it decodes the PUBLISHED blk.0.ffn_gate_shexp.weight (which the lane's
own WIDTH line reports as dtype=Q8_0) with tools/gguf_reader.dequant, and compares against the pinned
HF bf16 dump of the same tensor. Whichever rule reproduces the dump is the rule the artifact uses.

    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python work/dbg_published_q8_rule.py
Reads the published file (selftest-token gated) and the pinned dumps. No device, no lock, no timing.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(LANE))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, LANE)
import audit_published_quant as apq    # noqa: E402
import gguf_reader as gr               # noqa: E402
import oracle_arm as oa                # noqa: E402

GGUF = os.path.join(REPO, "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DUMP = os.path.join(REPO, "data/dumps/2026-10-26")
H, I = oa.H, oa.I


def rel(a, b):
    d = a - b
    return (float(np.abs(d).max()), float(np.linalg.norm(d) / np.linalg.norm(b)))


def main():
    f0 = oa.pinned_files(DUMP, "layer0")
    gw, _ = oa.load_bf16(f0["gate"], (I, H))
    gr.grant_selftest_token(apq.SELFTEST_TOKEN)
    gg = gr.Gguf(GGUF)
    t = gg.by_name["blk.0.ffn_gate_shexp.weight"]
    print("PUBLISHED tensor=%s type=%s dims_gguf=%s shape_hf=%s bytes=%d n_elts=%d"
          % (t.name, t.type_name, t.dims, t.shape_hf, t.row, t.n_elts))
    mm = gg.open_data(apq.SELFTEST_TOKEN, why="shared_expert lane: which Q8_0 scale rule the file uses")
    buf = bytes(mm[gg.data_start + t.offset: gg.data_start + t.offset + t.row])
    gg.close()
    pub = np.asarray(gr.dequant(buf, "Q8_0", t.n_elts), dtype=np.float32).reshape(I, H)
    ma, rl = rel(pub, gw)
    print("RULECHECK published_q8_0_decode_vs_pinned_hf maxabs=%.6g relL2=%.6g  "
          "(ggml rule = f16(amax/127)*int8: this is what tools/gguf_reader.py decodes)" % (ma, rl))

    import torch
    import moe_stats as ms
    prox = ms.q8_0_proxy(torch.from_numpy(gw.astype(np.float32))).numpy()
    ma2, rl2 = rel(prox, gw)
    print("RULECHECK repo_proxy(amax/127)_vs_pinned_hf  maxabs=%.6g relL2=%.6g" % (ma2, rl2))
    ma3, rl3 = rel(pub, prox)
    print("RULECHECK published_decode_vs_repo_proxy     maxabs=%.6g relL2=%.6g "
          "(two independent implementations of the same rule)" % (ma3, rl3))

    sc, q = oa.q8_quantize(gw)
    lane = oa.q8_decode(sc, q)
    ma4, rl4 = rel(lane, gw)
    print("RULECHECK lane_pack(amax/512)_decoded_vs_pinned_hf maxabs=%.6g relL2=%.6g  "
          "<-- what the kernel is actually reading from work/slab.bin" % (ma4, rl4))
    qmax = np.abs(q.astype(np.int32)).max()
    clipped = int((np.abs(q.astype(np.int32)) >= 127).sum())
    print("LANEPACK scale_blocks=%d scale_rms=%.6g weight_rms=%.6g q_absmax=%d q_at_clamp=%d/%d "
          "(a correct Q8_0 hits the clamp only on the block max: 1 in 32)"
          % (sc.size, float(np.sqrt((oa.f16_to_f32(sc) ** 2).mean())),
             float(np.sqrt((gw ** 2).mean())), int(qmax), clipped, q.size))
    print("RATIO lane_relL2_over_correct_relL2 = %.1fx" % (rl4 / rl2 if rl2 else float("inf")))
    print("no timing taken")
    return 0


if __name__ == "__main__":
    sys.exit(main())
