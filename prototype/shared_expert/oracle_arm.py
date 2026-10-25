#!/usr/bin/env python3
"""oracle_arm.py — the third side: numpy over the same declared route, plus the fixtures the arms read.

Run with the CPU venv (torch + safetensors + numpy live there):

    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python oracle_arm.py --mode derive
    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python oracle_arm.py --mode fixtures \
        --out work [--rows N]

`--mode derive` answers the questions this lane was told not to assume, from the files:
  * the geometry (config.json), cross-checked against the header's `#define`s by gate.py;
  * the width each tensor family actually gets in the shipping GGUF (via tools/gguf_reader.py, which
    requires the audit selftest token, so the audit marker is checked first and its absence is an
    error rather than a skip) — and the fact that the file holds `ffn_gate_shexp` and `ffn_up_shexp`
    as TWO tensors, which is the whole layout question in one line: the fused [2I, H] view the kernel
    reads is an assembly this project chooses, not something the artifact says;
  * which of the four candidate assemblies is correct, answered by construction plus a discriminator
    that says which candidates can be told apart and which cannot (see LAYOUT lines).

`--mode fixtures` writes slab.bin (Q8_0 bytes), inputs.bin (the expectation images), plan_pinned.tsv,
plan_synth.tsv and cases.tsv, then prints the populations every must-catch set is derived from.

The declared route, which is what "exactly equal" is a claim about:

    w      = f16(scale) * int8(q)            Q8_0 dequant; <= 11+7 significand bits, EXACT in f32
    gate_j = sum_k x_k * w[gate_j]_k        f32 accumulator, k ascending, one mul then one add
    up_j   = same, row j of the up half
    act_j  = bf16(bf16(silu(bf16(gate_j))) * bf16(up_j))          three roundings, then the product
    raw_h  = bf16(sum_k act_k * w[down_h]_k)                      same dot, over I
    t      = sum_k x_k * sg_k               sg is stored F32, so this dot is exact-input
    p      = bf16(sigmoid(bf16(t)))         the reference's order: sigmoid, then cast, then multiply
    out_h  = bf16(p * raw_h)
    CAST_AFTER (the wrong spelling, kept as an arm): out_h = bf16(sigmoid(bf16(t)) * raw_h)

Everything the reference itself rounds is rounded here too, in the same place. `shared_out_exact` in
the dump is the f64-route output stored as f32; it is reported as a distance, never as a target —
comparing a bf16-carrying output to it needs a band, and the band this lane uses comes from a floor
probe, not from what the build happens to produce.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))

H, I, TWO_I = 2048, 512, 1024
Q8_ELTS, Q8_BYTES = 32, 34
GATE_STRIDE = 64 * Q8_BYTES          # one gate/up row: 2048 columns = 64 whole Q8_0 blocks
DOWN_STRIDE = 16 * Q8_BYTES          # one down row: 512 columns = 16 whole Q8_0 blocks
CASE_FLOATS = 5633                   # gate|up|act|raw|out|prob, offsets in the header
OFF = {"gate": 0, "up": 512, "act": 1024, "raw": 1536, "out": 3584, "prob": 5632}
NAN_BITS = 0x7FC00000
MODES = ["correct", "layout_up_first", "layout_row_inter", "layout_pair_swap", "row_shift1",
         "f32_act", "tail_drop", "cols_half", "down_half", "wide_misread"]


# ---- the two half-precision formats, as bit operations -----------------------------------------
def bf16_rt(a):
    """Round-to-nearest-even to bf16, returned widened to f32 (the C helper's exact contract)."""
    a = np.asarray(a, dtype=np.float32)
    b = a.view(np.uint32)
    r = ((b >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    out = ((b + r) & np.uint32(0xFFFF0000)).view(np.float32).copy()
    return np.where(np.isnan(a), a, out)


def bf16_bits(a):
    return (np.asarray(a, dtype=np.float32).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def f16_to_f32(bits):
    """IEEE binary16 -> f32, exact including subnormals. The C side does it on the integer image; this
    is the same function by construction.

    `.view(np.float16)`, never `.astype(np.float16)`: the latter converts the *integer* 16384 to 16384.0
    instead of reinterpreting the bit pattern, which turns every Q8_0 scale into garbage and made the
    numpy route disagree with the C route on 5632 words of 5633 (pitfalls/30 §"`.view` bitcasts,
    `.to` casts").
    """
    return np.asarray(bits, dtype=np.uint16).view(np.float16).astype(np.float32)


def f16_bits_as_bf16(bits):
    """The wide-misread control: the same 16-bit image read as a bf16 instead of an f16."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


# ---- Q8_0: the quantizer that writes the slab, and the decoder that reads it back ----------------
def q8_quantize(w):
    """Q8_0 per 32-column block, the llama.cpp rule: scale = amax/512, round half away from zero,
    clamp to [-128, 127]. `w` is [rows, cols] f32 (already bf16-exact values). Returns
    (scales f16 bits, qs int8) with shapes [rows, cols/32] and [rows, cols]."""
    rows, cols = w.shape
    assert cols % Q8_ELTS == 0, "a ragged row would need a boundary path this kernel does not have"
    blk = w.reshape(rows, cols // Q8_ELTS, Q8_ELTS)
    amax = np.abs(blk).max(axis=2)
    scale = np.where(amax > 0, amax / np.float32(512.0), np.float32(0.0)).astype(np.float32)
    inv = np.where(amax > 0, np.float32(512.0) / np.where(amax > 0, amax, np.float32(1.0)),
                   np.float32(0.0)).astype(np.float32)
    prod = blk * inv[:, :, None]
    q = np.where(prod >= 0, np.floor(prod + 0.5), np.ceil(prod - 0.5))   # half away from zero
    q = np.clip(q, -128.0, 127.0).astype(np.int8).reshape(rows, cols)
    return np.asarray(scale, dtype=np.float16).view(np.uint16), q


def q8_pack(scale_bits, q):
    """[rows, cols/32, 2 bytes scale] + [rows, cols, 1 byte] interleaved per row, so row r's blocks
    are contiguous: that is the byte layout `se_q8_row_ptr` walks."""
    rows, nblk = scale_bits.shape
    out = np.zeros((rows, nblk * Q8_BYTES), dtype=np.uint8)
    for b in range(nblk):
        # the two scale bytes in little-endian order, which is how `se_q8_value` reassembles them
        out[:, b * Q8_BYTES:b * Q8_BYTES + 2] = np.ascontiguousarray(
            scale_bits[:, b:b + 1]).view(np.uint8)
        out[:, b * Q8_BYTES + 2:(b + 1) * Q8_BYTES] = np.ascontiguousarray(
            q[:, b * Q8_ELTS:(b + 1) * Q8_ELTS]).view(np.uint8)
    return out


def q8_decode(scale_bits, q, mode="exact"):
    """Values the arms will see, decoded from the same arrays that were packed. mode:
       exact -> f16(scale) * int8(q)                       (exact in f32: 11 significand bits x 7)
       wide  -> the f16 bit IMAGE reinterpreted as a bf16   (the wrong read, `WIDE_MISREAD`)"""
    sc = f16_to_f32(scale_bits)
    qf = np.asarray(q, dtype=np.int8).astype(np.float32).reshape(sc.shape[0], sc.shape[1], Q8_ELTS)
    if mode == "wide":
        # exactly what `se_q8_value` does with wide_misread: the scale's f16 bit image read as a bf16,
        # then multiplied by the same int8 — a legal mis-transcription of `block_q8_0::d`, not noise.
        sc = f16_bits_as_bf16(scale_bits)
    return (sc[:, :, None] * qf).reshape(scale_bits.shape[0], -1)

def load_bf16(path, shape):
    """A dumped bf16 tensor as (f32 image widened, uint16 bit image).

    The widening is a 16-bit LEFT shift. `astype(uint32).view(float32)` — which is what I wrote first
    — puts the bf16 image in the LOW half and yields denormal garbage around 2e-41, and every check
    downstream then compared zero to zero and printed a perfect pass. The non-degeneracy assertion at
    the bottom of this function is what makes that failure mode loud instead of green
    (pitfalls/40 §"a check that can produce no output cannot fail").
    """
    raw = np.fromfile(path, dtype=np.uint16)
    shape = list(shape)
    if -1 in shape:                      # an open-ended axis means "whatever the file holds"
        shape[shape.index(-1)] = raw.size // (int(np.prod([s for s in shape if s > 0])) or 1)
    want = int(np.prod(shape))
    if raw.size != want:
        raise SystemExit("%s holds %d elements, the manifest says %s (%d)"
                         % (path, raw.size, shape, want))
    vals = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(shape).copy()
    finite = np.isfinite(vals)
    if not finite.all() or float(np.abs(vals).max()) < 1e-20:
        raise SystemExit("%s widened to degenerate values (max |v| = %g): the bf16 widening is "
                         "wrong, and every comparison against it would be vacuous"
                         % (path, float(np.abs(vals).max())))
    return vals, raw.reshape(shape).copy()


def pinned_files(dump_dir, block):
    p = lambda suffix: os.path.join(dump_dir, "%s.%s.bin" % (block, suffix))
    return {
        "input": p("input"),
        "gate": p("mlp.shared_expert.gate_proj.weight"),
        "up": p("mlp.shared_expert.up_proj.weight"),
        "down": p("mlp.shared_expert.down_proj.weight"),
        "sg": p("mlp.shared_expert_gate.weight"),
        "shared_raw": p("shared_raw"),
        "shared_gate_prob": p("shared_gate_prob"),
        "shared_out": p("shared_out"),
        "shared_out_exact": p("shared_out_exact"),
    }


def validate_against_reference(blocks, dump_dir, threads):
    """The oracle's own agreement with the pinned HF numbers, before any fixture is written.

    This half is torch on purpose: the dump was written by torch's own `F.linear`, whose CPU matmul
    reduces in its own blocked order, so bit-identity with the pinned bytes is only achievable by
    calling the same routine with the same thread count the manifest records. The Q8 route the arms
    implement is a different, fully specified loop and is checked against these numbers only as a
    distance (`DIST` lines), never as bit-identity.
    """
    import torch
    torch.set_num_threads(threads)
    import torch.nn.functional as F
    worst = {}
    for name in blocks:
        f = pinned_files(dump_dir, name)
        x, _ = load_bf16(f["input"], (-1, H))
        gw, _ = load_bf16(f["gate"], (I, H))
        uw, _ = load_bf16(f["up"], (I, H))
        dw, _ = load_bf16(f["down"], (H, I))
        sgw, _ = load_bf16(f["sg"], (1, H))
        refs = {k: load_bf16(f[k], (-1, 1) if k == "shared_gate_prob" else (-1, H))[0]
                for k in ("shared_raw", "shared_gate_prob", "shared_out")}
        ex = np.fromfile(f["shared_out_exact"], dtype=np.float32).reshape(-1, H)
        if ex.shape != refs["shared_raw"].shape:
            raise SystemExit("shared_out_exact is %s, shared_raw is %s"
                             % (ex.shape, refs["shared_raw"].shape))

        def t(a):
            return torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16)
        xt, gwt, uwt, dwt, sgt = t(x), t(gw), t(uw), t(dw), t(sgw)
        gate = F.linear(xt, gwt)
        up = F.linear(xt, uwt)
        act = F.silu(gate) * up
        raw = F.linear(act, dwt)
        prob = torch.sigmoid(F.linear(xt, sgt))
        out = prob * raw
        got = {"shared_raw": raw.to(torch.float32).numpy(),
               "shared_gate_prob": prob.to(torch.float32).numpy(),
               "shared_out": out.to(torch.float32).numpy()}
        same = {}
        for k, want in refs.items():
            g_bits = np.asarray(got[k]).view(np.uint32) >> np.uint32(16)
            w_bits = np.asarray(want).view(np.uint32) >> np.uint32(16)
            same[k] = int((g_bits == w_bits).sum()), int(w_bits.size)
        d = np.abs(got["shared_out"] - ex)
        rel = d / np.maximum(np.abs(ex), 1e-30)
        worst[name] = dict(same=same, maxabs=float(d.max()), meanabs=float(d.mean()),
                           maxrel=float(rel.max()),
                           cos=float((got["shared_out"] * ex).sum() /
                                     (np.linalg.norm(got["shared_out"]) * np.linalg.norm(ex))))
        print("PINNED block=%s shared_raw=%d/%d shared_gate_prob=%d/%d shared_out=%d/%d "
              "out_vs_exact maxabs=%.4e meanabs=%.4e maxrel=%.4e cos=%.9f"
              % (name, same["shared_raw"][0], same["shared_raw"][1],
                 same["shared_gate_prob"][0], same["shared_gate_prob"][1],
                 same["shared_out"][0], same["shared_out"][1], worst[name]["maxabs"],
                 worst[name]["meanabs"], worst[name]["maxrel"], worst[name]["cos"]))
        for k, (a, b) in same.items():
            if a != b:
                raise SystemExit("PINNED block=%s field=%s disagrees in %d of %d words: the oracle "
                                 "no longer reproduces the pinned reference and must not be used to "
                                 "judge anything" % (name, k, b - a, b))
    return worst


# ---- the layout question, answered from the file -----------------------------------------------
LAYOUTS = ["gate_first", "up_first", "row_interleaved", "pair_swap"]


def layout_rows(layout, j):
    if layout == "up_first":
        return I + j, j
    if layout == "row_interleaved":
        return 2 * j, 2 * j + 1
    return j, I + j


def fused_of(wg, wu):
    return np.concatenate([wg, wu], axis=0)


def layout_row_order(fused, gate_vals, up_vals, n):
    """Does our assembly + Q8_0 round-trip keep row j where HF put it? Answered per candidate row map
    by nearest row, not bit equality: the fused image is a Q8_0 round-trip, so it is never bit-equal to
    the bf16 original and an equality test here would grade the quantizer instead of the row order.

    What this can and cannot say, stated plainly because the difference matters: the [2I,H] view is OUR
    assembly (the published file stores two separate tensors, printed by the WIDTH lines above), and
    the HF bf16 weight shards are not on this box, only the dump's copies are. So this test cannot
    discover a row order in a file; it verifies that the quantizer and the assembly are row-preserving,
    which is the failure a transposed Q8 pack would produce. The question "does the kernel read the row
    map the slab was written with" is answered by route_discriminates below and by the injected
    wrong-candidate cases, which is where an actual disagreement would show up.
    """
    for lay in LAYOUTS:
        hits = 0
        for src, which in ((gate_vals, 0), (up_vals, 1)):
            # nearest by full row distance. A norm test would not do: dozens of rows of a weight matrix
            # share a norm, which is exactly how a proxy looks like a failure on the honest candidate.
            d = np.sqrt(((fused[None, :, :] - src[:n][:, None, :]) ** 2).sum(axis=2))
            near = np.argmin(d, axis=1)
            for j in range(n):
                if near[j] == layout_rows(lay, j)[which]:
                    hits += 1
        print("LAYOUTROW candidate=%-16s rows_nearest_own_hf_row=%d/%d (2I=%d, %d rows each side)"
              % (lay, hits, 2 * n, TWO_I, n))
    marg = []
    for src in (gate_vals[:n], up_vals[:n]):
        for row in src:
            d = np.sqrt(((fused - row) ** 2).sum(axis=1))
            o = np.sort(d)
            marg.append(float(o[1] / o[0]) if o[0] > 0 else float("inf"))
    print("LAYOUTROW margin_nearest_vs_runner_up min=%.3f median=%.3f (the nearest-row test is only a "
          "test if the winner is clearly nearer; this is by how much)"
          % (min(marg), float(np.median(marg))))
    return min(marg)


def route_discriminates(x, ref_raw, wg, wu, wd):
    """Which kernel row map reproduces the pinned HF eager `shared_raw`, measured against HF's own
    bytes. Distances, not bit-identity: this route runs on Q8_0-decoded weights, so it cannot be
    bit-equal to an eager bf16 reference, and pretending otherwise would be a band typed to fit what the
    build produces. The claim that carries the answer is the ORDER of the distances: the honest map must
    be far closer than every wrong one, including the one-row-early control, or a real layout bug and an
    off-by-one would be indistinguishable. Vectorised matmul on purpose, because this half measures a
    distance to HF and does not produce expectations; the declared association order is not what is under
    test here."""
    import torch
    xt = torch.from_numpy(x.astype(np.float32))

    def bf(a):
        return a.to(torch.bfloat16).to(torch.float32)

    fused = fused_of(wg, wu)          # the [2I,H] slab the kernel actually reads; a candidate row map
                                      # is an index into THIS, not into the two tensors

    def raw_of(grows, urows):
        g = bf(xt @ torch.from_numpy(fused[grows]).T)   # row-major rows: the dot is x @ W^T
        u = bf(xt @ torch.from_numpy(fused[urows]).T)
        act = bf(bf(g * torch.sigmoid(g)) * u)
        return bf(act @ torch.from_numpy(wd.T))

    rows = {lay: np.array([layout_rows(lay, j) for j in range(I)]) for lay in LAYOUTS}
    out = {}
    for lay in LAYOUTS:
        raw = raw_of(rows[lay][:, 0], rows[lay][:, 1])
        d = (raw - torch.from_numpy(ref_raw)).abs()
        cos = float((raw * torch.from_numpy(ref_raw)).sum()
                    / (raw.norm() * torch.from_numpy(ref_raw).norm()))
        out[lay] = (float(d.max()), cos)
        print("LAYOUTRT candidate=%-16s shared_raw_vs_pinned_eager maxabs=%.6g cos=%.9f"
              % (lay, float(d.max()), cos))
    lim = TWO_I - 1
    ctl = raw_of(np.minimum(rows["gate_first"][:, 0] + 1, lim),
                 np.minimum(rows["gate_first"][:, 1] + 1, lim))
    dctl = float((ctl - torch.from_numpy(ref_raw)).abs().max())
    honest = out["gate_first"][0]
    worst_wrong = min(v[0] for l, v in out.items() if l != "gate_first")
    print("LAYOUTRT candidate=row_shift1(control) shared_raw_vs_pinned_eager maxabs=%.6g"
          "  (control: reading one row early must not be the best reading either)" % dctl)
    print("LAYOUTRT winner=gate_first nearest_wrong_ratio=%.1fx control_also_worse=%s"
          % (worst_wrong / honest if honest else float("inf"), "yes" if dctl > honest else "no"))
    if worst_wrong <= honest:
        raise SystemExit("the row map the kernel implements is not distinguishable from a wrong one: "
                         "honest maxabs=%.6g, closest wrong=%.6g" % (honest, worst_wrong))
    if dctl <= honest:
        raise SystemExit("the one-row-early control is not worse than the honest map, so this "
                         "discriminator cannot tell a layout bug from an off-by-one")


# ---- the declared route, vectorised over rows but never across k -------------------------------
def dot_rows(x, w, cols):
    """x [T,K] f32 against w [rows,K] f32 -> [T,rows] f32, one output element at a time, k ascending,
    one multiply then one add. No BLAS: `A @ B` reduces in a blocked order with contraction and would
    not be the same number, which is the whole reason this loop is written out."""
    acc = np.zeros((x.shape[0], w.shape[0]), dtype=np.float32)
    for k in range(cols):
        acc = (acc + x[:, k] * w[:, k])          # f32 mul, then f32 add; never fused
    return acc


def sigmoid_f32(z):
    return (np.float32(1.0) / (np.float32(1.0) + np.exp(-z))).astype(np.float32)


def silu_f32(z):
    return (z * (np.float32(1.0) / (np.float32(1.0) + np.exp(-z)))).astype(np.float32)


def cols_for(kind, mode):
    """`kind` is 'gateup' (K=H) or 'down' (K=I); same reduction the C `se_cols_of` makes."""
    full = H if kind == "gateup" else I
    if mode == "tail_drop":
        return full - Q8_ELTS
    if mode == "cols_half":
        return full // 2
    return full


def layout_of_mode(mode):
    return {"layout_up_first": "up_first", "layout_row_inter": "row_interleaved",
            "layout_pair_swap": "pair_swap"}.get(mode, "gate_first")


def route_block(x, W, sg, mode="correct", cast="before", sgate="scalar"):
    """One block's forward pass in the declared route, for one row. `W` is the six decoded weight
    matrices (normal and wide-misread for each of gate/up/down) so a mode can pick its own reading of
    the same bytes. Returns the six fields the arms write, as [1,n] arrays plus prob as [1]."""
    Wg, Wu, Wd, Wgw, Wuw, Wdw = W
    if mode == "wide_misread":
        Wg, Wu, Wd = Wgw, Wuw, Wdw
    lay = layout_of_mode(mode)
    shift = 1 if mode == "row_shift1" else 0
    wide = 1 if mode == "wide_misread" else 0
    fused = np.concatenate([Wg, Wu], axis=0)                       # [2I, H], the assembly under test
    grows = np.array([layout_rows(lay, j)[0] for j in range(I)])
    urows = np.array([layout_rows(lay, j)[1] for j in range(I)])
    if shift:                                                       # the C side clamps, never wraps
        grows = np.where(grows + shift >= TWO_I, grows + shift - TWO_I, grows + shift)
        urows = np.where(urows + shift >= TWO_I, urows + shift - TWO_I, urows + shift)
    T = x.shape[0]
    c1 = cols_for("gateup", mode)
    gate = dot_rows(x, fused[grows], c1)
    up = dot_rows(x, fused[urows], c1)
    pair = (lay == "pair_swap")
    if mode == "f32_act":
        act = (silu_f32(up if pair else gate) * (gate if pair else up)).astype(np.float32)
    else:
        gz, uz = (bf16_rt(up), bf16_rt(gate)) if pair else (bf16_rt(gate), bf16_rt(up))
        # three roundings, in the order the C body performs them: bf16(gate), bf16(silu(that)),
        # bf16(up), then the product rounded again. Dropping the middle one was my first version's bug
        # and it showed as 125 of 512 act words disagreeing with the C arm.
        act = bf16_rt(bf16_rt(silu_f32(gz)) * uz)
    c2 = cols_for("down", mode)
    raw_full = bf16_rt(dot_rows(act, Wd, c2))
    t = bf16_rt(dot_rows(x, sg.reshape(1, H), H))                   # the [T,1] scalar the reference takes
    prob = sigmoid_f32(t[:, 0]) if cast == "after" else bf16_rt(sigmoid_f32(t[:, 0]))
    nan = np.float32(struct.unpack("<f", struct.pack("<I", NAN_BITS))[0])
    out = np.full((T, H), nan, dtype=np.float32)
    raw_h = raw_full.copy()
    nrow = H // 2 if mode == "down_half" else H
    raw_h[:, nrow:] = nan                                          # `down_half` leaves holes, on purpose
    if sgate == "perchan":
        # the alternative reading of `shared_expert_gate`: one dot product per output channel, no
        # reduction at all. The `prob` field stays POISON in this form, which is the instrument.
        t_b = bf16_rt(x[:, :nrow] * sg.reshape(-1)[:nrow])
        p_used = sigmoid_f32(t_b) if cast == "after" else bf16_rt(sigmoid_f32(t_b))
        prob = np.full(T, nan, dtype=np.float32)
    else:
        p_used = np.repeat(prob[:, None], nrow, axis=1)
    out[:, :nrow] = bf16_rt(p_used * raw_h[:, :nrow])
    return dict(gate=gate, up=up, act=act, raw=raw_h, out=out, prob=prob,
                raw_full=raw_full, poison_rows=nrow)


# ---- the GGUF side: what width does each family actually get, and is gate|up fused? -------------
def derive_from_gguf(gguf_path):
    """Reads the published file's tensor table (metadata only, no tensor bytes) and answers the two
    file-anchored questions: the width class per family, and whether the file fuses the gate|up pair.
    The reader refuses data access without the audit selftest token, and the metadata read is the same
    reader, so the marker is checked here too rather than assumed."""
    import audit_published_quant as apq
    import gguf_reader as gr
    if not os.path.exists(apq.SELFTEST_MARK):
        raise SystemExit("no audit selftest marker at %s — run tools/audit_published_quant.py "
                         "--selftest first (this lane does not read a published file without it)"
                         % apq.SELFTEST_MARK)
    import hashlib

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()
    mark = json.load(open(apq.SELFTEST_MARK))
    if mark.get("gate_sha256") != sha(os.path.join(REPO, "tools/audit_published_quant.py")) or \
       mark.get("reader_sha256") != sha(os.path.join(REPO, "tools/gguf_reader.py")):
        raise SystemExit("the audit selftest marker is STALE against the current tools/")
    gr.grant_selftest_token(apq.SELFTEST_TOKEN)
    gg = gr.Gguf(gguf_path)
    want = {"ffn_gate_shexp.weight": "gate", "ffn_up_shexp.weight": "up",
            "ffn_down_shexp.weight": "down", "ffn_gate_inp_shexp.weight": "sgate"}
    found, blocks = {}, set()
    for t in gg.tensors:
        for suf, role in want.items():
            if t.name.endswith(suf):
                blk = int(t.name.split(".")[1]) if t.name.startswith("blk.") else -1
                blocks.add(blk)
                if role not in found:
                    found[role] = (t.name, t.type_name, list(t.dims), t.row)
    for role in ("gate", "up", "down", "sgate"):
        if role not in found:
            raise SystemExit("GGUF has no shared-expert %s tensor at all" % role)
        name, dt, dims, nb = found[role]
        print("WIDTH role=%s tensor=%s dtype=%s dims_gguf=%s shape_hf=%s bytes=%d"
              % (role, name, dt, dims, tuple(reversed(dims)), nb))
    print("WIDTH blocks_with_shared_expert=%d first=%d last=%d"
          % (len(blocks), min(blocks), max(blocks)))
    print("WIDTH fused_gate_up_in_file=%s (two tensors both present: %s, %s) — the [2I,H] view the "
          "kernel reads is OUR assembly, which is why the layout is a question this lane has to answer"
          % ("no", found["gate"][0], found["up"][0]))
    mtp = [t.name for t in gg.tensors if "mtp" in t.name.lower()]
    print("WIDTH mtp_tensors_in_file=%d %s" % (len(mtp), "the GGUF carries no MTP block: mtp0 is "
          "authority-only bf16 here" if not mtp else sorted(mtp)[:3]))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["derive", "fixtures"], required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "work"))
    ap.add_argument("--blocks", default="layer0,layer1,layer39,mtp0")
    ap.add_argument("--dump-dir", default=os.path.join(REPO, "data/dumps/2026-10-26"))
    ap.add_argument("--manifest", default=os.path.join(REPO, "checkpoints/dump-manifest-2026-10-26.json"))
    ap.add_argument("--gguf", default=os.path.join(REPO, "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"))
    ap.add_argument("--rows", type=int, default=32, help="rows per block used for expectations")
    a = ap.parse_args()
    blocks = [b.strip() for b in a.blocks.split(",") if b.strip()]
    man = json.load(open(a.manifest))
    geo = man["geometry"]
    cfg = json.load(open(geo["config"])) if os.path.exists(geo["config"]) else {}
    cfg = cfg.get("text_config", cfg) if cfg else {}
    print("GEOMETRY source=%s config=%s" % (a.manifest, geo["config"] if cfg else "MISSING"))
    for k in ("hidden_size", "shared_expert_intermediate_size", "num_hidden_layers",
              "mtp_num_hidden_layers"):
        if cfg and int(cfg[k]) != int(geo[k]):
            raise SystemExit("geometry disagreement on %s: manifest %s vs config %s"
                             % (k, geo[k], cfg[k]))
    print("GEOMETRY H=%d I=%d twoI=%d blocks=%d mtp=%d rows_pinned=%d"
          % (geo["hidden_size"], geo["shared_expert_intermediate_size"],
             2 * geo["shared_expert_intermediate_size"], geo["num_hidden_layers"],
             geo["mtp_num_hidden_layers"], man["input"]["n_tokens"]))
    if (geo["hidden_size"], geo["shared_expert_intermediate_size"]) != (H, I):
        raise SystemExit("this file's constants say H=%d I=%d, the geometry says %d/%d"
                         % (H, I, geo["hidden_size"], geo["shared_expert_intermediate_size"]))
    os.makedirs(a.out, exist_ok=True)
    derive_from_gguf(a.gguf)

    if a.mode == "derive":
        f0 = pinned_files(a.dump_dir, blocks[0])
        gw, _ = load_bf16(f0["gate"], (I, H))
        uw, _ = load_bf16(f0["up"], (I, H))
        dw, _ = load_bf16(f0["down"], (H, I))
        x0, _ = load_bf16(f0["input"], (-1, H))
        ref_raw, _ = load_bf16(f0["shared_raw"], (-1, H))
        gsc, gq = q8_quantize(gw)
        usc, uq = q8_quantize(uw)
        dsc, dq = q8_quantize(dw)
        Wg, Wu, Wd = q8_decode(gsc, gq), q8_decode(usc, uq), q8_decode(dsc, dq)
        n = 64
        layout_row_order(fused_of(Wg, Wu), gw, uw, n)
        route_discriminates(x0, ref_raw, Wg, Wu, Wd)
        return 0
    return build_fixtures(a, blocks, man)


if False:
    main()


# ---- fixtures -----------------------------------------------------------------------------------
MODE_ID = {m: i for i, m in enumerate(MODES)}
FIELD_ORDER = ["gate", "up", "act", "raw", "out", "prob"]


def field_widths(mode, cast):
    """Which declared width each field of a case carries. gate/up are raw f32 dots in every mode; act
    is bf16-carried unless the case IS the f32-act injection; prob is bf16-carried in the reference's
    cast order and f32 in the wrong one (that is how the field itself separates the two arms)."""
    return {"gate": "f32", "up": "f32", "act": ("f32" if mode == "f32_act" else "bf16"),
            "raw": "bf16", "out": "bf16", "prob": ("f32" if cast == "after" else "bf16")}


def pack_case_image(f):
    """One case's expectation span, in the arms' field order, as a flat float32 array."""
    span = np.full(CASE_FLOATS, np.float32(struct.unpack("<f", struct.pack("<I", NAN_BITS))[0]),
                   dtype=np.float32)
    span[OFF["gate"]:OFF["gate"] + I] = f["gate"][0]
    span[OFF["up"]:OFF["up"] + I] = f["up"][0]
    span[OFF["act"]:OFF["act"] + I] = f["act"][0]
    span[OFF["raw"]:OFF["raw"] + H] = f["raw"][0]
    span[OFF["out"]:OFF["out"] + H] = f["out"][0]
    span[OFF["prob"]] = f["prob"][0]
    return span


def build_fixtures(a, blocks, man):
    out = a.out
    threads = int(man["interpreter"].get("threads", 8))
    print("REFERENCE oracle-vs-pinned, torch %s threads=%d: the dump was written by this same "
          "routine, so bit-identity here is the oracle agreeing with HF eager on real rows"
          % (man["interpreter"]["torch"], threads))
    ref = validate_against_reference(blocks, a.dump_dir, threads)
    slab = bytearray()
    wlines, alines, clines, crows = [], [], [], []
    expects = []
    rows = min(a.rows, int(man["input"]["n_tokens"]))
    if rows < int(man["input"]["n_tokens"]):
        print("CUT rows=%d of %d per block (a numbers-only cut; no assertion depends on it)"
              % (rows, man["input"]["n_tokens"]))
    pops = []
    inj_modes = ["layout_up_first", "layout_row_inter", "layout_pair_swap", "row_shift1",
                 "tail_drop", "cols_half", "down_half", "wide_misread"]
    for bi, name in enumerate(blocks):
        f = pinned_files(a.dump_dir, name)
        x, xbits = load_bf16(f["input"], (-1, H))
        rms = float(np.sqrt((x.astype(np.float64) ** 2).mean(axis=1)).mean())
        if abs(rms - 1.0) > 5e-2:
            raise SystemExit("%s: input rows have RMS %.4f; the manifest pins them at exactly 1.0, "
                             "so the loader or the file is not what this oracle assumes" % (name, rms))
        gw, _ = load_bf16(f["gate"], (I, H))
        uw, _ = load_bf16(f["up"], (I, H))
        dw, _ = load_bf16(f["down"], (H, I))
        sgw, _ = load_bf16(f["sg"], (1, H))
        gsc, gq = q8_quantize(gw)
        usc, uq = q8_quantize(uw)
        dsc, dq = q8_quantize(dw)
        Wg, Wu, Wd = q8_decode(gsc, gq), q8_decode(usc, uq), q8_decode(dsc, dq)
        Wgw, Wuw, Wdw = q8_decode(gsc, gq, "wide"), q8_decode(usc, uq, "wide"), q8_decode(dsc, dq, "wide")
        # the byte image the arms read, and the proof it decodes to what the oracle computed with
        fused_b = np.concatenate([q8_pack(gsc, gq), q8_pack(usc, uq)], axis=0)
        down_b = q8_pack(dsc, dq)
        assert fused_b.shape == (TWO_I, GATE_STRIDE), fused_b.shape
        assert down_b.shape == (H, DOWN_STRIDE), down_b.shape
        sg_b = sgw.reshape(H).astype(np.float32)
        w_off = len(slab)
        slab += fused_b.tobytes(); slab += down_b.tobytes(); slab += sg_b.tobytes()
        w_fused_id, w_down_id, w_sg_id = 3 * bi, 3 * bi + 1, 3 * bi + 2
        wlines.append("W %d 0 %d %d 0 %d %d" % (w_fused_id, TWO_I, H, w_off, fused_b.nbytes))
        wlines.append("W %d 1 %d %d 0 %d %d" % (w_down_id, H, I, w_off + fused_b.nbytes, down_b.nbytes))
        wlines.append("W %d 2 1 %d 1 %d %d" % (w_sg_id, H, w_off + fused_b.nbytes + down_b.nbytes,
                                               sg_b.nbytes * 4))
        a_off = len(slab)
        slab += xbits[:rows].tobytes()
        for r in range(rows):
            alines.append("A %d %d %d" % (100 * bi + r, H, a_off + 2 * r * H))
        pops.append(dict(block=name, q8=[(gsc, gq), (usc, uq), (dsc, dq)], x=x[:rows]))
        # the case set for this block
        specs = [("honest", "correct", "before", "scalar")]
        if bi == 0:
            specs += [("cast_after", "correct", "after", "scalar"),
                      ("f32_act", "f32_act", "before", "scalar"),
                      ("perchan_gate", "correct", "before", "perchan")]
            specs += [(m, m, "before", "scalar") for m in inj_modes]
        for r in range(rows):
            for spec in specs:
                label, mode, cast, sgate = spec
                if r > 0 and label != "honest":
                    continue                       # injections are one row each; honest uses every row
                xrow = x[r:r + 1]
                F = route_block(xrow, (Wg, Wu, Wd, Wgw, Wuw, Wdw), sg_b, mode=mode,
                                cast=cast, sgate=sgate)
                img = pack_case_image(F)
                exp_off = sum(e.size for e in expects)
                expects.append(img)
                idx = len(clines)
                dst = CASE_FLOATS * len(clines)
                clines.append("C %d %d %d %d %d %d %d %d 1 %d %d %d all %d %d %d"
                              % (idx, w_fused_id, w_down_id, w_sg_id, 100 * bi + r,
                                 MODE_ID[mode], {"scalar": 0, "perchan": 1}[sgate],
                                 {"before": 0, "after": 1}[cast], dst, exp_off, CASE_FLOATS,
                                 0, bi, r))
                crows.append(dict(index=idx, label=label, mode=mode, cast=cast, sgate=sgate,
                                  block=name, block_index=bi, row=r, dst_off=dst, exp_off=exp_off,
                                  n_floats=CASE_FLOATS, widths=field_widths(mode, cast),
                                  poison_rows=F["poison_rows"]))
    n_out = CASE_FLOATS * len(clines)
    exp_flat = np.concatenate(expects)
    with open(os.path.join(out, "slab.bin"), "wb") as fh:
        fh.write(bytes(slab))
    with open(os.path.join(out, "inputs.bin"), "wb") as fh:
        fh.write(exp_flat.astype(np.float32).tobytes())
    with open(os.path.join(out, "plan_all.tsv"), "w") as fh:
        fh.write("# shared_expert plan: W id role rows cols kind byte_off n_bytes | "
                 "A id n_elems byte_off | C index w_fused w_down w_sg a mode sgate cast expect "
                 "dst_off exp_off n_floats field_exact cmp_case block row\n")
        fh.write("\n".join(wlines) + "\n" + "\n".join(alines) + "\n" + "\n".join(clines) + "\n")
    with open(os.path.join(out, "cases.tsv"), "w") as fh:
        fh.write("# index\tlabel\tmode\tcast\tsgate\tblock\trow\tdst_off\texp_off\tn_floats\t"
                 "gate\tup\tact\traw\tout\tprob\n")
        for c in crows:
            fh.write("%d\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n"
                     % (c["index"], c["label"], c["mode"], c["cast"], c["sgate"], c["block"],
                        c["row"], c["dst_off"], c["exp_off"], c["n_floats"],
                        *[c["widths"][k] for k in FIELD_ORDER]))
    with open(os.path.join(out, "cases.json"), "w") as fh:
        json.dump(crows, fh, indent=1, default=str)
    print("FIXTURES out=%s slab=%d expect=%d floats out=%d floats cases=%d weights=%d acts=%d"
          % (out, len(slab), exp_flat.size, n_out, len(clines), len(wlines), len(alines)))
    print_populations(pops, crows)
    for name, d in ref.items():
        print("DIST block=%s bf16_route_vs_f64_route maxabs=%.4e meanabs=%.4e cos=%.9f "
              "(shared_out vs shared_out_exact: a distance, never a target)"
              % (name, d["maxabs"], d["meanabs"], d["cos"]))
    print("no timing taken")
    return 0


def print_populations(pops, crows):
    """Every count a must-catch set is derived from, printed rather than asserted from prose."""
    for p in pops:
        sb = np.concatenate([s.ravel() for s, _ in p["q8"]]).astype(np.uint16)
        q = np.concatenate([qq.ravel().astype(np.int32) for _, qq in p["q8"]])
        sv = f16_to_f32(sb)
        wide = np.concatenate([(f16_bits_as_bf16(s)[:, :, None]
                                * qq.astype(np.int8).astype(np.float32).reshape(
                                    s.shape[0], s.shape[1], Q8_ELTS)).ravel()
                               for s, qq in p["q8"]])
        x = p["x"]
        print("POPQ block=%s q8_blocks=%d dequant_elements=%d zero_scale_blocks=%d "
              "distinct_scale_magnitudes=%d scale_min_nonzero=%.6g scale_max=%.6g "
              "q_min=%d q_max=%d q_at_minus128=%d q_zero=%d"
              % (p["block"], sb.size, q.size, int((sv == 0).sum()),
                 int(len(np.unique(np.abs(sv[sv != 0])))),
                 float(np.min(np.abs(sv[sv != 0]))) if (sv != 0).any() else 0.0,
                 float(np.max(sv)), int(q.min()), int(q.max()), int((q == -128).sum()),
                 int((q == 0).sum())))
        print("POPWIDE block=%s wide_values_read=%d wide_nonzero=%d wide_ge_1e30=%d wide_le_1e-30=%d "
              "wide_negative=%d wide_nan_inf=%d  (what reading the f16 image as bf16 actually yields)"
              % (p["block"], wide.size, int((wide != 0).sum()), int((np.abs(wide) >= 1e30).sum()),
                 int((np.abs(wide) <= 1e-30).sum()), int((wide < 0).sum()),
                 int((~np.isfinite(wide)).sum())))
        xf = x.astype(np.float32)
        print("POPIPUT block=%s rows=%d cols=%d elements=%d zeros=%d nonfinite=%d rms=%.6f "
              "absmax=%.6g" % (p["block"], x.shape[0], H, xf.size, int((xf == 0).sum()),
                               int((~np.isfinite(xf)).sum()), float(np.sqrt((xf * xf).mean())),
                               float(np.abs(xf).max())))
    kinds = {}
    for c in crows:
        kinds.setdefault(c["label"], []).append(c)
    for label, cs in sorted(kinds.items()):
        print("POPCASES label=%-16s cases=%d rows=%d" % (label, len(cs), len(cs)))
    poison = sum(1 for c in crows if c["poison_rows"] < H)
    print("POPPOISON cases_leaving_holes=%d of %d; a hole is 0x%08X and is counted, not skipped"
          % (poison, len(crows), NAN_BITS))
    print("MUSTCATCH families=%d names=%s" % (
        1 + len({c["label"] for c in crows if c["label"] != "honest"}),
        ",".join(sorted({c["label"] for c in crows})) + " (honest is the one that must NOT fire)"))


if __name__ == "__main__":
    sys.exit(main())
