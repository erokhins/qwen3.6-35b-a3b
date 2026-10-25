#!/usr/bin/env python3
"""oracle_arm.py — the CPU/reference side of prototype/moe_router's gate.

Run with the venv interpreter (system python3 has no torch):
    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python oracle_arm.py --work work

It imports the committed oracle, `speccheck/moe_ref.py`, and emits the files gate.py judges the
device words against:

  oracle_rows.tsv    one record per pinned row: its tie class, its 8th/9th margin in ULP, the
                     pinned arm's ids/weights/routed as bit patterns, and what the oracle's own
                     `route()` says under the reference Form and under the cast-after Form.
  oracle_synth.tsv   the synthetic rows, GENERATED HERE rather than typed in gate.py: nine
                     straddling-tie fixtures (j = 0..K members of the tied group inside the K), one
                     strictly-inside tie, and two id-order fixtures. Their logits, their synthetic
                     expert outputs and their expected outputs all come from this file, so the
                     fixture population is derived from the geometry, not from memory
                     (pitfalls/40-gates-and-admissions.md §"A fixture list typed from memory").
  oracle_floor.tsv   the floor probes: what legal re-associations of the reference's OWN fp32 math
                     cost, which is where every band gate.py uses comes from
                     (specification/10-reference-dumps-and-gates.md §"Bands are derived, not
                     invented"). Nothing here reads a device.
  oracle_p3.tsv      prediction 3: is `gate_up_proj` really [E, 2I, H] with `gate, up = chunk(2)`?
                     Answered by recomputing pinned `expert_raw` from the shard three ways — halves
                     as read, halves swapped, stack transposed — and reporting which one reproduces
                     the dump.

Every section prints the number of records it wrote; gate.py requires each printed count to equal
the number of records it parsed (pitfalls/40 §"A gate whose parser can parse nothing prints PASS").
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "speccheck"))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import moe_ref                                              # noqa: E402

MANIFEST = os.path.join(REPO, "checkpoints", "dump-manifest-2026-10-26.json")
BLOCKS = ["layer0", "layer1", "layer39", "mtp0"]
BF16 = "torch.bfloat16"
F32 = "torch.float32"
I64 = "torch.int64"


def shard_data_start(shard, hdr, header_len=None):
    """Where a safetensors data region begins: AFTER the 8-byte little-endian header length.

    `moe_ref.read_header` returns the LENGTH of the JSON header, which is not an offset. This lane
    once memmapped expert weights at `header_len + data_offsets[0]` and read four bf16 elements early,
    which is what made prediction 3 look unresolvable (`results.md §Corrigendum 2026-10-26`, witnessed
    by `work/verify_p3_offset.py`). The offset is therefore not taken on trust from either number: the
    length is re-read here out of the shard's own first 8 bytes and required to agree with the header
    reader's answer, and the shard's own invariant `data_start + max(data_offsets[1]) == filesize` is
    checked before anything memmaps — at `header_len` alone that identity is off by exactly 8.
    """
    size = os.path.getsize(shard)
    with open(shard, "rb") as fh:
        own_len = struct.unpack("<Q", fh.read(8))[0]
    if header_len is not None and int(header_len) != own_len:
        raise AssertionError("%s: the header reader says the JSON header is %s bytes, the file's own "
                             "8-byte prefix says %d — do not compute an offset from a number you "
                             "cannot reproduce" % (shard, header_len, own_len))
    start = 8 + own_len
    top = max([v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"] or [0])
    if start + top != size:
        raise AssertionError(
            "%s: data_start=%d (8 + header_len=%d) plus the last tensor's data_offsets[1]=%d is %d, "
            "but the file is %d bytes. The offset arithmetic is wrong by %d; at `header_len` without "
            "the 8-byte prefix it is wrong by exactly 8 and every value read sits four bf16 elements "
            "early" % (shard, start, own_len, top, start + top, size, size - (start + top)))
    return start


# --------------------------------------------------------------------- dump reading, by manifest
def load_manifest():
    m = json.load(open(MANIFEST))
    return m, {r["name"]: r for r in m["dumps"]}


def read_arr(rec):
    """Read a dump with the dtype the MANIFEST states, never off the filename — a 1 MiB file read as
    the wrong dtype still prints a plausible ndiff (pitfalls/30 §Read a dump's dtype)."""
    path = os.path.join(REPO, rec["file"])
    raw = open(path, "rb").read()
    if len(raw) != rec["bytes"]:
        raise AssertionError("%s: file is %d bytes, manifest says %d" % (rec["name"], len(raw),
                                                                         rec["bytes"]))
    if rec["dtype"] == BF16:
        a = np.frombuffer(raw, dtype=np.uint16).reshape(rec["shape"])
        return (np.asarray(a, dtype=np.uint32) << 16).view(np.float32), a
    if rec["dtype"] == F32:
        return np.frombuffer(raw, dtype=np.float32).reshape(rec["shape"]), None
    if rec["dtype"] == I64:
        return np.frombuffer(raw, dtype=np.int64).reshape(rec["shape"]), None
    raise AssertionError("%s: manifest dtype %r is not one this gate knows"
                         % (rec["name"], rec["dtype"]))


def u16_to_torch_bf16(a):
    """A bf16 BIT IMAGE to a real torch bf16 tensor, by VIEW. Passing the widened f32 array to the
    oracle instead would make it compute the logit dot in f32 rather than in bf16-with-f32-opmath,
    which is a different function and would have had me reporting a 3-element disagreement against
    the pinned logits that does not exist (pitfalls/30 §Tensor.to casts, .view bitcasts; the
    manifest's reading_rules.bf16 says the same thing in the manifest's own words)."""
    return torch.from_numpy(np.array(a, dtype=np.uint16, copy=True)).view(torch.bfloat16)


def u16_to_f32(a):
    """Widen a bf16 BIT IMAGE to f32 the way the dumps store it (bits << 16). A cast would be a
    different operation on a different type (pitfalls/30 §Tensor.to casts, .view bitcasts)."""
    return ((np.asarray(a, dtype=np.uint32)) << 16).view(np.float32)


def bf16_round_bits(f32):
    """Round-to-nearest-even to the 7 stored mantissa bits, in the bf16 BIT IMAGE (uint16)."""
    u = np.asarray(f32, dtype=np.float32).view(np.uint32).astype(np.uint32)
    nan = (u & 0x7F800000) == 0x7F800000
    mant = u & 0x007FFFFF
    lsb = (u >> 16) & 1
    r = ((u + 0x7FFF + lsb) >> 16).astype(np.uint32)
    r = np.where(nan & (mant != 0), 0x7FC0, r)
    return r.astype(np.uint16)


def ulpdist(a, b):
    """ULP distance between two f32 arrays, monotone-index form (works across the sign flip)."""
    ai = np.asarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    bi = np.asarray(b, dtype=np.float32).view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, 0x80000000 - ai, ai)
    bi = np.where(bi < 0, 0x80000000 - bi, bi)
    return np.abs(ai - bi)


# ------------------------------------------------------------------------ the documented rule,
# written independently of the kernel: descending probability, ties to the LOWER expert index,
# the K slots descending with ties in ascending expert index. moe_ref's docstring claims torch does
# this; §tie_break_below records where the pinned arm proves it does not.
def rule_topk(probs, K):
    T, E = probs.shape
    ids = np.zeros((T, K), dtype=np.int64)
    for t in range(T):
        used = np.zeros(E, dtype=bool)
        for k in range(K):
            p = np.where(used, -np.inf, probs[t])
            best = int(np.max(p))
            ids[t, k] = int(np.argmax(p))          # argmax returns the FIRST maximum: lower index
            used[ids[t, k]] = True
    return ids


def rule_route(probs, K):
    """Renormalise the K in fp32 in rank order, then the R5 cast — the documented engine rule."""
    T = probs.shape[0]
    ids = rule_topk(probs, K)
    pre = np.take_along_axis(probs, ids, axis=1).astype(np.float32)
    s = np.zeros(T, dtype=np.float32)
    for k in range(K):
        s = (s + pre[:, k]).astype(np.float32)
    w = (pre / s[:, None]).astype(np.float32)
    return ids, pre, w


def combine_bits(raw, wpre, wcast, ids, K, H, cast="before", order="expert_asc", acc="bf16"):
    """The reference combine, element-exact: prod bf16-rounded, addends bf16-added in the named
    order. `cast` is prediction 4's switch; `order` is §11.5's; `acc` is §11.5's other one."""
    T = ids.shape[0]
    if order == "expert_asc":
        o = np.argsort(ids, axis=1, kind="stable")
    else:
        o = np.tile(np.arange(K), (T, 1))
    # `wcast` is the bf16 BIT IMAGE (uint16), so widening is `<< 16` then a view. `.astype(float32)`
    # would numerically convert 0x3e59 to 15961.0 instead of to 0.2143 — the `.view` vs `.to` trap
    # (pitfalls/30-numerics.md), and the reason the synthetic fixtures' expected routed came out at
    # 10^4 while the kernel's sat at 10^-2.
    wf = ((wcast.astype(np.uint32) << 16).view(np.float32)
          if cast == "before" else wpre)
    out = np.zeros((T, H), dtype=np.float32)
    for t in range(T):
        a = np.zeros(H, dtype=np.float32)
        for j in range(K):
            k = int(o[t, j])
            prod = bf16_round_bits(raw[t, k] * wf[t, k]).astype(np.uint32)
            p = (prod << 16).view(np.float32)
            if acc == "bf16":
                a = bf16_round_bits(a + p).astype(np.uint32)
                a = (a << 16).view(np.float32)
            else:
                a = (a + p).astype(np.float32)
        out[t] = a
    return out


# -----------------------------------------------------------------------------------------------
# the floor probes
# -----------------------------------------------------------------------------------------------
def softmax_orders(logits_bits, K):
    """Every legal fp32 route at this stage, written without reference to what the kernel does:
    two exp routes x four denominator summation orders, plus the reference's own torch softmax.
    The probe's answer is the widest disagreement between them, which is what a band must cover."""
    x = np.asarray(logits_bits, dtype=np.float32)
    m = x.max(axis=-1, keepdims=True)
    routes = {}
    for ename, ef in (("expf32", lambda z: np.exp(z.astype(np.float32)).astype(np.float32)),
                      ("expf64", lambda z: np.exp(z.astype(np.float64)).astype(np.float32))):
        e = ef(x - m)
        T, E = e.shape
        sums = {}
        s = np.zeros(T, dtype=np.float32)
        for i in range(E):
            s = (s + e[:, i]).astype(np.float32)
        sums["seq"] = s
        s = np.zeros(T, dtype=np.float32)
        for i in reversed(range(E)):
            s = (s + e[:, i]).astype(np.float32)
        sums["rev"] = s
        a = e.copy()
        while a.shape[1] > 1:
            half = a.shape[1] // 2
            top = (a[:, :half] + a[:, a.shape[1] - half:]).astype(np.float32)
            if a.shape[1] % 2:
                top = np.concatenate([top, a[:, half:half + 1]], axis=1)
            a = top
        sums["pair"] = a[:, 0]
        acc = np.zeros((T, 32), dtype=np.float32)
        for i in range(E):
            acc[:, i % 32] = (acc[:, i % 32] + e[:, i]).astype(np.float32)
        s = np.zeros(T, dtype=np.float32)
        for i in range(32):
            s = (s + acc[:, i]).astype(np.float32)
        sums["blocked32"] = s
        for sname, sv in sums.items():
            routes["%s/%s" % (ename, sname)] = (e / sv[:, None]).astype(np.float32)
    return routes


def dot_orders(x, wg, ref_bits):
    """R1's 2048-term dot under every legal fp32 accumulation order, plus the correctly-rounded
    route, each rounded to the bf16 the reference stores, alongside the reference's own answer. The
    spread between them — and between each of them and the PINNED logits — is where the logits band
    comes from: a property of the reference's math measured here, not of what this build happens to
    produce (specification/10-reference-dumps-and-gates.md §"Bands are derived, not invented")."""
    T, H = x.shape
    E = wg.shape[0]
    xf = np.ascontiguousarray(x, dtype=np.float32)
    wf = np.ascontiguousarray(wg, dtype=np.float32)
    out = {}
    acc = np.zeros((T, E), dtype=np.float32)
    for j in range(H):                                   # ascending j, one multiply-add at a time
        acc = (acc + xf[:, None, j] * wf[None, :, j]).astype(np.float32)
    out["f32_seq"] = acc
    acc = np.zeros((T, E), dtype=np.float32)
    for j in reversed(range(H)):
        acc = (acc + xf[:, None, j] * wf[None, :, j]).astype(np.float32)
    out["f32_rev"] = acc
    nblk = 16
    acc = np.zeros((T, E, nblk), dtype=np.float32)
    for j in range(H):
        acc[:, :, j % nblk] = (acc[:, :, j % nblk] + xf[:, None, j] * wf[None, :, j]).astype(
            np.float32)
    s = np.zeros((T, E), dtype=np.float32)
    for b in range(nblk):
        s = (s + acc[:, :, b]).astype(np.float32)
    out["f32_blocked16"] = s
    out["f64_correct"] = (xf.astype(np.float64) @ wf.astype(np.float64).T).astype(np.float32)
    bits = {k: bf16_round_bits(v) for k, v in out.items()}
    bits["ref_linear"] = np.asarray(ref_bits, dtype=np.uint16)
    return bits


def tie_of(probs_row, K):
    """(tie_class, margin_ulp) for one row, read off the probabilities alone.
      0 no bit-identical pair among the top K+1
      1 a tie strictly inside the selected K
      2 a tie group that straddles the K / K+1 cut with exactly one member inside
      3 a tie group that straddles the cut with two or more inside
    margin_ulp is the 8th-minus-9th gap in ULPs of the 8th value, and is 0 for a straddle.
    This is the predicate A-03's population is counted with, so it is computed from the dump's own
    probabilities and never typed."""
    srt = np.sort(np.asarray(probs_row, dtype=np.float32))[::-1]
    v8, v9 = np.float32(srt[K - 1]), np.float32(srt[K])
    margin = float(v8 - v9)
    sp = float(np.spacing(v8))
    mu = 0 if margin == 0 else (int(round(margin / sp)) if sp > 0 else -1)
    n_in = int((srt[:K] == v8).sum())             # group members that made the cut
    if v8 == v9:                                  # the cut runs through the group
        return (3 if n_in >= 2 else 2), mu
    for i in range(K - 1):                        # a tie somewhere inside the eight
        if srt[i] == srt[i + 1]:
            return 1, mu
    return 0, mu


def floor_probe(rows, K):
    """rows: (tag, logits_bits, pinned_wpre, ref_probs, ref_logits, x, wg) per block.
    Returns the widest disagreement any LEGAL route pair shows at each stage."""
    out = {"softmax_denominator_reassoc_ulp": 0, "exp_route_ulp": 0,
           "renorm_sum_reassoc_ulp": 0, "weights_vs_pinned_f32_ulp": 0,
           "pinned_weight_to_bf16_boundary_f32ulp": 1 << 30,
           "logit_dot_route_disagreement_elements": 0, "logit_dot_route_ulp": 0,
           "logit_ref_vs_pinned_elements": 0, "logit_ref_vs_pinned_ulp": 0,
           "n_rows_probed": 0}
    for tag, lg, wpre_pin, probs_ref, pin_bits, x, wg, ref_bits in rows:
        routes = softmax_orders(lg, K)
        base = routes["expf32/seq"]
        for name, p in routes.items():
            d = int(ulpdist(p, base).max())
            if not name.endswith("/seq"):
                out["softmax_denominator_reassoc_ulp"] = max(
                    out["softmax_denominator_reassoc_ulp"], d)
            if name.startswith("expf64"):
                out["exp_route_ulp"] = max(out["exp_route_ulp"], d)
        # R4's own 8-term sum, re-associated, on the reference probabilities
        ids = rule_topk(probs_ref, K)
        pre = np.take_along_axis(probs_ref, ids, axis=1).astype(np.float32)
        T = pre.shape[0]
        sums = {}
        s = np.zeros(T, dtype=np.float32)
        for k in range(K):
            s = (s + pre[:, k]).astype(np.float32)
        sums["seq"] = s
        s = np.zeros(T, dtype=np.float32)
        for k in reversed(range(K)):
            s = (s + pre[:, k]).astype(np.float32)
        sums["rev"] = s
        a = pre.copy()
        while a.shape[1] > 1:
            half = a.shape[1] // 2
            a = (a[:, :half] + a[:, a.shape[1] - half:]).astype(np.float32)
        sums["pair"] = a[:, 0]
        srt = np.sort(pre, axis=1)
        s = np.zeros(T, dtype=np.float32)
        for k in range(K):
            s = (s + srt[:, k]).astype(np.float32)
        sums["sorted"] = s
        for nm, sv in sums.items():
            if nm == "seq":
                continue
            w = (pre / sv[:, None]).astype(np.float32)
            out["renorm_sum_reassoc_ulp"] = max(out["renorm_sum_reassoc_ulp"],
                                                int(ulpdist(w, pre / s[:, None]).max()))
        # every route's weights against the pinned f32 arm, and the pinned arm's distance to a
        # bf16 rounding boundary (the figure the zero-tolerance claim on R5 rests on)
        for name, p in routes.items():
            _, _, w = rule_route(p, K)
            out["weights_vs_pinned_f32_ulp"] = max(out["weights_vs_pinned_f32_ulp"],
                                                   int(ulpdist(w, wpre_pin).max()))
        v = np.asarray(wpre_pin, dtype=np.float32).view(np.uint32).astype(np.uint64)   # bits, not
        # a value cast: astype(uint32) on a float array TRUNCATES the number instead of reinterpreting
        # its bits, which is the pitfalls/30 trap and here it would silently report a huge margin.
        rem = (v & 0xFFFF).astype(np.int64)
        out["pinned_weight_to_bf16_boundary_f32ulp"] = min(
            out["pinned_weight_to_bf16_boundary_f32ulp"],
            int(np.minimum(np.abs(rem - 0x8000), 0x8000).min()))
        # R1: how far apart legal summation orders leave the STORED bf16 logits, and how far the
        # reference's own F.linear sits from the pinned dump
        step = 1                                     # every row the gate judges is probed too
        sub = slice(0, x.shape[0], step)
        bits = dot_orders(x[sub], wg, ref_bits[sub])
        pin = np.asarray(pin_bits[sub], dtype=np.uint16)
        ref = bits["f32_seq"]
        for nm, v2 in bits.items():
            if nm == "f32_seq":
                continue
            out["logit_dot_route_disagreement_elements"] = max(
                out["logit_dot_route_disagreement_elements"], int((v2 != ref).sum()))
            out["logit_dot_route_ulp"] = max(
                out["logit_dot_route_ulp"],
                int(ulpdist(u16_to_f32(v2), u16_to_f32(ref)).max()))
        out["logit_ref_vs_pinned_elements"] = max(
            out["logit_ref_vs_pinned_elements"], int((bits["ref_linear"] != pin).sum()))
        out["logit_ref_vs_pinned_ulp"] = max(
            out["logit_ref_vs_pinned_ulp"],
            int(ulpdist(u16_to_f32(bits["ref_linear"]), u16_to_f32(pin)).max()))
        out["n_rows_probed"] += int(np.ceil(x.shape[0] / step))
    return out


# -----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "work"))
    ap.add_argument("--skip-p3", action="store_true", help="skip the shard-reading layout probe")
    a = ap.parse_args()
    if not os.path.isdir(a.work):
        os.makedirs(a.work)
    torch.set_num_threads(1)                      # replaced by the manifest's recorded count below
    g = moe_ref.load_geometry()
    man, D = load_manifest()
    K, H, E = g.top_k, g.hidden_size, g.num_experts
    # A bf16 pin is only reproducible at the thread count that produced it: the CPU matmul's
    # reduction order follows the thread partitioning (speccheck/moe_ref.py §set_threads) and the
    # manifest records the number it used. Taking it from the manifest instead of typing it is what
    # lets the reference reproduce the pinned logits here at all.
    thr = int(man["interpreter"]["threads"])
    got = moe_ref.set_threads(thr)
    print("ORACLE threads_requested=%d threads_active=%d source=manifest.interpreter" % (thr, got))
    for nm, want in (("top_k", K), ("hidden_size", H), ("num_experts", E)):
        if man["geometry"][nm] != want:
            raise AssertionError("manifest %s=%s disagrees with the config's %s"
                                 % (nm, man["geometry"][nm], want))

    rows_tsv = open(os.path.join(a.work, "oracle_rows.tsv"), "w")
    rows_tsv.write("# src_key block token tie margin_ulp_pin margin_ulp_ref id_asc pair_identity "
                   "rule_ids pinned_ids rule_wpre_hex pinned_wpre_hex rule_wcast_hex "
                   "pinned_wcast_hex pinned_routed_hex castafter_routed_hex n_rule_ne_pin\n")
    n_rows = 0
    floor_rows = []
    tie_hist = {}
    tie_hist_ref = {}
    margins = []
    margins_ref = []
    for b, blk in enumerate(BLOCKS):
        x, xb = read_arr(D[blk + ".input"])
        wg, wgb = read_arr(D[blk + ".mlp.gate.weight"])
        lg, lgb = read_arr(D[blk + ".router_logits"])
        ids_pin = read_arr(D[blk + ".topk_ids"])[0]
        twb = read_arr(D[blk + ".topk_weights"])[1]
        twf = read_arr(D[blk + ".topk_weights_f32"])[0]
        raw, _ = read_arr(D[blk + ".expert_raw"])
        rs, rsb = read_arr(D[blk + ".routed_sum"])
        T = x.shape[0]
        x_t = u16_to_torch_bf16(xb)
        wg_t = u16_to_torch_bf16(wgb)
        form = moe_ref.Form.named("ref")
        r = moe_ref.route(x_t, wg_t, g, form)
        probs_ref = r["probs"].numpy()
        logits_ref = r["logits"].numpy() if r["logits"].dtype is not torch.bfloat16 \
            else r["logits"].view(torch.uint16).numpy()
        if tuple(r["logits"].shape) != (T, E) or r["logits"].dtype is not torch.bfloat16:
            raise AssertionError("route() returned %s %s; the reference form stores bf16 logits"
                                 % (r["logits"].dtype, tuple(r["logits"].shape)))
        logits_ref_bits = logits_ref              # already a bf16 bit image, not a value
        ids_orc = r["topk_ids"].numpy()
        rule_ids, rule_pre, rule_w = rule_route(probs_ref, K)
        rule_wcast = bf16_round_bits(rule_w)
        # cast-after's routed output, for the ALT comparison gate.py makes against mode 3
        ca = moe_ref.route(x_t, wg_t, g, moe_ref.Form.named("cast_after"))
        ca_w = ca["topk_weights"].numpy().astype(np.float32)
        ca_ids, _, _ = rule_route(ca["probs"].numpy(), K)
        ca_routed = combine_bits(raw, ca_w, rule_wcast, ca_ids, K, H, cast="after")
        floor_rows.append((blk, lg, twf, probs_ref, lgb, x, wg, logits_ref_bits))
        probs_pin = torch.softmax(torch.from_numpy(np.ascontiguousarray(lg)), dim=-1,
                                  dtype=torch.float32).numpy()
        for t in range(T):
            # the tie class and the boundary margin, from BOTH logit routes. The manifest does not
            # dump router_probs, so the probabilities that decided the pinned topk_ids can only be
            # reconstructed by re-softmaxing the pinned logits; a second route (the reference's own
            # R1, whose last bits differ) is measured alongside because a straddle tie is a
            # last-ULP phenomenon and the two routes do not always agree about it. See results.md
            # §A-03.
            tie_pin, mu_pin = tie_of(probs_pin[t], K)
            tie_ref, mu_ref = tie_of(probs_ref[t], K)
            tie = tie_pin
            mu = min(mu_pin, mu_ref)
            tie_hist[tie] = tie_hist.get(tie, 0) + 1
            tie_hist_ref[tie_ref] = tie_hist_ref.get(tie_ref, 0) + 1
            margins.append(mu_pin)
            margins_ref.append(mu_ref)
            asc = int(all(ids_pin[t, k] < ids_pin[t, k + 1] for k in range(K - 1)))
            pairid = int(all(int(ids_pin[t, k]) % K == k for k in range(K)))
            rn_ne = int((rule_ids[t] != ids_pin[t]).sum())
            rows_tsv.write("%d\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\n" % (
                b * T + t, blk, t, tie, mu_pin, mu_ref, asc, pairid,
                ",".join(str(int(v)) for v in rule_ids[t]),
                ",".join(str(int(v)) for v in ids_pin[t]),
                ",".join("%08x" % v for v in rule_w[t].view(np.uint32)),
                ",".join("%08x" % v for v in twf[t].view(np.uint32)),
                ",".join("%04x" % v for v in rule_wcast[t]),
                ",".join("%04x" % v for v in twb[t]),
                ",".join("%04x" % v for v in bf16_round_bits(rs[t])),
                ",".join("%04x" % v for v in bf16_round_bits(ca_routed[t])),
                rn_ne))
            n_rows += 1
    rows_tsv.close()
    print("ORACLESECTION section=rows records=%d blocks=%d K=%d" % (n_rows, len(BLOCKS), K))
    print("ORACLE tie_histogram_pinned_route " +
          " ".join("class%d=%d" % kv for kv in sorted(tie_hist.items())))
    print("ORACLE tie_histogram_reference_route " +
          " ".join("class%d=%d" % kv for kv in sorted(tie_hist_ref.items())))
    nz = [m for m in margins if m > 0]
    nzr = [m for m in margins_ref if m > 0]
    print("ORACLE margins pinned_min_nonzero_ulp=%d pinned_zero_rows=%d "
          "reference_min_nonzero_ulp=%d reference_zero_rows=%d"
          % (min(nz) if nz else -1, sum(1 for m in margins if m == 0),
             min(nzr) if nzr else -1, sum(1 for m in margins_ref if m == 0)))

    # ---- the floor probe ----------------------------------------------------------------------
    fl = floor_probe(floor_rows, K)
    with open(os.path.join(a.work, "oracle_floor.tsv"), "w") as fh:
        fh.write("# name n max_ulp\n")
        n_f = 0
        for k, v in sorted(fl.items()):
            fh.write("%s 1 %d\n" % (k, v))
            print("FLOOR name=%s n=1 max_ulp=%d" % (k, v))
            n_f += 1
    print("ORACLESECTION section=floor records=%d" % n_f)

    # ---- the synthetic fixtures, generated here ------------------------------------------------
    # j = how many members of the straddling tied group land inside the K. j in 0..K gives K+1
    # rows, one per position the boundary can cut a tie group at; then one strictly-inside tie and
    # two id-order fixtures. Derived from K, so the case list cannot be typed from memory.
    rng = np.random.default_rng(20261026)
    synth_tsv = open(os.path.join(a.work, "oracle_synth.tsv"), "w")
    synth_tsv.write("# key logits_hex expert_raw_hex rule_ids rule_wpre_hex rule_wcast_hex "
                    "routed_hex oracle_ids\n")
    n_s = 0

    def emit(key, logits, raw):
        p = softmax_orders(logits, K)["expf32/seq"]
        rids, rpre, rw = rule_route(p, K)
        rc = bf16_round_bits(rw)
        # The slab stores the synthetic expert outputs as bf16 bits, so the expectation must combine
        # the SAME rounded values the kernel reads. Combining the unrounded float32 draw instead made
        # the products differ on ~30% of the elements — the dtype-by-filesize trap
        # (pitfalls/30-numerics.md) in the fixture rather than in the kernel.
        raw = ((bf16_round_bits(raw).astype(np.uint32) << 16).view(np.float32)).astype(np.float32)
        rr = combine_bits(raw, rw, rc, rids, K, H, cast="before")
        orc = torch.topk(torch.from_numpy(np.ascontiguousarray(p)), K, dim=-1).indices.numpy()
        synth_tsv.write("%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
            key,
            ",".join("%04x" % v for v in bf16_round_bits(logits[0])),
            ",".join("%04x" % v for v in bf16_round_bits(raw.reshape(-1))),
            ",".join(str(int(v)) for v in rids[0]),
            ",".join("%08x" % v for v in rw[0].view(np.uint32)),
            ",".join("%04x" % v for v in rc[0]),
            ",".join("%04x" % v for v in bf16_round_bits(rr[0])),
            ",".join(str(int(v)) for v in orc[0])))
        return int(rids[0, 0])

    for j in range(0, K + 1):
        lg = np.full((1, E), -8.0, dtype=np.float32)
        n_above = K - j
        for i in range(n_above):
            lg[0, 200 + i] = np.float32(9.0 - 0.25 * i)
        gsize = max(2, j + 1)
        for i in range(gsize):
            lg[0, 20 + 3 * i] = np.float32(4.0)
        raw = (rng.random((1, K, H), dtype=np.float32) * 2.0 - 1.0).astype(np.float32)
        emit(10000 + j, lg, raw)
    # a tie strictly inside the K: two experts tied at rank 7/8, rank 9 strictly below
    lg = np.full((1, E), -8.0, dtype=np.float32)
    for i in range(6):
        lg[0, 100 + i] = np.float32(9.0 - 0.25 * i)
    lg[0, 60] = np.float32(5.0)
    lg[0, 61] = np.float32(5.0)
    lg[0, 62] = np.float32(4.5)
    raw = (rng.random((1, K, H), dtype=np.float32) * 2.0 - 1.0).astype(np.float32)
    emit(10100, lg, raw)
    # the two id-order fixtures: the K selected experts already ascending, and already descending.
    # These are what make the order injections' must-not-catch side reachable: no pinned row has
    # ascending ids (measured: 0 of 128), so without them the immunity classes would be empty.
    for key, base in ((10200, 0), (10201, 1)):
        lg = np.full((1, E), -8.0, dtype=np.float32)
        order = [9 * k for k in range(K)] if base == 0 else [9 * (K - 1 - k) for k in range(K)]
        for i, e in enumerate(order):
            lg[0, e] = np.float32(9.0 - 0.25 * i)
        raw = (rng.random((1, K, H), dtype=np.float32) * 2.0 - 1.0).astype(np.float32)
        emit(key, lg, raw)
    synth_tsv.close()
    n_s = (K + 1) + 1 + 2
    print("ORACLESECTION section=synth records=%d derived_from=K+1+1+2" % n_s)

    # ---- prediction 3: the fused layout, three ways --------------------------------------------
    # `gate_up_proj` is [E, 2I, H] with `gate, up = chunk(2)`, so the halves are blocks of the OUTPUT
    # dimension. That is checkable against the pinned `expert_raw` without implementing a gather: read
    # one selected expert three ways and see which reading reproduces the dump. The bytes come from a
    # np.memmap whose shape and offsets are read out of the shard's own header (moe_ref.read_header),
    # so the layout claim is checked against the file, not against a typed constant.
    if not a.skip_p3:
        n_p = 0
        I = g.moe_intermediate_size
        idx = moe_ref.load_index()
        with open(os.path.join(a.work, "oracle_p3.tsv"), "w") as fh:
            fh.write("# block token slot read max_abs one_minus_cos ndiff n\n")
            for blk in BLOCKS:
                blk_o = moe_ref.Block.parse(blk)
                gu_name = blk_o.prefix + moe_ref.SUFFIXES["gate_up"]
                dn_name = blk_o.prefix + moe_ref.SUFFIXES["down"]
                gu_shard = os.path.join(str(moe_ref.SHARD_DIR), moe_ref.shard_of(gu_name, idx))
                dn_shard = os.path.join(str(moe_ref.SHARD_DIR), moe_ref.shard_of(dn_name, idx))
                gu_hdr, gu_hn = moe_ref.read_header(gu_shard)
                dn_hdr, dn_hn = moe_ref.read_header(dn_shard)
                for nm, hdr, want in ((gu_name, gu_hdr, [E, g.two_i, H]),
                                      (dn_name, dn_hdr, [E, H, I])):
                    if list(hdr[nm]["shape"]) != want or hdr[nm]["dtype"] != "BF16":
                        raise AssertionError("%s is stored %s %s; config-derived is %s BF16"
                                             % (nm, hdr[nm]["dtype"], hdr[nm]["shape"], want))
                gu_lo, gu_hi = gu_hdr[gu_name]["data_offsets"]
                dn_lo, dn_hi = dn_hdr[dn_name]["data_offsets"]
                if (gu_hi - gu_lo) != E * g.two_i * H * 2 or (dn_hi - dn_lo) != E * H * I * 2:
                    raise AssertionError("%s byte span does not match [E,2I,H]+[E,H,I] bf16"
                                         % gu_name)
                # NOT `gu_hn + gu_lo`: that is the shifted read the corrigendum is about. The data
                # region begins at 8 + header length, and shard_data_start refuses to run if the
                # shard's own size does not say so.
                gu_data0 = shard_data_start(gu_shard, gu_hdr, gu_hn)
                dn_data0 = shard_data_start(dn_shard, dn_hdr, dn_hn)
                if (gu_data0 + gu_hi) > os.path.getsize(gu_shard) or \
                        (dn_data0 + dn_hi) > os.path.getsize(dn_shard):
                    raise AssertionError("%s: a tensor's data span runs past the end of its shard"
                                         % gu_name)
                gu = np.memmap(gu_shard, dtype=np.uint16, mode="r",
                               offset=gu_data0 + gu_lo,
                               shape=(E, g.two_i, H))
                dn = np.memmap(dn_shard, dtype=np.uint16, mode="r",
                               offset=dn_data0 + dn_lo,
                               shape=(E, H, I))
                ids_pin = read_arr(D[blk + ".topk_ids"])[0]
                raw_pin, _ = read_arr(D[blk + ".expert_raw"])
                x, xb = read_arr(D[blk + ".input"])   # the BIT IMAGE, not the widened values
                for (t, kslot) in ((0, 0), (7, 3)):
                    e = int(ids_pin[t, kslot])
                    xt = u16_to_torch_bf16(np.asarray(xb[t], dtype=np.uint16)).reshape(1, H)
                    for read in ("chunk2", "halves_swapped", "stack_transposed"):
                        if read == "stack_transposed":
                            # the wrong stride: read the stack as [2I, E, H] and [H, E, I]
                            gu_u = np.asarray(gu).reshape(g.two_i, E, H)[:, e, :]
                            dn_u = np.asarray(dn).reshape(g.hidden_size, E, I)[:, e, :]
                        else:
                            gu_u = np.asarray(gu[e], dtype=np.uint16)
                            dn_u = np.asarray(dn[e], dtype=np.uint16)
                        # The authoritative expert, not this file's re-derivation of one: speccheck is
                        # the independent implementation and the oracle, and its own
                        # `_expert_apply` asserts `out.shape[-1] == 2I` ("gate_up_proj is not
                        # [E, 2I, H]"), which is prediction 3's own check rather than a claim I
                        # wrote. The earlier version of this probe computed SwiGLU and the down
                        # projection here and reproduced nothing under ANY reading, including the
                        # correct one — a probe bug, and exactly the kind the oracle exists to catch.
                        y = moe_ref._expert_apply(
                            xt, u16_to_torch_bf16(gu_u), u16_to_torch_bf16(dn_u), g,
                            torch.float32, swap_halves=(read == "halves_swapped")).numpy()
                        y = np.asarray(y, dtype=np.float64).reshape(-1)
                        want = raw_pin[t, kslot].astype(np.float64)
                        d = np.abs(y - want)
                        nrm = float(np.linalg.norm(y) * np.linalg.norm(want))
                        cos = float(y @ want / nrm) if nrm else 0.0
                        fh.write("%s %d %d %s %.6e %.3e %d %d\n"
                                 % (blk, t, kslot, read, d.max(), max(0.0, 1.0 - cos),
                                    int((d > 0).sum()), want.size))
                        n_p += 1
        for line in open(os.path.join(a.work, "oracle_p3.tsv")):
            if line.startswith("#"):
                continue
            print("P3 %s" % line.rstrip())
        print("ORACLESECTION section=p3 records=%d" % n_p)

    print("RESULT PASS oracle arm rows=%d synth=%d" % (n_rows, n_s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
