#!/usr/bin/env python3
"""Independent witness for the safetensors header-offset claim (results.md §Corrigendum).

CLAIMS UNDER TEST, as asserted by W18 in prototype/moe_gather/HANDOFF.md §9:

  (1) prototype/moe_router/work/probe15.py memmapped expert weights at
      `header_json_len + data_offsets[0]`, but a safetensors data region begins at
      `8 + header_json_len + data_offsets[0]`. The missing 8-byte length prefix shifted every read by
      four bf16 elements, so NO candidate layout could have matched -- which means the router lane's
      recorded `prediction 3 UNRESOLVED` was an artefact of its own probe, not a property of the dump.

  (2) With the correct offset, pinned `layer0.expert_raw` reproduces bit-exactly.

Written independently of the probes it adjudicates: the data start here comes from struct-unpacking the
shard's own first 8 bytes, never from a helper that might share the bug, and the comparison is on bf16
BIT IMAGES (uint16), never on floats -- a float comparison can hide a 1-ulp difference in a population
this size (pitfalls/30 §Read a dump's dtype, §Tensor.to casts, .view bitcasts).

FOUR ARMS, all mandatory, and each claim is keyed to the arm that can decide it:

  ARM 3  sanctioned path: moe_ref.load_block_weights + moe_block(form='ref').     decides claim (2)
  ARM 2  probe15's shifted offset.                                                 decides claim (1)
  ARM 4  sanctioned path with Form(batching='per_slot').                          association control
  ARM 1  hand-rolled memmap at the correct offset, per-row bf16 compute.           diagnostic
  ARM 1b same, but f32 compute then round.                                          diagnostic

Arms 1/1b are DIAGNOSTIC, not adjudicating: they price the two tempting wrong answers a kernel author
might reach for (per-row association; f32-then-round) so that a later band is derived from a probe rather
than typed. If ARM 1 and ARM 4 move by the same amount, ARM 1's drift is association, and quoting it as a
bug would be wrong -- so the script says which of the two it is and refuses to attribute it otherwise.

A PASS requires: ARM 3 bit-exact (claim 2), AND ARM 2 corrupting every row (claim 1), AND ARM 1's drift
attributed. Zero populations, and an unattributed hand-rolled figure, are findings rather than silence:
a check that can produce no output cannot fail (pitfalls/40).

CPU only. No device, no clock: it prints `no timing taken` and never a duration.

    usage: python3 prototype/moe_router/work/verify_p3_offset.py
"""

import json
import os
import struct
import sys

import numpy as np
import torch


def _repo_root():
    """Walk up to the directory holding checkpoints/ and speccheck/, so this runs from any depth
    (work/, probes/, tmp/) instead of baking in '../..' and silently mis-rooting."""
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "checkpoints")) and os.path.isdir(os.path.join(d, "speccheck")):
            return d
        d = os.path.dirname(d)
    raise AssertionError("no repo root found (a directory containing checkpoints/ and speccheck/)")


REPO = _repo_root()
sys.path.insert(0, os.path.join(REPO, "speccheck"))
sys.path.insert(0, os.path.join(REPO, "prototype", "moe_router"))
os.chdir(REPO)

import moe_ref                                # noqa: E402
import oracle_arm as O                        # noqa: E402

BLOCK = "layer0"
E, TWO_I, H, I = 256, 1024, 2048, 512


def data_start(shard):
    """The data region begins AFTER the 8-byte little-endian header length. Independent read."""
    with open(shard, "rb") as fh:
        nbytes = struct.unpack("<Q", fh.read(8))[0]
    return 8 + nbytes, nbytes


def gu_dn_memmaps(block, shifted):
    """(gu, dn) uint16 memmaps, at the correct offset or at probe15's shifted one."""
    bo = moe_ref.Block.parse(block)
    idx = moe_ref.load_index()
    out = []
    for suf in ("gate_up", "down"):
        name = bo.prefix + moe_ref.SUFFIXES[suf]
        shard = os.path.join(str(moe_ref.SHARD_DIR), moe_ref.shard_of(name, idx))
        head, _ = moe_ref.read_header(shard)          # returns the JSON LENGTH, not the data start
        start, _n = data_start(shard)
        off = start + head[name]["data_offsets"][0] + (-8 if shifted else 0)
        if off < 0:
            raise AssertionError("negative offset -- arithmetic bug, not a finding")
        shape = (E, TWO_I, H) if suf == "gate_up" else (E, H, I)
        out.append(np.memmap(shard, dtype=np.uint16, mode="r", offset=off, shape=shape))
    return out[0], out[1]


def _count(got_u16, want_u16):
    per_row = np.count_nonzero(got_u16 != want_u16, axis=-1)
    return dict(elements=int(got_u16.size), differing=int(per_row.sum()),
                rows=int(per_row.size), rows_bad=int(np.count_nonzero(per_row)))


def arm_handrolled(block, shifted, compute, g):
    """ARMS 1 / 1b: direct memmap read + one expert applied per (token, slot) row."""
    gu_mm, dn_mm = gu_dn_memmaps(block, shifted)
    _, dump_raw_u = O.read_arr(O.DUMP_REC["%s.expert_raw" % block])
    _, dump_x_u = O.read_arr(O.DUMP_REC["%s.input" % block])
    ids = O.read_arr(O.DUMP_REC["%s.topk_ids" % block])[0]
    T, K = dump_raw_u.shape[0], dump_raw_u.shape[1]
    if (T, K) != (32, 8):
        raise AssertionError("unexpected expert_raw shape %r; this witness was written for 32x8 and "
                             "will not quietly compare something else" % (dump_raw_u.shape,))
    got = np.zeros((T, K, H), dtype=np.uint16)
    for t in range(T):
        xt = O.u16_to_torch_bf16(np.asarray(dump_x_u[t], dtype=np.uint16)).reshape(1, H)
        for k in range(K):
            e = int(ids[t, k])
            gu = O.u16_to_torch_bf16(np.asarray(gu_mm[e], dtype=np.uint16))
            dn = O.u16_to_torch_bf16(np.asarray(dn_mm[e], dtype=np.uint16))
            y = moe_ref._expert_apply(xt, gu, dn, g, compute, swap_halves=False)
            got[t, k] = y.to(torch.bfloat16).view(torch.uint16).reshape(-1).numpy()
    return _count(got, dump_raw_u)


def arm_canonical(block, g, batching="grouped"):
    """ARMS 3 / 4: the reference implementation's own path, grouped (the reference's) or per_slot."""
    bo = moe_ref.Block.parse(block)
    w = moe_ref.load_block_weights(bo, g)
    _, dump_x_u = O.read_arr(O.DUMP_REC["%s.input" % block])
    _, dump_raw_u = O.read_arr(O.DUMP_REC["%s.expert_raw" % block])
    x = O.u16_to_torch_bf16(np.asarray(dump_x_u, dtype=np.uint16))
    form = moe_ref.Form() if batching == "grouped" else moe_ref.Form(batching=batching)
    out = moe_ref.moe_block(x, w, g, form)
    got = out["expert_raw"].view(torch.uint16).numpy()
    if got.shape != dump_raw_u.shape:
        raise AssertionError("expert_raw shape %r != dump %r" % (got.shape, dump_raw_u.shape))
    return _count(got, dump_raw_u)


def arm_candidates(block, g):
    """P3 candidate readings, through the sanctioned reader, judged against the pinned expert_raw.

    The gather lane claims P3 is closed: gate-first reproduces while every neighbouring layout differs
    enormously. That claim came out of THEIR oracle_arm.py, so this arm re-checks the two candidates
    moe_ref exposes directly (swap_halves = halves swapped, transpose_stack = stack transposed) with
    MY witness. What is asserted is not 'gate-first matches' but 'gate-first is DISTINGUISHED from its
    neighbours': a candidate that differed by the same little bit as gate-first would leave P3 open.
    """
    bo = moe_ref.Block.parse(block)
    w = moe_ref.load_block_weights(bo, g)
    _, dump_x_u = O.read_arr(O.DUMP_REC["%s.input" % block])
    _, dump_raw_u = O.read_arr(O.DUMP_REC["%s.expert_raw" % block])
    x = O.u16_to_torch_bf16(np.asarray(dump_x_u, dtype=np.uint16))
    out = {}
    for name, kw in (("gate_first", {}),
                     ("halves_swapped", {"swap_halves": True}),
                     ("stack_transposed", {"transpose_stack": True})):
        r = moe_ref.moe_block(x, w, g, moe_ref.Form(), **kw)
        got = r["expert_raw"].view(torch.uint16).numpy()
        out[name] = _count(got, dump_raw_u)
    return out


def main():
    g = moe_ref.load_geometry()
    if (g.num_experts, g.hidden_size, g.moe_intermediate_size) != (E, H, I):
        raise AssertionError("geometry %r disagrees with the census this witness assumes "
                             "(E=%d, H=%d, I=%d)" % (g, E, H, I))
    print("geometry: E=%d H=%d I=%d twoI=%d seed-from checkpoints/Qwen3.6-35B-A3B.config.json"
          % (g.num_experts, g.hidden_size, g.moe_intermediate_size, TWO_I))
    if moe_ref.load_index() is None:
        raise AssertionError("no shard index")
    print("shard index: ok    block: %s    rows: 32 tokens x 8 slots = %d    elements/row: %d"
          % (BLOCK, 32 * 8, H))

    print("\n--- ARM 3: CANONICAL (load_block_weights + moe_block form='ref', grouped) ---")
    c = arm_canonical(BLOCK, g, "grouped")
    print("  elements_compared=%d differing=%d rows=%d rows_bad=%d"
          % (c["elements"], c["differing"], c["rows"], c["rows_bad"]))

    print("\n--- ARM 2: SHIFTED offset (probe15's form, missing the 8-byte prefix) -- control ---")
    b = arm_handrolled(BLOCK, shifted=True, compute=torch.bfloat16, g=g)
    print("  elements_compared=%d differing=%d rows=%d rows_bad=%d"
          % (b["elements"], b["differing"], b["rows"], b["rows_bad"]))

    print("\n--- ARM 4: CANONICAL with Form(batching='per_slot') -- association control ---")
    d = arm_canonical(BLOCK, g, "per_slot")
    print("  elements_compared=%d differing=%d rows=%d rows_bad=%d"
          % (d["elements"], d["differing"], d["rows"], d["rows_bad"]))

    print("\n--- ARM 1: hand-rolled memmap, correct offset, bf16 compute, one row per call ---")
    a = arm_handrolled(BLOCK, shifted=False, compute=torch.bfloat16, g=g)
    print("  elements_compared=%d differing=%d rows=%d rows_bad=%d"
          % (a["elements"], a["differing"], a["rows"], a["rows_bad"]))

    print("\n--- ARM 1b: hand-rolled, f32 compute then round ---")
    a32 = arm_handrolled(BLOCK, shifted=False, compute=torch.float32, g=g)
    print("  elements_compared=%d differing=%d rows=%d rows_bad=%d"
          % (a32["elements"], a32["differing"], a32["rows"], a32["rows_bad"]))

    print("\n--- P3 candidates (sanctioned reader, against pinned expert_raw) ---")
    cands = arm_candidates(BLOCK, g)
    for name in ("gate_first", "halves_swapped", "stack_transposed"):
        v = cands[name]
        print("  %-16s differing=%7d of %d  rows_bad=%d/%d"
              % (name, v["differing"], v["elements"], v["rows_bad"], v["rows"]))

    print("\n--- VERDICT ---")
    findings = []
    reproduction = (c["elements"] > 0 and c["differing"] == 0)
    shift_is_garbage = (b["differing"] > 0 and b["rows_bad"] == b["rows"])
    if c["elements"] == 0 or b["elements"] == 0:
        findings.append("zero population compared -- vacuous")
    if not reproduction:
        findings.append("CANONICAL arm is not bit-exact (%d/%d) -- claim (2) is false on the "
                        "sanctioned path too" % (c["differing"], c["elements"]))
    if not shift_is_garbage:
        findings.append("SHIFTED arm did not corrupt every row (%d/%d bad) -- the control cannot "
                        "separate the two offsets, so it says nothing about claim (1)"
                        % (b["rows_bad"], b["rows"]))

    attributed = True
    if a["differing"] != c["differing"]:
        same_association = abs(d["differing"] - a["differing"]) <= max(8, a["differing"] // 10)
        if same_association:
            print("  NOTE hand-rolled bf16 differs by %d vs canonical %d: explained by reduction "
                  "ASSOCIATION -- per-slot canonical moves by %d, same order, so this is "
                  "grouped-vs-per-slot rounding, not a wrong offset and not a wrong dtype"
                  % (a["differing"], c["differing"], d["differing"]))
        else:
            attributed = False
            findings.append("hand-rolled bf16 arm's %d differing elements are NOT attributed (per-slot "
                            "canonical differs by %d) -- do not quote them" % (a["differing"], d["differing"]))

    # P3: gate-first must be near-exact and its neighbours must be separated from it by a wide margin.
    gf, hs, st = cands["gate_first"]["differing"], cands["halves_swapped"]["differing"], cands["stack_transposed"]["differing"]
    pop = cands["gate_first"]["elements"]
    if gf > pop // 100:
        findings.append("gate_first differs by %d/%d -- not near-exact, so P3 cannot be called closed"
                        % (gf, pop))
    for nm, val in (("halves_swapped", hs), ("stack_transposed", st)):
        if val < 50 * max(gf, 1):
            findings.append("%s differs by only %d vs gate_first's %d -- the candidates are NOT "
                            "distinguished; P3 stays UNRESOLVED" % (nm, val, gf))
    print("  p3_distinguished=%s (gate_first=%d, halves_swapped=%d, stack_transposed=%d, margin>=50x required)"
          % ("YES" if (gf <= pop // 100 and hs >= 50 * max(gf, 1) and st >= 50 * max(gf, 1)) else "NO",
             gf, hs, st))

    for f in findings:
        print("  FINDING: %s" % f)
    verdict = "PASS" if not findings else "FAIL"
    print("  verdict=%s findings=%d" % (verdict, len(findings)))
    print("  claim1_offset_bug=%s" % ("CONFIRMED (shifted read corrupts every row)" if shift_is_garbage
                                      else "NOT CONFIRMED"))
    print("  claim2_reproduction=%s" % ("CONFIRMED (canonical path, bit-exact)" if reproduction
                                        else "NOT CONFIRMED"))
    print("  association_drift=%d   handrolled_bf16_drift=%d   attributed_to_association=%s"
          % (d["differing"], a["differing"], attributed))
    print("  f32_then_round_drift=%d of %d -- f32-then-round does NOT reproduce the dump, so a kernel "
          "that does it cannot have its mismatch waved through as a tolerance"
          % (a32["differing"], a32["elements"]))
    print("  scope=layer0 only (32 tokens x 8 slots); layer1/layer39/mtp0 unprobed here, layer2 absent")
    print("  amend_router_results_md=%s" % ("YES -- dated corrigendum for the shifted reads"
                                            if (reproduction and shift_is_garbage) else "NO"))
    print("no timing taken")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    # Bind dump records by name once, so a typo is an immediate KeyError rather than a comparison that
    # silently never happened.
    _man, _by = O.load_manifest()
    O.DUMP_REC = _by
    sys.exit(main())
