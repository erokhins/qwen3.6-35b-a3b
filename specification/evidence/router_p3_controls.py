#!/usr/bin/env python3
"""router_p3_controls.py — the negative controls for the repaired prediction-3 lane.

`prototype/moe_router/results.md` §Corrigendum records two defects that this lane repaired:

  1. `oracle_arm.py` §P3 memmapped expert weights at `header_len + data_offsets[0]`. A safetensors
     data region begins at `8 + header_len`, so every value was read four bf16 elements early and no
     reading could reproduce the pinned `expert_raw` — which is what made prediction 3 look
     unresolvable. Repaired by `oracle_arm.shard_data_start`, which derives the start from the shard's
     own 8-byte prefix and refuses to run unless `data_start + max(data_offsets[1]) == filesize`.
  2. `gate.py` §report_predictions carried the unresolved case as
     `v.check("prediction3-fused-layout-unresolved-is-reported", True, ...)`. A constant predicate is
     a note wearing a counter's clothes, and under defect 1 it was the ONLY P3 assertion that ran.

Every control below is an EXISTING predicate driven with tampered input, and each is required to
refuse — a check that has never returned False has been hoped for, not tested
(`pitfalls/40-gates-and-admissions.md`). The honest input is required to pass in the same breath, so a
predicate that refused everything could not pass this file either.

CPU only (torch via the oracle). Opens no device, reads no clock: no timing taken.

    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python \\
        specification/evidence/router_p3_controls.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "prototype", "moe_router"))
sys.path.insert(0, os.path.join(REPO, "speccheck"))
sys.path.insert(0, os.path.join(REPO, "tools"))

import numpy as np                                           # noqa: E402
import torch                                                 # noqa: E402
import gate as G                                             # noqa: E402
import oracle_arm as O                                       # noqa: E402
import moe_ref                                               # noqa: E402

BLOCK = "layer0"
E, H, I = 256, 2048, 512
WORK = os.path.join(REPO, "prototype", "moe_router", "work")


class V(object):
    def __init__(self):
        self.n = 0
        self.f = 0

    def check(self, name, ok, n, detail=""):
        self.n += 1
        if not ok:
            self.f += 1
        print("CHECK name=%s ok=%d n=%s %s" % (name, 1 if ok else 0, n, detail))
        return ok


def real_table():
    path = os.path.join(WORK, "oracle_p3.tsv")
    if not os.path.exists(path):
        raise SystemExit("no %s — run `./gate --check-only` first, this file adjudicates the table "
                         "the lane produced rather than regenerating it" % path)
    return G.p3_records(G.parse_tsv(path, 8))


def shifted_table(by_key):
    """The same table as the shifted read left it: every reading orthogonal to the dump. These are the
    numbers `work/oracle_p3.tsv` held on 2026-10-26, before the offset was repaired."""
    out = {}
    for k, v in by_key.items():
        out[k] = (1.005 if k[3] == "chunk2" else (0.9749 if k[3] == "halves_swapped" else 1.040),
                  v[1], v[2])
    return out


def expert_reproduction(block, shifted, g):
    """1-minus-cos of one (token, slot) expert recomputed from the shard, at the right offset or the
    lane's old shifted one. This is the behaviour the offset repair changed."""
    bo = moe_ref.Block.parse(block)
    idx = moe_ref.load_index()
    out = {}
    for role, suf, shape in (("gu", "gate_up", (E, g.two_i, H)), ("dn", "down", (E, H, I))):
        name = bo.prefix + moe_ref.SUFFIXES[suf]
        shard = os.path.join(str(moe_ref.SHARD_DIR), moe_ref.shard_of(name, idx))
        hdr, hlen = moe_ref.read_header(shard)
        start = O.shard_data_start(shard, hdr, hlen) + (-8 if shifted else 0)
        out[role] = np.memmap(shard, dtype=np.uint16, mode="r",
                              offset=start + hdr[name]["data_offsets"][0], shape=shape)
    _man, by = O.load_manifest()
    raw_pin, _ = O.read_arr(by[block + ".expert_raw"])
    x_u16, x_bits = O.read_arr(by[block + ".input"])
    ids = O.read_arr(by[block + ".topk_ids"])[0]
    e = int(ids[0, 0])
    xt = O.u16_to_torch_bf16(np.asarray(x_bits[0], dtype=np.uint16)).reshape(1, H)
    y = moe_ref._expert_apply(xt, O.u16_to_torch_bf16(np.asarray(out["gu"][e], dtype=np.uint16)),
                              O.u16_to_torch_bf16(np.asarray(out["dn"][e], dtype=np.uint16)),
                              g, torch.float32, swap_halves=False).numpy().astype(np.float64).ravel()
    want = raw_pin[0, 0].astype(np.float64)
    nrm = np.linalg.norm(y) * np.linalg.norm(want)
    return max(0.0, 1.0 - float(y @ want / nrm)), int((y != want).sum())


def main():
    v = V()
    by = real_table()
    print("TABLE records=%d probes=%d thresholds: 1-cos<%.1e separation>=%.0fx"
          % (len(by), len(by) // len(G.P3_READINGS), G.P3_ADMISSIBLE_ONE_MINUS_COS,
             G.P3_SEPARATION_MIN))

    # --- the population predicate, honest input first, then three tampered tables ---------------
    ok, detail = G.p3_population_ok(by)
    v.check("control:population accepts the real table", ok, len(by), detail)
    keys = sorted(by)
    dropped = dict((k, by[k]) for k in keys[1:])
    ok1, d1 = G.p3_population_ok(dropped)
    v.check("control:population refuses a truncated table", ok1 is False, len(dropped),
            "dropped one record -> %s" % d1)
    dup = dict(by)
    del dup[keys[-1]]
    dup[keys[-2]] = by[keys[-1]]                      # same count, one key duplicated, one missing
    ok2, d2 = G.p3_population_ok(dup)
    v.check("control:population refuses a duplicated key", ok2 is False, len(dup),
            "count still %d -> %s" % (len(dup), d2))
    bogus = dict(by)
    bogus[("layer7", 0, 0, "chunk2")] = (1e-9, 0, H)
    ok3, d3 = G.p3_population_ok(bogus)
    v.check("control:population refuses a block outside the manifest's four", ok3 is False, len(bogus),
            "-> %s" % d3)
    v.check("control:the three tampered tables are refused and the honest one is accepted",
            ok and not ok1 and not ok2 and not ok3, 3,
            "honest=True tampered=%s,%s,%s (a predicate that refuses everything fails this too)"
            % (ok1, ok2, ok3))

    # --- distinguishability and the layout verdict ---------------------------------------------
    ok, indistinct, n_probes = G.p3_distinguishable_ok(by)
    v.check("control:distinguishability accepts the real table", ok, n_probes,
            "probes where the readings are not three answers: %s" % indistinct)
    same = dict((k, (0.5, v2[1], v2[2])) for k, v2 in by.items())
    ok1, ind1, _n1 = G.p3_distinguishable_ok(same)
    v.check("control:distinguishability refuses one reading under three names", ok1 is False,
            len(same), "all three readings 0.5 -> %s" % ind1)
    adj, br, bw, wn, n_rec = G.p3_verdict(by)
    v.check("control:the repaired table is adjudicable", adj, n_rec,
            "worst chunk2 1-cos=%.3e, nearest wrong=%.3e" % (br, bw))
    adj_s, br_s, bw_s, _wn_s, _n_s = G.p3_verdict(shifted_table(by))
    v.check("control:the PRE-FIX (shifted-offset) table is NOT adjudicable", adj_s is False,
            len(by), "chunk2 1-cos=%.3f — the word the old lane reached for was UNRESOLVED, and the "
                     "only P3 assertion that then ran was the literal-`True` one" % br_s)
    # the report predicate must refuse to call that table resolved
    lied = G.p3_report(True, br_s, bw_s, len(by), 2048)
    honest_lie = G.p3_report_is_honest(lied, True, br_s, bw_s, len(by), 2048)
    v.check("control:the report predicate refuses RESOLVED on the pre-fix table",
            honest_lie[0] is False, 1, "refused: %s" % honest_lie[1])

    # --- the offset guard itself ---------------------------------------------------------------
    g = moe_ref.load_geometry()
    bo = moe_ref.Block.parse(BLOCK)
    idx = moe_ref.load_index()
    name = bo.prefix + moe_ref.SUFFIXES["gate_up"]
    shard = os.path.join(str(moe_ref.SHARD_DIR), moe_ref.shard_of(name, idx))
    hdr, hlen = moe_ref.read_header(shard)
    start = O.shard_data_start(shard, hdr, hlen)
    top = max([x["data_offsets"][1] for k, x in hdr.items() if k != "__metadata__"])
    v.check("control:data_start satisfies the shard's own size identity",
            start + top == os.path.getsize(shard) and start == 8 + hlen, start,
            "data_start=%d = 8 + header_len=%d; + last data_offsets[1]=%d == filesize=%d"
            % (start, hlen, top, os.path.getsize(shard)))
    shifted_hdr = dict(hdr)
    for k, x in shifted_hdr.items():
        if k != "__metadata__":
            shifted_hdr[k] = dict(x, data_offsets=(x["data_offsets"][0], x["data_offsets"][1] + 8))
    try:
        O.shard_data_start(shard, shifted_hdr, hlen)
        v.check("control:data_start refuses offsets that do not fill the shard", False, 0,
                "no exception — the guard would have let an 8-byte-shifted read through")
    except AssertionError as e:
        v.check("control:data_start refuses offsets that do not fill the shard", True, 1, str(e)[:150])
    try:
        O.shard_data_start(shard, hdr, hlen + 1)
        v.check("control:data_start refuses a header length the file contradicts", False, 0,
                "no exception — the two independent reads of the header length are not compared")
    except AssertionError as e:
        v.check("control:data_start refuses a header length the file contradicts", True, 1,
                str(e)[:150])

    cos_ok, ndiff_ok = expert_reproduction(BLOCK, False, g)
    cos_bad, ndiff_bad = expert_reproduction(BLOCK, True, g)
    print("OFFSET layer0/token0/slot0 correct: 1_minus_cos=%.3e elements_differing=%d/%d"
          % (cos_ok, ndiff_ok, H))
    print("OFFSET layer0/token0/slot0 shifted: 1_minus_cos=%.3e elements_differing=%d/%d"
          % (cos_bad, ndiff_bad, H))
    v.check("control:the correct offset reproduces and the 8-byte-shifted one does not",
            cos_ok < G.P3_ADMISSIBLE_ONE_MINUS_COS and cos_bad > 0.5
            and cos_bad > 100.0 * cos_ok, 2,
            "%.3e vs %.3e (>=100x apart) — element-differing counts are %d and %d of %d because the "
            "probe computes the expert in f32 while the dump is bf16, so 1-cos, not the element count, "
            "is the discriminator here (results.md §the four arms, arm 1b)"
            % (cos_ok, cos_bad, ndiff_ok, ndiff_bad, H))

    print("RESULT %s router P3 controls checks=%d findings=%d"
          % ("PASS" if v.f == 0 else "FAIL", v.n, v.f))
    return 0 if v.f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
