#!/usr/bin/env python3
"""floor_probe.py — where this lane's band comes from, and what it can and cannot see.

    <venv>/bin/python floor_probe.py [--work work] [--max-cases N]

CPU only. No device, no lock, no timer: this file prints no ms / tok/s / GB/s figure and runs no warm
loop (`AGENTS.md` §"GPU execution is serialized", `pitfalls/00-measurement.md`).

THE CLAIM
  Every BAND-classed word in this lane is judged against   band = gamma_K * fsum_k |h_k w_k|
  with                                 gamma_K = K*u/(1 - K*u),  u = 2**-24,  K = h_used terms.
  A band printed as a bare number is a number the reader has to trust, and a band taken from what the
  current build happens to produce is not a band at all (`AGENTS.md`: "Bands, floors and must-catch
  sets are derived from a floor probe and from geometry, never from what the current build happens to
  produce"). So this file derives it from the PINNED rows and from the geometry, five ways:

  1. THE POPULATION. plan.tsv, slab.bin, hidden.f32 and ids.i32 are re-read from work/ and checked
     against a fresh build of the population (every column but the expect column must agree); the
     hidden vectors probed are the pinned layer39.combined_out rows — indices [0, real_hidden) of
     hidden.f32 — never the synthetic shapes.
  2. THE GEOMETRY. gamma_K's K is h_used, read from the plan. The FORM of the bound needs the products
     to be exact in binary32: bf16 x bf16 is <= 16 significant bits and a Q8_0 decode (f16 delta x
     signed 8-bit quant) is <= 19. Otherwise a rounded product adds an error term that
     gamma_K * sum|x| does not carry, and lm_head_kernels.h §the dots' "FMA contraction is neutral"
     argument collapses with it. That is MEASURED here over the artifact's own bytes, per family; one
     inexact product is a refusal.
  3. THE DATA. sigma = fsum |h_k w_k| and the correctly-rounded expectation are recomputed from the
     slab bytes through the audited decoder (tools/gguf_reader, reached via gate.Data.row_values) and
     required to be BIT-IDENTICAL to band.f64 and expect.f32. The band on disk is thereby SHOWN to be
     gamma_K * sigma from these bytes rather than asserted.
  4. THE FLOOR. Five associations of the SAME terms — ascending, descending, 8 lanes combined in index
     order, 32-block pairwise, 64-block pairwise — are executed in f32 over the pinned rows. Their
     deviation from the correctly-rounded sum must stay INSIDE the band (that is what earns it), must
     be NON-ZERO somewhere (a probe whose arms never differ proves nothing), and the band must sit at
     or above the f32 grid at the answer (band >= ulp(|exact|)) or the BAND class would be unachievable.
  5. THE TWO SABOTAGE DIRECTIONS, on the same words: the 1-ulp floor is witnessed by
     band >= ulp(|exact|) with the arms' own non-zero deviations (a comparator sitting in a dead branch
     prints all-zeros and is refused — `pitfalls/30-numerics.md`), and a perturbation the band MUST
     refuse is refused: dropping the last summed term must exceed the band wherever that term is live.
     The print also names the words where doubling every weight of a family is invisible — a word whose
     terms cancel cannot be moved by a uniform rescale, which is precisely why relL2 and not maxabs is
     this lane's primary metric (evidence/three-arm-host-2026-10-27.md §"why maxabs is not load-bearing").
"""
import argparse
import math
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate as G                                              # noqa: E402  the lane's own readers

U32 = 2.0 ** -24                                              # binary32 unit roundoff
ARMS = ("seq", "rev", "split8", "tree32", "tree64")
MAX_ROWS, DEEP_ROWS = 4, 32                                   # the gate's own defaults


class Refuse(Exception):
    pass


def gamma(K):
    if K * U32 >= 1.0:
        raise Refuse("gamma is undefined at K=%d: K*u >= 1, the bound says nothing" % K)
    return K * U32 / (1.0 - K * U32)


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def max_significant_bits(a64):
    """Smallest b with mantissa * 2^b integral, over every non-zero value (vectorised, no sampling: a
    MAX quoted from a sample would be a lie)."""
    if a64.size == 0:
        return 0
    m, _e = np.frexp(np.abs(a64))
    for b in range(1, 54):
        s = m * (1 << b)
        if bool(np.all(s == np.floor(s))):
            return b
    return 53


def walk(W, X, ks, acc):
    """acc += W[:,k] * X[:,k] in f32. The product is written to its own array before the add so no
    ufunc can contract multiply-add into a single-rounded FMA and quietly change the association."""
    for k in ks:
        prod = (W[:, k][:, None] * X[:, k][None, :]).astype(np.float32)
        acc += prod
    return acc


def arms_for(W, X, h):
    nR, B = W.shape[0], X.shape[0]
    z = lambda: np.zeros((nR, B), dtype=np.float32)           # noqa: E731  (five call sites, one shape)
    out = {}
    out["seq"] = walk(W, X, range(h), z())
    out["rev"] = walk(W[:, ::-1], X[:, ::-1], range(h), z())
    lanes, per = 8, h // 8
    if h % lanes:
        raise Refuse("h=%d is not divisible by the 8 probe lanes" % h)
    acc = z()
    for l in range(lanes):
        lo = l * per
        hi = h if l == lanes - 1 else lo + per
        acc += walk(W, X, range(lo, hi), z())                 # combine in lane index order
    out["split8"] = acc
    for name, block in (("tree32", 32), ("tree64", 64)):
        if h % block:
            raise Refuse("h=%d is not divisible by the %d-block probe" % (h, block))
        acc = z()
        for i in range(h // block):
            lo, hi = i * block, (i + 1) * block
            blk = W[:, lo:hi][:, None, :] * X[:, lo:hi][None, :, :]
            acc += blk.astype(np.float32).sum(axis=2, dtype=np.float32)   # pairwise inside the block
        out[name] = acc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(HERE, "work"))
    ap.add_argument("--max-cases", type=int, default=0, help="probe at most N honest cases (0 = all)")
    a = ap.parse_args()
    work = a.work

    fc, en, fields, modes, fmts = G.contract()
    cfg = G.read_config()
    p = G.build(cfg, fc, en, MAX_ROWS, DEEP_ROWS)

    # 1 ---- the bytes on disk are the population this probe reasons about -------------------------
    mine = G.write_plan(os.path.join(work, "floorplan.tsv"), p,
                        dict((c["idx"], "BAND" if c["mode"] == en["LMH_MODE_CORRECT"] else "FIRES")
                             for c in p.cases))
    disk = [l for l in open(os.path.join(work, "plan.tsv")).read().splitlines() if not l.startswith("#")]
    if len(mine) != len(disk):
        raise Refuse("work/plan.tsv has %d records, this build makes %d: the probe would be reading "
                     "bytes for a population that no longer exists" % (len(disk), len(mine)))
    moved = []
    for i, (x, y) in enumerate(zip(mine, disk)):
        fx, fy = x.split("\t"), y.split("\t")
        if len(fx) != len(fy):
            moved.append((i, "field count %d/%d" % (len(fx), len(fy))))
        elif fx[:17] != fy[:17] or fx[18:] != fy[18:]:
            moved.append((i, [k for k in range(len(fx)) if fx[k] != fy[k]][:6]))
    if moved:
        raise Refuse("work/plan.tsv differs from this build outside the expect column at %s"
                     % (moved[:4],))
    print("FLOORPOP plan_records=%d tables=%d cases=%d hidden=%d pinned_rows=%d synth_rows=%d h=%d "
          "head_dim=%d arms=%s matches_build=1"
          % (len(mine), len(p.tables), len(p.cases), p.n_hidden, p.real_hidden, p.synth_hidden,
             cfg["h"], cfg["head_dim"], ",".join(ARMS)))

    d = G.Data(work, p, fc, en)
    out_floats = p.out_floats
    expw = G.words_of(os.path.join(work, "expect.f32"), out_floats)
    bandw = G.doubles_of(os.path.join(work, "band.f64"), out_floats)

    cases = [c for c in p.cases
             if c["mode"] == en["LMH_MODE_CORRECT"]
             and p.tables[c["table"]]["real"] == 1 and p.tables[c["table"]]["kind"] == "real"
             and c["b0"] >= 0 and c["b0"] + c["bcount"] <= p.real_hidden]
    if a.max_cases:
        cases = cases[:a.max_cases]
    if not cases:
        raise Refuse("no honest case over a real table with pinned hidden rows exists: the population "
                     "does not contain the claim this file is here to make")

    n_anchor_bad = n_band_bad = n_dev_zero = 0
    worst_ratio, worst_where = 0.0, "none"
    fam, printed = {}, []
    for c in cases:
        t = p.tables[c["table"]]
        h_used = d.h_used(c)
        K, gam = h_used, gamma(h_used)
        rows = d.rows_of(c)
        slot = p.slot(c)
        vals = [d.row_values(t, r, h_used) for r in rows]
        W = np.array(vals, dtype=np.float32)
        X = np.array([d.hidden[c["b0"] + b] for b in range(c["bcount"])], dtype=np.float32)
        if W.shape[1] != h_used or X.shape[1] != h_used:
            raise Refuse("case %d: decoded width %d/%d != h_used %d"
                         % (c["idx"], W.shape[1], X.shape[1], h_used))
        # one product per (row, batch, k) triple: the same products the kernel forms. Exact in binary64
        # (<= 19 significant bits), so comparing them against the binary32 grid below is a measurement
        # and not a round-trip through the ruler being tested.
        P64 = W.astype(np.float64)[:, None, :] * X.astype(np.float64)[None, :, :]
        inexact = int(np.count_nonzero(P64.astype(np.float32).astype(np.float64) != P64))
        maxbits = max_significant_bits(P64[np.nonzero(P64)])
        # Where the bits come from, printed rather than argued: a bf16 table multiplies an 8-bit hidden
        # element by an 8-bit weight (16 bits, exact in f32); a Q8_0 table multiplies that same 8-bit
        # hidden element by a DECODED value that is itself f16-delta x 8-bit quant = up to 19 bits, so
        # the term needs 8 + 19 = 27 bits and is NOT exact in f32.
        wbits = max_significant_bits(W[np.nonzero(W)].astype(np.float64))
        hbits = max_significant_bits(X[np.nonzero(X)].astype(np.float64))
        # A recursive sum of K terms takes K-1 roundings when the products are exact and one more per
        # term when they are not (Higham): gamma_{K-1} versus gamma_{2K-1}. The oracle carries gamma_K
        # for every family, which is the right form for bf16 and one rounding short for Q8_0.
        gidx_req = (K - 1) if inexact == 0 else (2 * K - 1)
        f = fam.setdefault(t["idx"], dict(enc=t["enc"], K=h_used, words=0, pairs=0, inexact=0,
                                          maxbits=0, wbits=0, hbits=0, gidx=0, gamma_req=0.0,
                                          dev_max=0.0, dev_nonzero=0, rescale_blind=0, dropped=0,
                                          dropped_live=0, band_min=float("inf"),
                                          ulp_min=float("inf"), ratio_band_ulp=float("inf")))
        f["pairs"] += int(P64.size)
        f["inexact"] += inexact
        f["maxbits"] = max(f["maxbits"], maxbits)
        f["wbits"] = max(f["wbits"], wbits)
        f["hbits"] = max(f["hbits"], hbits)
        f["gidx"] = max(f["gidx"], gidx_req)
        f["gamma_req"] = max(f["gamma_req"], gamma(gidx_req))
        arms = arms_for(W, X, h_used)
        worst_case, band_min, ulp_min, ratio_bu, sigma_max = {}, float("inf"), float("inf"), \
            float("inf"), 0.0
        for jj in range(len(rows)):
            for b in range(c["bcount"]):
                at = c["out_off"] + b * slot + jj
                if at >= out_floats:
                    raise Refuse("case %d: word %d is outside out_floats=%d" % (c["idx"], at, out_floats))
                anchor = struct.unpack("<f", struct.pack("<I", expw[at]))[0]
                band = bandw[at]
                if not (band > 0.0):
                    raise Refuse("case %d word %d has band %r: a non-positive band bounds nothing"
                                 % (c["idx"], at, band))
                terms = [d.hidden[c["b0"] + b][k] * vals[jj][k] for k in range(h_used)]
                if f32(math.fsum(terms)) != anchor:
                    n_anchor_bad += 1
                sigma = math.fsum(abs(x) for x in terms)
                sigma_max = max(sigma_max, sigma)
                if gam * sigma != band:
                    n_band_bad += 1
                ulp = float(np.spacing(np.float32(abs(anchor)) if anchor != 0.0 else np.float32(0.0)))
                band_min = min(band_min, band)
                ulp_min = min(ulp_min, ulp)
                ratio_bu = min(ratio_bu, band / ulp)
                f["words"] += 1
                for nm in ARMS:
                    dev = abs(float(arms[nm][jj, b]) - anchor)
                    ratio = dev / band
                    worst_case[nm] = max(worst_case.get(nm, 0.0), ratio)
                    if ratio > worst_ratio:
                        worst_ratio, worst_where = ratio, "case %d word %d arm %s" % (c["idx"], at, nm)
                    if dev > f["dev_max"]:
                        f["dev_max"] = dev
                    if nm == "seq":
                        if dev > 0.0:
                            f["dev_nonzero"] += 1
                        else:
                            n_dev_zero += 1
                    if nm == "tree64":                         # the two sabotages, once per word
                        if terms[-1] != 0.0:
                            f["dropped_live"] += 1
                            if abs(f32(math.fsum(terms[:-1])) - anchor) > band:
                                f["dropped"] += 1
                        if abs(f32(math.fsum([2.0 * x for x in terms])) - anchor) <= band:
                            f["rescale_blind"] += 1
        f["band_min"] = min(f["band_min"], band_min)
        f["ulp_min"] = min(f["ulp_min"], ulp_min)
        f["ratio_band_ulp"] = min(f["ratio_band_ulp"], ratio_bu)
        printed.append("BANDPROBE case=%d table=%d enc=%d rows=%d bcount=%d K=%d u=2^-24 gamma=%.9g "
                       "gamma_required_index=%d sigma_max=%.9g band_min=%.9g words=%d ulp_min=%.9g "
                       "band_over_ulp_min=%.4g products=%d products_inexact=%d product_max_bits=%d "
                       "weight_max_bits=%d hidden_max_bits=%d %s"
                       % (c["idx"], t["idx"], t["enc"], len(rows), c["bcount"], K, gam, gidx_req,
                          sigma_max, band_min, len(rows) * c["bcount"], ulp_min, ratio_bu,
                          int(P64.size), inexact, maxbits, wbits, hbits,
                          " ".join("%s=%.4g" % (nm, worst_case.get(nm, 0.0)) for nm in ARMS)))
    for line in printed:
        print(line)

    for ti, f in sorted(fam.items()):
        print("FLOORSUM table=%d enc=%d words=%d term_pairs=%d products_inexact=%d "
              "product_max_bits=%d weight_max_bits=%d hidden_max_bits=%d gamma_index_used=%d "
              "gamma_index_required=%d gamma_required=%.9g max_arm_deviation=%.9g band_min=%.9g "
              "assoc_arms=%d arms_dev_nonzero_words=%d drop_last_live=%d drop_last_rejected=%d "
              "rescale_blind_words=%d ulp_min=%.9g band_over_ulp_min=%.4g"
              % (ti, f["enc"], f["words"], f["pairs"], f["inexact"], f["maxbits"], f["wbits"],
                 f["hbits"], f["K"], f["gidx"], f["gamma_req"], f["dev_max"], f["band_min"],
                 len(ARMS), f["dev_nonzero"], f["dropped_live"], f["dropped"], f["rescale_blind"],
                 f["ulp_min"], f["ratio_band_ulp"]))
    print("FLOORAGG probed_cases=%d printed_bandprobe=%d words=%d arms=%d dev_nonzero=%d dev_zero=%d "
          "worst_ratio=%.4g headroom_x=%.4g worst_at=%s band_over_ulp_min=%.4g "
          "anchor_recompute_bad=%d band_recompute_bad=%d"
          % (len(cases), len(printed), sum(f["words"] for f in fam.values()), len(ARMS),
             sum(f["dev_nonzero"] for f in fam.values()), n_dev_zero, worst_ratio,
             (1.0 / worst_ratio) if worst_ratio > 0 else float("inf"), worst_where,
             min(f["ratio_band_ulp"] for f in fam.values()), n_anchor_bad, n_band_bad))

    # a parse-back of what was just printed: a probe that printed nothing cannot fail either
    reparsed = [l for l in printed if l.startswith("BANDPROBE case=")]
    if len(reparsed) != len(printed) or len(printed) != len(cases):
        raise Refuse("printed %d BANDPROBE records for %d probed cases (parsed back %d): the count "
                     "that the verdict is taken over is not the count that was printed"
                     % (len(printed), len(cases), len(reparsed)))
    for l in reparsed[:1]:
        keys = dict(tok.split("=", 1) for tok in l.split() if "=" in tok)
        need = ("case", "K", "gamma", "gamma_required_index", "sigma_max", "band_min",
                "products_inexact", "weight_max_bits", "hidden_max_bits", "seq", "rev", "split8",
                "tree32", "tree64")
        if not all(k in keys for k in need):
            raise Refuse("BANDPROBE record is missing %s: the derivation is not readable off the print"
                         % [k for k in need if k not in keys])

    if n_anchor_bad:
        raise Refuse("%d expectation words are not fsum(terms) rounded to f32: expect.f32 was not made "
                     "by the arithmetic this band bounds" % n_anchor_bad)
    if n_band_bad:
        raise Refuse("%d bands are not gamma_K * fsum(|h_k w_k|) recomputed from these slab bytes"
                     % n_band_bad)

    bad = []
    for ti, f in sorted(fam.items()):
        if f["inexact"]:
            bad.append("table %d (enc=%d): %d of %d dot terms h_k*w_k are NOT exact in binary32 "
                       "(weight %d bits x hidden %d bits = up to %d). The DECODE is exact, the term is "
                       "not, so the bound needs gamma_{2K-1} (index %d, gamma=%.9g) where the oracle "
                       "carries gamma_K (index %d); lm_head_kernels.h's 'Products are exact' block "
                       "extends the exact-decode argument to the dot, which this measurement refutes. "
                       "Observed deviation %.3g against a band of %.3g, so this is a gap in the proof "
                       "and not a wrong number; closing it changes a band, which is not this file's "
                       "call to make"
                       % (ti, f["enc"], f["inexact"], f["pairs"], f["wbits"], f["hbits"], f["maxbits"],
                          f["gidx"], f["gamma_req"], f["K"], f["dev_max"], f["band_min"]))
        elif f["maxbits"] > 24:
            bad.append("table %d: every term compared exact yet one needs %d significant bits"
                       % (ti, f["maxbits"]))
        if f["words"] == 0:
            bad.append("table %d: no word probed, so nothing about its band is certified" % ti)
        if f["dev_nonzero"] == 0:
            bad.append("table %d: no re-associated arm ever moved a word — the probe is vacuous there, "
                       "and a comparator in a dead branch prints exactly this" % ti)
        if f["dropped_live"] == 0:
            bad.append("table %d: no word has a live last term, so the drop-last-term direction cannot "
                       "fire and is not evidence" % ti)
        elif f["dropped"] == 0:
            bad.append("table %d: dropping the last summed term exceeded the band on NONE of the %d "
                       "words where it was live — the band is too loose to refuse a term-sized error"
                       % (ti, f["dropped_live"]))
        if not (f["ratio_band_ulp"] >= 1.0):
            bad.append("table %d: some word's band (%.9g) is below one ulp of its own answer (%.9g) — "
                       "no f32 value could ever satisfy the BAND class there"
                       % (ti, f["band_min"], f["ulp_min"]))
    if worst_ratio > 1.0:
        bad.append("a legal re-association escaped the band at %s (ratio %.4g) — the band is not a "
                   "bound, and it was NOT loosened to make this pass" % (worst_where, worst_ratio))
    if worst_ratio == 0.0:
        bad.append("no arm ever moved a word anywhere in the population: the band's tightness is "
                   "unmeasured, which is the same hole as a check that cannot produce output")
    for b in bad:
        print("FINDING %s" % b)
    if bad:
        print("RESULT FAIL lm_head floor probe findings=%d" % len(bad))
        return 1
    print("RESULT PASS lm_head floor probe cases=%d words=%d arms=%d worst_ratio=%.4g "
          "band_over_ulp_min=%.4g products_exact=1"
          % (len(cases), sum(f["words"] for f in fam.values()), len(ARMS), worst_ratio,
             min(f["ratio_band_ulp"] for f in fam.values())))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refuse as e:
        print("REFUSE floor probe: %s" % e)
        sys.exit(1)
